//! Step 4 — mutual identity.
//!
//! Everything both peers need to trust each other is minted here and
//! written twice, once in each side's own dialect:
//!
//! - the node's `handoff.json` (schema 6: containerised exporter + DFlash),
//!   satisfying `muser_prefilld.py:load_config` field for field;
//! - this Mac's `cluster.json` (`ReceiverConfigV2`), at
//!   `~/.muser/nodes/<name>/cluster.json`.
//!
//! The two files are generated from one set of digests, so the pins, the
//! HMAC identity and the exact-identity block agree by construction rather
//! than by an operator copying twenty fields by hand.
//!
//! `lane_dir/generation.json` is the producer's replay ledger. It is created
//! only when absent and never overwritten: rewinding it would let an already
//! committed generation number be reused.

use std::io::{BufReader, Read as _};
use std::path::Path;

use muser_engine::dflash::{DFlashConfig, DFlashContextGeometry};
use serde_json::json;
use sha2::{Digest as _, Sha256};

use super::artifacts::{EXPORT_BINARY, MAX_CONTEXT};
use super::pki::{self, PRODUCER_SERVER_NAME, RECEIVER_SERVER_NAME};
use super::progress::{Status, Step};
use super::registry::{
    create_private_dir, node_dir, write_private, NodeEntry, ProducerKind, Registry, DAEMON_PORT,
    RECEIVER_PORT, STATE_ENROLLED, STATE_NEEDS_REENROLLMENT,
};
use super::{Ctx, Result};

/// Producer and receiver both cap a transfer here.
const TIMEOUT_SECONDS: i64 = 900;
const TIMEOUT_MS: i64 = 300_000;

#[derive(Debug, Clone, Copy)]
struct EnrolledDFlashIdentity {
    context: DFlashContextGeometry,
    kv_heads: usize,
    head_dim: usize,
}

const PROBE_DOCKER: &str = r#"set -eu
command -v docker
"#;

// The ledger body is written by the remote script itself. Remote arguments
// travel through a shell command line, so nothing quoted or spaced is ever
// passed as one.
const LEDGER_IF_ABSENT: &str = r#"set -eu
umask 077
if [ -e "$1/generation.json" ]; then
    printf 'present\n'
else
    printf '{"next_generation": 1}\n' > "$1/generation.json"
    printf 'created\n'
fi
"#;

const NODE_CSR: &str = r#"set -eu
umask 077
stage="$1/.enroll-v2-$2"
mkdir -p "$stage/pki"
chmod 700 "$stage" "$stage/pki"
if [ ! -f "$stage/pki/gx10.key.pem" ]; then
    openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$stage/pki/gx10.key.pem"
fi
chmod 600 "$stage/pki/gx10.key.pem"
openssl req -new -key "$stage/pki/gx10.key.pem" -subj /CN=muser-prefilld -out "$stage/pki/gx10.csr.pem"
cat "$stage/pki/gx10.csr.pem"
"#;

const ACTIVATE_ENROLLMENT: &str = r#"set -eu
lane="$1"
run="$2"
stage="$lane/.enroll-v2-$run"
test -f "$stage/pki/gx10.key.pem"
test -f "$stage/pki/gx10.cert.pem"
test -f "$stage/pki/ca.cert.pem"
test -f "$stage/pki/hmac.key"
test -f "$stage/handoff.json"
test -f "$stage/container.json"
openssl verify -CAfile "$stage/pki/ca.cert.pem" "$stage/pki/gx10.cert.pem" >/dev/null
openssl pkey -in "$stage/pki/gx10.key.pem" -pubout -out "$stage/key.pub"
openssl x509 -in "$stage/pki/gx10.cert.pem" -pubkey -noout > "$stage/cert.pub"
cmp "$stage/key.pub" "$stage/cert.pub"
rm -f "$stage/key.pub" "$stage/cert.pub"
chmod 700 "$stage" "$stage/pki"
chmod 600 "$stage/pki/gx10.key.pem" "$stage/pki/hmac.key" "$stage/handoff.json"
chmod 644 "$stage/pki/ca.cert.pem" "$stage/pki/gx10.cert.pem" "$stage/container.json"
for name in handoff.json pki container.json; do
    if [ -e "$lane/$name" ] && [ ! -L "$lane/$name" ]; then
        test ! -e "$lane/$name.pre-v2"
        mv "$lane/$name" "$lane/$name.pre-v2"
    fi
done
rm -f "$lane/.enrollment-active.next-$run" "$lane/.handoff.next-$run" "$lane/.pki.next-$run" "$lane/.container.next-$run"
ln -s ".enrollment-active/handoff.json" "$lane/.handoff.next-$run"
ln -s ".enrollment-active/pki" "$lane/.pki.next-$run"
ln -s ".enrollment-active/container.json" "$lane/.container.next-$run"
mv -Tf "$lane/.handoff.next-$run" "$lane/handoff.json"
mv -Tf "$lane/.pki.next-$run" "$lane/pki"
mv -Tf "$lane/.container.next-$run" "$lane/container.json"
ln -s ".enroll-v2-$run" "$lane/.enrollment-active.next-$run"
mv -Tf "$lane/.enrollment-active.next-$run" "$lane/.enrollment-active"
printf 'activated\n'
"#;

pub fn run(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    // Exhaustive recipe selection happens before SSH or key mutation. A
    // registry containing an unknown serialized lane is rejected by
    // `Registry::load` before this function can be reached.
    let producer = entry.producer_kind();
    if producer == ProducerKind::Native {
        return run_native(ctx, entry);
    }
    let recipe = producer.qualification_recipe();
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    let local = node_dir(&ctx.muser_home, &entry.name);
    let pki_dir = local.join("pki");
    let release = ctx.release()?;
    let receipt = ctx.receipt()?;
    let ca = pki::Ca::paths(&ctx.muser_home);
    let next_epoch = entry.hmac_epoch.max(0) + 1;
    let run_id = format!("epoch-{next_epoch}");
    let key_id = format!("muser-{}-{}-e{next_epoch}", entry.name, today());

    ctx.progress.emit(
        Step::Enroll,
        Status::Start,
        &format!(
            "minting the lab PKI and both configs for recipe {}",
            recipe.public_name()
        ),
    );

    if ctx.dry_run {
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "create the 10-year lab CA at {} if it does not exist (O_EXCL on {})",
                ca.dir.display(),
                ca.key.display()
            ),
        );
        ctx.progress.plan_command(
            Step::Enroll,
            &format!(
                "issue the Mac leaf (CN={RECEIVER_SERVER_NAME}, serverAuth+clientAuth) under {}",
                pki_dir.display()
            ),
            &pki::openssl_argv(&["x509", "-req", "-CA", &ca.cert.display().to_string()]),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "generate the GX10 private key inside {lane}/.enroll-v2-{run_id}, retrieve only its CSR, validate it, and sign CN={PRODUCER_SERVER_NAME}"
            ),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "read 32 bytes from /dev/urandom into {} as key id {key_id}, epoch {next_epoch}",
                pki_dir.join("hmac.key").display()
            ),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "stage {}/.enroll-v2-{run_id}/handoff.json (schema 6, image {}, adapter {})",
                lane,
                receipt.image_id,
                &receipt.adapter_sha256[..12]
            ),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "write {} pinned to model {} at revision {}",
                local.join("cluster.json").display(),
                &release.target.sha256[..12],
                release.revision
            ),
        );
        ctx.progress.plan_command(
            Step::Enroll,
            &format!("install at 0600 and atomically activate {lane}/.enrollment-active"),
            &ssh.argv(&[&lane]),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!("create {lane}/generation.json only if it is absent"),
        );
        ctx.progress
            .plan(Step::Enroll, "finish without creating key material");
        return Ok(());
    }

    // Geometry is trusted only after the exact artifact named by the release
    // identity has been reverified. Do this before rotating any enrollment
    // state so a stale or modified sidecar cannot leave half-minted keys.
    let dflash = verified_dflash_identity(ctx, &release)?;
    ctx.progress.emit_data(
        Step::Enroll,
        Status::Info,
        &format!(
            "DFlash identity declares {} layers, {} elements/token, sink {}, window {}",
            dflash.context.layers,
            dflash.context.elements_per_token,
            dflash.context.sink_size,
            dflash.context.window_size
        ),
        json!({
            "dflash_identity_sha256": release.dflash.sha256,
            "dflash_context_geometry": dflash.context,
        }),
    );

    // Persist this before rotating any local secret. A crash or interrupted
    // activation can therefore never leave a formerly healthy registry entry
    // claiming health against half-rotated credentials.
    entry.touch(STATE_NEEDS_REENROLLMENT);
    entry.last_error = Some(format!("enrollment {run_id} has not completed activation"));
    let mut registry = Registry::load(&ctx.muser_home)?;
    registry.upsert(entry.clone());
    registry.save(&ctx.muser_home)?;

    let (ca, minted) = pki::ensure_ca(&ctx.muser_home)?;
    ctx.progress.emit(
        Step::Enroll,
        Status::Info,
        &if minted {
            format!("minted the lab CA at {}", ca.dir.display())
        } else {
            format!("reusing the lab CA at {}", ca.dir.display())
        },
    );

    create_private_dir(&pki_dir)?;
    let mac = pki::issue_leaf(
        &ca,
        &pki_dir,
        &format!("mac-e{next_epoch}"),
        RECEIVER_SERVER_NAME,
    )?;
    // The node creates its own private key in a resumable staging directory.
    // SSH returns only the signed CSR; private key bytes never leave GX10.
    let csr_pem = ssh.run(NODE_CSR, &[&lane, &run_id])?;
    let csr = pki_dir.join(format!("gx10-e{next_epoch}.csr.pem"));
    write_private(&csr, csr_pem.as_bytes())?;
    let node = pki::sign_csr(
        &ca,
        &pki_dir,
        &format!("gx10-e{next_epoch}"),
        PRODUCER_SERVER_NAME,
        &csr,
    )?;
    ctx.progress.emit_data(
        Step::Enroll,
        Status::Info,
        &format!("leaf pins mac={} node={}", &mac.pin[..12], &node.pin[..12]),
        json!({ "mac_leaf_sha256": mac.pin, "node_leaf_sha256": node.pin }),
    );

    let hmac_key = pki_dir.join("hmac.key");
    pki::mint_hmac_key(&hmac_key)?;
    ctx.progress.emit(
        Step::Enroll,
        Status::Info,
        &format!("HMAC key {key_id} at epoch {next_epoch}"),
    );

    let container_runtime = ssh.run(PROBE_DOCKER, &[])?.trim().to_string();
    if container_runtime.is_empty() {
        return Err("the node no longer reports a docker binary — rerun preflight".into());
    }

    let handoff = handoff_config(
        &release,
        &receipt,
        &container_runtime,
        &mac.pin,
        &key_id,
        next_epoch,
        dflash,
    );
    let handoff_local = local.join("handoff.json");
    write_private(&handoff_local, pretty(&handoff)?.as_bytes())?;

    // This Mac's receiver config. `producer_control` is what lets the Mac
    // *drive* a prefill instead of waiting for an unsolicited producer; it
    // needs an address, so the node's host is resolved here.
    let control = resolve_control(entry.connect_host.as_deref().unwrap_or(&entry.host));
    let advertised = match &control {
        Some(_) => Some(super::preflight::advertised_receiver_host(&ssh)?),
        None => {
            ctx.progress.emit(
                Step::Enroll,
                Status::Info,
                &format!(
                    "WARN: {} does not resolve to an address; the Mac cannot open the control \
                     channel and will only accept an unsolicited producer",
                    entry.host
                ),
            );
            None
        }
    };
    let cluster = cluster_config_value(
        &release,
        &receipt,
        &ca.cert,
        &mac,
        &node.pin,
        &hmac_key,
        &key_id,
        next_epoch,
        &local.join("replay.json"),
        control.as_deref().zip(advertised.as_deref()),
        producer,
        Some(dflash.context),
    );
    let cluster_local = local.join("cluster.json");
    write_private(&cluster_local, pretty(&cluster)?.as_bytes())?;
    ctx.progress.emit_data(
        Step::Enroll,
        Status::Info,
        &format!("wrote {}", cluster_local.display()),
        json!({ "cluster_config": cluster_local, "handoff_config": handoff_local }),
    );

    // Node-side install. The receipt travels too: the producer refuses to
    // arm an exporter whose receipt does not match its own config.
    let stage = format!("{lane}/.enroll-v2-{run_id}");
    ssh.scp(&ca.cert, &format!("{stage}/pki/ca.cert.pem"))?;
    ssh.scp(&node.cert, &format!("{stage}/pki/gx10.cert.pem"))?;
    ssh.scp(&hmac_key, &format!("{stage}/pki/hmac.key"))?;
    ssh.scp(&handoff_local, &format!("{stage}/handoff.json"))?;
    ssh.scp(&receipt.path, &format!("{stage}/container.json"))?;
    let activation = ssh.run(ACTIVATE_ENROLLMENT, &[&lane, &run_id])?;
    if activation.trim() != "activated" {
        return Err("node enrollment activation did not return its terminal marker".into());
    }

    let outcome = ssh.run(LEDGER_IF_ABSENT, &[&lane])?;
    ctx.progress.emit(
        Step::Enroll,
        Status::Info,
        &if outcome.trim() == "created" {
            format!("{lane}/generation.json created at generation 0")
        } else {
            format!("{lane}/generation.json already exists and was left alone")
        },
    );

    entry.pki_dir = pki_dir.display().to_string();
    entry.hmac_key_id = key_id.clone();
    entry.hmac_epoch = next_epoch;
    entry.enrollment_version = 2;
    entry.touch(STATE_ENROLLED);
    entry.last_error = None;
    ctx.progress.emit_data(
        Step::Enroll,
        Status::Ok,
        &format!("{} enrolled with key {key_id}", entry.name),
        json!({ "hmac_key_id": key_id, "hmac_epoch": next_epoch, "pki_dir": entry.pki_dir }),
    );
    Ok(())
}

fn run_native(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    let recipe = entry.producer_kind().qualification_recipe();
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    let local = node_dir(&ctx.muser_home, &entry.name);
    let pki_dir = local.join("pki");
    let identity = ctx.native_identity()?;
    let ca = pki::Ca::paths(&ctx.muser_home);
    let next_epoch = entry.hmac_epoch.max(0) + 1;
    let run_id = format!("epoch-{next_epoch}");
    let key_id = format!("muser-{}-{}-e{next_epoch}", entry.name, today());

    ctx.progress.emit(
        Step::Enroll,
        Status::Start,
        &format!(
            "minting the lab PKI and native configs for recipe {}",
            recipe.public_name()
        ),
    );
    if ctx.dry_run {
        ctx.progress.plan(
            Step::Enroll,
            &format!("create the lab CA at {} if absent", ca.dir.display()),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "generate the GX10 TLS key inside {lane}/.enroll-v2-{run_id}; only its CSR leaves the node"
            ),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "stage a native vLLM control config pinned to image {} and checkpoint {}",
                identity.image_id, identity.checkpoint_artifact_sha256
            ),
        );
        ctx.progress.plan(
            Step::Enroll,
            &format!(
                "write {} for native Mac artifact {}",
                local.join("cluster.json").display(),
                identity.consumer.sha256
            ),
        );
        ctx.progress
            .plan(Step::Enroll, "finish without creating key material");
        return Ok(());
    }

    // Verify the consumer artifact before rotating any enrollment state. Its
    // digest is the model identity the producer will put on every Begin.
    let consumer_path = ctx.model_dir()?.join(&identity.consumer.filename);
    crate::model::validate_configured_artifact(&consumer_path, &identity.consumer.sha256)
        .map_err(|error| format!("verify native consumer identity: {error}"))?;

    entry.touch(STATE_NEEDS_REENROLLMENT);
    entry.last_error = Some(format!("enrollment {run_id} has not completed activation"));
    let mut registry = Registry::load(&ctx.muser_home)?;
    registry.upsert(entry.clone());
    registry.save(&ctx.muser_home)?;

    let (ca, minted) = pki::ensure_ca(&ctx.muser_home)?;
    ctx.progress.emit(
        Step::Enroll,
        Status::Info,
        &if minted {
            format!("minted the lab CA at {}", ca.dir.display())
        } else {
            format!("reusing the lab CA at {}", ca.dir.display())
        },
    );
    create_private_dir(&pki_dir)?;
    let mac = pki::issue_leaf(
        &ca,
        &pki_dir,
        &format!("mac-e{next_epoch}"),
        RECEIVER_SERVER_NAME,
    )?;
    let csr_pem = ssh.run(NODE_CSR, &[&lane, &run_id])?;
    let csr = pki_dir.join(format!("gx10-e{next_epoch}.csr.pem"));
    write_private(&csr, csr_pem.as_bytes())?;
    let node = pki::sign_csr(
        &ca,
        &pki_dir,
        &format!("gx10-e{next_epoch}"),
        PRODUCER_SERVER_NAME,
        &csr,
    )?;
    ctx.progress.emit_data(
        Step::Enroll,
        Status::Info,
        &format!("leaf pins mac={} node={}", &mac.pin[..12], &node.pin[..12]),
        json!({ "mac_leaf_sha256": mac.pin, "node_leaf_sha256": node.pin }),
    );

    let hmac_key = pki_dir.join("hmac.key");
    pki::mint_hmac_key(&hmac_key)?;
    let container_runtime = ssh.run(PROBE_DOCKER, &[])?.trim().to_string();
    if container_runtime.is_empty() {
        return Err("the node no longer reports a docker binary — rerun preflight".into());
    }
    let checkpoint = format!("{lane}/models/{}", identity.checkpoint_directory);
    let container_name = format!("muser-native-{}", entry.name);
    let handoff = native_handoff_config(
        &identity,
        &container_runtime,
        &container_name,
        &checkpoint,
        &mac.pin,
        &key_id,
        next_epoch,
    );
    let handoff_local = local.join("handoff.json");
    write_private(&handoff_local, pretty(&handoff)?.as_bytes())?;

    let control = resolve_control(entry.connect_host.as_deref().unwrap_or(&entry.host));
    let advertised = match &control {
        Some(_) => Some(super::preflight::advertised_receiver_host(&ssh)?),
        None => {
            ctx.progress.emit(
                Step::Enroll,
                Status::Info,
                &format!(
                    "WARN: {} does not resolve; the Mac cannot open native producer control",
                    entry.host
                ),
            );
            None
        }
    };
    let cluster = native_cluster_config_value(
        &identity,
        &ca.cert,
        &mac,
        &node.pin,
        &hmac_key,
        &key_id,
        next_epoch,
        &local.join("replay.json"),
        control.as_deref().zip(advertised.as_deref()),
    );
    let cluster_local = local.join("cluster.json");
    write_private(&cluster_local, pretty(&cluster)?.as_bytes())?;

    let stage = format!("{lane}/.enroll-v2-{run_id}");
    ssh.scp(&ca.cert, &format!("{stage}/pki/ca.cert.pem"))?;
    ssh.scp(&node.cert, &format!("{stage}/pki/gx10.cert.pem"))?;
    ssh.scp(&hmac_key, &format!("{stage}/pki/hmac.key"))?;
    ssh.scp(&handoff_local, &format!("{stage}/handoff.json"))?;
    ssh.scp(&identity.path, &format!("{stage}/container.json"))?;
    let activation = ssh.run(ACTIVATE_ENROLLMENT, &[&lane, &run_id])?;
    if activation.trim() != "activated" {
        return Err("node enrollment activation did not return its terminal marker".into());
    }
    let outcome = ssh.run(LEDGER_IF_ABSENT, &[&lane])?;
    ctx.progress.emit(
        Step::Enroll,
        Status::Info,
        &if outcome.trim() == "created" {
            format!("{lane}/generation.json created at generation 0")
        } else {
            format!("{lane}/generation.json already exists and was left alone")
        },
    );

    entry.pki_dir = pki_dir.display().to_string();
    entry.hmac_key_id = key_id.clone();
    entry.hmac_epoch = next_epoch;
    entry.enrollment_version = 2;
    entry.touch(STATE_ENROLLED);
    entry.last_error = None;
    ctx.progress.emit_data(
        Step::Enroll,
        Status::Ok,
        &format!("{} enrolled with native key {key_id}", entry.name),
        json!({
            "hmac_key_id": key_id,
            "hmac_epoch": next_epoch,
            "producer_mode": "native",
            "checkpoint_artifact_sha256": identity.checkpoint_artifact_sha256,
            "consumer_sha256": identity.consumer.sha256,
        }),
    );
    Ok(())
}

fn native_handoff_config(
    identity: &super::artifacts::NativeIdentity,
    container_runtime: &str,
    container_name: &str,
    checkpoint_dir: &str,
    mac_pin: &str,
    key_id: &str,
    hmac_epoch: i64,
) -> serde_json::Value {
    json!({
        "schema": "muser.native-prefilld.v1",
        "schema_version": 1,
        "listen_host": "0.0.0.0",
        "listen_port": DAEMON_PORT,
        "certificate_chain": "pki/gx10.cert.pem",
        "private_key": "pki/gx10.key.pem",
        "peer_ca": "pki/ca.cert.pem",
        "peer_leaf_sha256": [mac_pin],
        "receiver_server_name": RECEIVER_SERVER_NAME,
        "receiver_leaf_sha256": mac_pin,
        "hmac_key_file": "pki/hmac.key",
        "hmac_key_id": key_id,
        "hmac_epoch": hmac_epoch,
        "generation_ledger": "generation.json",
        "work_dir": "work",
        "container_runtime": container_runtime,
        "container_image": identity.image_id,
        "container_name": container_name,
        "runtime_identity": ".enrollment-active/container.json",
        "checkpoint_dir": checkpoint_dir,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_context": MAX_CONTEXT,
        "checkpoint_artifact_sha256": identity.checkpoint_artifact_sha256,
        "checkpoint_revision": identity.checkpoint_revision,
        "model_sha256": identity.consumer.sha256,
        "model_revision": identity.checkpoint_revision,
        "tokenizer_revision": identity.checkpoint_revision,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "chat_template_sha256": identity.chat_template_sha256,
        "context_policy_sha256": identity.context_policy_sha256,
        "adapter_sha256": identity.adapter_sha256,
        "target_cache_identity_sha256": identity.target_cache_identity_sha256,
        "vllm_commit": identity.vllm_commit,
        "producer_socket": "work/producer.sock",
        "startup_receipt": "work/native-startup-receipt.json",
        "rope_cache_output": "work/native-rope-cache-f32le.bin",
        "rope_cache_bytes": identity.rope_cache_bytes,
        "rope_cache_sha256": identity.rope_cache_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
fn native_cluster_config_value(
    identity: &super::artifacts::NativeIdentity,
    ca_cert: &Path,
    mac: &pki::Leaf,
    node_pin: &str,
    hmac_key: &Path,
    key_id: &str,
    hmac_epoch: i64,
    replay_ledger: &Path,
    control: Option<(&str, &str)>,
) -> serde_json::Value {
    let mut value = json!({
        "schema_version": 1,
        "listen": format!("0.0.0.0:{RECEIVER_PORT}"),
        "certificate_chain": mac.cert,
        "private_key": mac.key,
        "peer_ca": ca_cert,
        "peer_leaf_sha256": [node_pin],
        "hmac_key_file": hmac_key,
        "hmac_key_id": key_id,
        "minimum_hmac_epoch": hmac_epoch,
        "replay_ledger": replay_ledger,
        "timeout_ms": TIMEOUT_MS,
        "wait_for_producer_ms": TIMEOUT_MS,
        "producer_mode": "native",
        "identity": {
            "adapter_sha256": identity.adapter_sha256,
            "chat_template_sha256": identity.chat_template_sha256,
            "context_policy_sha256": identity.context_policy_sha256,
            "model_revision": identity.checkpoint_revision,
            "model_sha256": identity.consumer.sha256,
            "tokenizer_revision": identity.checkpoint_revision,
            "tokenizer_sha256": identity.tokenizer_sha256,
        },
        "target_cache_identity_sha256": identity.target_cache_identity_sha256,
    });
    if let Some((address, advertised)) = control {
        value["advertised_receiver_host"] = json!(advertised);
        value["producer_control"] = json!({
            "address": address,
            "server_name": PRODUCER_SERVER_NAME,
        });
    }
    value
}

/// The node's `handoff.json`, schema 6 — a containerised exporter with
/// DFlash. Every field `muser_prefilld.py:load_config` demands is present
/// and no field it does not know is added; its path fields are relative to
/// the config, which is what `load_config` resolves them against.
fn handoff_config(
    release: &super::artifacts::Release,
    receipt: &super::artifacts::ContainerReceipt,
    container_runtime: &str,
    mac_pin: &str,
    key_id: &str,
    hmac_epoch: i64,
    dflash: EnrolledDFlashIdentity,
) -> serde_json::Value {
    json!({
        "schema_version": 6,
        "listen_host": "0.0.0.0",
        "listen_port": DAEMON_PORT,
        "certificate_chain": "pki/gx10.cert.pem",
        "private_key": "pki/gx10.key.pem",
        "peer_ca": "pki/ca.cert.pem",
        "peer_leaf_sha256": [mac_pin],
        "receiver_server_name": RECEIVER_SERVER_NAME,
        "receiver_leaf_sha256": mac_pin,
        "hmac_key_file": "pki/hmac.key",
        "hmac_key_id": key_id,
        "hmac_epoch": hmac_epoch,
        "generation_ledger": "generation.json",
        "work_dir": "work",
        "export_binary": EXPORT_BINARY,
        "container_runtime": container_runtime,
        "container_image": receipt.image_id,
        // The public `container.json` name is an atomic activation symlink.
        // Point through the active-generation directory so the final path is
        // the retained regular receipt; the producer deliberately rejects a
        // symlink as the receipt itself.
        "container_receipt": ".enrollment-active/container.json",
        "sender_script": "llamacpp/muser_v2_send.py",
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_context": MAX_CONTEXT,
        "model_sha256": release.target.sha256,
        "model_revision": release.revision,
        "tokenizer_revision": release.revision,
        "tokenizer_sha256": release.tokenizer_sha256,
        "chat_template_sha256": release.chat_template_sha256,
        "context_policy_sha256": release.context_policy_sha256,
        "adapter_sha256": receipt.adapter_sha256,
        "target_cache_identity_sha256": release.target_cache_identity_sha256,
        // The DFlash artifact is a single GGUF, so its file digest is also
        // its component identity (see `state.rs:dflash_identity`).
        "dflash_identity_sha256": release.dflash.sha256,
        "dflash_gguf_sha256": release.dflash.sha256,
        "dflash_kv_heads": dflash.kv_heads,
        "dflash_head_dim": dflash.head_dim,
        "dflash_context_geometry": dflash.context,
    })
}

/// This Mac's `cluster.json` (`ReceiverConfigV2`), built from the same
/// digests as the handoff config above so the two agree by construction.
/// The native NVFP4 lane is the deliberate exception: it ships plain decode
/// only, so it names `producer_mode` and enrolls no DFlash identity at all.
#[allow(clippy::too_many_arguments)]
fn cluster_config_value(
    release: &super::artifacts::Release,
    receipt: &super::artifacts::ContainerReceipt,
    ca_cert: &Path,
    mac: &pki::Leaf,
    node_pin: &str,
    hmac_key: &Path,
    key_id: &str,
    hmac_epoch: i64,
    replay_ledger: &Path,
    control: Option<(&str, &str)>,
    producer: ProducerKind,
    dflash_geometry: Option<DFlashContextGeometry>,
) -> serde_json::Value {
    let mut value = json!({
        "schema_version": 1,
        "listen": format!("0.0.0.0:{RECEIVER_PORT}"),
        "certificate_chain": mac.cert,
        "private_key": mac.key,
        "peer_ca": ca_cert,
        "peer_leaf_sha256": [node_pin],
        "hmac_key_file": hmac_key,
        "hmac_key_id": key_id,
        "minimum_hmac_epoch": hmac_epoch,
        "replay_ledger": replay_ledger,
        "timeout_ms": TIMEOUT_MS,
        "wait_for_producer_ms": TIMEOUT_MS,
        "identity": {
            "adapter_sha256": receipt.adapter_sha256,
            "chat_template_sha256": release.chat_template_sha256,
            "context_policy_sha256": release.context_policy_sha256,
            "model_revision": release.revision,
            "model_sha256": release.target.sha256,
            "tokenizer_revision": release.revision,
            "tokenizer_sha256": release.tokenizer_sha256,
        },
        "target_cache_identity_sha256": release.target_cache_identity_sha256,
    });
    match producer {
        ProducerKind::Llamacpp => {
            // Combined target+DFlash transfers: the receiver pins the DFlash
            // identity so admission refuses a cache built by any other one.
            value["dflash_identity_sha256"] = json!(release.dflash.sha256);
            value["dflash_context_geometry"] = json!(
                dflash_geometry.expect("combined enrollment supplies verified DFlash geometry")
            );
        }
        ProducerKind::Native => {
            // Fallback B: the native lane refuses DFlash at serve time, so
            // no DFlash identity is enrolled here either.
            value["producer_mode"] = json!("native");
        }
    }
    if let Some((address, advertised)) = control {
        value["advertised_receiver_host"] = json!(advertised);
        value["producer_control"] = json!({
            "address": address,
            "server_name": PRODUCER_SERVER_NAME,
        });
    }
    value
}

/// Reverify and parse the exact sidecar whose digest enrollment will pin.
fn verified_dflash_identity(
    ctx: &Ctx,
    release: &super::artifacts::Release,
) -> Result<EnrolledDFlashIdentity> {
    let path = ctx.model_dir()?.join(&release.dflash.filename);
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|error| format!("inspect pinned DFlash sidecar {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "pinned DFlash sidecar is not a regular file: {} (symlinks are \
             rejected — copy the artifact to this path)",
            path.display()
        ));
    }
    if metadata.len() != release.dflash.bytes {
        return Err(format!(
            "pinned DFlash sidecar byte count mismatch: expected {}, got {}",
            release.dflash.bytes,
            metadata.len()
        ));
    }
    let file = std::fs::File::open(&path)
        .map_err(|error| format!("open pinned DFlash sidecar {}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("hash pinned DFlash sidecar {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != release.dflash.sha256 {
        return Err(format!(
            "pinned DFlash sidecar SHA-256 mismatch: expected {}, got {actual}",
            release.dflash.sha256
        ));
    }
    let config = DFlashConfig::from_artifact(&path)
        .map_err(|error| format!("parse verified DFlash sidecar identity: {error}"))?;
    let context = config.context_geometry();
    context.validate()?;
    Ok(EnrolledDFlashIdentity {
        context,
        kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
    })
}

/// `producer_control.address` is a socket address, so the node's host has to
/// resolve here and now. An unresolvable host is not fatal — it just costs
/// the control channel.
fn resolve_control(host: &str) -> Option<String> {
    use std::net::ToSocketAddrs;
    (host, DAEMON_PORT)
        .to_socket_addrs()
        .ok()?
        .next()
        .map(|address| address.to_string())
}

/// `hmac_key_id` carries the date it was minted: `muser-<name>-<YYYYMMDD>`.
fn today() -> String {
    crate::timefmt::now_rfc3339()
        .chars()
        .take(10)
        .filter(|character| *character != '-')
        .collect()
}

fn pretty(value: &serde_json::Value) -> Result<String> {
    serde_json::to_string_pretty(value).map_err(|error| format!("encode config: {error}"))
}

/// The local half of an enrolment, for the steps that follow it.
pub fn cluster_config(home: &Path, name: &str) -> std::path::PathBuf {
    node_dir(home, name).join("cluster.json")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use muser_cluster::config::ReceiverConfigV2;

    #[test]
    fn enrollment_v2_remote_scripts_are_valid_shell() {
        for script in [NODE_CSR, ACTIVATE_ENROLLMENT] {
            let mut child = std::process::Command::new("bash")
                .arg("-n")
                .stdin(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            use std::io::Write as _;
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(script.as_bytes())
                .unwrap();
            assert!(child.wait().unwrap().success());
        }
    }

    #[test]
    fn the_key_id_carries_a_compact_mint_date() {
        let id = format!("muser-gx10-{}-e2", today());
        assert_eq!(id.len(), "muser-gx10-".len() + 8 + "-e2".len());
        assert!(today().chars().all(|c| c.is_ascii_digit()));
    }

    /// The whole point of this step is that both files agree. A field that
    /// drifts on one side and not the other is a prefill that fails
    /// admission on the node, hours later.
    #[test]
    fn both_configs_are_minted_from_one_set_of_digests() {
        let (release, receipt) = fixtures();
        let handoff = handoff_config(
            &release,
            &receipt,
            "/usr/bin/docker",
            &"a".repeat(64),
            "k",
            2,
            fixture_dflash(),
        );
        let mac = pki::Leaf {
            key: PathBuf::from("/pki/mac.key.pem"),
            cert: PathBuf::from("/pki/mac.cert.pem"),
            pin: "a".repeat(64),
        };
        let cluster = cluster_config_value(
            &release,
            &receipt,
            Path::new("/pki/ca.cert.pem"),
            &mac,
            &"b".repeat(64),
            Path::new("/pki/hmac.key"),
            "k",
            2,
            Path::new("/replay.json"),
            Some(("10.0.0.9:29591", "10.0.0.2")),
            ProducerKind::Llamacpp,
            Some(fixture_dflash().context),
        );
        for field in [
            "model_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "context_policy_sha256",
            "adapter_sha256",
            "model_revision",
            "tokenizer_revision",
        ] {
            assert_eq!(
                handoff[field], cluster["identity"][field],
                "{field} differs between the two configs"
            );
        }
        assert_eq!(
            handoff["target_cache_identity_sha256"],
            cluster["target_cache_identity_sha256"]
        );
        assert_eq!(
            handoff["dflash_identity_sha256"],
            cluster["dflash_identity_sha256"]
        );
        assert_eq!(
            handoff["dflash_context_geometry"],
            cluster["dflash_context_geometry"]
        );
        assert_eq!(handoff["hmac_key_id"], cluster["hmac_key_id"]);
        assert_eq!(handoff["hmac_epoch"], cluster["minimum_hmac_epoch"]);
        // Each side pins the *other* leaf.
        assert_eq!(handoff["peer_leaf_sha256"][0], json!(mac.pin));
        assert_eq!(cluster["peer_leaf_sha256"][0], json!("b".repeat(64)));
    }

    /// The handoff config carries exactly the schema-6 field set
    /// `muser_prefilld.py:load_config` accepts — no more, no less.
    #[test]
    fn the_handoff_config_matches_the_producers_schema_six_field_set() {
        let (release, receipt) = fixtures();
        let handoff = handoff_config(
            &release,
            &receipt,
            "/usr/bin/docker",
            &"a".repeat(64),
            "k",
            2,
            fixture_dflash(),
        );
        let mut actual = handoff
            .as_object()
            .expect("object")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        actual.sort();
        let mut expected = vec![
            "schema_version",
            "listen_host",
            "listen_port",
            "certificate_chain",
            "private_key",
            "peer_ca",
            "peer_leaf_sha256",
            "receiver_server_name",
            "receiver_leaf_sha256",
            "hmac_key_file",
            "hmac_key_id",
            "hmac_epoch",
            "generation_ledger",
            "work_dir",
            "export_binary",
            "container_runtime",
            "container_image",
            "container_receipt",
            "sender_script",
            "timeout_seconds",
            "max_context",
            "model_sha256",
            "model_revision",
            "tokenizer_revision",
            "tokenizer_sha256",
            "chat_template_sha256",
            "context_policy_sha256",
            "adapter_sha256",
            "target_cache_identity_sha256",
            "dflash_identity_sha256",
            "dflash_gguf_sha256",
            "dflash_kv_heads",
            "dflash_head_dim",
            "dflash_context_geometry",
        ];
        expected.sort_unstable();
        assert_eq!(actual, expected);
        assert_eq!(handoff["export_binary"], json!(EXPORT_BINARY));
        assert_eq!(handoff["container_image"], json!(receipt.image_id));
    }

    /// The generated `cluster.json` is fed to the real loader, with real
    /// minted certificates, so a shape this Mac cannot serve is caught here
    /// rather than at the first prefill.
    #[test]
    fn the_generated_cluster_config_loads_in_the_production_receiver() {
        let Some(home) = temp_home("cluster-loads") else {
            return;
        };
        let Ok((ca, _)) = pki::ensure_ca(&home) else {
            eprintln!("openssl unavailable; skipping");
            return;
        };
        let pki_dir = home.join("pki");
        let mac = pki::issue_leaf(&ca, &pki_dir, "mac", RECEIVER_SERVER_NAME).expect("mac leaf");
        let node = pki::issue_leaf(&ca, &pki_dir, "gx10", PRODUCER_SERVER_NAME).expect("node leaf");
        assert_ne!(mac.pin, node.pin, "each side must present its own leaf");
        assert_eq!(mac.pin.len(), 64);
        let hmac_key = pki_dir.join("hmac.key");
        pki::mint_hmac_key(&hmac_key).expect("hmac key");

        let (release, receipt) = fixtures();
        let value = cluster_config_value(
            &release,
            &receipt,
            &ca.cert,
            &mac,
            &node.pin,
            &hmac_key,
            "muser-test-20260813",
            1,
            &home.join("replay.json"),
            Some(("127.0.0.1:29591", "127.0.0.1")),
            ProducerKind::Llamacpp,
            Some(fixture_dflash().context),
        );
        let path = home.join("cluster.json");
        write_private(&path, pretty(&value).unwrap().as_bytes()).expect("write");
        let config =
            ReceiverConfigV2::load(&path).expect("the receiver must accept its own config");
        assert_eq!(config.listen.port(), RECEIVER_PORT);
        assert_eq!(config.minimum_hmac_epoch, 1);
        assert!(config.peer_leaf_sha256.contains(&node.pin));
        assert_eq!(config.identity.model_sha256, release.target.sha256);
        // The default lane enrolls the DFlash identity and names no mode.
        assert_eq!(
            config.dflash_identity_sha256.as_deref(),
            Some(release.dflash.sha256.as_str())
        );
        assert_eq!(config.producer_mode, None);
        assert_eq!(
            config.dflash_context_geometry,
            Some(fixture_dflash().context)
        );
        assert_eq!(
            config.producer_control.expect("control").server_name,
            PRODUCER_SERVER_NAME
        );
        let mut stale = value;
        stale
            .as_object_mut()
            .unwrap()
            .remove("dflash_context_geometry");
        write_private(&path, pretty(&stale).unwrap().as_bytes()).expect("write stale config");
        let error = ReceiverConfigV2::load(&path).unwrap_err();
        assert!(error.contains("declared together"), "{error}");
        let _ = std::fs::remove_dir_all(&home);
    }

    /// The native NVFP4 lane enrolls plain decode only: the receiver config
    /// names `producer_mode` and carries no DFlash identity, so a native
    /// cache is never mistaken for a combined one at admission time.
    #[test]
    fn the_native_cluster_config_names_its_mode_and_omits_dflash() {
        let (release, receipt) = fixtures();
        let mac = pki::Leaf {
            key: PathBuf::from("/pki/mac.key.pem"),
            cert: PathBuf::from("/pki/mac.cert.pem"),
            pin: "a".repeat(64),
        };
        let cluster = cluster_config_value(
            &release,
            &receipt,
            Path::new("/pki/ca.cert.pem"),
            &mac,
            &"b".repeat(64),
            Path::new("/pki/hmac.key"),
            "k",
            2,
            Path::new("/replay.json"),
            Some(("10.0.0.9:29591", "10.0.0.2")),
            ProducerKind::Native,
            None,
        );
        assert_eq!(cluster["producer_mode"], json!("native"));
        assert!(cluster.get("dflash_identity_sha256").is_none());
        assert!(cluster.get("dflash_context_geometry").is_none());
        // Every other pin still comes from the same set of digests.
        assert_eq!(
            cluster["identity"]["model_sha256"],
            json!(release.target.sha256)
        );

        let Some(home) = temp_home("cluster-loads-native") else {
            return;
        };
        let Ok((ca, _)) = pki::ensure_ca(&home) else {
            eprintln!("openssl unavailable; skipping");
            return;
        };
        let pki_dir = home.join("pki");
        let mac = pki::issue_leaf(&ca, &pki_dir, "mac", RECEIVER_SERVER_NAME).expect("mac leaf");
        let node = pki::issue_leaf(&ca, &pki_dir, "gx10", PRODUCER_SERVER_NAME).expect("node leaf");
        let hmac_key = pki_dir.join("hmac.key");
        pki::mint_hmac_key(&hmac_key).expect("hmac key");
        let value = cluster_config_value(
            &release,
            &receipt,
            &ca.cert,
            &mac,
            &node.pin,
            &hmac_key,
            "muser-test-20260819",
            1,
            &home.join("replay.json"),
            Some(("127.0.0.1:29591", "127.0.0.1")),
            ProducerKind::Native,
            None,
        );
        let path = home.join("cluster.json");
        write_private(&path, pretty(&value).unwrap().as_bytes()).expect("write");
        let config =
            ReceiverConfigV2::load(&path).expect("the receiver must accept its own native config");
        assert_eq!(
            config.producer_mode,
            Some(muser_cluster::config::Nvfp4ProducerMode::Native)
        );
        assert_eq!(config.dflash_identity_sha256, None);
        assert_eq!(config.dflash_context_geometry, None);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn native_configs_bind_the_frozen_checkpoint_image_and_consumer() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap();
        let identity = super::super::artifacts::NativeIdentity::load(root).unwrap();
        let handoff = native_handoff_config(
            &identity,
            "/usr/bin/docker",
            "muser-native-fixture",
            "/home/muser/.muser/lane/fixture/models/checkpoint",
            &"a".repeat(64),
            "fixture-key",
            1,
        );
        assert_eq!(handoff["schema"], json!("muser.native-prefilld.v1"));
        assert_eq!(handoff["container_image"], json!(identity.image_id));
        assert_eq!(
            handoff["checkpoint_artifact_sha256"],
            json!(identity.checkpoint_artifact_sha256)
        );
        assert_eq!(handoff["model_sha256"], json!(identity.consumer.sha256));
        assert!(handoff.get("dflash_identity_sha256").is_none());

        let Some(home) = temp_home("cluster-loads-real-native") else {
            return;
        };
        let Ok((ca, _)) = pki::ensure_ca(&home) else {
            eprintln!("openssl unavailable; skipping");
            return;
        };
        let pki_dir = home.join("pki");
        let mac = pki::issue_leaf(&ca, &pki_dir, "mac", RECEIVER_SERVER_NAME).unwrap();
        let node = pki::issue_leaf(&ca, &pki_dir, "node", PRODUCER_SERVER_NAME).unwrap();
        let hmac = pki_dir.join("hmac.key");
        pki::mint_hmac_key(&hmac).unwrap();
        let value = native_cluster_config_value(
            &identity,
            &ca.cert,
            &mac,
            &node.pin,
            &hmac,
            "fixture-key",
            1,
            &home.join("replay.json"),
            Some(("127.0.0.1:29591", "127.0.0.1")),
        );
        let path = home.join("cluster.json");
        write_private(&path, pretty(&value).unwrap().as_bytes()).unwrap();
        let loaded = ReceiverConfigV2::load(&path).unwrap();
        assert_eq!(
            loaded.producer_mode,
            Some(muser_cluster::config::Nvfp4ProducerMode::Native)
        );
        assert_eq!(loaded.identity.model_sha256, identity.consumer.sha256);
        assert_eq!(loaded.identity.adapter_sha256, identity.adapter_sha256);
        assert!(loaded.dflash_identity_sha256.is_none());
        assert_eq!(
            loaded.producer_control.unwrap().address.to_string(),
            "127.0.0.1:29591"
        );
        let _ = std::fs::remove_dir_all(home);
    }

    fn fixtures() -> (
        super::super::artifacts::Release,
        super::super::artifacts::ContainerReceipt,
    ) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root");
        let release = super::super::artifacts::Release::load(root).expect("release pins");
        let receipt = serde_json::from_value(json!({
            "schema": super::super::artifacts::RECEIPT_SCHEMA,
            "status": "built",
            "architecture": "arm64",
            "image_id": format!("sha256:{}", "c".repeat(64)),
            "image_tag": "muser-gx10-prefill:test",
            "adapter_sha256": "d".repeat(64),
            "cuda_matmul": "default",
            "entrypoint": [EXPORT_BINARY],
            "source_commit": "e".repeat(40),
        }))
        .expect("receipt");
        (release, receipt)
    }

    fn fixture_dflash() -> EnrolledDFlashIdentity {
        EnrolledDFlashIdentity {
            context: DFlashContextGeometry {
                layers: 5,
                elements_per_token: 8 * 128,
                sink_size: 64,
                window_size: 2_048,
            },
            kv_heads: 8,
            head_dim: 128,
        }
    }

    fn temp_home(label: &str) -> Option<PathBuf> {
        let home =
            std::env::temp_dir().join(format!("muser-node-enroll-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        create_private_dir(&home).ok()?;
        Some(home)
    }
}
