//! Authenticated, encrypted logical-session state and revision admission.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chacha20poly1305::aead::{Aead, KeyInit, OsRng};
use chacha20poly1305::{AeadCore, XChaCha20Poly1305, XNonce};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};

const MAX_LOGICAL_SESSIONS: usize = 64;
const MAGIC: &[u8; 16] = b"MUSER-SESSION-V3";
const MAX_TRANSFER_BYTES: u64 = 32 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SessionBundle {
    pub schema: String,
    pub session_id: String,
    pub revision: u64,
    pub context_epoch: u64,
    pub model_sha256: String,
    pub tokenizer_sha256: [u8; 32],
    pub template_sha256: [u8; 32],
    pub layout_abi: String,
    pub dflash_identity_sha256: Option<String>,
    pub vision_projector_sha256: Option<String>,
    pub vision_preprocessing_sha256: Option<String>,
    pub target: muser_engine::cache::SessionCacheSnapshot,
    pub target_logits: Vec<f32>,
    pub dflash: Option<muser_engine::dflash::DFlashContextSnapshot>,
    pub position_witnesses: Vec<u32>,
    pub rng_seed: u64,
    pub sampler_state: SamplerStateSnapshot,
    pub sampler_config_sha256: [u8; 32],
    pub sampler_history: Vec<u32>,
    pub detokenizer_pending: String,
    pub stop_matcher_pending: String,
    pub grammar_state: Option<crate::grammar::GrammarMatcher>,
    pub grammar_sha256: Option<[u8; 32]>,
    /// Canonical JSON is stored as text because Postcard intentionally does
    /// not implement Serde's self-describing `deserialize_any` data model.
    pub canonical_replay_plan_json: String,
    pub vision_rows: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SamplerStateSnapshot {
    pub distribution_rng: muser_engine::sampling::Mt19937Snapshot,
    pub xtc_rng: muser_engine::sampling::Mt19937Snapshot,
    pub mirostat_rng: muser_engine::sampling::Mt19937Snapshot,
    pub adaptive_rng: muser_engine::sampling::Mt19937Snapshot,
    pub mirostat_mu: f32,
    pub adaptive_weighted_sum: f32,
    pub adaptive_total_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CachedGeneration {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub seed: u64,
    pub revision: u64,
    pub context: Vec<u32>,
    #[serde(default)]
    pub sampled_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SessionView {
    pub id: String,
    pub revision: u64,
    pub context_epoch: u64,
    pub tokens: usize,
    pub busy: bool,
    pub saved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TransferView {
    pub id: String,
    pub session_id: String,
    pub direction: String,
    pub mode: String,
    pub tier: String,
    pub destination: String,
    pub status: String,
    pub bytes: u64,
    pub sha256: String,
    pub source_deleted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransferRecord {
    view: TransferView,
    transport_key: [u8; 32],
    model_sha256: String,
    tokenizer_sha256: [u8; 32],
    template_sha256: [u8; 32],
    layout_abi: String,
    dflash_identity_sha256: Option<String>,
    vision_projector_sha256: Option<String>,
    vision_preprocessing_sha256: Option<String>,
}

pub(crate) struct TransferExport {
    pub view: TransferView,
    pub payload: PathBuf,
    pub transport_key: [u8; 32],
    pub model_sha256: String,
    pub tokenizer_sha256: [u8; 32],
    pub template_sha256: [u8; 32],
    pub layout_abi: String,
    pub dflash_identity_sha256: Option<String>,
    pub vision_projector_sha256: Option<String>,
    pub vision_preprocessing_sha256: Option<String>,
}

struct Record {
    revision: u64,
    context_epoch: u64,
    bundle: Option<SessionBundle>,
    busy: bool,
    saved: bool,
    idempotency: HashMap<String, IdempotencyRecord>,
}

struct IdempotencyRecord {
    expected_revision: u64,
    request_sha256: [u8; 32],
    result: CachedGeneration,
}

pub(crate) enum BeginMutation {
    Started(Option<Box<SessionBundle>>),
    Replay(CachedGeneration),
}

pub(crate) struct SessionStore {
    root: PathBuf,
    key_path: PathBuf,
    transfer_root: PathBuf,
    records: Mutex<HashMap<String, Record>>,
}

impl SessionStore {
    pub(crate) fn new() -> Self {
        let home = std::env::var_os("MUSER_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".muser")))
            .unwrap_or_else(|| PathBuf::from(".muser"));
        Self {
            root: home.join("sessions"),
            key_path: home.join("session.key"),
            transfer_root: home.join("session-transfers"),
            records: Mutex::new(HashMap::new()),
        }
    }

    #[cfg(test)]
    fn under(home: &Path) -> Self {
        Self {
            root: home.join("sessions"),
            key_path: home.join("session.key"),
            transfer_root: home.join("session-transfers"),
            records: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn create(&self, requested: Option<&str>) -> Result<SessionView, String> {
        let id = match requested {
            Some(id) => {
                validate_id(id)?;
                id.to_owned()
            }
            None => random_id(),
        };
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        if records.contains_key(&id) {
            return Err("session already exists".into());
        }
        if records.len() >= MAX_LOGICAL_SESSIONS {
            return Err("logical session limit (64) reached".into());
        }
        records.insert(
            id.clone(),
            Record {
                revision: 0,
                context_epoch: 0,
                bundle: None,
                busy: false,
                saved: false,
                idempotency: HashMap::new(),
            },
        );
        Ok(view(&id, records.get(&id).expect("inserted")))
    }

    pub(crate) fn list(&self) -> Result<Vec<SessionView>, String> {
        let records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        let mut views = records
            .iter()
            .map(|(id, record)| view(id, record))
            .collect::<Vec<_>>();
        views.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(views)
    }

    pub(crate) fn get(&self, id: &str) -> Result<Option<SessionView>, String> {
        validate_id(id)?;
        let records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        Ok(records.get(id).map(|record| view(id, record)))
    }

    pub(crate) fn delete(&self, id: &str) -> Result<bool, String> {
        validate_id(id)?;
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        if records.get(id).is_some_and(|record| record.busy) {
            return Err("session is busy".into());
        }
        Ok(records.remove(id).is_some())
    }

    pub(crate) fn delete_after_transfer(
        &self,
        id: &str,
        transfer_id: &str,
    ) -> Result<bool, String> {
        validate_id(id)?;
        let transfer = self.read_transfer(transfer_id)?;
        if transfer.view.direction != "outgoing"
            || transfer.view.session_id != id
            || transfer.view.mode != "move"
            || !matches!(
                transfer.view.status.as_str(),
                "destination_committed" | "destination_committed_source_retained"
            )
            || transfer.view.source_deleted
        {
            return Err(
                "source deletion requires a matching durable destination-committed move journal"
                    .into(),
            );
        }
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        let Some(record) = records.get(id) else {
            return Ok(false);
        };
        if record.busy {
            return Err("session is busy".into());
        }
        let path = self.root.join(format!("{id}.bundle"));
        match fs::remove_file(&path) {
            Ok(()) => sync_directory(&self.root)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("remove {}: {error}", path.display())),
        }
        records.remove(id);
        Ok(true)
    }

    /// Return whether an outgoing transfer is already terminal or has a
    /// durable destination ACK whose source-side move can be reconciled
    /// without retransmitting the payload.
    pub(crate) fn reconcile_outgoing_after_ack(&self, transfer_id: &str) -> Result<bool, String> {
        let view = self.transfer(transfer_id)?;
        if view.direction != "outgoing" {
            return Err("cannot reconcile a non-outgoing transfer".into());
        }
        if view.status == "completed" {
            return Ok(true);
        }
        if !matches!(
            view.status.as_str(),
            "destination_committed" | "destination_committed_source_retained"
        ) {
            return Ok(false);
        }
        if view.mode == "move" {
            match self.delete_after_transfer(&view.session_id, transfer_id) {
                Ok(_) => {
                    self.update_transfer(transfer_id, "completed", None, true)?;
                }
                Err(error) => {
                    self.update_transfer(
                        transfer_id,
                        "destination_committed_source_retained",
                        Some(error),
                        false,
                    )?;
                }
            }
        } else if view.mode == "copy" {
            self.update_transfer(transfer_id, "completed", None, false)?;
        } else {
            return Err("outgoing transfer journal has an invalid migration mode".into());
        }
        Ok(true)
    }

    pub(crate) fn begin(
        &self,
        id: &str,
        expected_revision: u64,
        idempotency_key: &str,
        request_sha256: [u8; 32],
    ) -> Result<BeginMutation, String> {
        validate_id(id)?;
        validate_idempotency_key(idempotency_key)?;
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        let record = records.get_mut(id).ok_or("session does not exist")?;
        if let Some(cached) = record.idempotency.get(idempotency_key) {
            if cached.expected_revision != expected_revision
                || cached.request_sha256 != request_sha256
            {
                return Err(
                    "Idempotency-Key is already bound to a different session mutation".into(),
                );
            }
            return Ok(BeginMutation::Replay(cached.result.clone()));
        }
        if record.busy {
            return Err("session is busy".into());
        }
        if record.revision != expected_revision {
            return Err(format!(
                "session revision conflict: expected {expected_revision}, current {}",
                record.revision
            ));
        }
        record.busy = true;
        Ok(BeginMutation::Started(record.bundle.clone().map(Box::new)))
    }

    pub(crate) fn abort(&self, id: &str) {
        if let Ok(mut records) = self.records.lock() {
            if let Some(record) = records.get_mut(id) {
                record.busy = false;
            }
        }
    }

    pub(crate) fn commit(
        &self,
        id: &str,
        expected_revision: u64,
        idempotency_key: &str,
        request_sha256: [u8; 32],
        mut bundle: SessionBundle,
        mut result: CachedGeneration,
    ) -> Result<u64, String> {
        self.validate_bundle(&bundle, id)?;
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        let record = records.get_mut(id).ok_or("session disappeared")?;
        if !record.busy || record.revision != expected_revision {
            return Err("session mutation lost its revision lease".into());
        }
        let revision = expected_revision
            .checked_add(1)
            .ok_or("session revision overflow")?;
        bundle.revision = revision;
        result.revision = revision;
        record.revision = revision;
        record.context_epoch = bundle.context_epoch;
        record.bundle = Some(bundle);
        record.busy = false;
        record.saved = false;
        if record.idempotency.len() >= 64 {
            record.idempotency.clear();
        }
        record.idempotency.insert(
            idempotency_key.into(),
            IdempotencyRecord {
                expected_revision,
                request_sha256,
                result,
            },
        );
        Ok(revision)
    }

    pub(crate) fn save(&self, id: &str) -> Result<PathBuf, String> {
        validate_id(id)?;
        let bundle = {
            let records = self
                .records
                .lock()
                .map_err(|_| "session registry poisoned")?;
            let record = records.get(id).ok_or("session does not exist")?;
            if record.busy {
                return Err("session is busy".into());
            }
            record
                .bundle
                .clone()
                .ok_or("session has no committed state")?
        };
        let key = load_or_create_key(&self.key_path)?;
        let plaintext = postcard::to_stdvec(&bundle).map_err(|error| error.to_string())?;
        fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
        set_mode(&self.root, 0o700)?;
        let path = self.root.join(format!("{id}.bundle"));
        let bytes = encrypt_envelope(&plaintext, &key)?;
        atomic_private_write(&path, &bytes)?;
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        if let Some(record) = records.get_mut(id) {
            record.saved = true;
        }
        Ok(path)
    }

    pub(crate) fn restore(&self, id: &str) -> Result<SessionView, String> {
        validate_id(id)?;
        let path = self.root.join(format!("{id}.bundle"));
        require_private_regular_file(&path)?;
        let bytes = fs::read(&path).map_err(|error| error.to_string())?;
        if bytes.len() <= MAGIC.len() + 24 || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid encrypted session envelope".into());
        }
        let key = load_or_create_key(&self.key_path)?;
        let plaintext = decrypt_envelope(&bytes, &key)?;
        let bundle: SessionBundle =
            postcard::from_bytes(&plaintext).map_err(|error| error.to_string())?;
        self.validate_bundle(&bundle, id)?;
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        if records.len() >= MAX_LOGICAL_SESSIONS && !records.contains_key(id) {
            return Err("logical session limit (64) reached".into());
        }
        records.insert(
            id.into(),
            Record {
                revision: bundle.revision,
                context_epoch: bundle.context_epoch,
                bundle: Some(bundle),
                busy: false,
                saved: true,
                idempotency: HashMap::new(),
            },
        );
        Ok(view(id, records.get(id).expect("restored")))
    }

    pub(crate) fn begin_export(
        &self,
        transfer_id: &str,
        session_id: &str,
        destination: &str,
        mode: &str,
        tier: &str,
    ) -> Result<TransferExport, String> {
        validate_id(transfer_id)?;
        validate_id(session_id)?;
        if !matches!(mode, "copy" | "move") {
            return Err("migration mode must be 'copy' or 'move'".into());
        }
        validate_transfer_tier(tier)?;
        let journal = self.transfer_record_path(transfer_id);
        if journal
            .try_exists()
            .map_err(|error| format!("inspect transfer journal: {error}"))?
        {
            let existing = self.read_transfer(transfer_id)?;
            if existing.view.direction != "outgoing"
                || existing.view.session_id != session_id
                || existing.view.destination != destination
                || existing.view.mode != mode
                || existing.view.tier != tier
            {
                return Err("transfer ID is already bound to a different migration".into());
            }
            if existing.view.status != "starting"
                || existing.view.bytes != 0
                || !existing.view.sha256.is_empty()
            {
                return Err(
                    "transfer already has durable material; resume it instead of replacing it"
                        .into(),
                );
            }
        }
        let bundle = {
            let records = self
                .records
                .lock()
                .map_err(|_| "session registry poisoned")?;
            let record = records.get(session_id).ok_or("session does not exist")?;
            if record.busy {
                return Err("session is busy".into());
            }
            record
                .bundle
                .clone()
                .ok_or("session has no committed state")?
        };
        fs::create_dir_all(&self.transfer_root).map_err(|error| error.to_string())?;
        set_mode(&self.transfer_root, 0o700)?;
        let payload = self.transfer_root.join(format!("{transfer_id}.payload"));
        let mut transport_key = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut transport_key);
        let plaintext = postcard::to_stdvec(&bundle).map_err(|error| error.to_string())?;
        let envelope = encrypt_envelope(&plaintext, &transport_key)?;
        atomic_private_write(&payload, &envelope)?;
        let bytes = envelope.len() as u64;
        let digest = sha256_file(&payload)?;
        let view = TransferView {
            id: transfer_id.into(),
            session_id: session_id.into(),
            direction: "outgoing".into(),
            mode: mode.into(),
            tier: tier.into(),
            destination: destination.into(),
            status: "prepared".into(),
            bytes,
            sha256: digest,
            source_deleted: false,
            last_error: None,
        };
        let record = TransferRecord {
            view: view.clone(),
            transport_key,
            model_sha256: bundle.model_sha256.clone(),
            tokenizer_sha256: bundle.tokenizer_sha256,
            template_sha256: bundle.template_sha256,
            layout_abi: bundle.layout_abi.clone(),
            dflash_identity_sha256: bundle.dflash_identity_sha256.clone(),
            vision_projector_sha256: bundle.vision_projector_sha256.clone(),
            vision_preprocessing_sha256: bundle.vision_preprocessing_sha256.clone(),
        };
        self.write_transfer(&record)?;
        Ok(TransferExport {
            view,
            payload,
            transport_key,
            model_sha256: bundle.model_sha256,
            tokenizer_sha256: bundle.tokenizer_sha256,
            template_sha256: bundle.template_sha256,
            layout_abi: bundle.layout_abi,
            dflash_identity_sha256: bundle.dflash_identity_sha256,
            vision_projector_sha256: bundle.vision_projector_sha256,
            vision_preprocessing_sha256: bundle.vision_preprocessing_sha256,
        })
    }

    pub(crate) fn register_outgoing(
        &self,
        transfer_id: &str,
        session_id: &str,
        destination: &str,
        mode: &str,
        tier: &str,
    ) -> Result<TransferView, String> {
        validate_id(transfer_id)?;
        validate_id(session_id)?;
        if !matches!(mode, "copy" | "move") {
            return Err("migration mode must be 'copy' or 'move'".into());
        }
        validate_transfer_tier(tier)?;
        let journal = self.transfer_record_path(transfer_id);
        if journal
            .try_exists()
            .map_err(|error| format!("inspect transfer journal: {error}"))?
        {
            // A caller may resume only after successfully reading and matching
            // the existing journal.  Never replace corrupt or mismatched
            // durable state as though it were a new transfer.
            self.read_transfer(transfer_id)?;
            return Err("transfer ID is already registered".into());
        }
        {
            let records = self
                .records
                .lock()
                .map_err(|_| "session registry poisoned")?;
            let record = records.get(session_id).ok_or("session does not exist")?;
            if record.busy || record.bundle.is_none() {
                return Err("session is busy or has no committed state".into());
            }
        }
        let record = TransferRecord {
            view: TransferView {
                id: transfer_id.into(),
                session_id: session_id.into(),
                direction: "outgoing".into(),
                mode: mode.into(),
                tier: tier.into(),
                destination: destination.into(),
                status: "starting".into(),
                bytes: 0,
                sha256: String::new(),
                source_deleted: false,
                last_error: None,
            },
            transport_key: [0; 32],
            model_sha256: String::new(),
            tokenizer_sha256: [0; 32],
            template_sha256: [0; 32],
            layout_abi: String::new(),
            dflash_identity_sha256: None,
            vision_projector_sha256: None,
            vision_preprocessing_sha256: None,
        };
        self.write_transfer(&record)?;
        Ok(record.view)
    }

    pub(crate) fn resume_export(&self, transfer_id: &str) -> Result<TransferExport, String> {
        let record = self.read_transfer(transfer_id)?;
        if record.view.direction != "outgoing"
            || record.view.bytes == 0
            || !is_sha256(&record.view.sha256)
        {
            return Err("outgoing transfer has no resumable prepared payload".into());
        }
        let payload = self.transfer_root.join(format!("{transfer_id}.payload"));
        require_private_regular_file(&payload)?;
        if payload.metadata().map_err(|error| error.to_string())?.len() != record.view.bytes
            || sha256_file(&payload)? != record.view.sha256
        {
            return Err("resumable transfer payload no longer matches its journal".into());
        }
        Ok(TransferExport {
            view: record.view,
            payload,
            transport_key: record.transport_key,
            model_sha256: record.model_sha256,
            tokenizer_sha256: record.tokenizer_sha256,
            template_sha256: record.template_sha256,
            layout_abi: record.layout_abi,
            dflash_identity_sha256: record.dflash_identity_sha256,
            vision_projector_sha256: record.vision_projector_sha256,
            vision_preprocessing_sha256: record.vision_preprocessing_sha256,
        })
    }

    pub(crate) fn transfer_payload_path(&self, transfer_id: &str) -> Result<PathBuf, String> {
        validate_id(transfer_id)?;
        Ok(self.transfer_root.join(format!("{transfer_id}.payload")))
    }

    pub(crate) fn remove_transfer_payload(&self, transfer_id: &str) -> Result<(), String> {
        let path = self.transfer_payload_path(transfer_id)?;
        match fs::remove_file(&path) {
            Ok(()) => sync_directory(&self.transfer_root),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }

    pub(crate) fn adopt_export_payload(
        &self,
        transfer_id: &str,
        staging: &Path,
    ) -> Result<PathBuf, String> {
        let record = self.read_transfer(transfer_id)?;
        let metadata = fs::symlink_metadata(staging).map_err(|error| error.to_string())?;
        if !metadata.file_type().is_file()
            || metadata.len() != record.view.bytes
            || sha256_file(staging)? != record.view.sha256
        {
            return Err("retrieved storage payload size or SHA-256 mismatch".into());
        }
        set_mode(staging, 0o600)?;
        let target = self.transfer_payload_path(transfer_id)?;
        fs::rename(staging, &target).map_err(|error| error.to_string())?;
        sync_directory(&self.transfer_root)?;
        Ok(target)
    }

    pub(crate) fn restore_export(&self, transfer_id: &str) -> Result<SessionView, String> {
        let export = self.resume_export(transfer_id)?;
        let encrypted = fs::read(&export.payload).map_err(|error| error.to_string())?;
        let plaintext = decrypt_envelope(&encrypted, &export.transport_key)?;
        let bundle: SessionBundle =
            postcard::from_bytes(&plaintext).map_err(|error| error.to_string())?;
        self.validate_bundle(&bundle, &export.view.session_id)?;
        if bundle.model_sha256 != export.model_sha256
            || bundle.tokenizer_sha256 != export.tokenizer_sha256
            || bundle.template_sha256 != export.template_sha256
            || bundle.layout_abi != export.layout_abi
            || bundle.dflash_identity_sha256 != export.dflash_identity_sha256
            || bundle.vision_projector_sha256 != export.vision_projector_sha256
            || bundle.vision_preprocessing_sha256 != export.vision_preprocessing_sha256
        {
            return Err("storage-tier session identity differs from its transfer journal".into());
        }
        let id = bundle.session_id.clone();
        self.install_imported_bundle(bundle)?;
        self.get(&id)?.ok_or("restored session disappeared".into())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_import(
        &self,
        transfer_id: &str,
        session_id: &str,
        source: &str,
        mode: &str,
        tier: &str,
        bytes: u64,
        sha256: &str,
        transport_key: [u8; 32],
        model_sha256: &str,
        tokenizer_sha256: [u8; 32],
        template_sha256: [u8; 32],
        layout_abi: &str,
        dflash_identity_sha256: Option<&str>,
        vision_projector_sha256: Option<&str>,
        vision_preprocessing_sha256: Option<&str>,
    ) -> Result<TransferView, String> {
        validate_id(transfer_id)?;
        validate_id(session_id)?;
        if !matches!(mode, "copy" | "move") {
            return Err("migration mode must be 'copy' or 'move'".into());
        }
        if tier != "decode" {
            return Err("decode-node transfer tier must be 'decode'".into());
        }
        if bytes == 0 || bytes > MAX_TRANSFER_BYTES || !is_sha256(sha256) {
            return Err("transfer payload size or SHA-256 is invalid".into());
        }
        if !is_sha256(model_sha256)
            || layout_abi != "muse-kv-layout-v1"
            || [
                dflash_identity_sha256,
                vision_projector_sha256,
                vision_preprocessing_sha256,
            ]
            .into_iter()
            .flatten()
            .any(|digest| !is_sha256(digest))
            || vision_projector_sha256.is_some() != vision_preprocessing_sha256.is_some()
        {
            return Err("transfer model or layout identity is invalid".into());
        }
        let journal = self.transfer_record_path(transfer_id);
        if journal
            .try_exists()
            .map_err(|error| format!("inspect transfer journal: {error}"))?
        {
            let existing = self.read_transfer(transfer_id)?;
            if existing.view.direction == "incoming"
                && existing.view.session_id == session_id
                && existing.view.destination == source
                && existing.view.mode == mode
                && existing.view.tier == tier
                && existing.view.bytes == bytes
                && existing.view.sha256 == sha256
                && existing.transport_key == transport_key
                && existing.model_sha256 == model_sha256
                && existing.tokenizer_sha256 == tokenizer_sha256
                && existing.template_sha256 == template_sha256
                && existing.layout_abi == layout_abi
                && existing.dflash_identity_sha256.as_deref() == dflash_identity_sha256
                && existing.vision_projector_sha256.as_deref() == vision_projector_sha256
                && existing.vision_preprocessing_sha256.as_deref() == vision_preprocessing_sha256
            {
                return Ok(existing.view);
            }
            return Err("transfer ID is already bound to different material".into());
        }
        fs::create_dir_all(&self.transfer_root).map_err(|error| error.to_string())?;
        set_mode(&self.transfer_root, 0o700)?;
        let record = TransferRecord {
            view: TransferView {
                id: transfer_id.into(),
                session_id: session_id.into(),
                direction: "incoming".into(),
                mode: mode.into(),
                tier: tier.into(),
                destination: source.into(),
                status: "prepared".into(),
                bytes,
                sha256: sha256.into(),
                source_deleted: false,
                last_error: None,
            },
            transport_key,
            model_sha256: model_sha256.into(),
            tokenizer_sha256,
            template_sha256,
            layout_abi: layout_abi.into(),
            dflash_identity_sha256: dflash_identity_sha256.map(str::to_owned),
            vision_projector_sha256: vision_projector_sha256.map(str::to_owned),
            vision_preprocessing_sha256: vision_preprocessing_sha256.map(str::to_owned),
        };
        self.write_transfer(&record)?;
        Ok(record.view)
    }

    pub(crate) fn payload_path(&self, transfer_id: &str) -> Result<(PathBuf, u64), String> {
        let record = self.read_transfer(transfer_id)?;
        if record.view.direction != "incoming" || record.view.status == "committed" {
            return Err("transfer is not awaiting an incoming payload".into());
        }
        Ok((
            self.transfer_root
                .join(format!(".{transfer_id}.incoming-{}", random_id())),
            record.view.bytes,
        ))
    }

    pub(crate) fn accept_payload(
        &self,
        transfer_id: &str,
        staging: &Path,
    ) -> Result<TransferView, String> {
        let mut record = self.read_transfer(transfer_id)?;
        let metadata = fs::symlink_metadata(staging).map_err(|error| error.to_string())?;
        if !metadata.file_type().is_file()
            || metadata.len() != record.view.bytes
            || sha256_file(staging)? != record.view.sha256
        {
            return Err("incoming transfer payload size or SHA-256 mismatch".into());
        }
        let target = self.transfer_root.join(format!("{transfer_id}.payload"));
        fs::rename(staging, &target).map_err(|error| error.to_string())?;
        sync_directory(&self.transfer_root)?;
        record.view.status = "uploaded".into();
        record.view.last_error = None;
        self.write_transfer(&record)?;
        Ok(record.view)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_import(
        &self,
        transfer_id: &str,
        expected_model: &str,
        expected_tokenizer: [u8; 32],
        expected_template: [u8; 32],
        expected_layout: &str,
        expected_dflash: Option<&str>,
        expected_vision_projector: Option<&str>,
        expected_vision_preprocessing: Option<&str>,
    ) -> Result<TransferView, String> {
        let mut record = self.read_transfer(transfer_id)?;
        if record.model_sha256 != expected_model
            || record.tokenizer_sha256 != expected_tokenizer
            || record.template_sha256 != expected_template
            || record.layout_abi != expected_layout
            || record
                .dflash_identity_sha256
                .as_deref()
                .is_some_and(|value| Some(value) != expected_dflash)
            || record
                .vision_projector_sha256
                .as_deref()
                .is_some_and(|value| Some(value) != expected_vision_projector)
            || record
                .vision_preprocessing_sha256
                .as_deref()
                .is_some_and(|value| Some(value) != expected_vision_preprocessing)
        {
            return Err("incoming session model/template/layout identity mismatch".into());
        }
        if record.view.status == "committed" {
            return Ok(record.view);
        }
        if record.view.status != "uploaded" {
            return Err("transfer payload has not been durably uploaded".into());
        }
        let payload = self.transfer_root.join(format!("{transfer_id}.payload"));
        require_private_regular_file(&payload)?;
        let encrypted = fs::read(&payload).map_err(|error| error.to_string())?;
        let plaintext = decrypt_envelope(&encrypted, &record.transport_key)?;
        let bundle: SessionBundle =
            postcard::from_bytes(&plaintext).map_err(|error| error.to_string())?;
        self.validate_bundle(&bundle, &record.view.session_id)?;
        if bundle.model_sha256 != expected_model
            || bundle.tokenizer_sha256 != expected_tokenizer
            || bundle.template_sha256 != expected_template
            || bundle.layout_abi != expected_layout
            || bundle.dflash_identity_sha256 != record.dflash_identity_sha256
            || bundle.vision_projector_sha256 != record.vision_projector_sha256
            || bundle.vision_preprocessing_sha256 != record.vision_preprocessing_sha256
        {
            return Err("decrypted session identity differs from transfer preparation".into());
        }
        self.install_imported_bundle(bundle)?;
        record.view.status = "committed".into();
        record.view.last_error = None;
        self.write_transfer(&record)?;
        Ok(record.view)
    }

    pub(crate) fn transfer(&self, transfer_id: &str) -> Result<TransferView, String> {
        validate_id(transfer_id)?;
        Ok(self.read_transfer(transfer_id)?.view)
    }

    pub(crate) fn update_transfer(
        &self,
        transfer_id: &str,
        status: &str,
        error: Option<String>,
        source_deleted: bool,
    ) -> Result<TransferView, String> {
        let mut record = self.read_transfer(transfer_id)?;
        record.view.status = status.into();
        record.view.last_error = error;
        record.view.source_deleted = source_deleted;
        self.write_transfer(&record)?;
        Ok(record.view)
    }

    fn validate_bundle(&self, bundle: &SessionBundle, id: &str) -> Result<(), String> {
        if bundle.schema != "muser.session-bundle.v3" || bundle.session_id != id {
            return Err("session bundle identity mismatch".into());
        }
        if !is_sha256(&bundle.model_sha256) || bundle.layout_abi != "muse-kv-layout-v1" {
            return Err("session bundle model or layout identity is invalid".into());
        }
        if [
            bundle.dflash_identity_sha256.as_deref(),
            bundle.vision_projector_sha256.as_deref(),
            bundle.vision_preprocessing_sha256.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|digest| !is_sha256(digest))
            || bundle.dflash.is_some() != bundle.dflash_identity_sha256.is_some()
            || bundle.vision_projector_sha256.is_some()
                != bundle.vision_preprocessing_sha256.is_some()
            || (!bundle.vision_rows.is_empty() && bundle.vision_projector_sha256.is_none())
        {
            return Err("session bundle assistant or vision identity is invalid".into());
        }
        bundle.target.validate()?;
        for rng in [
            &bundle.sampler_state.distribution_rng,
            &bundle.sampler_state.xtc_rng,
            &bundle.sampler_state.mirostat_rng,
            &bundle.sampler_state.adaptive_rng,
        ] {
            muser_engine::sampling::Mt19937::from_snapshot(rng)
                .map_err(|error| error.to_string())?;
        }
        if !bundle.sampler_state.mirostat_mu.is_finite()
            || !bundle.sampler_state.adaptive_weighted_sum.is_finite()
            || !bundle.sampler_state.adaptive_total_weight.is_finite()
            || bundle.sampler_state.adaptive_total_weight <= 0.0
        {
            return Err("session bundle sampler scalar state is invalid".into());
        }
        if bundle.target_logits.is_empty()
            || bundle.target_logits.iter().any(|value| !value.is_finite())
            || bundle.position_witnesses.as_slice() != bundle.target.tokens.as_ref()
            || bundle.sampler_history.as_slice() != bundle.target.tokens.as_ref()
            || !bundle.detokenizer_pending.is_empty()
            || !bundle.stop_matcher_pending.is_empty()
            || bundle
                .grammar_state
                .as_ref()
                .is_some_and(crate::grammar::GrammarMatcher::has_pending_utf8)
            || bundle.grammar_state.is_some() != bundle.grammar_sha256.is_some()
        {
            return Err("session bundle frontier state is incomplete or inconsistent".into());
        }
        serde_json::from_str::<Vec<crate::openai::Message>>(&bundle.canonical_replay_plan_json)
            .map_err(|error| format!("session bundle canonical replay plan is invalid: {error}"))?;
        if let Some(dflash) = &bundle.dflash {
            dflash.validate()?;
            let target_position = bundle.target.position as usize;
            if dflash.position > target_position || target_position - dflash.position > 1 {
                return Err(
                    "DFlash and target session frontiers differ by more than one boundary token"
                        .into(),
                );
            }
        }
        let vision_positions = bundle
            .position_witnesses
            .iter()
            .filter(|&&value| value == muser_engine::EMBEDDING_POSITION_WITNESS)
            .count();
        if bundle.vision_rows.len() != vision_positions
            || bundle.vision_rows.first().is_some_and(|first| {
                first.is_empty()
                    || bundle.vision_rows.iter().any(|row| {
                        row.len() != first.len() || row.iter().any(|value| !value.is_finite())
                    })
            })
        {
            return Err("session bundle vision rows do not match embedding witnesses".into());
        }
        Ok(())
    }

    fn install_imported_bundle(&self, bundle: SessionBundle) -> Result<(), String> {
        let id = bundle.session_id.clone();
        let mut records = self
            .records
            .lock()
            .map_err(|_| "session registry poisoned")?;
        if records.len() >= MAX_LOGICAL_SESSIONS && !records.contains_key(&id) {
            return Err("logical session limit (64) reached".into());
        }
        if records.get(&id).is_some_and(|existing| existing.busy) {
            return Err("destination session is busy".into());
        }
        let plaintext = postcard::to_stdvec(&bundle).map_err(|error| error.to_string())?;
        let key = load_or_create_key(&self.key_path)?;
        let encrypted = encrypt_envelope(&plaintext, &key)?;
        fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
        set_mode(&self.root, 0o700)?;
        atomic_private_write(&self.root.join(format!("{id}.bundle")), &encrypted)?;
        records.insert(
            id,
            Record {
                revision: bundle.revision,
                context_epoch: bundle.context_epoch,
                bundle: Some(bundle),
                busy: false,
                saved: true,
                idempotency: HashMap::new(),
            },
        );
        Ok(())
    }

    fn transfer_record_path(&self, transfer_id: &str) -> PathBuf {
        self.transfer_root.join(format!("{transfer_id}.json"))
    }

    fn read_transfer(&self, transfer_id: &str) -> Result<TransferRecord, String> {
        validate_id(transfer_id)?;
        let path = self.transfer_record_path(transfer_id);
        require_private_regular_file(&path)?;
        serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())
    }

    fn write_transfer(&self, record: &TransferRecord) -> Result<(), String> {
        fs::create_dir_all(&self.transfer_root).map_err(|error| error.to_string())?;
        set_mode(&self.transfer_root, 0o700)?;
        let bytes = serde_json::to_vec_pretty(record).map_err(|error| error.to_string())?;
        atomic_private_write(&self.transfer_record_path(&record.view.id), &bytes)
    }
}

fn view(id: &str, record: &Record) -> SessionView {
    SessionView {
        id: id.into(),
        revision: record.revision,
        context_epoch: record.context_epoch,
        tokens: record
            .bundle
            .as_ref()
            .map_or(0, |bundle| bundle.target.position as usize),
        busy: record.busy,
        saved: record.saved,
    }
}

fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty()
        || id.len() > 128
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err("session ID must contain 1..=128 ASCII letters, digits, '-' or '_'".into());
    }
    Ok(())
}

fn validate_idempotency_key(key: &str) -> Result<(), String> {
    if key.is_empty() || key.len() > 256 || key.bytes().any(|byte| byte.is_ascii_control()) {
        return Err("Idempotency-Key must contain 1..=256 non-control bytes".into());
    }
    Ok(())
}

fn validate_transfer_tier(tier: &str) -> Result<(), String> {
    if matches!(tier, "decode" | "storage") {
        Ok(())
    } else {
        Err("migration tier must be 'decode' or 'storage'".into())
    }
}

fn random_id() -> String {
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    format!("sess-{}", hex(&bytes))
}

fn encrypt_envelope(plaintext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, String> {
    let cipher = XChaCha20Poly1305::new(key.into());
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|_| "encrypt session bundle")?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + nonce.len() + ciphertext.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&nonce);
    bytes.extend_from_slice(&ciphertext);
    Ok(bytes)
}

fn decrypt_envelope(bytes: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, String> {
    if bytes.len() <= MAGIC.len() + 24 || &bytes[..MAGIC.len()] != MAGIC {
        return Err("invalid encrypted session envelope".into());
    }
    let cipher = XChaCha20Poly1305::new(key.into());
    let nonce = XNonce::from_slice(&bytes[MAGIC.len()..MAGIC.len() + 24]);
    cipher
        .decrypt(nonce, &bytes[MAGIC.len() + 24..])
        .map_err(|_| "session bundle authentication failed".into())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest as _, Sha256};
    use std::io::Read as _;
    let mut input = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sync_directory(path: &Path) -> Result<(), String> {
    OpenOptions::new()
        .read(true)
        .open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| error.to_string())
}

fn load_or_create_key(path: &Path) -> Result<[u8; 32], String> {
    if path.exists() {
        require_private_regular_file(path)?;
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        return bytes
            .try_into()
            .map_err(|_| "session key must contain exactly 32 bytes".into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        set_mode(parent, 0o700)?;
    }
    let mut key = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut key);
    atomic_private_write(path, &key)?;
    Ok(key)
}

pub(crate) fn atomic_private_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    atomic_private_file(path, |file| {
        file.write_all(bytes).map_err(|error| error.to_string())
    })
}

/// Stream a potentially multi-gigabyte cache bundle to a private temporary
/// file and expose it only after file and directory durability complete.
/// Callers avoid materializing a second copy of every KV plane in memory.
pub(crate) fn atomic_private_file(
    path: &Path,
    write: impl FnOnce(&mut std::fs::File) -> Result<(), String>,
) -> Result<(), String> {
    let parent = path.parent().ok_or("output path has no parent")?;
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("bundle"),
        random_id()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let result = (|| {
        let mut file = options.open(&temp).map_err(|error| error.to_string())?;
        maybe_fail_atomic("write")?;
        write(&mut file)?;
        maybe_fail_atomic("file-fsync")?;
        file.sync_all().map_err(|error| error.to_string())?;
        maybe_fail_atomic("rename")?;
        fs::rename(&temp, path).map_err(|error| error.to_string())?;
        maybe_fail_atomic("directory-fsync")?;
        OpenOptions::new()
            .read(true)
            .open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
thread_local! {
    static ATOMIC_FAILURE: std::cell::RefCell<Option<&'static str>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn inject_atomic_failure(stage: &'static str) {
    ATOMIC_FAILURE.with(|value| *value.borrow_mut() = Some(stage));
}

#[cfg(test)]
fn maybe_fail_atomic(stage: &str) -> Result<(), String> {
    ATOMIC_FAILURE.with(|value| {
        let mut value = value.borrow_mut();
        if value.as_ref().is_some_and(|candidate| *candidate == stage) {
            *value = None;
            return Err(format!("injected atomic {stage} failure"));
        }
        Ok(())
    })
}

#[cfg(not(test))]
fn maybe_fail_atomic(_stage: &str) -> Result<(), String> {
    Ok(())
}

fn require_private_regular_file(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err("session material must be a regular non-symlink file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err("session material permissions are looser than 0600".into());
        }
    }
    Ok(())
}

fn set_mode(path: &Path, mode: u32) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(path, fs::Permissions::from_mode(mode))
            .map_err(|error| error.to_string())?;
    }
    let _ = mode;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use muser_engine::cache::{CachePlaneSnapshot, PlaneEncoding, SessionCacheSnapshot};
    use muser_engine::dflash::DFlashContextSnapshot;

    use super::*;

    fn sampler_state(seed: u32) -> SamplerStateSnapshot {
        let rng = muser_engine::sampling::Mt19937::new(seed).snapshot();
        SamplerStateSnapshot {
            distribution_rng: rng.clone(),
            xtc_rng: rng.clone(),
            mirostat_rng: rng.clone(),
            adaptive_rng: rng,
            mirostat_mu: 10.0,
            adaptive_weighted_sum: 0.0,
            adaptive_total_weight: 10.0,
        }
    }

    fn bundle(id: &str) -> SessionBundle {
        let layers = (0..52)
            .map(|layer| CachePlaneSnapshot {
                layer,
                logical_start: 0,
                logical_count: 1,
                encoding: PlaneEncoding::F16Le,
                key: Arc::from([0u8, 0]),
                value: Arc::from([0u8, 0]),
            })
            .collect::<Vec<_>>();
        SessionBundle {
            schema: "muser.session-bundle.v3".into(),
            session_id: id.into(),
            revision: 0,
            context_epoch: 0,
            // P-series native NVFP4 artifact identity. Decode migration is
            // keyed by the model digest, so its transaction fixtures should
            // exercise the product identity instead of a generic zero hash.
            model_sha256: "c979f33e706862d09af662ba980930e58ee0801ee70c048f7ddbbf9589b18ee8".into(),
            tokenizer_sha256: [1; 32],
            template_sha256: [2; 32],
            layout_abi: "muse-kv-layout-v1".into(),
            dflash_identity_sha256: Some("11".repeat(32)),
            vision_projector_sha256: None,
            vision_preprocessing_sha256: None,
            target: SessionCacheSnapshot {
                position: 1,
                tokens: Arc::from([42u32]),
                elements_per_token: 1,
                layers: Arc::from(layers),
            },
            target_logits: vec![0.0, 1.0],
            dflash: Some(DFlashContextSnapshot {
                position: 1,
                sink_size: 64,
                window_size: 1024,
                elements_per_token: 1,
                layers: vec![(vec![0.0], vec![0.0]); 5],
            }),
            position_witnesses: vec![42],
            rng_seed: 7,
            sampler_state: sampler_state(7),
            sampler_config_sha256: [3; 32],
            sampler_history: vec![42],
            detokenizer_pending: String::new(),
            stop_matcher_pending: String::new(),
            grammar_state: None,
            grammar_sha256: None,
            canonical_replay_plan_json: "[]".into(),
            vision_rows: Vec::new(),
        }
    }

    fn persist_test_bundle(store: &SessionStore, bundle: &SessionBundle) -> PathBuf {
        fs::create_dir_all(&store.root).unwrap();
        set_mode(&store.root, 0o700).unwrap();
        let key = load_or_create_key(&store.key_path).unwrap();
        let plaintext = postcard::to_stdvec(bundle).unwrap();
        let encrypted = encrypt_envelope(&plaintext, &key).unwrap();
        let path = store.root.join(format!("{}.bundle", bundle.session_id));
        atomic_private_write(&path, &encrypted).unwrap();
        path
    }

    #[test]
    fn revision_cas_busy_and_idempotency_are_fail_closed() {
        let store = SessionStore::new();
        let session = store.create(None).unwrap();
        assert!(matches!(
            store.begin(&session.id, 0, "key-1", [1; 32]).unwrap(),
            BeginMutation::Started(None)
        ));
        assert!(store.begin(&session.id, 0, "key-2", [2; 32]).is_err());
        store.abort(&session.id);
        assert!(store.begin(&session.id, 1, "key-2", [2; 32]).is_err());
    }

    #[test]
    fn aborted_staging_generation_preserves_the_atomic_committed_cut() {
        let store = SessionStore::new();
        let id = store.create(None).unwrap().id;
        assert!(matches!(
            store.begin(&id, 0, "first", [1; 32]).unwrap(),
            BeginMutation::Started(None)
        ));
        store
            .commit(
                &id,
                0,
                "first",
                [1; 32],
                bundle(&id),
                CachedGeneration {
                    text: "committed".into(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    finish_reason: "stop".into(),
                    seed: 1,
                    revision: 0,
                    context: vec![42],
                    sampled_tokens: vec![42],
                },
            )
            .unwrap();
        assert!(matches!(
            store.begin(&id, 0, "first", [1; 32]).unwrap(),
            BeginMutation::Replay(_)
        ));
        assert!(store.begin(&id, 0, "first", [9; 32]).is_err());
        assert!(store.begin(&id, 1, "first", [1; 32]).is_err());
        let staged = store.begin(&id, 1, "shift", [2; 32]).unwrap();
        let BeginMutation::Started(Some(previous)) = staged else {
            panic!("committed cut was not staged");
        };
        assert_eq!(previous.revision, 1);
        store.abort(&id);
        let view = store.get(&id).unwrap().unwrap();
        assert_eq!(view.revision, 1);
        assert_eq!(view.context_epoch, 0);
        assert!(!view.busy);
        assert!(matches!(
            store.begin(&id, 1, "retry", [3; 32]).unwrap(),
            BeginMutation::Started(Some(_))
        ));
        store.abort(&id);
    }

    #[test]
    fn encrypted_postcard_bundle_round_trips_and_rejects_tampering() {
        let home = std::env::temp_dir().join(format!("muser-session-test-{}", random_id()));
        let store = SessionStore::under(&home);
        let id = "round-trip";
        store.create(Some(id)).unwrap();
        assert!(matches!(
            store.begin(id, 0, "request-1", [1; 32]).unwrap(),
            BeginMutation::Started(None)
        ));
        let mut complete = bundle(id);
        complete.target.tokens = Arc::from([muser_engine::EMBEDDING_POSITION_WITNESS]);
        complete.position_witnesses = vec![muser_engine::EMBEDDING_POSITION_WITNESS];
        complete.sampler_history = vec![muser_engine::EMBEDDING_POSITION_WITNESS];
        complete.vision_rows = vec![vec![0.25, -0.5]];
        complete.vision_projector_sha256 = Some("22".repeat(32));
        complete.vision_preprocessing_sha256 = Some("33".repeat(32));
        store
            .commit(
                id,
                0,
                "request-1",
                [1; 32],
                complete,
                CachedGeneration {
                    text: "ok".into(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    finish_reason: "stop".into(),
                    seed: 7,
                    revision: 0,
                    context: vec![muser_engine::EMBEDDING_POSITION_WITNESS],
                    sampled_tokens: vec![muser_engine::EMBEDDING_POSITION_WITNESS],
                },
            )
            .unwrap();
        let path = store.save(id).unwrap();

        let restored = SessionStore::under(&home);
        let view = restored.restore(id).unwrap();
        assert_eq!(view.revision, 1);
        assert_eq!(view.tokens, 1);

        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        atomic_private_write(&path, &bytes).unwrap();
        assert!(SessionStore::under(&home).restore(id).is_err());
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn restart_restore_runs_the_complete_bundle_validator() {
        let home = std::env::temp_dir().join(format!("muser-invalid-bundle-{}", random_id()));
        let store = SessionStore::under(&home);
        let id = "invalid-state";

        let mut invalid = bundle(id);
        invalid.target_logits[0] = f32::NAN;
        persist_test_bundle(&store, &invalid);
        assert!(store.restore(id).unwrap_err().contains("frontier state"));
        assert!(store.get(id).unwrap().is_none());

        let mut invalid = bundle(id);
        invalid.canonical_replay_plan_json = "{".into();
        persist_test_bundle(&store, &invalid);
        assert!(store.restore(id).unwrap_err().contains("replay plan"));
        assert!(store.get(id).unwrap().is_none());

        let mut invalid = bundle(id);
        invalid.target.tokens = Arc::from([muser_engine::EMBEDDING_POSITION_WITNESS]);
        invalid.position_witnesses = vec![muser_engine::EMBEDDING_POSITION_WITNESS];
        invalid.sampler_history = vec![muser_engine::EMBEDDING_POSITION_WITNESS];
        invalid.vision_rows = vec![vec![f32::INFINITY]];
        invalid.vision_projector_sha256 = Some("22".repeat(32));
        invalid.vision_preprocessing_sha256 = Some("33".repeat(32));
        persist_test_bundle(&store, &invalid);
        assert!(store.restore(id).unwrap_err().contains("vision rows"));
        assert!(store.get(id).unwrap().is_none());

        let mut invalid = bundle(id);
        invalid.layout_abi = "future-layout".into();
        persist_test_bundle(&store, &invalid);
        assert!(store.restore(id).unwrap_err().contains("model or layout"));
        assert!(store.get(id).unwrap().is_none());

        let mut invalid = bundle(id);
        invalid.dflash_identity_sha256 = None;
        persist_test_bundle(&store, &invalid);
        assert!(store
            .restore(id)
            .unwrap_err()
            .contains("assistant or vision identity"));
        assert!(store.get(id).unwrap().is_none());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let valid = bundle(id);
            let bundle_path = persist_test_bundle(&store, &valid);
            assert_eq!(
                fs::metadata(&bundle_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(&store.key_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(&store.root).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn materialized_transfer_is_never_replaced_after_resume_failure() {
        let home = std::env::temp_dir().join(format!("muser-transfer-resume-{}", random_id()));
        let store = SessionStore::under(&home);
        let session_id = "resume-session";
        let transfer_id = "resume-transfer";
        store.create(Some(session_id)).unwrap();
        assert!(matches!(
            store.begin(session_id, 0, "request", [1; 32]).unwrap(),
            BeginMutation::Started(None)
        ));
        store
            .commit(
                session_id,
                0,
                "request",
                [1; 32],
                bundle(session_id),
                CachedGeneration {
                    text: "ok".into(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    finish_reason: "stop".into(),
                    seed: 1,
                    revision: 0,
                    context: vec![42],
                    sampled_tokens: vec![42],
                },
            )
            .unwrap();
        let export = store
            .begin_export(
                transfer_id,
                session_id,
                "https://destination",
                "move",
                "decode",
            )
            .unwrap();
        atomic_private_write(&export.payload, b"corrupt-but-private").unwrap();
        let journal = store.transfer_record_path(transfer_id);
        let journal_before = fs::read(&journal).unwrap();
        let payload_before = fs::read(&export.payload).unwrap();

        assert!(store.resume_export(transfer_id).is_err());
        let replacement_error = store
            .begin_export(
                transfer_id,
                session_id,
                "https://destination",
                "move",
                "decode",
            )
            .err()
            .expect("materialized transfer replacement must fail");
        assert!(replacement_error.contains("resume it instead"));
        assert_eq!(fs::read(&journal).unwrap(), journal_before);
        assert_eq!(fs::read(&export.payload).unwrap(), payload_before);
        assert!(store
            .register_outgoing(
                "wrong-tier",
                session_id,
                "destination",
                "copy",
                "gx10-decode",
            )
            .is_err());
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn two_phase_transfer_commits_before_source_move_and_is_idempotent() {
        let base = std::env::temp_dir().join(format!("muser-transfer-test-{}", random_id()));
        let source_home = base.join("source");
        let destination_home = base.join("destination");
        let source = SessionStore::under(&source_home);
        let destination = SessionStore::under(&destination_home);
        let id = "migrated-session";
        let transfer_id = "transfer-one";
        source.create(Some(id)).unwrap();
        assert!(matches!(
            source.begin(id, 0, "request", [1; 32]).unwrap(),
            BeginMutation::Started(None)
        ));
        source
            .commit(
                id,
                0,
                "request",
                [1; 32],
                bundle(id),
                CachedGeneration {
                    text: "ok".into(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    finish_reason: "stop".into(),
                    seed: 3,
                    revision: 0,
                    context: vec![42],
                    sampled_tokens: vec![42],
                },
            )
            .unwrap();
        source
            .register_outgoing(transfer_id, id, "https://destination", "move", "decode")
            .unwrap();
        let export = source
            .begin_export(transfer_id, id, "https://destination", "move", "decode")
            .unwrap();
        assert!(destination
            .prepare_import(
                transfer_id,
                id,
                "source",
                "rename",
                "decode",
                export.view.bytes,
                &export.view.sha256,
                export.transport_key,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .is_err());
        assert!(destination
            .prepare_import(
                transfer_id,
                id,
                "source",
                "move",
                "gx10-decode",
                export.view.bytes,
                &export.view.sha256,
                export.transport_key,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .is_err());
        destination
            .prepare_import(
                transfer_id,
                id,
                "source",
                "move",
                "decode",
                export.view.bytes,
                &export.view.sha256,
                export.transport_key,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .unwrap();
        // Identical prepare retries are idempotent, but a transfer ID can
        // never be rebound by changing any semantic field while reusing the
        // same encrypted payload.
        destination
            .prepare_import(
                transfer_id,
                id,
                "source",
                "move",
                "decode",
                export.view.bytes,
                &export.view.sha256,
                export.transport_key,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .unwrap();
        assert!(destination
            .prepare_import(
                transfer_id,
                id,
                "source",
                "copy",
                "decode",
                export.view.bytes,
                &export.view.sha256,
                export.transport_key,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .is_err());
        let (staging, _) = destination.payload_path(transfer_id).unwrap();
        fs::copy(&export.payload, &staging).unwrap();
        set_mode(&staging, 0o600).unwrap();
        destination.accept_payload(transfer_id, &staging).unwrap();
        let wrong_dflash = "ff".repeat(32);
        assert!(destination
            .commit_import(
                transfer_id,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                Some(&wrong_dflash),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .is_err());
        assert!(destination.get(id).unwrap().is_none());
        assert_eq!(
            destination.transfer(transfer_id).unwrap().status,
            "uploaded"
        );
        let committed = destination
            .commit_import(
                transfer_id,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .unwrap();
        assert_eq!(committed.status, "committed");
        assert_eq!(destination.get(id).unwrap().unwrap().revision, 1);
        assert!(destination
            .commit_import(
                transfer_id,
                "wrong-model-after-commit",
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .is_err());
        // ACK reconciliation repeats the commit without changing the cut.
        assert_eq!(
            destination
                .commit_import(
                    transfer_id,
                    &export.model_sha256,
                    export.tokenizer_sha256,
                    export.template_sha256,
                    &export.layout_abi,
                    export.dflash_identity_sha256.as_deref(),
                    export.vision_projector_sha256.as_deref(),
                    export.vision_preprocessing_sha256.as_deref(),
                )
                .unwrap()
                .status,
            "committed"
        );
        assert!(source.get(id).unwrap().is_some());
        let remote_copy = base.join("remote-storage.bundle");
        fs::copy(&export.payload, &remote_copy).unwrap();
        assert!(source.delete_after_transfer(id, transfer_id).is_err());
        assert!(source.get(id).unwrap().is_some());
        source
            .update_transfer(transfer_id, "destination_committed", None, false)
            .unwrap();
        assert!(source.reconcile_outgoing_after_ack(transfer_id).unwrap());
        assert!(source.get(id).unwrap().is_none());
        let terminal = source.transfer(transfer_id).unwrap();
        assert_eq!(terminal.status, "completed");
        assert!(terminal.source_deleted);
        // A retry after an ambiguous client failure is terminal and must not
        // require the source bundle or destination to still be reachable.
        assert!(source.reconcile_outgoing_after_ack(transfer_id).unwrap());
        source.remove_transfer_payload(transfer_id).unwrap();
        let staging = source.transfer_root.join("storage-restore.staging");
        fs::copy(&remote_copy, &staging).unwrap();
        set_mode(&staging, 0o600).unwrap();
        source.adopt_export_payload(transfer_id, &staging).unwrap();
        assert_eq!(source.restore_export(transfer_id).unwrap().revision, 1);
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn transfer_rejects_tamper_and_wrong_abi_without_publishing_a_session() {
        let base = std::env::temp_dir().join(format!("muser-transfer-fail-{}", random_id()));
        let source = SessionStore::under(&base.join("source"));
        let destination = SessionStore::under(&base.join("destination"));
        let id = "fail-closed";
        let transfer_id = "transfer-failure";
        source.create(Some(id)).unwrap();
        assert!(matches!(
            source.begin(id, 0, "request", [1; 32]).unwrap(),
            BeginMutation::Started(None)
        ));
        source
            .commit(
                id,
                0,
                "request",
                [1; 32],
                bundle(id),
                CachedGeneration {
                    text: "ok".into(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    finish_reason: "stop".into(),
                    seed: 1,
                    revision: 0,
                    context: vec![42],
                    sampled_tokens: vec![42],
                },
            )
            .unwrap();
        let export = source
            .begin_export(transfer_id, id, "destination", "copy", "decode")
            .unwrap();
        destination
            .prepare_import(
                transfer_id,
                id,
                "source",
                "copy",
                "decode",
                export.view.bytes,
                &export.view.sha256,
                export.transport_key,
                &export.model_sha256,
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .unwrap();
        let (staging, _) = destination.payload_path(transfer_id).unwrap();
        let mut bytes = fs::read(&export.payload).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        atomic_private_write(&staging, &bytes).unwrap();
        assert!(destination.accept_payload(transfer_id, &staging).is_err());
        assert!(destination.get(id).unwrap().is_none());

        let (staging, _) = destination.payload_path(transfer_id).unwrap();
        fs::copy(&export.payload, &staging).unwrap();
        set_mode(&staging, 0o600).unwrap();
        destination.accept_payload(transfer_id, &staging).unwrap();
        assert!(destination
            .commit_import(
                transfer_id,
                "wrong-model",
                export.tokenizer_sha256,
                export.template_sha256,
                &export.layout_abi,
                export.dflash_identity_sha256.as_deref(),
                export.vision_projector_sha256.as_deref(),
                export.vision_preprocessing_sha256.as_deref(),
            )
            .is_err());
        assert!(destination.get(id).unwrap().is_none());
        assert_eq!(
            destination.transfer(transfer_id).unwrap().status,
            "uploaded"
        );
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn corrupt_transfer_journals_are_never_replaced_as_new_or_resumed() {
        let base = std::env::temp_dir().join(format!("muser-transfer-journal-{}", random_id()));
        let store = SessionStore::under(&base);
        let session_id = "journal-session";
        let transfer_id = "journal-transfer";
        store.create(Some(session_id)).unwrap();
        assert!(matches!(
            store.begin(session_id, 0, "request", [1; 32]).unwrap(),
            BeginMutation::Started(None)
        ));
        store
            .commit(
                session_id,
                0,
                "request",
                [1; 32],
                bundle(session_id),
                CachedGeneration {
                    text: "ok".into(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    finish_reason: "stop".into(),
                    seed: 1,
                    revision: 0,
                    context: vec![42],
                    sampled_tokens: vec![42],
                },
            )
            .unwrap();
        fs::create_dir_all(&store.transfer_root).unwrap();
        set_mode(&store.transfer_root, 0o700).unwrap();
        let journal = store.transfer_record_path(transfer_id);
        atomic_private_write(&journal, b"not-json").unwrap();
        let original = fs::read(&journal).unwrap();

        assert!(store
            .register_outgoing(
                transfer_id,
                session_id,
                "https://destination",
                "copy",
                "decode",
            )
            .is_err());
        assert!(store
            .prepare_import(
                transfer_id,
                session_id,
                "source",
                "copy",
                "decode",
                1,
                &"0".repeat(64),
                [0; 32],
                "model",
                [0; 32],
                [0; 32],
                "abi",
                None,
                None,
                None,
            )
            .is_err());
        assert_eq!(fs::read(&journal).unwrap(), original);
        fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn atomic_private_write_reports_every_durability_phase_failure() {
        let base = std::env::temp_dir().join(format!("muser-atomic-fail-{}", random_id()));
        fs::create_dir_all(&base).unwrap();
        for stage in ["write", "file-fsync", "rename", "directory-fsync"] {
            let path = base.join(format!("{stage}.json"));
            inject_atomic_failure(stage);
            assert!(atomic_private_write(&path, b"material").is_err(), "{stage}");
            assert!(
                !base
                    .read_dir()
                    .unwrap()
                    .flatten()
                    .any(|entry| entry.file_name().to_string_lossy().contains(".tmp-")),
                "{stage} left a staging file"
            );
        }
        fs::remove_dir_all(&base).unwrap();
    }
}
