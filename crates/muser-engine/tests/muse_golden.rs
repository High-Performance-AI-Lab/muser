#![cfg(feature = "release-real-model")]

//! Mandatory pinned-artifact Muse correctness gates. Ordinary CI does not
//! enable this explicit release feature; when enabled, absence is failure.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, SessionConfig};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

const MODEL_SHA256: &str = "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8";
const MODEL_BYTES: u64 = 16_756_681_056;
const LLAMA_COMMIT: &str = "89e0aa6fd362617d9073e0dafc18e41241521572";

fn model_path() -> &'static Path {
    static MODEL: OnceLock<PathBuf> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let declared = std::env::var("MUSER_MODEL_SHA256")
                .expect("release-real-model requires MUSER_MODEL_SHA256");
            assert_eq!(declared, MODEL_SHA256, "unknown release-model identity");
            let path = PathBuf::from(
                std::env::var("MUSER_MODEL").expect("release-real-model requires MUSER_MODEL"),
            );
            let metadata = std::fs::symlink_metadata(&path)
                .expect("release-real-model MUSER_MODEL must exist");
            assert!(
                metadata.file_type().is_file(),
                "MUSER_MODEL must be a regular file"
            );
            assert_eq!(metadata.len(), MODEL_BYTES, "release model byte size");
            let mut file = std::fs::File::open(&path).expect("open release model");
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 1024 * 1024];
            loop {
                let count = file.read(&mut buffer).expect("hash release model");
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            assert_eq!(format!("{:x}", hash.finalize()), MODEL_SHA256);
            path
        })
        .as_path()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GreedyFixture {
    id: String,
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
    oracle: GreedyOracle,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GreedyOracle {
    implementation: String,
    commit: String,
    model_sha256: String,
    temperature: f64,
    generated_tokens_sha256: String,
    parity_receipt: String,
}

fn greedy_fixture() -> GreedyFixture {
    let fixture: GreedyFixture =
        serde_json::from_str(include_str!("fixtures/greedy_p1_7e9b74b7.json"))
            .expect("checked-in release-model greedy fixture");
    assert_eq!(fixture.id, "p1-official");
    assert_eq!(fixture.oracle.implementation, "llama.cpp");
    assert_eq!(fixture.oracle.commit, LLAMA_COMMIT);
    assert_eq!(fixture.oracle.model_sha256, MODEL_SHA256);
    assert_eq!(fixture.oracle.temperature, 0.0);
    assert_eq!(fixture.oracle.parity_receipt, "muser.greedy-parity.v1");
    let mut digest = Sha256::new();
    for token in &fixture.gen_tokens {
        digest.update(token.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", digest.finalize()),
        fixture.oracle.generated_tokens_sha256
    );
    fixture
}

#[test]
fn historical_muse_gguf_loads_with_release_geometry() {
    let path = model_path();
    let model = Model::load(ModelConfig::new(path)).expect("Muse GGUF must load");
    let config = model.config();
    assert_eq!(config.n_layers, 52);
    assert_eq!(config.hidden_dim, 6_656);
    assert_eq!(config.n_heads, 32);
    assert_eq!(config.n_kv_heads, 2);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.intermediate_dim, 19_968);
    assert_eq!(config.vocab_size, 202_048);
    assert_eq!(config.sliding_window, 2_048);
    assert_eq!(config.context_length, 131_072);
    // The pinned GGUF declares EOS and EOT. Its vocabulary also contains an
    // EOM control piece, but no tokenizer.ggml.eom_token_id metadata key.
    assert_eq!(config.eos_tokens, vec![200_001, 200_008]);
    assert_eq!(
        config
            .layer_kinds
            .iter()
            .filter(|kind| kind.is_swa())
            .count(),
        39
    );
}

#[test]
fn cpu_reference_matches_exact_greedy_fixture() {
    let path = model_path();
    let GreedyFixture {
        prompt_tokens: prompt,
        gen_tokens: expected,
        ..
    } = greedy_fixture();
    assert!(!prompt.is_empty(), "fixture prompt cannot be empty");
    assert!(!expected.is_empty(), "fixture output cannot be empty");

    let model = Model::load(ModelConfig::new(path)).expect("Muse GGUF must load");
    let mut session = model
        .new_session(SessionConfig {
            max_context: prompt.len() + expected.len(),
        })
        .expect("session");
    let prefill = session
        .prefill(PrefillBatch::tokens(prompt))
        .expect("prefill");
    assert!(prefill.logits.iter().all(|value| value.is_finite()));
    let first = prefill
        .last_logits()
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .expect("nonempty logits");
    assert_eq!(first, expected[0], "greedy token 0");

    for (step, pair) in expected.windows(2).enumerate() {
        let decoded = session
            .decode(DecodeInput { token_id: pair[0] })
            .expect("decode");
        assert!(decoded.logits.iter().all(|value| value.is_finite()));
        assert_eq!(decoded.next_token, pair[1], "greedy token {}", step + 1);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_one_token_matches_cpu_reference_logits() {
    let path = model_path();
    let token = greedy_fixture().prompt_tokens[0];
    let model = Model::load(ModelConfig::new(path)).expect("Muse GGUF must load");
    let config = SessionConfig { max_context: 1 };
    let mut cpu = model.new_session(config).expect("CPU session");
    let mut metal = model.new_metal_session(config).expect("Metal session");

    let cpu_logits = cpu
        .prefill(PrefillBatch::tokens(vec![token]))
        .expect("CPU prefill")
        .logits;
    let metal_logits = metal
        .prefill(PrefillBatch::tokens(vec![token]))
        .expect("Metal prefill")
        .logits;
    assert_eq!(metal.position(), 1);
    assert!(metal_logits.iter().all(|value| value.is_finite()));
    let max_error = cpu_logits
        .iter()
        .zip(&metal_logits)
        .map(|(cpu, gpu)| (cpu - gpu).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 0.5, "CPU/Metal max logit error {max_error}");
    assert_eq!(greedy_id(&metal_logits), greedy_id(&cpu_logits));

    metal.reset();
    let repeated = metal
        .prefill(PrefillBatch::tokens(vec![token]))
        .expect("Metal prefill after reset")
        .logits;
    assert_eq!(metal_logits, repeated, "reset must reproduce exact logits");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_batched_prompt_matches_cpu_and_llama_greedy() {
    let path = model_path();
    let fixture = greedy_fixture();
    let prompt_len = fixture.prompt_tokens.len();
    let model = Model::load(ModelConfig::new(path)).expect("Muse GGUF must load");
    let config = SessionConfig {
        max_context: fixture.prompt_tokens.len(),
    };
    let mut cpu = model.new_session(config).expect("CPU session");
    let mut metal = model.new_metal_session(config).expect("Metal session");
    let cpu = cpu
        .prefill(PrefillBatch::tokens(fixture.prompt_tokens.clone()))
        .expect("CPU prompt");
    let gpu = metal
        .prefill(PrefillBatch::tokens(fixture.prompt_tokens))
        .expect("Metal prompt");

    assert_eq!(gpu.tokens_processed, prompt_len);
    assert_eq!(gpu.logits.len(), model.config().vocab_size);
    assert!(gpu.logits.iter().all(|value| value.is_finite()));
    let max_error = cpu
        .last_logits()
        .iter()
        .zip(gpu.last_logits())
        .map(|(cpu, gpu)| (cpu - gpu).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 0.5, "CPU/Metal max logit error {max_error}");
    assert_eq!(greedy_id(gpu.last_logits()), fixture.gen_tokens[0] as usize);
    assert_eq!(greedy_id(gpu.last_logits()), greedy_id(cpu.last_logits()));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_prefill_and_decode_match_exact_llama_greedy_fixture() {
    let path = model_path();
    let fixture = greedy_fixture();
    let model = Model::load(ModelConfig::new(path)).expect("Muse GGUF must load");
    let mut metal = model
        .new_metal_session(SessionConfig {
            max_context: fixture.prompt_tokens.len() + fixture.gen_tokens.len(),
        })
        .expect("Metal session");
    let prefill = metal
        .prefill(PrefillBatch::tokens(fixture.prompt_tokens))
        .expect("Metal prompt");
    assert_eq!(
        greedy_id(prefill.last_logits()),
        fixture.gen_tokens[0] as usize,
        "greedy token 0"
    );
    for (step, pair) in fixture.gen_tokens.windows(2).enumerate() {
        let decoded = metal
            .decode(DecodeInput { token_id: pair[0] })
            .expect("Metal decode");
        assert!(decoded.logits.iter().all(|value| value.is_finite()));
        assert_eq!(decoded.next_token, pair[1], "greedy token {}", step + 1);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn greedy_id(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("nonempty logits")
}
