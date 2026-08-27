# muser-engine

`muser-engine` is the reusable Muse Glimmer inference core from Muser. It
loads the pinned Muse GGUF shape, tokenizes text, and exposes a small
`Model`/`Session` API for CPU-reference inference and opt-in Apple Metal
inference. HTTP serving, node enrollment, telemetry, and release orchestration
belong to the other Muser crates.

The dependency boundary is engine-only. This crate does not depend on
kvpack. Projects that need durable or remote KV reuse should add
`muser-kvpack` separately and keep its policy/storage lifecycle outside the
model session.

## Use it by path

The crate is currently `publish = false`. Organization projects should use a
path or private git dependency; there is no crates.io or docs.rs publication
in the v0.1 plan.

```toml
[dependencies]
muser-engine = { path = "../muser/crates/muser-engine", default-features = false }
```

The default feature set is deliberately empty, so omitting
`default-features = false` has the same CPU-safe result. Keeping it explicit
in downstream manifests makes the selected backend easy to audit.

## Model and Session

`Model` owns immutable, memory-mapped weights, model metadata, and tokenizer
state. `Session` owns one mutable sequence: its KV cache, context position,
token history, and retained next-token distribution. Create a separate
session for each concurrently active sequence.

```rust,no_run
use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, SessionConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::load(ModelConfig::new("/models/muse-glimmer.gguf"))?;
    let prompt = model.encode("Explain why KV caching helps decode.");
    let mut session = model.new_session(SessionConfig {
        max_context: prompt.len() + 32,
    })?;

    let prefill = session.prefill(PrefillBatch::tokens(prompt))?;
    assert_eq!(prefill.last_logits().len(), model.config().vocab_size);

    let first_token = session.greedy_next_token()?;
    let decoded = session.decode(DecodeInput {
        token_id: first_token,
    })?;
    println!("{}", model.decode_tokens(&[first_token, decoded.next_token]));
    Ok(())
}
```

`new_session` is always the CPU correctness path. It is intentionally simple
and deterministic; it is not the recommended high-throughput serving path
for the 30B model.

## Features

| Feature | Default | Effect |
|---|---:|---|
| `metal` | no | On macOS, adds the Metal decode/prefill backend and `new_metal_session` APIs. Metal runtime creation fails closed when a tensor needs the pinned llama.cpp metallib and `MUSER_GGML_METALLIB` does not name it. |
| `ane-coreml` | no | On macOS, adds the experimental public-CoreML ANE modules. It is outside the v0.1 automatic serving route. |
| `release-real-model` | no | Enables repository release tests that require the exact pinned model and explicit identity environment. Downstream applications should not enable it. |

Enable Metal explicitly in a downstream crate:

```toml
[dependencies]
muser-engine = { path = "../muser/crates/muser-engine", default-features = false, features = ["metal"] }
```

The `metal` feature is compile-time capability only. Loading a Metal session
is a runtime action and must follow Muser's accelerator lock discipline when
the machine is shared.

## Verify the downstream boundary

The tracked fixture under `tests/downstream` depends on this crate exactly as
another repository would. From the Muser root, both dependency shapes build
without loading a model or starting an engine:

```sh
cargo check --manifest-path crates/muser-engine/tests/downstream/Cargo.toml
cargo check --manifest-path crates/muser-engine/tests/downstream/Cargo.toml --features metal
```
