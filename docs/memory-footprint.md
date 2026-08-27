# Memory allocation arithmetic

This is a topology-derived allocation estimate, not a measured peak-RSS result
and not launch guidance for unqualified Macs. The v0.1 release contract is four
full-context slots on the 96 GB M3 Ultra. No smaller-memory configuration may
be advertised as supported until a retained hardware qualification measures
it.

## Pinned geometry

The release GGUF is the authority. A real-model load test asserts:

| Field | Value |
|---|---:|
| layers | 52 |
| sliding/RoPE layers | 39 |
| full/NoPE layers | 13 |
| sliding window | 2,048 tokens |
| KV heads | 2 |
| head dimension | 128 |
| maximum context | 131,072 tokens |

`MuseConfig::sliding_window` is parsed from GGUF metadata and the pinned
artifact resolves to 2,048. Test cuts at 2,559/2,560 exercise behavior well
after ring wrap; they are not evidence of a 2,560-token window.

## KV formula

K and V are separate f16 buffers, so one cached row in one layer is:

```text
2 KV heads * 128 values * 2 bytes * (K + V) = 1,024 bytes
```

For one slot configured with context `C`:

```text
swa_rows  = min(C, 2,048)
nope_rows = C
slot_kv_bytes = (39 * swa_rows + 13 * nope_rows) * 1,024
```

The release configuration has four independently stateful slots. The KV
allocation ceiling is therefore four times the one-slot number:

| Context per slot | One-slot KV | Four-slot KV |
|---:|---:|---:|
| 8,192 | 0.191 GB | 0.763 GB |
| 32,768 | 0.518 GB | 2.072 GB |
| 131,072 | 1.827 GB | 7.306 GB |

These are decimal GB and describe target KV planes only. A staging generation
is used for restore/context shift, so an operation can temporarily require
additional state even though it is never a fifth serving slot.

## Other material allocations

The artifact manifest records these on-disk sizes:

| Artifact | Bytes | Loading behavior |
|---|---:|---|
| target GGUF | 16,756,681,056 | mmap/page-cache backed |
| DFlash GGUF | 1,631,205,312 | loaded only when configured |
| vision projector | 1,400,328,928 | loaded only when configured |

On-disk size is not the same as resident or wired memory. The process also
owns shared Metal pipelines and workspaces, per-slot logits/sampler state,
DFlash state, image embeddings, network buffers, and temporary
restore/migration material. An explicitly selected experimental/post-release
ANE process also owns `MLState`, but ANE is absent from the v0.1 candidate and
never selected by `auto`. Peaks depend on request mix and backend. Summing
artifact sizes with the KV formula is therefore only a lower bound, not a safe
RAM recommendation.

The prefill driver chunks at 512 positions. A simple sum of the major f32
batch-activation widths is about 0.99 GB, but buffers are reused and additional
temporary/capture buffers exist. That arithmetic must not be labeled peak RSS.

## Allocation behavior

Metal KV buffers are allocated without a CPU memset. Pages are committed as
the GPU touches rows instead of being eagerly zero-filled at session creation.
This reduces startup pressure but does not reduce the eventual full-context
allocation ceiling.

## Release requirement

The final serving benchmark must retain process and system memory evidence for
the exact four-slot binary, artifacts, context cell, v0.1 Metal DFlash/vision
route, and concurrency. Experimental ANE has its own post-release research
evidence and is not a v0.1 memory gate. Until that matrix passes, this document
supports engineering capacity checks only. It does not authorize 24 GB, 32 GB,
or 64 GB product claims.
