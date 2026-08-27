---
name: muser-bench-ladder
description: Run or audit Muser's local performance matrix, DFlash ladder, disaggregated prefill cells, and kvpack reuse ladder with exactness gates and retained receipts. Use when reproducing benchmarks, comparing Muser with pinned llama.cpp, validating a performance change, or preparing evidence without inventing launch claims.
---

# Run the Muser benchmark ladder

Work from the repository root. Read `AGENTS.md`, `docs/benchmarks.md`, and
`docs/kvpack.md`. For disaggregated cells, also read
`docs/disaggregated-prefill.md`; for tool arguments, use each script's
`--help` rather than copying a historical command blindly.

## Define the question before running

Choose the smallest ladder that answers the task:

- plain local target: decode and prefill at the requested context depths;
- local DFlash: target-plus-draft exactness, acceptance, decode, prefill, and
  wall time;
- disaggregated prefill: local-versus-remote TTFT plus handoff exactness and
  transport health;
- kvpack warm reuse: cold, warm-hit, and unrelated-prompt miss control;
- kvpack delta: full and suffix-only arms with identical output digest.

Record the source commit, frozen model/engine/hardware identity, comparator
revision, model digests, metallib receipt, fixtures, depths, output length,
repetition policy, and acceptance thresholds before execution. Do not change
the matrix after seeing results.

## Preserve comparability

- Use the source-pinned llama.cpp comparator and the same pinned model.
- Treat ratios as `llama / muser`; record raw measurements as well as ratios.
- Gate every synthetic performance cell on exact target tokens. Apply the
  lane's required full-logit and DFlash-token gates where specified.
- Keep warmups, counted repetitions, cooldowns, context lengths, and output
  lengths identical across arms.
- Report wall time when phase-accounting boundaries differ. Never promote an
  asymmetric phase ratio as an end-to-end speedup.
- Retain misses and rejected cells. A failed exactness gate cannot become a
  performance result.

Representative entry points are:

```sh
python3 scripts/representative_target_smoke.py --help
python3 scripts/representative_dflash_smoke.py --help
python3 scripts/qualify_nvfp4_fast.py --help
```

Use `rg --files scripts | rg 'kvpack|prefix|delta|handoff'` to locate the
current kvpack and handoff drivers, then read their `--help` and referenced
methodology before use. Do not infer a command from a receipt produced by an
older revision.

## Serialize accelerator cells

All Metal, llama.cpp, Core ML, and remote qualification execution must be a
child of `scripts/accelerator_safe.py`. It is dry-run by default. Give every
cell a unique identity and fresh evidence directory, review the planned
command, then add `--execute` only when the operator authorized the run.

Do not run parallel Metal tests on one host. For node cells, coordinate the
whole attempt, enforce the single-producer rule, and retain node state before
and after. Keep replay ledgers, locks, sockets, and other operational state on
the internal disk; evidence belongs in the append-only evidence volume.

## Apply the ladder gates

For every cell, retain:

- the accelerator command log and result receipt;
- per-repetition raw JSON/JSONL and exactness fields;
- means, medians where specified, CV, warmup disposition, and failures;
- target-token agreement and required logit/DFlash comparisons;
- for remote cells, installed bytes, transfer time, payload rate, route, and
  before/after node state;
- for kvpack, prefix identity, hit/miss disposition, full/delta bytes, and
  output digest equality.

Warm-prefix evidence is incomplete without an unrelated-prompt miss control.
Delta evidence is incomplete without a full-handoff arm and identical output
digest. Remote TTFT evidence is incomplete if the receiver fell back locally.

## Audit and report

Use `scripts/gx10/handoff_report.py` to attribute remote phases. Compare the
result only with the scope and caveats already recorded in
`docs/benchmarks.md` and the append-only campaign ledger. If a new run changes
a publishable number, add the receipt-backed ledger entry first and leave
launch wording to the operator-review process in `docs/launch-claims.md`.

End with a cell-by-cell pass/fail table, exact receipt paths, exactness
results, rejected cells, and environmental caveats. Never average away a
failed gate or copy a number without its depth, lane, and denominator.

Blind performance testing and the final public-claim decision remain
operator-side.
