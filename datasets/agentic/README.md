# muser agentic dataset

A small, self-contained, **deterministically-scorable** agentic task set so muser can *visibly
execute* agentic work in the executive demo — prompt → tool calls → verified PASS/FAIL — not just
emit text. It exercises the tool-use / multi-step-planning path of Muse Glimmer-30B, an open
agentic model, through muser-server's OpenAI-compatible endpoint.

- **Size:** 24 tasks · **5 categories** · easy/medium/hard.
- **Provenance:** **100% original synthetic content** authored for muser. **0 third-party records
  redistributed.** Deliberate — see [§ Provenance & licenses](#provenance--licenses).
- **Scoring:** deterministic checkers only (exact-match / numeric / set / JSON / predicate /
  final-state). **No LLM-judge** on the demo path.
- **Self-consistent:** every task's checker is proven to pass on its reference solution, offline,
  with no model in the loop (`harness/validate.py` → 24/24).

---

## Layout

```
datasets/agentic/
├─ README.md                 # this file
├─ RUNNER.md                 # how muser executes + scores a task; 3 golden examples worked
├─ schema.json               # JSON Schema for one task record
├─ tasks.jsonl               # the 24 tasks (one JSON object per line)
├─ golden/                   # 3 fully-worked end-to-end trajectories + checker verdicts
│  ├─ golden-01-lookup.json
│  ├─ golden-02-calc-chain.json
│  └─ golden-03-code-edit.json
└─ harness/                  # stdlib-only, zero-dependency reference tooling
   ├─ agentic_harness.py     # deterministic stub executor + checkers + answer extraction (the semantics)
   ├─ build_tasks.py         # reproducible generator → tasks.jsonl (asserts every checker passes)
   ├─ validate.py            # schema + no-leak + replay + checker gate (CI); --emit-golden
   └─ run.py                 # live runner vs muser-server /v1/chat/completions; --mock for offline
```

## The record schema (`schema.json`)

Each JSONL line is one task. Fields:

| field | purpose |
|-------|---------|
| `id` | stable id, `^[a-z]+-[0-9]{3}$` (e.g. `plan-004`) |
| `category` | one of `information_lookup`, `calculation`, `multi_tool_planning`, `code_edit`, `verification_chain` |
| `difficulty` | `easy` \| `medium` \| `hard` |
| `prompt` | the user turn shown to the model |
| `world` *(opt)* | seed/mutable state (`files`, `kv`) for stateful builtins |
| `tools` | tool defs the runner exposes: `name`, `description`, JSON-Schema `parameters`, and a deterministic **`stub`** (`fixture` table or `builtin` fn) |
| `expected` | oracle target for the checker |
| `checker` | deterministic pass/fail rule (`kind` + options) |
| `reference_solution` | ideal tool-call trajectory — proves the task is solvable and the checker correct; **runner-side only** |

**The model is shown only `prompt` + each tool's `name/description/parameters`.** `stub`,
`expected`, `checker`, and `reference_solution` are oracle data and never enter the model context
(`validate.py` asserts the projection leaks none of them). Full loop, stub semantics, checker
table, and the Muse dual-EOS (`200001`/`200008`) template notes are in **[RUNNER.md](RUNNER.md)**.

## Contents

**By category (24 total):**

| category | count | what it exercises | example checkers |
|----------|:----:|-------------------|------------------|
| `information_lookup` | 6 | `search`/`open_page`/`kb_lookup`; extract a fact | numeric, exact, set, contains_all |
| `calculation` | 5 | `calculator` (+ `convert_units`, kb); compute/verify | numeric, bool |
| `multi_tool_planning` | 6 | 2–4 tools in sequence, incl. a conditional branch and a stateful write | numeric, exact, set, final_state |
| `code_edit` | 4 | mock filesystem: read → patch/write | final_state |
| `verification_chain` | 3 | check a claim / validate JSON across tools | bool, json_equal |

**By difficulty:** easy 6 · medium 12 · hard 6.

**Checkers used:** `numeric_equal` (9), `final_state` (5), `set_equal` (3), `exact_match` (2),
`bool_equal` (2), `json_equal` (2), `contains_all` (1).

---

## Provenance & licenses

### The split: 24 synthetic / 0 sourced

Every task is **original content authored for muser** and generated reproducibly by
`harness/build_tasks.py` (no scraping, no network, no clock, no randomness). **No third-party
dataset record is copied or redistributed.** The dataset's own license is the muser repo license,
**Apache-2.0 OR MIT** (see workspace `Cargo.toml`). Names like *Testburg / Testcorp / Mount
Testmore* are invented; factual tasks (boiling point of water, Rust's first-appearance year,
speed of light) rely on public-domain facts, not on any dataset's expression of them.

### Why synthesize instead of source

The task brief said to prefer permissively-licensed sources and, **when unsure, to synthesize
rather than ship license-dirty data to an executive demo.** I researched the public landscape and
concluded that *for this specific use* — a redistributable, self-contained, single-box live demo —
authoring original tasks is both cleaner and better-fit than adapting any external set:

- **Gating** breaks "self-contained." The most-cited function-calling set (Salesforce xLAM) is
  CC-BY-4.0 **but gated** behind Hugging Face login + terms acceptance — you cannot drop a
  redistributable file into the repo cleanly.
- **Data provenance** invites awkward questions. Several permissive sets (Glaive, xLAM, ToolBench)
  are **model-generated or API-derived**; "was this produced within the upstream provider's ToS?"
  is not a question you want raised in front of executives.
- **Copyleft / attribution chains.** Share-alike (HotpotQA, CC-BY-SA-4.0) and CC-BY attribution
  add carry-forward obligations to any shipped derivative.
- **Weight / hosted-env.** WebArena, SWE-bench, and GAIA need Docker or live web environments (and
  GAIA withholds test answers) — none is a self-contained, deterministic, one-box demo.
- **Exact fit.** Synthetic tasks are tailored to muser's OpenAI tool schema and to **deterministic
  checkers with in-record stubs**, which is exactly what makes the score live and reproducible.

Task **formats** (function-calling turns, GSM8K-style verification, filesystem edits) are common
ideas, not copyrightable expression; I drew on the clean-license benchmarks below as **format
inspiration only** and copied no data.

### Candidates researched (licenses recorded)

Verified **this session via WebFetch of the LICENSE file / dataset card on 2026-08-11** unless
marked *(from documented knowledge)*. WebSearch budget for the session was exhausted, so broad
discovery was limited to WebFetch verification of these high-profile candidates plus prior
knowledge; that uncertainty is itself a reason the shipped data is 100% original.

| Candidate | License | Redistribute cleanly for this demo? | URL |
|-----------|---------|-------------------------------------|-----|
| **GSM8K** (grade-school math; calc/verify format) | **MIT** ✅ verified | Clean license, but word-problems ≠ agentic tool-use; would need re-wrapping. Used as **format inspiration** for the calculation/verification chains. | https://github.com/openai/grade-school-math |
| **τ-bench (tau-bench)** (tool-agent tasks) | **MIT** ✅ verified | Clean, but domains (airline/retail) need hosted DB state + are heavier than a live tile. **Format inspiration** for multi-tool planning. | https://github.com/sierra-research/tau-bench |
| **Gorilla / BFCL** (function-calling leaderboard) | **Apache-2.0** ✅ verified | Code Apache-2.0; entry provenance mixed across sources. **Format inspiration** for the tool-schema shape. | https://github.com/ShishirPatil/gorilla |
| **Salesforce xLAM function-calling-60k** | **CC-BY-4.0**, **gated** ✅ verified | Commercial-OK **but** HF login + terms gate breaks self-containment; model-generated. **Not shipped.** | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k |
| **Glaive function-calling-v2** | **Apache-2.0** ✅ verified | Permissive, but synthetically GPT-generated (provenance/ToS) and large/noisy. **Not shipped.** | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 |
| **ToolBench (OpenBMB)** | Apache-2.0 code; data **RapidAPI-derived** *(documented knowledge)* | Redistributing third-party API data is a concern. **Not shipped.** | https://github.com/OpenBMB/ToolBench |
| **WebArena** | Apache-2.0 code *(documented knowledge)* | Needs hosted web environments — not self-contained. **Not shipped.** | https://github.com/web-arena-x/webarena |
| **GAIA** | CC-BY-4.0, **gated**, answers withheld *(documented knowledge)* | Gated + hidden test answers → can't score locally. **Not shipped.** | https://huggingface.co/datasets/gaia-benchmark/GAIA |
| **SWE-bench** | Data derived from GitHub issues (mixed upstream licenses); needs Docker *(documented knowledge)* | Heavy + mixed provenance. **Not shipped.** | https://github.com/princeton-nlp/SWE-bench |
| **HotpotQA** (multi-hop QA) | **CC-BY-SA-4.0** *(documented knowledge)* | Share-alike copyleft on derivatives. **Not shipped.** | https://hotpotqa.github.io/ |

Net: several sources are *usable* under their licenses, but **none is a clean, self-contained,
un-gated, deterministically-scorable fit** for a redistributable executive demo. Synthesizing is
the honest, lowest-risk call — which is also what the brief asked for when in doubt.

---

## How to run

```bash
cd datasets/agentic/harness

python3 validate.py          # gate: schema + no-leak + replay + checker → 24/24, exit 0
python3 run.py --mock        # end-to-end scoring pipeline, no server → 24/24 PASS

# live, once muser-server is up (Phase 2+ of the roadmap):
MUSER_BASE_URL=http://127.0.0.1:8080/v1 MUSER_MODEL=muse-glimmer-30b python3 run.py
python3 run.py --json results.json    # dashboard-ready results
```

No third-party Python packages are required (standard library only). Regenerate everything
reproducibly with `python3 build_tasks.py` (data) and `python3 validate.py --emit-golden`
(fixtures). See **[RUNNER.md](RUNNER.md)** for the execution model and the three worked golden
examples.
