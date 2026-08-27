# RUNNER — how muser executes and scores an agentic task

This spec defines the loop the demo uses to turn **one task record → one live PASS/FAIL**.
The reference implementation is [`harness/run.py`](harness/run.py) (standard-library Python,
engine-agnostic); its semantics come entirely from [`harness/agentic_harness.py`](harness/agentic_harness.py),
which the validator and the golden fixtures share, so "scored live" and "known-solvable" mean
the same thing byte-for-byte.

---

## 1. The loop

```
                 ┌──────────────────────── runner (harness/run.py) ────────────────────────┐
 task record ──► build messages: [system prompt] + [user = task.prompt]                     │
                 project task.tools → OpenAI tool schemas (name/description/parameters ONLY) │
                 │                                                                           │
                 ▼                                                                           │
      ┌───► POST /v1/chat/completions {model, messages, tools, tool_choice:auto, temp:0} ────┼─► muser-server
      │          │                                                                          │
      │          ▼  assistant message                                                       │
      │     tool_calls present? ──no──► final answer text ─► extract `ANSWER:` ─► CHECKER ─► PASS/FAIL
      │          │yes                                                                        │
      │          ▼                                                                           │
      │     for each tool_call: execute the task's STUB locally (deterministic)              │
      │     append {role:"tool", tool_call_id, content: json(observation)}                   │
      └──────────┘  (loop, capped at MAX_TURNS = 8)                                          │
                                                                                             ┘
```

Key properties for a demo shown to executives:

- **The model never sees the answer.** Only `name/description/parameters` of each tool are sent
  (`to_openai_tools`). `stub`, `expected`, `checker`, and `reference_solution` stay runner-side.
  `validate.py` asserts this projection leaks none of those fields.
- **The runner, not the model, executes tools.** Stub outputs are computed locally and fed back
  as `role:"tool"` messages — the model cannot fake a tool result.
- **Scoring is deterministic.** No LLM-judge on the demo path. Every checker is exact-match, a
  numeric/set/JSON comparison, or a tiny predicate. `temperature=0` for repeatable decode.

---

## 2. Where the Muse chat/tool template matters

muser-server exposes an **OpenAI-compatible** `/v1/chat/completions` (see
`docs/muser-architecture.md` §G and `crates/muser-server/src/openai.rs`), so this runner speaks
plain OpenAI JSON. Underneath, the server must render `{system, tools, messages}` into the **Muse
Glimmer chat/tool template** (`crates/muser-engine/src/tokenizer.rs`), decode, then parse the raw
model turn back into OpenAI `tool_calls`. Three things the integrator must get right, and where:

1. **Tool serialization (into the template).** The `tools` array has to be rendered into the Muse
   prompt in the exact tool-template form the weights were trained on. If the schema block is
   malformed, the model emits prose instead of a parseable tool call and the loop stalls.
2. **Dual-EOS stop (decode).** Muse Glimmer has **two** terminal token ids —
   **`eos = 200001`** and **`eot = 200008`**. Both end a decode turn. A tool-call turn typically
   closes with **EOT (200008)**; a final natural-language turn may close with **EOS (200001)**.
   The server must treat **either** as turn-final and must not hard-code a single stop id (a
   common llama.cpp-for-this-arch bug). Whether the loop continues is decided by *did this turn
   parse into a tool_call?*, not by which EOS fired.
3. **Tool-call parse (out of the template).** The raw assistant turn's tool-call segment is mapped
   to `{id, type:"function", function:{name, arguments}}` with `arguments` as a **JSON string**
   (OpenAI shape). This runner depends only on that shape; the raw in-template delimiter tokens are
   a server-internal detail.

> Honesty note: Muse's exact in-template tool-call delimiter syntax is a model-internal detail not
> reproduced in this dataset. The dataset and runner are deliberately decoupled from it — they ride
> on muser-server's OpenAI-compatible parse layer. What this spec pins down is the contract
> (tools-in, dual-EOS stop, tool_calls-out), which is what the demo needs.

---

## 3. Tool stubs (deterministic, in-record)

Each tool carries a `stub`. Two kinds, both pure functions of the call args + `world`:

- **`fixture`** — a table of `cases`. A case matches when every key in `when` equals the arg
  (compared after `lower`+`collapse_ws`+`trim`) **and** every substring list in `when_contains`
  is present in the arg. First match wins; else `default`. Used for `search`, `open_page`,
  `kb_lookup`, `get_weather`.
- **`builtin`** — a named pure function the runner implements:
  - `calc.eval` — safe arithmetic (`+ - * / // % **`, `round/abs/min/max/floor/ceil/sqrt`) via a
    whitelisted AST walk. **Rejects any non-arithmetic node** (no `import`, no attribute access,
    no names) — verified in the harness tests.
  - `unit.convert` — fixed factor/affine table (kg↔g, m↔ft, mi↔km, lb↔kg, °C↔°F, l↔ml, h↔min, min↔s).
  - `fs.read` / `fs.write` / `fs.apply_patch` — a mock filesystem living in `world.files`
    (`{path: content}`). `apply_patch` replaces the first occurrence of `find` with `replace`.

`world` (optional per task) seeds mutable state; stateful builtins mutate it in place, and the
`final_state` checker reads it after the run.

## 4. Answer extraction & checkers

The system prompt requires the model to end with a line `ANSWER: <result>`. The checker reads
**the text after the last `ANSWER:`** (falling back to the last non-empty line if the marker is
missing, so a sloppy model is still scoreable). Checker kinds (see `run_checker`):

| kind            | reads               | passes when |
|-----------------|---------------------|-------------|
| `exact_match`   | answer text         | normalized answer == `expected.answer` |
| `numeric_equal` | first number in answer | `\|got − expected.value\| ≤ tol` (after `strip` tokens) |
| `set_equal`     | answer split by `sep` | item set == `set(expected.set)` |
| `contains_all`  | answer text         | every string in `expected.substrings` present |
| `bool_equal`    | answer text         | true/false/yes/no parse == `expected.value` |
| `json_equal`    | JSON in answer      | deep-equals `expected.value` (numbers ≈, keys exact) |
| `final_state`   | `world.files`       | each `expected.files[path]` matches (trailing-newline-insensitive by default) |

---

## 5. Three golden examples, fully worked

Each is a real fixture emitted by `python3 harness/validate.py --emit-golden` and re-scored on
every validation run. The full OpenAI-style trajectories live in [`golden/`](golden/).

### Golden 1 — `lookup-003` · search → open_page → numeric_equal · [`golden/golden-01-lookup.json`](golden/golden-01-lookup.json)

*Task:* find the year Rust **first appeared**, via `search` then `open_page`.

| turn | content |
|------|---------|
| assistant | `tool_call search{"query":"Rust programming language history"}` |
| tool `search` | `[{"id":"rust_hist","title":"History of Rust","snippet":"Timeline of the Rust programming language."}]` |
| assistant | `tool_call open_page{"id":"rust_hist"}` |
| tool `open_page` | `{...,"content":"Rust ... first appeared in 2010. Version 1.0.0 was released on May 15, 2015."}` |
| assistant | `The page states Rust first appeared in 2010.`<br>`ANSWER: 2010` |

Extract → `2010`. Checker `numeric_equal(tol=0)` vs `expected.value=2010` → **PASS**. (Note the
distractor "2015" in the same page — the checker rewards reading `first appeared`, not the first
number seen.)

### Golden 2 — `calc-005` · kb_lookup → calculator → numeric_equal · [`golden/golden-02-calc-chain.json`](golden/golden-02-calc-chain.json)

*Task:* look up the speed of light, compute metres travelled in 3 µs, round.

| turn | content |
|------|---------|
| assistant | `tool_call kb_lookup{"entity":"physics_constants","field":"speed_of_light_m_s"}` |
| tool `kb_lookup` | `299792458` |
| assistant | `tool_call calculator{"expression":"round(299792458*0.000003)"}` |
| tool `calculator` | `899` |
| assistant | `ANSWER: 899` |

Extract → `899`. Checker `numeric_equal(tol=0, strip=["m","meters"])` vs `899` → **PASS**. A
two-tool verification chain: the fact comes from the KB, the arithmetic from the calculator, and
neither is trusted to the model's head.

### Golden 3 — `edit-001` · read_file → apply_patch → final_state · [`golden/golden-03-code-edit.json`](golden/golden-03-code-edit.json)

*Task:* fix `Helo`→`Hello` in `greeting.py`. Seed `world.files["greeting.py"] = "print('Helo, world!')\n"`.

| turn | content |
|------|---------|
| assistant | `tool_call read_file{"path":"greeting.py"}` |
| tool `read_file` | `{"path":"greeting.py","content":"print('Helo, world!')\n"}` |
| assistant | `tool_call apply_patch{"path":"greeting.py","find":"Helo","replace":"Hello"}` |
| tool `apply_patch` | `{"ok":true,"path":"greeting.py"}` |
| assistant | `ANSWER: done` |

After the run `world.files["greeting.py"] == "print('Hello, world!')\n"`. Checker `final_state` vs
`expected.files` → **PASS**. Here the score is the **world state**, not the answer text — the demo
shows the file actually changed.

---

## 6. How to run

```bash
cd datasets/agentic/harness

# 1) Prove the dataset is self-consistent (schema + no-leak + replay + checker). CI-usable.
python3 validate.py                      # 24/24 expected; exit 0

# 2) Prove the scoring pipeline end-to-end WITHOUT a model (reference replay):
python3 run.py --mock                    # 24/24 PASS

# 3) Score the live model once muser-server is up (Phase 2+):
MUSER_BASE_URL=http://127.0.0.1:8080/v1 MUSER_MODEL=muse-glimmer-30b \
  python3 run.py                         # full suite
python3 run.py --task calc-005           # one task
python3 run.py --category code_edit      # one category
python3 run.py --json results.json       # machine-readable, for the dashboard

# Regenerate data / fixtures (reproducible; no network):
python3 build_tasks.py                   # rewrites ../tasks.jsonl (asserts every checker passes)
python3 validate.py --emit-golden        # rewrites ../golden/*.json
```

`run.py --json` emits `{model, score:{passed,total}, results:[{id,passed,answer,n_tool_calls,…}]}`
— the shape `web/muser-dashboard.html` can render as a live "agentic pass-rate" panel next to the
tokens/sec and KV-economics tiles.
