"""agentic_harness — deterministic stub executor + checkers for muser's agentic dataset.

Pure Python standard library. No third-party dependencies, no network, no clock,
no randomness. Everything here is a pure function of the task record + the model's
output, which is exactly what makes the demo scores reproducible and defensible.

Two consumers import this module:

  * validate.py  — replays each task's `reference_solution` against the stubs and
                   asserts the `checker` passes. Proves the dataset is internally
                   consistent WITHOUT a model in the loop (the "golden gate").
  * run.py       — drives a live OpenAI-compatible endpoint (muser-server), executes
                   the same stubs on the model's real tool calls, and scores with the
                   same checkers.

Because both paths share this one module, "the reference solution passes" and "the
live model is scored correctly" are guaranteed to use identical semantics.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #

def normalize(s: Any, ops: list[str] | None) -> str:
    """Apply an ordered list of normalization ops to a value stringified.

    Supported ops: "trim", "lower", "collapse_ws" (collapse runs of whitespace
    to a single space), "strip_punct_edges" (strip leading/trailing .,:;! ).
    """
    out = "" if s is None else str(s)
    for op in (ops or []):
        if op == "trim":
            out = out.strip()
        elif op == "lower":
            out = out.lower()
        elif op == "collapse_ws":
            out = re.sub(r"\s+", " ", out)
        elif op == "strip_punct_edges":
            out = out.strip(" .,:;!?\t\n")
        else:
            raise ValueError(f"unknown normalize op: {op}")
    return out


_MATCH_NORM = ["lower", "collapse_ws", "trim"]


def _match_str(v: Any) -> str:
    return normalize(v, _MATCH_NORM)


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #

def _norm_num(x: float | int) -> float | int:
    """Round to 10 decimals and collapse integral floats to int for clean fixtures."""
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    r = round(float(x), 10)
    if r == int(r):
        return int(r)
    return r


def parse_number(s: str, strip: list[str] | None = None) -> float:
    """Extract the first numeric literal from a string, after removing `strip` tokens
    and thousands separators. Raises ValueError if no number is present."""
    t = s
    for tok in (strip or []):
        t = t.replace(tok, "")
    t = t.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        raise ValueError(f"no number found in: {s!r}")
    return float(m.group(0))


def _nums_close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# Safe arithmetic evaluator (calc.eval builtin)
# --------------------------------------------------------------------------- #

_CALC_FUNCS = {
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "floor": __import__("math").floor,
    "ceil": __import__("math").ceil,
    "sqrt": __import__("math").sqrt,
    "pow": pow,
}

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


def _eval_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left, right = _eval_arith(node.left), _eval_arith(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        v = _eval_arith(node.operand)
        return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _CALC_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"function not allowed: {node.func.id}")
        args = [_eval_arith(a) for a in node.args]
        return fn(*args)
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def calc_eval(expression: str) -> float | int:
    tree = ast.parse(expression, mode="eval")
    return _norm_num(_eval_arith(tree))


# --------------------------------------------------------------------------- #
# Unit conversion (unit.convert builtin) — fixed, deterministic factor table
# --------------------------------------------------------------------------- #

_UNIT_FACTORS = {
    ("kg", "g"): 1000.0, ("g", "kg"): 0.001,
    ("km", "m"): 1000.0, ("m", "km"): 0.001,
    ("m", "ft"): 3.280839895, ("ft", "m"): 0.3048,
    ("mi", "km"): 1.609344, ("km", "mi"): 0.621371192,
    ("lb", "kg"): 0.45359237, ("kg", "lb"): 2.204622622,
    ("h", "min"): 60.0, ("min", "h"): 1.0 / 60.0,
    ("min", "s"): 60.0, ("s", "min"): 1.0 / 60.0,
    ("l", "ml"): 1000.0, ("ml", "l"): 0.001,
    ("c", "f"): None,  # handled specially below (affine, not linear)
}


def unit_convert(value: float, from_unit: str, to_unit: str) -> Any:
    fu, tu = from_unit.strip().lower(), to_unit.strip().lower()
    if fu == tu:
        return _norm_num(value)
    if (fu, tu) == ("c", "f"):
        return _norm_num(value * 9.0 / 5.0 + 32.0)
    if (fu, tu) == ("f", "c"):
        return _norm_num((value - 32.0) * 5.0 / 9.0)
    factor = _UNIT_FACTORS.get((fu, tu))
    if factor is None:
        return {"error": f"unsupported conversion {from_unit}->{to_unit}"}
    return _norm_num(value * factor)


# --------------------------------------------------------------------------- #
# Mock filesystem (fs.* builtins) operate on world["files"]: {path: content}
# --------------------------------------------------------------------------- #

def _files(world: dict) -> dict:
    return world.setdefault("files", {})


def fs_read(world: dict, path: str) -> Any:
    files = _files(world)
    if path not in files:
        return {"error": f"no such file: {path}"}
    return {"path": path, "content": files[path]}


def fs_write(world: dict, path: str, content: str) -> Any:
    _files(world)[path] = content
    return {"ok": True, "path": path, "bytes": len(content)}


def fs_apply_patch(world: dict, path: str, find: str, replace: str, count: int = 1) -> Any:
    files = _files(world)
    if path not in files:
        return {"ok": False, "error": f"no such file: {path}"}
    text = files[path]
    if find not in text:
        return {"ok": False, "error": "find string not present"}
    files[path] = text.replace(find, replace, count if count and count > 0 else -1)
    return {"ok": True, "path": path}


# --------------------------------------------------------------------------- #
# Fixture matching (search / open_page / kb_lookup / get_weather etc.)
# --------------------------------------------------------------------------- #

def _fixture_case_matches(args: dict, case: dict) -> bool:
    for k, v in case.get("when", {}).items():
        if _match_str(args.get(k)) != _match_str(v):
            return False
    for k, subs in case.get("when_contains", {}).items():
        hay = _match_str(args.get(k))
        for sub in subs:
            if _match_str(sub) not in hay:
                return False
    return True


def _run_fixture(stub: dict, args: dict) -> Any:
    for case in stub.get("cases", []):
        if _fixture_case_matches(args, case):
            return case["return"]
    if "default" in stub:
        return stub["default"]
    return {"error": "no fixture matched"}


# --------------------------------------------------------------------------- #
# Stub dispatch
# --------------------------------------------------------------------------- #

def execute_stub(tool: dict, args: dict, world: dict) -> Any:
    """Execute one tool call deterministically. Mutates `world` for stateful builtins."""
    stub = tool["stub"]
    kind = stub["kind"]
    if kind == "fixture":
        return _run_fixture(stub, args)
    if kind == "builtin":
        fn = stub["fn"]
        if fn == "calc.eval":
            return calc_eval(str(args["expression"]))
        if fn == "unit.convert":
            return unit_convert(float(args["value"]), str(args["from_unit"]), str(args["to_unit"]))
        if fn == "fs.read":
            return fs_read(world, str(args["path"]))
        if fn == "fs.write":
            return fs_write(world, str(args["path"]), str(args["content"]))
        if fn == "fs.apply_patch":
            return fs_apply_patch(world, str(args["path"]), str(args["find"]), str(args["replace"]),
                                  int(args.get("count", 1)))
        raise ValueError(f"unknown builtin fn: {fn}")
    raise ValueError(f"unknown stub kind: {kind}")


def find_tool(task: dict, name: str) -> dict:
    for t in task["tools"]:
        if t["name"] == name:
            return t
    raise KeyError(f"task {task['id']} has no tool named {name}")


# --------------------------------------------------------------------------- #
# Answer extraction & deep comparison
# --------------------------------------------------------------------------- #

_ANSWER_RE = re.compile(r"answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    """Everything after the LAST `ANSWER:` marker, trimmed. Falls back to the last
    non-empty line if the marker is absent (keeps a sloppy model scoreable)."""
    matches = list(re.finditer(r"answer\s*:\s*", text, re.IGNORECASE))
    if matches:
        return text[matches[-1].end():].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"no JSON found in: {text!r}")


def deep_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _nums_close(a, b, tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y, tol) for x, y in zip(a, b))
    return a == b


def _to_bool(s: str):
    v = normalize(s, ["trim", "lower", "strip_punct_edges"])
    if v in ("true", "yes", "1", "correct", "valid"):
        return True
    if v in ("false", "no", "0", "incorrect", "invalid"):
        return False
    return None


# --------------------------------------------------------------------------- #
# Checkers — the only place a pass/fail decision is made
# --------------------------------------------------------------------------- #

def run_checker(checker: dict, answer: str, world: dict, expected: dict) -> tuple[bool, str]:
    kind = checker["kind"]

    if kind == "exact_match":
        norm = checker.get("normalize", ["trim", "lower"])
        got, want = normalize(answer, norm), normalize(expected["answer"], norm)
        return got == want, f"exact_match got={got!r} want={want!r}"

    if kind == "numeric_equal":
        tol = checker.get("tol", 1e-9)
        try:
            got = parse_number(answer, checker.get("strip"))
        except ValueError as e:
            return False, f"numeric_equal parse-fail: {e}"
        want = float(expected["value"])
        return abs(got - want) <= tol, f"numeric_equal got={got} want={want} tol={tol}"

    if kind == "set_equal":
        sep = checker.get("sep", ",")
        norm = checker.get("normalize", ["trim", "lower"])
        got = {normalize(x, norm) for x in answer.split(sep) if normalize(x, norm)}
        want = {normalize(x, norm) for x in expected["set"]}
        return got == want, f"set_equal got={sorted(got)} want={sorted(want)}"

    if kind == "contains_all":
        norm = checker.get("normalize", ["trim", "lower"])
        hay = normalize(answer, norm)
        missing = [s for s in expected["substrings"] if normalize(s, norm) not in hay]
        return not missing, f"contains_all missing={missing}"

    if kind == "bool_equal":
        got = _to_bool(answer)
        want = bool(expected["value"])
        return got is want or got == want, f"bool_equal got={got} want={want}"

    if kind == "json_equal":
        try:
            got = _extract_json(answer)
        except ValueError as e:
            return False, f"json_equal parse-fail: {e}"
        ok = deep_equal(got, expected["value"])
        return ok, f"json_equal got={got} want={expected['value']}"

    if kind == "final_state":
        strip_nl = checker.get("normalize_trailing_newline", True)
        files = _files(world)
        for path, want in expected["files"].items():
            got = files.get(path)
            if got is None:
                return False, f"final_state missing file {path}"
            if strip_nl:
                if got.rstrip("\n") != want.rstrip("\n"):
                    return False, f"final_state {path} got={got!r} want={want!r}"
            elif got != want:
                return False, f"final_state {path} got={got!r} want={want!r}"
        return True, "final_state ok"

    raise ValueError(f"unknown checker kind: {kind}")


# --------------------------------------------------------------------------- #
# Reference-solution replay (used by validate.py and to emit golden fixtures)
# --------------------------------------------------------------------------- #

def fresh_world(task: dict) -> dict:
    w = copy.deepcopy(task.get("world", {}))
    w.setdefault("files", {})
    w.setdefault("kv", {})
    return w


# --------------------------------------------------------------------------- #
# Runner policy shared by the live runner (run.py), the validator, and the
# golden-fixture emitter so all three agree byte-for-byte on semantics.
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are muser's agentic assistant. Solve the user's task by calling the provided tools. "
    "Rules:\n"
    "1. Use tools to obtain facts and to change state; never fabricate a tool result.\n"
    "2. Call one or more tools as needed, then reason over their results.\n"
    "3. When the task is complete, output your reasoning followed by a final line of the "
    "exact form `ANSWER: <result>` and nothing after it.\n"
    "4. For a JSON answer, put the raw JSON object after `ANSWER:`.\n"
    "5. Be concise and deterministic; do not ask the user questions."
)


def to_openai_tools(task: dict) -> list[dict]:
    """Model-facing tool projection: name/description/parameters ONLY.

    The stub implementation, fixtures, `expected`, `checker`, and `reference_solution`
    are deliberately excluded — the model must never see the answer or the oracle."""
    out = []
    for t in task["tools"]:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        })
    return out


def openai_trajectory_from_reference(task: dict) -> tuple[list[dict], str, dict]:
    """Build the IDEAL OpenAI-style message trajectory from reference_solution.

    Returns (messages, extracted_answer, final_world). Used to emit golden fixtures
    and to document exactly what a correct model turn sequence looks like."""
    world = fresh_world(task)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    n = 0
    final_text = ""
    for step in task["reference_solution"]["steps"]:
        if "call" in step:
            n += 1
            c = step["call"]
            call_id = f"call_{n}"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": c["tool"], "arguments": json.dumps(c["args"])},
                }],
            })
            tool = find_tool(task, c["tool"])
            obs = execute_stub(tool, c["args"], world)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": c["tool"],
                "content": json.dumps(obs),
            })
        elif "final" in step:
            final_text = step["final"]
    messages.append({"role": "assistant", "content": final_text})
    return messages, extract_answer(final_text), world


def replay(task: dict) -> tuple[str, dict, list[dict]]:
    """Execute the task's reference_solution against the stubs.

    Returns (final_answer_text, final_world, trajectory). Raises AssertionError if a
    declared `observe` does not match what the stub actually returns — that mismatch
    means the fixture and the hand-written expected observation disagree, which is a
    dataset bug we want to fail loudly on.
    """
    world = fresh_world(task)
    final_text = ""
    trajectory: list[dict] = []
    for step in task["reference_solution"]["steps"]:
        if "call" in step:
            call = step["call"]
            tool = find_tool(task, call["tool"])
            got = execute_stub(tool, call["args"], world)
            want = step.get("observe", got)
            if not deep_equal(got, want):
                raise AssertionError(
                    f"{task['id']}: stub {call['tool']}{call['args']} returned {got!r}, "
                    f"reference_solution declared observe={want!r}")
            trajectory.append({"tool": call["tool"], "args": call["args"], "observation": got})
        elif "final" in step:
            final_text = step["final"]
        else:
            raise ValueError(f"{task['id']}: malformed reference step {step}")
    return extract_answer(final_text), world, trajectory
