#!/usr/bin/env python3
"""Validate muser's agentic dataset — the dataset's golden gate.

Zero third-party dependencies. For every task it:
  * checks structural conformance to schema.json (required fields, known enums);
  * confirms the model-facing tool projection leaks NO oracle data;
  * replays the reference_solution against the real stubs; and
  * asserts the checker PASSES on the replayed answer.

Exit code is non-zero if any task fails, so this doubles as CI.

  python3 validate.py                 # validate all tasks
  python3 validate.py --emit-golden   # also (re)write ../golden/*.json fixtures
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_harness as H  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
TASKS_PATH = os.path.join(ROOT, "tasks.jsonl")
GOLDEN_DIR = os.path.join(ROOT, "golden")

CATEGORIES = {"information_lookup", "calculation", "multi_tool_planning", "code_edit", "verification_chain"}
DIFFICULTIES = {"easy", "medium", "hard"}
CHECKERS = {"exact_match", "numeric_equal", "set_equal", "contains_all", "bool_equal", "json_equal", "final_state"}
CHECKER_NEEDS = {
    "exact_match": "answer", "numeric_equal": "value", "set_equal": "set",
    "contains_all": "substrings", "bool_equal": "value", "json_equal": "value",
    "final_state": "files",
}
BUILTINS = {"calc.eval", "unit.convert", "fs.read", "fs.write", "fs.apply_patch"}

GOLDEN = [
    ("lookup-003", "golden-01-lookup.json"),
    ("calc-005", "golden-02-calc-chain.json"),
    ("edit-001", "golden-03-code-edit.json"),
]


def load_tasks() -> list[dict]:
    tasks = []
    with open(TASKS_PATH) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"tasks.jsonl line {i}: invalid JSON: {e}")
    return tasks


def structural_errors(task: dict) -> list[str]:
    errs = []
    for req in ("id", "category", "difficulty", "prompt", "tools", "expected", "checker"):
        if req not in task:
            errs.append(f"missing required field '{req}'")
    if task.get("category") not in CATEGORIES:
        errs.append(f"bad category {task.get('category')!r}")
    if task.get("difficulty") not in DIFFICULTIES:
        errs.append(f"bad difficulty {task.get('difficulty')!r}")
    for t in task.get("tools", []):
        for req in ("name", "description", "parameters", "stub"):
            if req not in t:
                errs.append(f"tool {t.get('name')!r} missing '{req}'")
        stub = t.get("stub", {})
        if stub.get("kind") == "builtin" and stub.get("fn") not in BUILTINS:
            errs.append(f"tool {t.get('name')!r} unknown builtin {stub.get('fn')!r}")
        elif stub.get("kind") == "fixture" and "cases" not in stub:
            errs.append(f"tool {t.get('name')!r} fixture missing 'cases'")
        elif stub.get("kind") not in ("builtin", "fixture"):
            errs.append(f"tool {t.get('name')!r} bad stub kind {stub.get('kind')!r}")
    ck = task.get("checker", {}).get("kind")
    if ck not in CHECKERS:
        errs.append(f"unknown checker kind {ck!r}")
    else:
        need = CHECKER_NEEDS[ck]
        if need not in task.get("expected", {}):
            errs.append(f"checker '{ck}' needs expected.{need}")
    # tools referenced by the reference solution must exist
    names = {t["name"] for t in task.get("tools", [])}
    for step in task.get("reference_solution", {}).get("steps", []):
        if "call" in step and step["call"]["tool"] not in names:
            errs.append(f"reference calls unknown tool {step['call']['tool']!r}")
    return errs


def leak_errors(task: dict) -> list[str]:
    """The model-facing projection must not carry oracle fields."""
    errs = []
    blob = json.dumps(H.to_openai_tools(task))
    for banned in ('"stub"', '"return"', '"observe"', '"expected"', '"reference_solution"'):
        if banned in blob:
            errs.append(f"model-facing tools leak {banned}")
    return errs


def emit_golden() -> None:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    tasks = {t["id"]: t for t in load_tasks()}
    for tid, fname in GOLDEN:
        task = tasks[tid]
        messages, answer, world = H.openai_trajectory_from_reference(task)
        passed, detail = H.run_checker(task["checker"], answer, world, task["expected"])
        fixture = {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "prompt": task["prompt"],
            "tools_shown_to_model": H.to_openai_tools(task),
            "expected_trajectory": messages,
            "final_world": world,
            "extracted_answer": answer,
            "checker": task["checker"],
            "expected": task["expected"],
            "checker_result": {"passed": passed, "detail": detail},
        }
        with open(os.path.join(GOLDEN_DIR, fname), "w") as f:
            json.dump(fixture, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  emitted golden/{fname}  (checker passed={passed})")


def main() -> int:
    tasks = load_tasks()
    ids = set()
    n_fail = 0
    by_cat: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    print(f"validating {len(tasks)} tasks from {os.path.relpath(TASKS_PATH, ROOT)}\n")
    for task in tasks:
        tid = task.get("id", "<no-id>")
        problems = structural_errors(task) + leak_errors(task)
        if tid in ids:
            problems.append("duplicate id")
        ids.add(tid)
        if not problems:
            try:
                answer, world, _ = H.replay(task)
                passed, detail = H.run_checker(task["checker"], answer, world, task["expected"])
                if not passed:
                    problems.append(f"checker did not pass: {detail}")
            except (AssertionError, Exception) as e:  # noqa: BLE001
                problems.append(f"replay error: {e}")
        status = "PASS" if not problems else "FAIL"
        if problems:
            n_fail += 1
        by_cat[task.get("category", "?")] = by_cat.get(task.get("category", "?"), 0) + 1
        by_diff[task.get("difficulty", "?")] = by_diff.get(task.get("difficulty", "?"), 0) + 1
        print(f"  [{status}] {tid:<12} {task.get('category',''):<20} {task.get('checker',{}).get('kind','')}")
        for p in problems:
            print(f"         - {p}")
    print(f"\nsummary: {len(tasks)-n_fail}/{len(tasks)} passed")
    print("by category:  ", dict(sorted(by_cat.items())))
    print("by difficulty:", dict(sorted(by_diff.items())))
    if "--emit-golden" in sys.argv and n_fail == 0:
        print("\nemitting golden fixtures:")
        emit_golden()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
