#!/usr/bin/env python3
"""Live agentic runner for muser — prompt -> tool-call loop -> stub -> checker -> score.

Engine-agnostic: speaks the OpenAI-compatible `/v1/chat/completions` API that muser-server
exposes (docs/muser-architecture.md G). Standard library only (urllib for HTTP). The tool
STUBS and CHECKERS are executed locally by agentic_harness, so scoring is deterministic and
never depends on the model to grade itself.

Usage:
  # against a running muser-server:
  MUSER_BASE_URL=http://127.0.0.1:8080/v1 MUSER_MODEL=muse-glimmer-30b python3 run.py
  python3 run.py --task calc-005            # one task
  python3 run.py --category code_edit       # one category
  python3 run.py --json results.json        # machine-readable results for the dashboard

  # no server yet? replay the reference trajectory through the SAME scoring path:
  python3 run.py --mock                     # proves the pipeline end-to-end offline

Env:
  MUSER_BASE_URL  default http://127.0.0.1:8080/v1
  MUSER_MODEL     default muse-glimmer-30b
  MUSER_API_KEY   optional; sent as `Authorization: Bearer ...` if set
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_harness as H  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_PATH = os.path.normpath(os.path.join(HERE, "..", "tasks.jsonl"))
BASE_URL = os.environ.get("MUSER_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
MODEL = os.environ.get("MUSER_MODEL", "muse-glimmer-30b")
API_KEY = os.environ.get("MUSER_API_KEY")
MAX_TURNS = 8


def load_tasks() -> list[dict]:
    with open(TASKS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_completion(messages: list[dict], tools: list[dict]) -> dict:
    """POST one turn to muser-server. Returns the first choice's `message`."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,        # deterministic decode for a repeatable demo
        "max_tokens": 512,       # same bounded turn budget as native D2 control
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    if API_KEY:
        req.add_header("Authorization", f"Bearer {API_KEY}")
    with urllib.request.urlopen(
        req, timeout=int(os.environ.get("MUSER_AGENTIC_TIMEOUT", "120"))
    ) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]


def run_task_live(task: dict) -> dict:
    """Drive the model through the tool loop, execute stubs locally, then score."""
    world = H.fresh_world(task)
    tools = H.to_openai_tools(task)
    messages = [
        {"role": "system", "content": H.SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    trace = []
    final_text = ""
    for _ in range(MAX_TURNS):
        msg = chat_completion(messages, tools)
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            final_text = msg.get("content") or ""
            break
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                tool = H.find_tool(task, name)
                observation = H.execute_stub(tool, args, world)
            except Exception as e:  # noqa: BLE001 — surface a bad tool call to the model
                observation = {"error": str(e)}
            trace.append({"tool": name, "args": args, "observation": observation})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{len(trace)}"),
                "name": name,
                "content": json.dumps(observation),
            })
    answer = H.extract_answer(final_text)
    passed, detail = H.run_checker(task["checker"], answer, world, task["expected"])
    return {"id": task["id"], "category": task["category"], "difficulty": task["difficulty"],
            "passed": passed, "answer": answer, "detail": detail, "n_tool_calls": len(trace),
            "trace": trace}


def run_task_mock(task: dict) -> dict:
    """Replay the reference trajectory through the identical scoring path (no server)."""
    _, answer, world = H.openai_trajectory_from_reference(task)
    passed, detail = H.run_checker(task["checker"], answer, world, task["expected"])
    n = sum(1 for s in task["reference_solution"]["steps"] if "call" in s)
    return {"id": task["id"], "category": task["category"], "difficulty": task["difficulty"],
            "passed": passed, "answer": answer, "detail": detail, "n_tool_calls": n, "trace": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="run a single task by id")
    ap.add_argument("--category", help="run only tasks in this category")
    ap.add_argument("--mock", action="store_true", help="replay reference trajectory (no server)")
    ap.add_argument("--json", help="write results to this JSON file")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    mode = "MOCK (reference replay)" if args.mock else f"LIVE {BASE_URL} model={MODEL}"
    print(f"running {len(tasks)} task(s) — {mode}\n")
    results = []
    transport_failures = 0
    for task in tasks:
        try:
            r = run_task_mock(task) if args.mock else run_task_live(task)
            transport_failures = 0
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            r = {"id": task["id"], "category": task["category"],
                 "difficulty": task["difficulty"], "passed": False, "answer": "",
                 "detail": f"transport: muser-server returned HTTP {e.code}: {detail}",
                 "n_tool_calls": 0, "trace": []}
            transport_failures += 1
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            r = {"id": task["id"], "category": task["category"],
                 "difficulty": task["difficulty"], "passed": False, "answer": "",
                 "detail": f"transport: {e}", "n_tool_calls": 0, "trace": []}
            transport_failures += 1
        results.append(r)
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{flag}] {r['id']:<12} tools={r['n_tool_calls']}  answer={r['answer']!r}", flush=True)
        if not r["passed"]:
            print(f"         {r['detail']}", flush=True)
        if transport_failures >= 3:
            print("aborting: 3 consecutive transport failures — server likely wedged",
                  file=sys.stderr)
            break

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\nscore: {n_pass}/{len(results)} passed")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": MODEL, "base_url": BASE_URL, "mock": args.mock,
                       "score": {"passed": n_pass, "total": len(results)},
                       "results": results}, f, indent=2)
        print(f"wrote {args.json}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
