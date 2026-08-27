#!/usr/bin/env python3
"""Reproducible generator for muser's agentic dataset (tasks.jsonl).

Every task here is ORIGINAL synthetic content authored for muser — no scraping,
no third-party data. Task *formats* are inspired by public agentic benchmarks
(BFCL / tau-bench / GSM8K-style verification), but no data is copied from them.

The generator does three jobs:
  1. Declare each task (tools, expected, checker, reference_solution).
  2. Execute each reference step against the real stubs to POPULATE `observe`
     (so observations in tasks.jsonl are correct by construction, never hand-typed).
  3. Assert every task's checker PASSES on its replayed answer — the build fails
     loudly if any task is not self-consistent.

Run:  python3 build_tasks.py   (writes ../tasks.jsonl)
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_harness as H  # noqa: E402


# --------------------------------------------------------------------------- #
# Reusable tool builders (OpenAI function-schema shape + deterministic stub)
# --------------------------------------------------------------------------- #

def t_search(cases, default=None):
    return {
        "name": "search",
        "description": "Search a reference corpus. Returns a list of {id,title,snippet}.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search terms"}},
            "required": ["query"],
        },
        "stub": {"kind": "fixture", "cases": cases, "default": default if default is not None else []},
    }


def t_open_page(cases):
    return {
        "name": "open_page",
        "description": "Open a reference page by its id. Returns {id,title,content}.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        "stub": {"kind": "fixture", "cases": cases, "default": {"error": "no such page"}},
    }


def t_kb_lookup(cases):
    return {
        "name": "kb_lookup",
        "description": "Look up a single field for an entity in the structured knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {"entity": {"type": "string"}, "field": {"type": "string"}},
            "required": ["entity", "field"],
        },
        "stub": {"kind": "fixture", "cases": cases, "default": None},
    }


def t_get_weather(cases):
    return {
        "name": "get_weather",
        "description": "Get the current weather snapshot for a city: {condition,temp_c}.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "stub": {"kind": "fixture", "cases": cases, "default": {"error": "unknown city"}},
    }


def t_calculator():
    return {
        "name": "calculator",
        "description": ("Evaluate an arithmetic expression and return the number. "
                        "Supports + - * / // % ** parentheses and round/abs/min/max/floor/ceil/sqrt."),
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        "stub": {"kind": "builtin", "fn": "calc.eval"},
    }


def t_convert_units():
    return {
        "name": "convert_units",
        "description": "Convert a numeric value between units (kg/g, m/ft, mi/km, lb/kg, C/F, l/ml, h/min, min/s).",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        "stub": {"kind": "builtin", "fn": "unit.convert"},
    }


def t_read_file():
    return {
        "name": "read_file",
        "description": "Read a file from the workspace. Returns {path,content}.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "stub": {"kind": "builtin", "fn": "fs.read"},
    }


def t_write_file():
    return {
        "name": "write_file",
        "description": "Write (create or overwrite) a file with exact content. Returns {ok,path,bytes}.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        "stub": {"kind": "builtin", "fn": "fs.write"},
    }


def t_apply_patch():
    return {
        "name": "apply_patch",
        "description": "Replace the first occurrence of `find` with `replace` in a file. Returns {ok,path}.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "find": {"type": "string"},
                "replace": {"type": "string"},
            },
            "required": ["path", "find", "replace"],
        },
        "stub": {"kind": "builtin", "fn": "fs.apply_patch"},
    }


def call(tool, **args):
    return {"call": {"tool": tool, "args": args}}


def final(text):
    return {"final": text}


TASKS: list[dict] = []


def add(**task):
    TASKS.append(task)


# --------------------------------------------------------------------------- #
# information_lookup (6)
# --------------------------------------------------------------------------- #

add(
    id="lookup-001", category="information_lookup", difficulty="easy",
    prompt="Use the search tool to find the boiling point of water at sea level in degrees Celsius. Report just the number.",
    tools=[t_search([
        {"when_contains": {"query": ["boiling point", "water"]},
         "return": [{"id": "wp1", "title": "Boiling point of water",
                     "snippet": "At sea level (101.325 kPa) water boils at 100 degrees Celsius (212 F)."}]},
    ])],
    expected={"value": 100},
    checker={"kind": "numeric_equal", "tol": 0, "strip": ["C", "F", "degrees", "celsius"]},
    reference_solution={"steps": [
        call("search", query="boiling point of water at sea level"),
        final("The reference says water boils at 100 degrees Celsius at sea level.\nANSWER: 100"),
    ]},
)

add(
    id="lookup-002", category="information_lookup", difficulty="easy",
    prompt="Look up the capital city of the country 'Australia' using the knowledge base and report it.",
    tools=[t_kb_lookup([
        {"when": {"entity": "australia", "field": "capital"}, "return": "Canberra"},
    ])],
    expected={"answer": "Canberra"},
    checker={"kind": "exact_match", "normalize": ["trim", "lower"]},
    reference_solution={"steps": [
        call("kb_lookup", entity="Australia", field="capital"),
        final("ANSWER: Canberra"),
    ]},
)

add(
    id="lookup-003", category="information_lookup", difficulty="medium",
    prompt=("Find out what year the programming language 'Rust' first appeared. Use search to locate the "
            "history page, open it, then report the year Rust FIRST APPEARED (not the 1.0 release)."),
    tools=[
        t_search([
            {"when_contains": {"query": ["rust"]},
             "return": [{"id": "rust_hist", "title": "History of Rust",
                         "snippet": "Timeline of the Rust programming language."}]},
        ]),
        t_open_page([
            {"when": {"id": "rust_hist"},
             "return": {"id": "rust_hist", "title": "History of Rust",
                        "content": ("Rust is a systems programming language. It first appeared in 2010. "
                                    "Version 1.0.0 was released on May 15, 2015.")}},
        ]),
    ],
    expected={"value": 2010},
    checker={"kind": "numeric_equal", "tol": 0},
    reference_solution={"steps": [
        call("search", query="Rust programming language history"),
        call("open_page", id="rust_hist"),
        final("The page states Rust first appeared in 2010.\nANSWER: 2010"),
    ]},
)

add(
    id="lookup-004", category="information_lookup", difficulty="medium",
    prompt="Use the knowledge base to list the three primary additive colors. Report them as a comma-separated list.",
    tools=[t_kb_lookup([
        {"when": {"entity": "additive colors", "field": "members"}, "return": ["red", "green", "blue"]},
    ])],
    expected={"set": ["red", "green", "blue"]},
    checker={"kind": "set_equal", "sep": ",", "normalize": ["trim", "lower"]},
    reference_solution={"steps": [
        call("kb_lookup", entity="additive colors", field="members"),
        final("ANSWER: red, green, blue"),
    ]},
)

add(
    id="lookup-005", category="information_lookup", difficulty="medium",
    prompt=("Using the knowledge base, find the chemical symbol for Sodium and for Potassium. "
            "Report both symbols separated by a comma."),
    tools=[t_kb_lookup([
        {"when": {"entity": "sodium", "field": "symbol"}, "return": "Na"},
        {"when": {"entity": "potassium", "field": "symbol"}, "return": "K"},
    ])],
    expected={"set": ["Na", "K"]},
    checker={"kind": "set_equal", "sep": ",", "normalize": ["trim"]},
    reference_solution={"steps": [
        call("kb_lookup", entity="Sodium", field="symbol"),
        call("kb_lookup", entity="Potassium", field="symbol"),
        final("ANSWER: Na, K"),
    ]},
)

add(
    id="lookup-006", category="information_lookup", difficulty="hard",
    prompt=("Which planet in our solar system has the most confirmed moons according to the reference, and how "
            "many does it have? Use search, then open the reference page. Report exactly as 'Planet: N'."),
    tools=[
        t_search([
            {"when_contains": {"query": ["moons"]},
             "return": [{"id": "moons_ref", "title": "Confirmed moons per planet",
                         "snippet": "Reference snapshot table of confirmed moon counts."}]},
        ]),
        t_open_page([
            {"when": {"id": "moons_ref"},
             "return": {"id": "moons_ref", "title": "Confirmed moons per planet",
                        "content": ("Confirmed moons (reference snapshot): Mercury 0, Venus 0, Earth 1, "
                                    "Mars 2, Jupiter 95, Saturn 146, Uranus 28, Neptune 16.")}},
        ]),
    ],
    expected={"substrings": ["saturn", "146"]},
    checker={"kind": "contains_all", "normalize": ["trim", "lower"]},
    reference_solution={"steps": [
        call("search", query="planet with the most moons"),
        call("open_page", id="moons_ref"),
        final("Saturn has 146, the largest count in the table.\nANSWER: Saturn: 146"),
    ]},
)


# --------------------------------------------------------------------------- #
# calculation (5)
# --------------------------------------------------------------------------- #

add(
    id="calc-001", category="calculation", difficulty="easy",
    prompt="Compute 17 * 23 + 4 using the calculator and report the result.",
    tools=[t_calculator()],
    expected={"value": 395},
    checker={"kind": "numeric_equal", "tol": 0},
    reference_solution={"steps": [
        call("calculator", expression="17*23+4"),
        final("ANSWER: 395"),
    ]},
)

add(
    id="calc-002", category="calculation", difficulty="easy",
    prompt="A jacket costs $80 and is discounted 25%. Use the calculator to find the final price in dollars.",
    tools=[t_calculator()],
    expected={"value": 60},
    checker={"kind": "numeric_equal", "tol": 0, "strip": ["$"]},
    reference_solution={"steps": [
        call("calculator", expression="80*(1-0.25)"),
        final("ANSWER: 60"),
    ]},
)

add(
    id="calc-003", category="calculation", difficulty="medium",
    prompt=("A colleague claims the average of the numbers 12, 15, 21, and 8 is 15. Use the calculator to compute "
            "the true average, then report whether the claim is correct: answer 'true' or 'false'."),
    tools=[t_calculator()],
    expected={"value": False},
    checker={"kind": "bool_equal"},
    reference_solution={"steps": [
        call("calculator", expression="(12+15+21+8)/4"),
        final("The true average is 14, not 15, so the claim is wrong.\nANSWER: false"),
    ]},
)

add(
    id="calc-004", category="calculation", difficulty="medium",
    prompt="A recipe needs 2.5 kilograms of flour. Convert that to grams using convert_units and report the number of grams.",
    tools=[t_convert_units()],
    expected={"value": 2500},
    checker={"kind": "numeric_equal", "tol": 0, "strip": ["g", "grams"]},
    reference_solution={"steps": [
        call("convert_units", value=2.5, from_unit="kg", to_unit="g"),
        final("ANSWER: 2500"),
    ]},
)

add(
    id="calc-005", category="calculation", difficulty="hard",
    prompt=("Look up the constant 'speed_of_light_m_s' for entity 'physics_constants' in the knowledge base, then "
            "use the calculator to compute how many meters light travels in 3 microseconds. Report meters rounded "
            "to the nearest whole meter."),
    tools=[
        t_kb_lookup([
            {"when": {"entity": "physics_constants", "field": "speed_of_light_m_s"}, "return": 299792458},
        ]),
        t_calculator(),
    ],
    expected={"value": 899},
    checker={"kind": "numeric_equal", "tol": 0, "strip": ["m", "meters"]},
    reference_solution={"steps": [
        call("kb_lookup", entity="physics_constants", field="speed_of_light_m_s"),
        call("calculator", expression="round(299792458*0.000003)"),
        final("ANSWER: 899"),
    ]},
)


# --------------------------------------------------------------------------- #
# multi_tool_planning (6)
# --------------------------------------------------------------------------- #

add(
    id="plan-001", category="multi_tool_planning", difficulty="medium",
    prompt=("Find the recorded population of the reference city 'Testburg' via search, then use the calculator to "
            "compute 12% of that population. Report the rounded number of people."),
    tools=[
        t_search([
            {"when_contains": {"query": ["testburg"]},
             "return": [{"id": "tb", "title": "Testburg",
                         "snippet": "Testburg has a recorded population of 250000 residents."}]},
        ]),
        t_calculator(),
    ],
    expected={"value": 30000},
    checker={"kind": "numeric_equal", "tol": 0},
    reference_solution={"steps": [
        call("search", query="Testburg population"),
        call("calculator", expression="round(250000*0.12)"),
        final("ANSWER: 30000"),
    ]},
)

add(
    id="plan-002", category="multi_tool_planning", difficulty="medium",
    prompt=("Use kb_lookup to get the height in meters of 'Mount Testmore' (field 'height_m'), convert it to feet "
            "with convert_units, and report the height in feet rounded to the nearest foot."),
    tools=[
        t_kb_lookup([
            {"when": {"entity": "mount testmore", "field": "height_m"}, "return": 1200},
        ]),
        t_convert_units(),
    ],
    expected={"value": 3937},
    checker={"kind": "numeric_equal", "tol": 0.5},
    reference_solution={"steps": [
        call("kb_lookup", entity="Mount Testmore", field="height_m"),
        call("convert_units", value=1200, from_unit="m", to_unit="ft"),
        final("1200 m is about 3937.0 ft, which rounds to 3937.\nANSWER: 3937"),
    ]},
)

add(
    id="plan-003", category="multi_tool_planning", difficulty="medium",
    prompt=("Plan and execute: (1) search for 'Testcorp quarterly revenue', (2) kb_lookup Testcorp's "
            "'employee_count', (3) compute revenue divided by employees with the calculator. Report dollars of "
            "revenue per employee, rounded to the nearest dollar."),
    tools=[
        t_search([
            {"when_contains": {"query": ["testcorp", "revenue"]},
             "return": [{"id": "tc_rev", "title": "Testcorp earnings",
                         "snippet": "Testcorp reported quarterly revenue of $48,000,000."}]},
        ]),
        t_kb_lookup([
            {"when": {"entity": "testcorp", "field": "employee_count"}, "return": 1600},
        ]),
        t_calculator(),
    ],
    expected={"value": 30000},
    checker={"kind": "numeric_equal", "tol": 0},
    reference_solution={"steps": [
        call("search", query="Testcorp quarterly revenue"),
        call("kb_lookup", entity="Testcorp", field="employee_count"),
        call("calculator", expression="round(48000000/1600)"),
        final("ANSWER: 30000"),
    ]},
)

add(
    id="plan-004", category="multi_tool_planning", difficulty="hard",
    prompt=("Check the weather in 'Test City' with get_weather. If the condition is 'rain', report the recommendation "
            "from kb_lookup(entity='recommendations', field='rain'); otherwise report kb_lookup(entity='recommendations', "
            "field='clear'). Report the recommendation text exactly."),
    tools=[
        t_get_weather([
            {"when": {"city": "test city"}, "return": {"condition": "rain", "temp_c": 14}},
        ]),
        t_kb_lookup([
            {"when": {"entity": "recommendations", "field": "rain"}, "return": "Bring an umbrella."},
            {"when": {"entity": "recommendations", "field": "clear"}, "return": "No umbrella needed."},
        ]),
    ],
    expected={"answer": "Bring an umbrella."},
    checker={"kind": "exact_match", "normalize": ["trim", "lower"]},
    reference_solution={"steps": [
        call("get_weather", city="Test City"),
        call("kb_lookup", entity="recommendations", field="rain"),
        final("Condition is rain, so:\nANSWER: Bring an umbrella."),
    ]},
)

add(
    id="plan-005", category="multi_tool_planning", difficulty="hard",
    prompt=("Compile a report. (1) kb_lookup Testcorp's 'q3_sales' and 'q4_sales'. (2) Use the calculator to compute "
            "their total. (3) Write that total into the file 'report.txt' as a single line exactly 'TOTAL=<n>' using "
            "write_file. Then report the total number."),
    world={"files": {"report.txt": ""}},
    tools=[
        t_kb_lookup([
            {"when": {"entity": "testcorp", "field": "q3_sales"}, "return": 120000},
            {"when": {"entity": "testcorp", "field": "q4_sales"}, "return": 180000},
        ]),
        t_calculator(),
        t_write_file(),
    ],
    expected={"files": {"report.txt": "TOTAL=300000"}},
    checker={"kind": "final_state"},
    reference_solution={"steps": [
        call("kb_lookup", entity="Testcorp", field="q3_sales"),
        call("kb_lookup", entity="Testcorp", field="q4_sales"),
        call("calculator", expression="120000+180000"),
        call("write_file", path="report.txt", content="TOTAL=300000"),
        final("ANSWER: 300000"),
    ]},
)

add(
    id="plan-006", category="multi_tool_planning", difficulty="hard",
    prompt=("Find the 'Test Marathon' results via search, open the results page, and report the names of all runners "
            "who finished in under 3 hours (a time like 2:58 is under 3 hours; 3:05 is not). Report the names as a "
            "comma-separated list."),
    tools=[
        t_search([
            {"when_contains": {"query": ["test marathon"]},
             "return": [{"id": "tm", "title": "Test Marathon Results", "snippet": "Official finish times."}]},
        ]),
        t_open_page([
            {"when": {"id": "tm"},
             "return": {"id": "tm", "title": "Test Marathon Results",
                        "content": ("Finish times -- Alice 2:58, Bob 3:05, Carol 2:45, Dave 3:20, Eve 2:59.")}},
        ]),
    ],
    expected={"set": ["Alice", "Carol", "Eve"]},
    checker={"kind": "set_equal", "sep": ",", "normalize": ["trim", "lower"]},
    reference_solution={"steps": [
        call("search", query="Test Marathon results"),
        call("open_page", id="tm"),
        final("Under 3:00 -> Alice 2:58, Carol 2:45, Eve 2:59.\nANSWER: Alice, Carol, Eve"),
    ]},
)


# --------------------------------------------------------------------------- #
# code_edit (4)
# --------------------------------------------------------------------------- #

add(
    id="edit-001", category="code_edit", difficulty="easy",
    prompt=("The file 'greeting.py' prints a misspelled 'Helo' instead of 'Hello'. Read the file, fix the typo using "
            "apply_patch, and then report 'done'."),
    world={"files": {"greeting.py": "print('Helo, world!')\n"}},
    tools=[t_read_file(), t_apply_patch()],
    expected={"files": {"greeting.py": "print('Hello, world!')\n"}},
    checker={"kind": "final_state"},
    reference_solution={"steps": [
        call("read_file", path="greeting.py"),
        call("apply_patch", path="greeting.py", find="Helo", replace="Hello"),
        final("ANSWER: done"),
    ]},
)

add(
    id="edit-002", category="code_edit", difficulty="easy",
    prompt="Create a file named 'VERSION' whose content is exactly the text 1.0.0 (no trailing newline) using write_file, then report 'created'.",
    world={"files": {}},
    tools=[t_write_file()],
    expected={"files": {"VERSION": "1.0.0"}},
    checker={"kind": "final_state", "normalize_trailing_newline": False},
    reference_solution={"steps": [
        call("write_file", path="VERSION", content="1.0.0"),
        final("ANSWER: created"),
    ]},
)

add(
    id="edit-003", category="code_edit", difficulty="medium",
    prompt=("In the file 'config.json', change the value of \"debug\" from true to false, keeping the rest of the file "
            "identical. Use read_file to inspect first, then apply_patch. Report 'updated'."),
    world={"files": {"config.json": '{\n  "debug": true,\n  "port": 8080\n}\n'}},
    tools=[t_read_file(), t_apply_patch()],
    expected={"files": {"config.json": '{\n  "debug": false,\n  "port": 8080\n}\n'}},
    checker={"kind": "final_state"},
    reference_solution={"steps": [
        call("read_file", path="config.json"),
        call("apply_patch", path="config.json", find='"debug": true', replace='"debug": false'),
        final("ANSWER: updated"),
    ]},
)

add(
    id="edit-004", category="code_edit", difficulty="medium",
    prompt=("Append the line 'export PATH=$PATH:/usr/local/bin' as a new final line of the file '.profile', preserving "
            "all existing content. Use read_file to get the current content, then write_file with the full new content. "
            "Report 'appended'."),
    world={"files": {".profile": "# user profile\nexport EDITOR=vim\n"}},
    tools=[t_read_file(), t_write_file()],
    expected={"files": {".profile": "# user profile\nexport EDITOR=vim\nexport PATH=$PATH:/usr/local/bin\n"}},
    checker={"kind": "final_state"},
    reference_solution={"steps": [
        call("read_file", path=".profile"),
        call("write_file", path=".profile",
             content="# user profile\nexport EDITOR=vim\nexport PATH=$PATH:/usr/local/bin\n"),
        final("ANSWER: appended"),
    ]},
)


# --------------------------------------------------------------------------- #
# verification_chain (3)
# --------------------------------------------------------------------------- #

add(
    id="verify-001", category="verification_chain", difficulty="medium",
    prompt=("Read the file 'user.json'. Report a JSON object {\"valid\": <bool>} where valid is true only if the file "
            "parses as JSON AND has a top-level 'email' field. Report only that JSON object."),
    world={"files": {"user.json": '{"name":"Ada","email":"ada@example.com"}\n'}},
    tools=[t_read_file()],
    expected={"value": {"valid": True}},
    checker={"kind": "json_equal"},
    reference_solution={"steps": [
        call("read_file", path="user.json"),
        final('The file parses and has an "email" field.\nANSWER: {"valid": true}'),
    ]},
)

add(
    id="verify-002", category="verification_chain", difficulty="medium",
    prompt="A colleague claims that 2 to the power of 10 equals 1000. Use the calculator to check, and report 'true' or 'false'.",
    tools=[t_calculator()],
    expected={"value": False},
    checker={"kind": "bool_equal"},
    reference_solution={"steps": [
        call("calculator", expression="2**10"),
        final("2**10 is 1024, not 1000.\nANSWER: false"),
    ]},
)

add(
    id="verify-003", category="verification_chain", difficulty="hard",
    prompt=("Verify a project budget. Use kb_lookup to get entity 'project_alpha' fields 'budget_usd' and 'spent_usd'. "
            "Use the calculator to compute the remaining budget. Report a JSON object "
            "{\"within_budget\": <bool>, \"remaining\": <number>} where within_budget is true if spent <= budget."),
    tools=[
        t_kb_lookup([
            {"when": {"entity": "project_alpha", "field": "budget_usd"}, "return": 50000},
            {"when": {"entity": "project_alpha", "field": "spent_usd"}, "return": 42500},
        ]),
        t_calculator(),
    ],
    expected={"value": {"within_budget": True, "remaining": 7500}},
    checker={"kind": "json_equal"},
    reference_solution={"steps": [
        call("kb_lookup", entity="project_alpha", field="budget_usd"),
        call("kb_lookup", entity="project_alpha", field="spent_usd"),
        call("calculator", expression="50000-42500"),
        final('spent 42500 <= budget 50000; remaining 7500.\nANSWER: {"within_budget": true, "remaining": 7500}'),
    ]},
)


# --------------------------------------------------------------------------- #
# Finalize: populate observations by executing stubs, and assert every checker passes.
# --------------------------------------------------------------------------- #

def finalize(task: dict) -> dict:
    world = H.fresh_world(task)
    final_text = ""
    for step in task["reference_solution"]["steps"]:
        if "call" in step:
            c = step["call"]
            tool = H.find_tool(task, c["tool"])
            step["observe"] = H.execute_stub(tool, c["args"], world)
        elif "final" in step:
            final_text = step["final"]
    answer = H.extract_answer(final_text)
    ok, detail = H.run_checker(task["checker"], answer, world, task["expected"])
    if not ok:
        raise SystemExit(f"BUILD FAIL {task['id']}: checker did not pass on reference answer -> {detail}")
    return task


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "tasks.jsonl")
    ids = set()
    with open(out, "w") as f:
        for task in TASKS:
            if task["id"] in ids:
                raise SystemExit(f"duplicate id {task['id']}")
            ids.add(task["id"])
            finalize(task)
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    by_cat: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    for t in TASKS:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1
    print(f"wrote {len(TASKS)} tasks -> {os.path.normpath(out)}")
    print("by category:", dict(sorted(by_cat.items())))
    print("by difficulty:", dict(sorted(by_diff.items())))


if __name__ == "__main__":
    main()
