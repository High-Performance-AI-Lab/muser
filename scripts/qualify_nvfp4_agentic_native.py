#!/usr/bin/env python3
"""Score the checked agentic set with the pinned native NVFP4 vLLM engine.

This is deliberately an in-process qualification driver, not a serving path.
It uses the checkpoint's immutable chat template, vLLM greedy generation, and
the same deterministic stubs/checkers as datasets/agentic/harness/run.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


MAX_TURNS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def parse_scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_atem(text: str) -> dict[str, Any]:
    """Parse the checkpoint's ATEM call format into OpenAI message shape."""
    cleaned = text.replace("<|eot|>", "")
    content: list[str] = []
    calls: list[dict[str, Any]] = []
    for raw_phase in cleaned.split("<|eom|>"):
        phase = raw_phase.removeprefix("<|start|>assistant")
        recipient = "user"
        if phase.startswith(" to="):
            header, marker, body = phase[4:].partition("<|message|>")
            if not marker:
                raise ValueError("assistant recipient header omits <|message|>")
            recipient = header.strip()
            phase = body
        elif phase.startswith("<|message|>"):
            phase = phase[len("<|message|>") :]
        if not phase:
            continue
        if recipient == "self":
            continue
        if "<atem:function_calls>" not in phase:
            if recipient != "user":
                raise ValueError(f"unexpected assistant recipient {recipient!r}")
            content.append(phase)
            continue
        match = re.fullmatch(
            r"\s*<atem:function_calls>(.*)</atem:function_calls>\s*",
            phase,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError("ATEM function-call phase is not one closed block")
        body = match.group(1)
        invoke_pattern = re.compile(
            r"\s*<atem:invoke name=\"([A-Za-z0-9_.-]+)\">(.*?)</atem:invoke>",
            flags=re.DOTALL,
        )
        offset = 0
        phase_calls: list[dict[str, Any]] = []
        while offset < len(body):
            invoke = invoke_pattern.match(body, offset)
            if invoke is None:
                if body[offset:].strip():
                    raise ValueError("unexpected text outside an ATEM invoke")
                break
            name, parameter_body = invoke.groups()
            parameters: dict[str, Any] = {}
            parameter_pattern = re.compile(
                r"\s*<atem:parameter name=\"([^\"]+)\">(.*?)</atem:parameter>",
                flags=re.DOTALL,
            )
            parameter_offset = 0
            while parameter_offset < len(parameter_body):
                parameter = parameter_pattern.match(parameter_body, parameter_offset)
                if parameter is None:
                    if parameter_body[parameter_offset:].strip():
                        raise ValueError("unexpected text outside an ATEM parameter")
                    break
                parameter_name, raw = parameter.groups()
                if parameter_name in parameters:
                    raise ValueError(f"duplicate ATEM parameter {parameter_name!r}")
                if "<atem:" in raw or "</atem:" in raw:
                    raise ValueError("ATEM parameter contains a structural tag")
                parameters[parameter_name] = parse_scalar(raw)
                parameter_offset = parameter.end()
            arguments = json.dumps(parameters, separators=(",", ":"), sort_keys=True)
            call_digest = hashlib.sha256(
                f"{name}\0{arguments}\0{len(calls) + len(phase_calls)}".encode()
            ).hexdigest()[:24]
            phase_calls.append(
                {
                    "id": f"call_{call_digest}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            offset = invoke.end()
        if not phase_calls:
            raise ValueError("ATEM function-call block contains no invokes")
        if recipient not in ("tool", phase_calls[0]["function"]["name"]):
            raise ValueError("assistant recipient does not match first ATEM invoke")
        calls.extend(phase_calls)
    return {"role": "assistant", "content": "".join(content), "tool_calls": calls}


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """HF's template requires tool-call arguments as mappings, not strings."""
    output = json.loads(json.dumps(messages))
    for message in output:
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments or "{}")
    return output


def load_harness(repo: Path) -> Any:
    harness = repo / "datasets" / "agentic" / "harness"
    sys.path.insert(0, str(harness))
    import agentic_harness  # type: ignore

    return agentic_harness


def run_task(
    task: dict[str, Any],
    *,
    engine: Any,
    tokenizer: Any,
    sampling_params_type: Any,
    text_prompt_type: Any,
    harness: Any,
    current_date: str,
) -> dict[str, Any]:
    world = harness.fresh_world(task)
    tools = harness.to_openai_tools(task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": harness.SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    trace: list[dict[str, Any]] = []
    final_text = ""
    raw_turns: list[str] = []
    started = time.perf_counter_ns()
    for _ in range(MAX_TURNS):
        prompt = tokenizer.apply_chat_template(
            template_messages(messages),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            current_date=current_date,
            reasoning_strength="high",
        )
        generated = engine.generate(
            text_prompt_type(prompt=prompt),
            sampling_params_type(temperature=0, max_tokens=512, seed=0),
            use_tqdm=False,
        )[0].outputs[0]
        # vLLM's convenience text drops the Muse structural special tokens
        # (`<|message|>`, `<|eom|>`, `<|eot|>`).  They are the ATEM framing,
        # not presentation whitespace, so qualification must decode the
        # returned IDs with special tokens preserved exactly as muser-server
        # does internally.
        raw_turn = tokenizer.decode(generated.token_ids, skip_special_tokens=False)
        raw_turns.append(raw_turn)
        message = parse_atem(raw_turn)
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            final_text = message.get("content") or ""
            break
        for call in tool_calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"] or "{}")
            try:
                tool = harness.find_tool(task, name)
                observation = harness.execute_stub(tool, arguments, world)
            except Exception as error:  # surface a bad call back to the model
                observation = {"error": str(error)}
            trace.append(
                {"tool": name, "args": arguments, "observation": observation}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(observation),
                }
            )
    answer = harness.extract_answer(final_text)
    passed, detail = harness.run_checker(
        task["checker"], answer, world, task["expected"]
    )
    return {
        "id": task["id"],
        "category": task["category"],
        "difficulty": task["difficulty"],
        "passed": passed,
        "answer": answer,
        "detail": detail,
        "n_tool_calls": len(trace),
        "turns": len(raw_turns),
        "elapsed_ns": time.perf_counter_ns() - started,
        "trace": trace,
        "raw_turns": raw_turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task")
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
        raise RuntimeError("must run below scripts/accelerator_safe.py")
    if args.output.exists():
        raise FileExistsError(args.output)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MUSER_NVFP4_PRODUCER_MODE", "native")

    harness = load_harness(args.repo)
    tasks_path = args.repo / "datasets" / "agentic" / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line]
    if args.task:
        tasks = [task for task in tasks if task["id"] == args.task]
    if not tasks:
        raise ValueError("no selected agentic tasks")

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams, TextPrompt

    checkpoint_config = json.loads((args.model / "config.json").read_text())
    revision = args.expected_model_revision
    if checkpoint_config.get("_commit_hash") not in (None, revision):
        raise ValueError("checkpoint config revision differs from expected revision")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False
    )
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        load_format="safetensors",
        quantization=None,
        kv_cache_dtype="auto",
        dtype="float16",
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=True,
        enable_flashinfer_autotune=False,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=0.82,
        kv_cache_memory_bytes=3_221_225_472,
        kernel_config={"enable_cutedsl_warmup": False, "enable_jit_warmup": False},
        seed=0,
    )
    current_date = "2026-08-17"
    started = time.perf_counter_ns()
    results = []
    for task in tasks:
        result = run_task(
            task,
            engine=engine,
            tokenizer=tokenizer,
            sampling_params_type=SamplingParams,
            text_prompt_type=TextPrompt,
            harness=harness,
            current_date=current_date,
        )
        results.append(result)
        print(
            f"[{('PASS' if result['passed'] else 'FAIL')}] {result['id']} "
            f"tools={result['n_tool_calls']} turns={result['turns']} "
            f"answer={result['answer']!r}",
            flush=True,
        )
    passed = sum(result["passed"] for result in results)
    report = {
        "schema": "muser.nvfp4-agentic-native-qualification.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lane": "native-w4a4-direct-semantic-control",
        "scope_note": (
            "task-level native W4A4 semantic control; product transport and Mac "
            "decode are qualified separately"
        ),
        "identity": {
            "model": str(args.model.resolve()),
            "model_revision": revision,
            "checkpoint_artifact_sha256": args.expected_checkpoint_sha256,
            "container_image_id": args.expected_image_id,
            "tasks_sha256": sha256(tasks_path),
            "driver_sha256": sha256(Path(__file__)),
            "chat_template_sha256": sha256(args.model / "chat_template.jinja"),
        },
        "determinism": {
            "temperature": 0,
            "seed": 0,
            "enforce_eager": True,
            "chunked_prefill": False,
            "prefix_cache": False,
            "max_num_seqs": 1,
            "current_date": current_date,
        },
        "runtime": {
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "score": {"passed": passed, "total": len(results)},
        "elapsed_ns": time.perf_counter_ns() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(f"score={passed}/{len(results)} output={args.output}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
