#!/usr/bin/env python3
"""Retain a fail-closed differential against the pinned llama-server.

This is an unsealed qualification lane.  It deliberately has no option that
creates a seal and it writes a truthful report even when an individual route
fails, so a partial smoke run cannot be mistaken for API qualification.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import http.client
import json
import math
import os
from pathlib import Path
import ssl
import struct
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


LOGPROB_ABS_LIMIT = 1.0e-4
EMBEDDING_COSINE_MIN = 0.999999
EMBEDDING_ABS_LIMIT = 1.0e-4


class ParityFailure(RuntimeError):
    pass


def request_json(
    base: str,
    route: str,
    payload: dict | None,
    timeout: float,
    *,
    headers: dict[str, str] | None = None,
    method: str | None = None,
) -> tuple[int, object]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request_headers = dict(headers or {})
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw.decode("utf-8", errors="replace")
        return error.code, parsed


def request_sse(
    base: str,
    route: str,
    payload: dict | None,
    timeout: float,
    *,
    headers: dict[str, str] | None = None,
) -> list[dict]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request_headers = dict(headers or {})
    request_headers["Accept"] = "text/event-stream"
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=body,
        headers=request_headers,
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ParityFailure(f"stream returned HTTP {response.status}")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            events.append(json.loads(data))
    return events


def request_ndjson(base: str, route: str, payload: dict, timeout: float) -> list[dict]:
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
    )
    rows: list[dict] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ParityFailure(f"NDJSON route returned HTTP {response.status}")
        for raw in response:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def require_equal(label: str, reference: object, candidate: object) -> None:
    if reference != candidate:
        raise ParityFailure(f"{label} differs: reference={reference!r} candidate={candidate!r}")


def native_logprobs(response: dict) -> list[dict]:
    value = response.get("completion_probabilities")
    if not isinstance(value, list):
        raise ParityFailure("native completion omitted completion_probabilities")
    return value


def compare_logprobs(reference: list[dict], candidate: list[dict]) -> float:
    require_equal("logprob row count", len(reference), len(candidate))
    maximum = 0.0
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        for field in ("id", "token", "bytes"):
            require_equal(f"logprob[{index}].{field}", left.get(field), right.get(field))
        maximum = max(maximum, abs(float(left["logprob"]) - float(right["logprob"])))
        left_top = left.get("top_logprobs", [])
        right_top = right.get("top_logprobs", [])
        require_equal(f"logprob[{index}] top count", len(left_top), len(right_top))
        for top_index, (a, b) in enumerate(zip(left_top, right_top, strict=True)):
            for field in ("id", "token", "bytes"):
                require_equal(
                    f"logprob[{index}].top[{top_index}].{field}", a.get(field), b.get(field)
                )
            maximum = max(maximum, abs(float(a["logprob"]) - float(b["logprob"])))
    if maximum > LOGPROB_ABS_LIMIT:
        raise ParityFailure(f"maximum logprob error {maximum} exceeds {LOGPROB_ABS_LIMIT}")
    return maximum


def embedding_values(response: object) -> list[float]:
    data = response.get("data") if isinstance(response, dict) else response
    if not isinstance(data, list) or len(data) != 1:
        raise ParityFailure("embedding response must contain exactly one row")
    value = data[0].get("embedding")
    # The two native aliases retain llama-server's batch-shaped payload:
    # each response row contains one embedding row per input sequence.
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, str):
        raw = base64.b64decode(value, validate=True)
        if len(raw) % 4:
            raise ParityFailure("base64 embedding is not float32 aligned")
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if not isinstance(value, list):
        raise ParityFailure("embedding row is neither float array nor base64")
    return [float(item) for item in value]


def compare_embeddings(reference: object, candidate: object) -> tuple[float, float, int]:
    left = embedding_values(reference)
    right = embedding_values(candidate)
    require_equal("embedding dimension", len(left), len(right))
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    cosine = dot / (norm_left * norm_right)
    maximum = max(abs(a - b) for a, b in zip(left, right, strict=True))
    if cosine < EMBEDDING_COSINE_MIN or maximum > EMBEDDING_ABS_LIMIT:
        raise ParityFailure(
            f"embedding cosine/max_abs {cosine}/{maximum} misses "
            f"{EMBEDDING_COSINE_MIN}/{EMBEDDING_ABS_LIMIT}"
        )
    return cosine, maximum, len(left)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(
    reference_url: str,
    candidate_url: str,
    timeout: float,
    candidate_api_key: str | None,
    candidate_slot_dir: Path | None,
) -> list[dict]:
    results: list[dict] = []
    candidate_headers = (
        {"Authorization": f"Bearer {candidate_api_key}"} if candidate_api_key else {}
    )

    def candidate_request(
        route: str, payload: dict | None, *, method: str | None = None
    ) -> tuple[int, object]:
        return request_json(
            candidate_url,
            route,
            payload,
            timeout,
            headers=candidate_headers,
            method=method,
        )

    def check(name: str, operation) -> None:
        try:
            evidence = operation()
            results.append({"name": name, "status": "passed", "evidence": evidence or {}})
        except Exception as error:  # keep the complete differential in one report
            results.append({"name": name, "status": "failed", "error": str(error)})

    def both(route: str, payload: dict) -> tuple[dict, dict]:
        reference_status, reference = request_json(reference_url, route, payload, timeout)
        candidate_status, candidate = candidate_request(route, payload)
        require_equal(f"{route} HTTP status", reference_status, candidate_status)
        if reference_status != 200 or not isinstance(reference, dict) or not isinstance(candidate, dict):
            raise ParityFailure(f"{route} did not return two JSON objects")
        return reference, candidate

    def template(payload: dict) -> dict:
        left, right = both("/apply-template", payload)
        require_equal("template prompt", left.get("prompt"), right.get("prompt"))
        prompt = left.get("prompt")
        return {"bytes": len(prompt.encode()) if isinstance(prompt, str) else 0}

    check(
        "apply-template/plain",
        lambda: template({"messages": [{"role": "user", "content": "Hello"}]}),
    )
    check(
        "apply-template/tools",
        lambda: template(
            {
                "messages": [{"role": "user", "content": "What time is it?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "clock.now",
                            "description": "Read clock",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            }
        ),
    )

    def exact_json(route: str, payload: dict, fields: tuple[str, ...]) -> dict:
        left, right = both(route, payload)
        for field in fields:
            require_equal(f"{route}.{field}", left.get(field), right.get(field))
        return {field: left.get(field) for field in fields}

    for health_route in ("/health", "/v1/health"):
        check(
            f"health:{health_route}",
            lambda route=health_route: exact_json(route, None, ("status",)),
        )

    def models(route: str) -> dict:
        left, right = both(route, None)
        require_equal(f"{route}.object", left.get("object"), right.get("object"))
        left_models, right_models = left.get("models", []), right.get("models", [])
        left_data, right_data = left.get("data", []), right.get("data", [])
        require_equal(f"{route} model count", len(left_models), len(right_models))
        require_equal(f"{route} data count", len(left_data), len(right_data))
        if len(left_models) != 1 or len(left_data) != 1:
            raise ParityFailure(f"{route} must expose exactly one model")
        for field in ("type", "capabilities", "parameters", "details"):
            require_equal(f"{route}.models[0].{field}", left_models[0].get(field), right_models[0].get(field))
        for field in ("tags", "object", "meta"):
            require_equal(f"{route}.data[0].{field}", left_data[0].get(field), right_data[0].get(field))
        return {"meta": left_data[0].get("meta")}

    for models_route in ("/models", "/v1/models"):
        check(f"models:{models_route}", lambda route=models_route: models(route))

    def props() -> dict:
        left, right = both("/props", None)
        for field in ("total_slots", "model_ftype", "chat_template", "modalities"):
            require_equal(f"props.{field}", left.get(field), right.get(field))
        require_equal(
            "props.n_ctx",
            left.get("default_generation_settings", {}).get("n_ctx"),
            right.get("default_generation_settings", {}).get("n_ctx"),
        )
        return {"total_slots": left.get("total_slots")}

    check("props", props)

    def slots() -> dict:
        reference_status, left = request_json(reference_url, "/slots", None, timeout)
        candidate_status, right = candidate_request("/slots", None)
        require_equal("slots HTTP status", reference_status, candidate_status)
        if not isinstance(left, list) or not isinstance(right, list):
            raise ParityFailure("slots response must be an array")
        require_equal("slot count", len(left), len(right))
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            for field in ("id", "n_ctx", "speculative"):
                require_equal(f"slots[{index}].{field}", a.get(field), b.get(field))
        return {"slots": len(left), "n_ctx": [slot.get("n_ctx") for slot in left]}

    check("slots", slots)

    def physical_slot_actions() -> dict:
        if not candidate_api_key or candidate_slot_dir is None:
            raise ParityFailure(
                "physical slot actions require --candidate-api-key-file and --candidate-slot-dir"
            )
        filename = f"api-parity-{os.getpid()}-{time.time_ns()}.slot"
        snapshot_path = candidate_slot_dir / filename
        try:
            generation = {
                "prompt": [19873],
                "id_slot": 0,
                "max_tokens": 2,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": False,
                "return_tokens": True,
            }
            status, generated = candidate_request("/completion", generation)
            if status != 200 or not isinstance(generated, dict):
                raise ParityFailure(f"slot qualification generation failed: {status} {generated}")
            prompt_tokens = generated.get("tokens_evaluated")
            completion_tokens = generated.get("tokens_predicted")
            if (
                generated.get("id_slot") != 0
                or not isinstance(prompt_tokens, int)
                or not isinstance(completion_tokens, int)
            ):
                raise ParityFailure("slot qualification generation omitted its resident frontier")
            expected_frontier = prompt_tokens + completion_tokens

            status, saved = candidate_request(
                "/slots/0?action=save", {"filename": filename}
            )
            if status != 200 or not isinstance(saved, dict):
                raise ParityFailure(f"slot save failed: {status} {saved}")
            n_saved = saved.get("n_saved")
            if n_saved != expected_frontier:
                raise ParityFailure(
                    "slot save frontier differs from evaluated plus predicted tokens: "
                    f"{n_saved} != {expected_frontier}"
                )
            metadata = snapshot_path.stat()
            if not snapshot_path.is_file() or metadata.st_mode & 0o077:
                raise ParityFailure("slot snapshot is not a private regular file")

            status, erased = candidate_request(
                "/slots/0?action=erase", None, method="POST"
            )
            if status != 200 or not isinstance(erased, dict):
                raise ParityFailure(f"slot erase failed: {status} {erased}")

            status, restored = candidate_request(
                "/slots/0?action=restore", {"filename": filename}
            )
            if status != 200 or not isinstance(restored, dict):
                raise ParityFailure(f"slot restore failed: {status} {restored}")
            if restored.get("n_restored") != n_saved:
                raise ParityFailure(
                    f"slot restore frontier differs: {restored.get('n_restored')} != {n_saved}"
                )
            return {
                "slot": 0,
                "n_saved": n_saved,
                "n_restored": restored.get("n_restored"),
                "snapshot_mode": oct(metadata.st_mode & 0o777),
            }
        finally:
            try:
                snapshot_path.unlink()
            except FileNotFoundError:
                pass

    check("slots:save-erase-restore", physical_slot_actions)

    check(
        "tokenize",
        lambda: exact_json("/tokenize", {"content": "Hello, Muse!", "add_special": True}, ("tokens",)),
    )
    check(
        "detokenize",
        lambda: exact_json("/detokenize", {"tokens": [19873, 24, 22570, 2]}, ("content",)),
    )

    completion = {
        "prompt": [19873],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
        "logprobs": 5,
    }

    def native_completion(route: str, payload: dict = completion) -> dict:
        left, right = both(route, payload)
        for field in ("content", "tokens", "stop", "tokens_predicted"):
            require_equal(f"{route}.{field}", left.get(field), right.get(field))
        maximum = compare_logprobs(native_logprobs(left), native_logprobs(right))
        return {"tokens": left.get("tokens"), "max_logprob_abs_error": maximum}

    for route in ("/completion", "/completions"):
        check(f"completion-alias:{route}", lambda route=route: native_completion(route))

    schema_payload = dict(completion)
    schema_payload["max_tokens"] = 32
    schema_payload["json_schema"] = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    check("json-schema", lambda: native_completion("/completion", schema_payload))

    grammar_payload = dict(completion)
    grammar_payload["grammar"] = 'root ::= "Hello"'
    check("gbnf", lambda: native_completion("/completion", grammar_payload))

    sampler_cases = {
        "default": {},
        "penalties": {
            "samplers": ["penalties", "temperature"],
            "repeat_penalty": 1.15,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
        },
        "dry": {
            "prompt": [19873, 19873, 19873],
            "samplers": ["dry", "temperature"],
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
        },
        "top-n-sigma": {"samplers": ["top_n_sigma", "temperature"], "top_n_sigma": 1.5},
        "top-k": {"samplers": ["top_k", "temperature"], "top_k": 10},
        "typical-p": {"samplers": ["typ_p", "temperature"], "typical_p": 0.5},
        "top-p": {"samplers": ["top_p", "temperature"], "top_p": 0.5},
        "min-p": {"samplers": ["min_p", "temperature"], "min_p": 0.1},
        "xtc": {
            "samplers": ["xtc", "temperature"],
            "xtc_probability": 1.0,
            "xtc_threshold": 0.1,
        },
        "dynamic-temperature": {
            "samplers": ["temperature"],
            "dynatemp_range": 0.3,
            "dynatemp_exponent": 1.0,
        },
        "mirostat-1": {"mirostat": 1, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
        "mirostat-2": {"mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
        "adaptive-p": {
            "samplers": ["adaptive_p", "temperature"],
            "adaptive_target": 0.5,
            "adaptive_decay": 0.9,
        },
        "composed": {
            "repeat_penalty": 1.1,
            "dry_multiplier": 0.2,
            "top_n_sigma": 2.0,
            "top_k": 30,
            "typical_p": 0.95,
            "top_p": 0.9,
            "min_p": 0.03,
            "xtc_probability": 0.1,
            "xtc_threshold": 0.1,
            "dynatemp_range": 0.1,
        },
    }

    def sampler_case(name: str, extra: dict) -> dict:
        payload = {
            "prompt": [19873],
            "max_tokens": 8,
            "temperature": 0.8,
            "seed": 123,
            "cache_prompt": False,
            "return_tokens": True,
        }
        payload.update(extra)
        left, right = both("/completion", payload)
        for field in ("tokens", "content", "stop", "tokens_predicted"):
            require_equal(f"sampler {name}.{field}", left.get(field), right.get(field))
        return {"tokens": left.get("tokens")}

    for sampler_name, sampler_extra in sampler_cases.items():
        check(
            f"sampler:{sampler_name}",
            lambda name=sampler_name, extra=sampler_extra: sampler_case(name, extra),
        )

    def openai_completion() -> dict:
        payload = {
            "model": "muse-glimmer-30b",
            "prompt": [19873],
            "max_tokens": 8,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
            "logprobs": 5,
        }
        left, right = both("/v1/completions", payload)
        left_choices = left.get("choices", [])
        right_choices = right.get("choices", [])
        require_equal("completion choice count", len(left_choices), len(right_choices))
        maximum = 0.0
        for index, (a, b) in enumerate(zip(left_choices, right_choices, strict=True)):
            for field in ("text", "index", "finish_reason"):
                require_equal(f"completion[{index}].{field}", a.get(field), b.get(field))
            a_logprobs = a.get("logprobs", {}).get("content", [])
            b_logprobs = b.get("logprobs", {}).get("content", [])
            maximum = max(maximum, compare_logprobs(a_logprobs, b_logprobs))
        require_equal("completion usage", left.get("usage"), right.get("usage"))
        return {"choices": len(left_choices), "max_logprob_abs_error": maximum}

    check("v1-completions", openai_completion)

    def multiple_completions() -> dict:
        payload = {
            "model": "muse-glimmer-30b",
            "prompt": [19873],
            "max_tokens": 4,
            "temperature": 0.8,
            "seed": 123,
            "cache_prompt": False,
            "n": 2,
        }
        left, right = both("/v1/completions", payload)
        left_choices, right_choices = left.get("choices", []), right.get("choices", [])
        require_equal("n=2 choice count", len(left_choices), len(right_choices))
        for index, (a, b) in enumerate(zip(left_choices, right_choices, strict=True)):
            for field in ("text", "index", "finish_reason"):
                require_equal(f"n=2[{index}].{field}", a.get(field), b.get(field))
        require_equal("n=2 usage", left.get("usage"), right.get("usage"))
        return {"choices": len(left_choices), "usage": left.get("usage")}

    check("v1-completions:n=2", multiple_completions)

    chat_payload = {
        "model": "muse-glimmer-30b",
        "messages": [{"role": "user", "content": "Reply with one word: hello"}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
    }

    def chat() -> dict:
        left, right = both("/v1/chat/completions", chat_payload)
        left_choices = left.get("choices", [])
        right_choices = right.get("choices", [])
        require_equal("chat choice count", len(left_choices), len(right_choices))
        for index, (a, b) in enumerate(zip(left_choices, right_choices, strict=True)):
            require_equal(f"chat[{index}].message", a.get("message"), b.get("message"))
            require_equal(f"chat[{index}].finish_reason", a.get("finish_reason"), b.get("finish_reason"))
        require_equal("chat usage", left.get("usage"), right.get("usage"))
        return {"choices": len(left_choices)}

    check("v1-chat-completions", chat)

    response_format_payload = {
        "model": "muse-glimmer-30b",
        "messages": [{"role": "user", "content": "Return whether this is okay."}],
        "max_tokens": 96,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        },
    }

    def chat_response_format() -> dict:
        left, right = both("/v1/chat/completions", response_format_payload)
        left_choice, right_choice = left.get("choices", [{}])[0], right.get("choices", [{}])[0]
        require_equal("response_format message", left_choice.get("message"), right_choice.get("message"))
        require_equal(
            "response_format finish_reason",
            left_choice.get("finish_reason"),
            right_choice.get("finish_reason"),
        )
        require_equal("response_format usage", left.get("usage"), right.get("usage"))
        return {"usage": left.get("usage")}

    check("v1-chat-response-format", chat_response_format)

    def chat_alias() -> dict:
        left, right = both("/chat/completions", chat_payload)
        left_choices, right_choices = left.get("choices", []), right.get("choices", [])
        require_equal("chat alias choices", left_choices, right_choices)
        require_equal("chat alias usage", left.get("usage"), right.get("usage"))
        return {"choices": len(left_choices)}

    check("chat-completions-alias", chat_alias)

    tool_payload = {
        "model": "muse-glimmer-30b",
        "messages": [{"role": "user", "content": "Call clock.now now."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "clock.now",
                    "description": "Read clock",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
        "max_tokens": 96,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
    }

    def tool_completion(payload: dict, label: str) -> dict:
        left, right = both("/v1/chat/completions", payload)
        left_choice = left.get("choices", [{}])[0]
        right_choice = right.get("choices", [{}])[0]
        require_equal(f"{label} finish_reason", left_choice.get("finish_reason"), right_choice.get("finish_reason"))
        left_message, right_message = left_choice.get("message", {}), right_choice.get("message", {})
        for field in ("role", "content", "reasoning_content"):
            require_equal(f"{label} message.{field}", left_message.get(field), right_message.get(field))
        left_calls, right_calls = left_message.get("tool_calls", []), right_message.get("tool_calls", [])
        require_equal(f"{label} call count", len(left_calls), len(right_calls))
        for index, (a, b) in enumerate(zip(left_calls, right_calls, strict=True)):
            require_equal(f"{label}[{index}].type", a.get("type"), b.get("type"))
            require_equal(f"{label}[{index}].function", a.get("function"), b.get("function"))
        require_equal(f"{label} usage", left.get("usage"), right.get("usage"))
        return {"calls": len(left_calls), "usage": left.get("usage")}

    check("tools:required", lambda: tool_completion(tool_payload, "required tool"))
    named_tool_payload = dict(
        tool_payload,
        tool_choice={"type": "function", "function": {"name": "clock.now"}},
    )
    check("tools:named", lambda: tool_completion(named_tool_payload, "named tool"))

    def streamed_tool() -> dict:
        payload = dict(tool_payload, stream=True, stream_options={"include_usage": True})
        left = request_sse(reference_url, "/v1/chat/completions", payload, timeout)
        right = request_sse(candidate_url, "/v1/chat/completions", payload, timeout)

        def semantic(events: list[dict]) -> dict:
            reasoning: list[str] = []
            content: list[str] = []
            calls: dict[int, dict] = {}
            finish_reason = None
            usage = None
            opening_has_null_content = False
            for event in events:
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta", {})
                    if delta.get("role") == "assistant":
                        opening_has_null_content = "content" in delta and delta["content"] is None
                    if isinstance(delta.get("reasoning_content"), str):
                        reasoning.append(delta["reasoning_content"])
                    if isinstance(delta.get("content"), str):
                        content.append(delta["content"])
                    for call in delta.get("tool_calls", []):
                        index = int(call["index"])
                        current = calls.setdefault(index, {"type": None, "name": "", "arguments": ""})
                        current["type"] = call.get("type", current["type"])
                        function = call.get("function", {})
                        current["name"] += function.get("name", "")
                        current["arguments"] += function.get("arguments", "")
            return {
                "reasoning": "".join(reasoning),
                "content": "".join(content),
                "calls": [calls[index] for index in sorted(calls)],
                "finish_reason": finish_reason,
                "usage": usage,
                "opening_has_null_content": opening_has_null_content,
            }

        left_semantic, right_semantic = semantic(left), semantic(right)
        require_equal("streamed tool semantics", left_semantic, right_semantic)
        return {
            "reference_events": len(left),
            "candidate_events": len(right),
            "calls": left_semantic["calls"],
        }

    check("tools:required-stream", streamed_tool)

    def disconnect_cancellation() -> dict:
        parsed = urllib.parse.urlsplit(candidate_url)
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        kwargs = {"timeout": timeout}
        if parsed.scheme == "https":
            kwargs["context"] = ssl.create_default_context()
        connection = connection_type(parsed.hostname, parsed.port, **kwargs)
        payload = dict(tool_payload)
        body = json.dumps(payload, separators=(",", ":")).encode()
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders()
        connection.send(body)

        def busy_slots() -> int:
            status, slots = candidate_request("/slots", None)
            if status != 200 or not isinstance(slots, list):
                raise ParityFailure("could not observe slots during disconnect qualification")
            return sum(bool(slot.get("is_processing")) for slot in slots)

        began = time.monotonic()
        while busy_slots() == 0 and time.monotonic() - began < 5.0:
            time.sleep(0.05)
        if busy_slots() == 0:
            connection.close()
            raise ParityFailure("disconnect request never acquired a serving slot")
        connection.close()
        disconnected = time.monotonic()
        while busy_slots() != 0 and time.monotonic() - disconnected < 5.0:
            time.sleep(0.05)
        elapsed = time.monotonic() - disconnected
        if busy_slots() != 0:
            raise ParityFailure(f"slot remained busy {elapsed:.3f}s after client disconnect")
        return {"slot_release_seconds": elapsed}

    check("cancellation:nonstream-disconnect", disconnect_cancellation)

    ollama_payload = {
        "model": "muse-glimmer-30b",
        "prompt": "Hello",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 4, "seed": 42},
    }

    def ollama_aliases() -> dict:
        status_a, left = candidate_request("/api/generate", ollama_payload)
        status_b, right = candidate_request("/generate", ollama_payload)
        require_equal("Ollama alias status", status_a, status_b)
        if status_a != 200 or not isinstance(left, dict) or not isinstance(right, dict):
            raise ParityFailure("Ollama aliases did not return JSON objects")
        for value in (left, right):
            for field in (
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "created_at",
            ):
                value.pop(field, None)
        require_equal("Ollama aliases", left, right)
        if not left.get("done") or left.get("eval_count") != 4 or not left.get("context"):
            raise ParityFailure("Ollama terminal record omitted done/eval/context evidence")
        return {"eval_count": left["eval_count"], "context_tokens": len(left["context"])}

    check("ollama:generate-aliases", ollama_aliases)

    def ollama_stream() -> dict:
        payload = dict(ollama_payload)
        payload.pop("stream")
        rows = request_ndjson(candidate_url, "/api/generate", payload, timeout)
        if not rows or not rows[-1].get("done"):
            raise ParityFailure("Ollama NDJSON omitted its terminal done record")
        if any(row.get("done") for row in rows[:-1]):
            raise ParityFailure("Ollama NDJSON emitted an early terminal record")
        visible = "".join(str(row.get("response", "")) for row in rows)
        thinking = "".join(str(row.get("thinking", "")) for row in rows)
        status, nonstream = candidate_request("/api/generate", ollama_payload)
        if status != 200 or not isinstance(nonstream, dict):
            raise ParityFailure("Ollama nonstream equivalence request failed")
        require_equal("Ollama stream response", visible, nonstream.get("response", ""))
        require_equal("Ollama stream thinking", thinking, nonstream.get("thinking", ""))
        require_equal("Ollama stream context", rows[-1].get("context"), nonstream.get("context"))
        require_equal("Ollama stream eval_count", rows[-1].get("eval_count"), nonstream.get("eval_count"))
        return {"records": len(rows), "eval_count": rows[-1].get("eval_count")}

    check("ollama:ndjson-equivalence", ollama_stream)

    def ollama_rejections() -> dict:
        rejected = {
            "suffix": dict(ollama_payload, suffix="infill"),
            "keep_alive": dict(ollama_payload, keep_alive="0"),
            "template": dict(ollama_payload, template="{{ .Prompt }}"),
        }
        evidence = {}
        for field, payload in rejected.items():
            status, value = candidate_request("/api/generate", payload)
            if status != 400 or field not in json.dumps(value):
                raise ParityFailure(f"Ollama unsupported {field} was not rejected by name")
            evidence[field] = status
        return evidence

    check("ollama:intentional-rejections", ollama_rejections)

    def ollama_structured_format() -> dict:
        payload = {
            "model": "muse-glimmer-30b",
            "prompt": "Hello",
            "raw": True,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            "options": {"temperature": 0, "num_predict": 32, "seed": 42},
        }
        status, value = candidate_request("/api/generate", payload)
        if status != 200 or not isinstance(value, dict):
            raise ParityFailure("Ollama structured format request failed")
        parsed = json.loads(value.get("response", ""))
        if set(parsed) != {"ok"} or not isinstance(parsed["ok"], bool):
            raise ParityFailure("Ollama structured response violates its requested schema")
        if not value.get("done") or value.get("done_reason") != "stop":
            raise ParityFailure("Ollama structured response omitted its stop terminal")
        return {"response": value["response"], "eval_count": value.get("eval_count")}

    check("ollama:structured-format", ollama_structured_format)

    def stream() -> dict:
        payload = dict(chat_payload, stream=True, stream_options={"include_usage": True})
        left = request_sse(reference_url, "/v1/chat/completions", payload, timeout)
        right = request_sse(candidate_url, "/v1/chat/completions", payload, timeout)

        def visible(events: list[dict]) -> tuple[list[object], object]:
            deltas = []
            usage = None
            for event in events:
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") is not None:
                        deltas.append(delta["content"])
            return deltas, usage

        require_equal("stream visible deltas/usage", visible(left), visible(right))
        return {"reference_events": len(left), "candidate_events": len(right)}

    check("v1-chat-stream", stream)

    def resumable_stream() -> dict:
        conversation_id = f"api-parity-{os.getpid()}-{time.time_ns()}"
        payload = dict(chat_payload, stream=True, stream_options={"include_usage": True})
        live = request_sse(
            candidate_url,
            "/v1/chat/completions",
            payload,
            timeout,
            headers={**candidate_headers, "X-Conversation-Id": conversation_id},
        )
        status, views = candidate_request(
            "/v1/streams/lookup", {"conversation_ids": [conversation_id]}
        )
        if status != 200 or not isinstance(views, list) or len(views) != 1:
            raise ParityFailure(f"resumable stream lookup failed: {status} {views}")
        view = views[0]
        if (
            view.get("conversation_id") != conversation_id
            or view.get("is_done") is not True
            or not isinstance(view.get("total_bytes"), int)
            or view["total_bytes"] <= 0
        ):
            raise ParityFailure(f"resumable stream lookup returned invalid state: {view}")
        query_id = urllib.parse.quote(conversation_id, safe="")
        replay = request_sse(
            candidate_url,
            f"/v1/stream?conv_id={query_id}&from=0",
            None,
            timeout,
            headers=candidate_headers,
        )
        require_equal("resumable stream replay", live, replay)
        status, _ = candidate_request(
            f"/v1/stream?conv_id={query_id}", None, method="DELETE"
        )
        if status != 204:
            raise ParityFailure(f"resumable stream deletion returned HTTP {status}")
        status, after = candidate_request(
            "/v1/streams/lookup", {"conversation_ids": [conversation_id]}
        )
        if status != 200 or after != []:
            raise ParityFailure("deleted resumable stream remained discoverable")
        return {"events": len(live), "retained_bytes": view["total_bytes"]}

    check("stream:resume-lookup-delete", resumable_stream)

    def active_reasoning_control() -> dict:
        parsed = urllib.parse.urlsplit(candidate_url)
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        kwargs = {"timeout": timeout}
        if parsed.scheme == "https":
            kwargs["context"] = ssl.create_default_context()
        connection = connection_type(parsed.hostname, parsed.port, **kwargs)
        payload = dict(chat_payload, stream=True, reasoning_control=True, max_tokens=32)
        body = json.dumps(payload, separators=(",", ":")).encode()
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Accept", "text/event-stream")
        connection.putheader("Content-Length", str(len(body)))
        for name, value in candidate_headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        connection.send(body)
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read().decode("utf-8", errors="replace")
            connection.close()
            raise ParityFailure(f"reasoning-control stream failed: {response.status} {detail}")
        completion_id = None
        events = 0
        controlled = False
        try:
            while True:
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                events += 1
                if completion_id is None and isinstance(event.get("id"), str):
                    completion_id = event["id"]
                    status, result = candidate_request(
                        "/v1/chat/completions/control",
                        {"id": completion_id, "action": "reasoning_end"},
                    )
                    if status != 200 or not isinstance(result, dict) or result.get("success") is not True:
                        raise ParityFailure(
                            f"active reasoning control was not accepted: {status} {result}"
                        )
                    controlled = True
        finally:
            connection.close()
        if not controlled or completion_id is None or events < 2:
            raise ParityFailure("reasoning control did not observe a complete active stream")
        return {"events": events, "control_accepted": controlled}

    check("reasoning-control:active", active_reasoning_control)

    def embedding(route: str = "/v1/embeddings", encoding_format: str = "float") -> dict:
        payload = {"model": "muse-glimmer-30b", "input": [19873], "encoding_format": encoding_format}
        reference_status, left = request_json(reference_url, route, payload, timeout)
        candidate_status, right = candidate_request(route, payload)
        require_equal(f"{route} HTTP status", reference_status, candidate_status)
        if reference_status != 200:
            raise ParityFailure(f"{route} did not return HTTP 200")
        cosine, maximum, dimension = compare_embeddings(left, right)
        return {"dimension": dimension, "cosine": cosine, "max_abs_error": maximum}

    check("v1-embeddings", embedding)
    check("v1-embeddings:base64", lambda: embedding("/v1/embeddings", "base64"))
    check("embedding-alias", lambda: embedding("/embedding"))
    check("embeddings-alias", lambda: embedding("/embeddings"))

    def strict_unknown() -> dict:
        status, value = candidate_request(
            "/v1/chat/completions", dict(chat_payload, muser_unknown_field=True)
        )
        if status != 400 or "muser_unknown_field" not in json.dumps(value):
            raise ParityFailure("candidate did not reject and name an unknown nested DTO field")
        return {"status": status}

    check("strict-unknown-field", strict_unknown)

    def strict_error_contracts() -> dict:
        cases = [
            (
                "n",
                "/v1/chat/completions",
                dict(chat_payload, n=5),
                400,
                "invalid_request_error",
            ),
            (
                "model",
                "/v1/chat/completions",
                dict(chat_payload, model="missing-model"),
                404,
                "model_not_found",
            ),
            (
                "grammar",
                "/completion",
                {
                    "prompt": [19873],
                    "max_tokens": 1,
                    "grammar": 'root ::= "x"',
                    "json_schema": {"type": "string"},
                },
                400,
                "invalid_request_error",
            ),
            (
                "top_p",
                "/completion",
                {"prompt": [19873], "max_tokens": 1, "top_p": 2.0},
                400,
                "invalid_request_error",
            ),
        ]
        evidence = {}
        for field, route, payload, expected_status, expected_type in cases:
            status, value = candidate_request(route, payload)
            error = value.get("error", {}) if isinstance(value, dict) else {}
            if (
                status != expected_status
                or error.get("type") != expected_type
                or field not in str(error.get("message", ""))
            ):
                raise ParityFailure(f"strict error contract failed for {field}: {status} {value}")
            evidence[field] = status
        return evidence

    check("strict-error-contracts", strict_error_contracts)

    def raw_context_shift() -> dict:
        status, value = candidate_request(
            "/completion",
            {
                "prompt": [19873] * 513,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": False,
            },
        )
        if status != 200 or not isinstance(value, dict):
            raise ParityFailure("raw context shift request failed")
        if value.get("tokens_evaluated") != 511 or value.get("tokens_predicted") != 1:
            raise ParityFailure(f"raw context shift retained the wrong frontier: {value}")
        return {
            "input_tokens": 513,
            "retained_tokens": value["tokens_evaluated"],
            "completion_tokens": value["tokens_predicted"],
        }

    check("context-shift:raw-prefix-suffix", raw_context_shift)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--candidate-api-key-file", type=Path, required=True)
    parser.add_argument("--candidate-slot-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        key_metadata = args.candidate_api_key_file.stat()
        if not args.candidate_api_key_file.is_file() or key_metadata.st_mode & 0o077:
            raise ParityFailure("candidate API key file must be a private regular file")
        candidate_api_key = args.candidate_api_key_file.read_text(encoding="utf-8").strip()
        if not candidate_api_key:
            raise ParityFailure("candidate API key file is empty")
        checks = run(
            args.reference_url,
            args.candidate_url,
            args.timeout,
            candidate_api_key,
            args.candidate_slot_dir,
        )
        passed = all(check["status"] == "passed" for check in checks)
        report = {
            "schema": "muser.unsealed-qualification.v1",
            "lane": "api-parity",
            "status": "passed" if passed else "failed",
            "seal_eligible": False,
            "identity": args.identity,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reference_url": args.reference_url,
            "candidate_url": args.candidate_url,
            "thresholds": {
                "logprob_absolute_error": LOGPROB_ABS_LIMIT,
                "embedding_cosine": EMBEDDING_COSINE_MIN,
                "embedding_max_absolute_error": EMBEDDING_ABS_LIMIT,
            },
            "checks": checks,
        }
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if passed else 1
    except Exception as error:
        report = {
            "schema": "muser.unsealed-qualification.v1",
            "lane": "api-parity",
            "status": "failed",
            "seal_eligible": False,
            "identity": args.identity,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": str(error),
        }
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
