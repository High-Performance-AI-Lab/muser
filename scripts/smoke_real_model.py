#!/usr/bin/env python3
"""Human-facing, model-backed Muser API smoke.
This is deliberately smaller than api_parity.py: it checks one running Muser
server for coherent public behavior instead of comparing it to llama-server.
It never emits release evidence or a seal.
Run with --help for the required server URL, private API key, and report path.
If /props advertises vision, --image is mandatory and one real request runs.
"""
from __future__ import annotations
import argparse
import base64
import http.client
import json
import math
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from api_parity import ParityFailure, atomic_json, embedding_values
MODEL = "muse-glimmer-30b"
class Client:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        ca_file: Path | None,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or parsed.hostname not in ("127.0.0.1", "::1", "localhost")
            or parsed.port is None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ParityFailure(
                "--base-url must be one loopback HTTP(S) origin with an explicit port"
            )
        self.base_url = base_url.rstrip("/")
        self.parsed = parsed
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.ssl_context = None
        if parsed.scheme == "https":
            self.ssl_context = ssl.create_default_context(
                cafile=str(ca_file) if ca_file else None
            )
    def request(
        self,
        route: str,
        payload: dict | None = None,
        *,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object]:
        body = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        request_headers = {**self.headers, **(headers or {})}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + route,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self.ssl_context
            ) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                value = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                value = raw.decode("utf-8", errors="replace")
            return error.code, value
    def require_json(
        self,
        route: str,
        payload: dict | None = None,
        *,
        method: str | None = None,
        headers: dict[str, str] | None = None,
        status: int = 200,
    ) -> dict:
        actual, value = self.request(route, payload, method=method, headers=headers)
        if actual != status or not isinstance(value, dict):
            raise ParityFailure(f"{route}: expected HTTP {status} JSON, got {actual} {value}")
        return value
    def sse(
        self,
        route: str,
        payload: dict,
        *,
        headers: dict[str, str] | None = None,
    ) -> list[dict]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route,
            data=body,
            headers={
                **self.headers,
                **(headers or {}),
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        events: list[dict] = []
        with urllib.request.urlopen(
            request, timeout=self.timeout, context=self.ssl_context
        ) as response:
            if response.status != 200:
                raise ParityFailure(f"{route}: stream returned HTTP {response.status}")
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                events.append(json.loads(data))
        if not events:
            raise ParityFailure(f"{route}: stream returned no JSON events")
        return events
    def ndjson(self, route: str, payload: dict) -> list[dict]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route,
            data=body,
            headers={
                **self.headers,
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
            },
        )
        rows: list[dict] = []
        with urllib.request.urlopen(
            request, timeout=self.timeout, context=self.ssl_context
        ) as response:
            if response.status != 200:
                raise ParityFailure(f"{route}: NDJSON returned HTTP {response.status}")
            for raw in response:
                if raw.strip():
                    rows.append(json.loads(raw))
        if not rows:
            raise ParityFailure(f"{route}: NDJSON returned no records")
        return rows
    def connection(self) -> http.client.HTTPConnection:
        if self.parsed.scheme == "https":
            return http.client.HTTPSConnection(
                self.parsed.hostname,
                self.parsed.port,
                timeout=self.timeout,
                context=self.ssl_context,
            )
        return http.client.HTTPConnection(
            self.parsed.hostname, self.parsed.port, timeout=self.timeout
        )
def one_choice(value: dict, label: str) -> dict:
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ParityFailure(f"{label}: expected exactly one choice")
    if not isinstance(choices[0], dict):
        raise ParityFailure(f"{label}: choice is not an object")
    return choices[0]
def streamed_chat_semantics(events: list[dict]) -> dict:
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    finish_reason = None
    usage = None
    for event in events:
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {})
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning.append(delta["reasoning_content"])
            for call in delta.get("tool_calls", []):
                index = int(call["index"])
                current = calls.setdefault(
                    index, {"type": None, "name": "", "arguments": ""}
                )
                current["type"] = call.get("type", current["type"])
                function = call.get("function", {})
                current["name"] += function.get("name", "")
                current["arguments"] += function.get("arguments", "")
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": [calls[index] for index in sorted(calls)],
        "finish_reason": finish_reason,
        "usage": usage,
    }
def check_chat_equivalence(client: Client) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: human-smoke-ok"}],
        "max_tokens": 24,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
    }
    plain = client.require_json("/v1/chat/completions", payload)
    choice = one_choice(plain, "chat nonstream")
    message = choice.get("message", {})
    streamed = client.sse(
        "/v1/chat/completions",
        {**payload, "stream": True, "stream_options": {"include_usage": True}},
    )
    semantics = streamed_chat_semantics(streamed)
    if semantics["content"] != (message.get("content") or ""):
        raise ParityFailure("chat stream content differs from nonstream")
    if semantics["reasoning_content"] != (message.get("reasoning_content") or ""):
        raise ParityFailure("chat stream reasoning differs from nonstream")
    if semantics["finish_reason"] != choice.get("finish_reason"):
        raise ParityFailure("chat stream finish reason differs from nonstream")
    if semantics["usage"] != plain.get("usage"):
        raise ParityFailure("chat stream usage differs from nonstream")
    return {
        "events": len(streamed),
        "content": message.get("content"),
        "reasoning_bytes": len((message.get("reasoning_content") or "").encode()),
    }
def check_completion_aliases(client: Client) -> dict:
    payload = {
        "prompt": [19873],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
    }
    values = [
        client.require_json(route, payload)
        for route in ("/completion", "/completions")
    ]
    for field in ("content", "tokens", "stop", "tokens_predicted"):
        if values[0].get(field) != values[1].get(field):
            raise ParityFailure(f"native completion aliases differ at {field}")
    openai = client.require_json(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": [19873],
            "max_tokens": 8,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
        },
    )
    one_choice(openai, "v1 completion")
    return {"native_tokens": values[0].get("tokens"), "v1_usage": openai.get("usage")}
def check_ollama(client: Client) -> dict:
    payload = {
        "model": MODEL,
        "prompt": "Reply briefly with hello.",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8, "seed": 42},
    }
    plain = client.require_json("/api/generate", payload)
    alias = client.require_json("/generate", payload)
    for value in (plain, alias):
        for field in (
            "created_at",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        ):
            value.pop(field, None)
    if plain != alias:
        raise ParityFailure("Ollama nonstream aliases differ")
    rows = client.ndjson("/api/generate", {key: value for key, value in payload.items() if key != "stream"})
    if rows[-1].get("done") is not True or any(row.get("done") for row in rows[:-1]):
        raise ParityFailure("Ollama NDJSON terminal record is malformed")
    visible = "".join(str(row.get("response", "")) for row in rows)
    thinking = "".join(str(row.get("thinking", "")) for row in rows)
    if visible != plain.get("response", "") or thinking != plain.get("thinking", ""):
        raise ParityFailure("Ollama NDJSON visible output differs from nonstream")
    for field in ("context", "eval_count"):
        if rows[-1].get(field) != plain.get(field):
            raise ParityFailure(f"Ollama NDJSON differs at {field}")
    return {"records": len(rows), "eval_count": plain.get("eval_count")}
def check_constraints(client: Client) -> dict:
    common = {
        "prompt": [19873],
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
    }
    grammar = client.require_json(
        "/completion",
        {**common, "max_tokens": 16, "grammar": 'root ::= "Hello"'},
    )
    if grammar.get("content") != "Hello":
        raise ParityFailure(f"GBNF produced {grammar.get('content')!r}, not 'Hello'")
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    structured = client.require_json(
        "/completion", {**common, "max_tokens": 32, "json_schema": schema}
    )
    parsed = json.loads(structured.get("content", ""))
    if set(parsed) != {"ok"} or not isinstance(parsed["ok"], bool):
        raise ParityFailure("JSON-schema completion violated the schema")
    return {"grammar": grammar["content"], "json": parsed}
def check_tools_and_reasoning(client: Client) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Think briefly, then call clock.now."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "clock.now",
                    "description": "Read the current clock",
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
    plain = client.require_json("/v1/chat/completions", payload)
    choice = one_choice(plain, "tool chat")
    message = choice.get("message", {})
    calls = message.get("tool_calls")
    if choice.get("finish_reason") != "tool_calls" or not isinstance(calls, list) or len(calls) != 1:
        raise ParityFailure("required tool request did not return exactly one tool call")
    function = calls[0].get("function", {})
    if function.get("name") != "clock.now":
        raise ParityFailure("tool parser returned the wrong function")
    arguments = json.loads(function.get("arguments", ""))
    if arguments != {}:
        raise ParityFailure("clock.now tool arguments are not the requested empty object")
    transport = json.dumps(message)
    if "<|recipient|" in transport or "<|channel|" in transport:
        raise ParityFailure("ATEM transport markers leaked into the public response")
    events = client.sse(
        "/v1/chat/completions",
        {**payload, "stream": True, "stream_options": {"include_usage": True}},
    )
    semantic = streamed_chat_semantics(events)
    streamed_calls = semantic["tool_calls"]
    if len(streamed_calls) != 1 or streamed_calls[0]["name"] != "clock.now":
        raise ParityFailure("streamed tool deltas did not reconstruct clock.now")
    json.loads(streamed_calls[0]["arguments"])
    if semantic["reasoning_content"] != (message.get("reasoning_content") or ""):
        raise ParityFailure("streamed reasoning differs from nonstream reasoning")
    return {
        "tool": function["name"],
        "reasoning_bytes": len((message.get("reasoning_content") or "").encode()),
        "stream_events": len(events),
    }
def check_logprobs(client: Client) -> dict:
    value = client.require_json(
        "/completion",
        {
            "prompt": [19873],
            "max_tokens": 8,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
            "return_tokens": True,
            "logprobs": 5,
        },
    )
    rows = value.get("completion_probabilities")
    if not isinstance(rows, list) or not rows:
        raise ParityFailure("completion omitted logprob rows")
    for index, row in enumerate(rows):
        if not isinstance(row.get("id"), int) or not math.isfinite(float(row["logprob"])):
            raise ParityFailure(f"logprob row {index} omitted chosen token evidence")
        if not isinstance(row.get("bytes"), list) or not isinstance(row.get("top_logprobs"), list):
            raise ParityFailure(f"logprob row {index} omitted bytes/top entries")
        if len(row["top_logprobs"]) > 5:
            raise ParityFailure(f"logprob row {index} exceeded requested top-5")
    return {"rows": len(rows), "top_entries": [len(row["top_logprobs"]) for row in rows]}
def check_embeddings(client: Client) -> dict:
    common = {"model": MODEL, "input": [19873]}
    floats = client.require_json(
        "/v1/embeddings", {**common, "encoding_format": "float"}
    )
    encoded = client.require_json(
        "/v1/embeddings", {**common, "encoding_format": "base64"}
    )
    left = embedding_values(floats)
    right = embedding_values(encoded)
    if len(left) != 6656 or len(right) != 6656:
        raise ParityFailure(f"embedding dimension is {len(left)}/{len(right)}, expected 6656")
    norm = math.sqrt(sum(value * value for value in left))
    maximum = max(abs(a - b) for a, b in zip(left, right, strict=True))
    if abs(norm - 1.0) > 1.0e-4 or maximum > 1.0e-6:
        raise ParityFailure(
            f"embedding normalization/encoding mismatch: norm={norm} max_abs={maximum}"
        )
    return {"dimension": len(left), "l2_norm": norm, "encoding_max_abs": maximum}
def check_sessions(client: Client) -> dict:
    session_id = f"human-smoke-{uuid.uuid4().hex[:12]}"
    idempotency_key = f"human-smoke-{uuid.uuid4().hex}"
    created = False
    try:
        session = client.require_json(
            "/v1/sessions", {"id": session_id}, status=201
        )
        created = True
        if session.get("revision") != 0:
            raise ParityFailure("new session did not begin at revision zero")
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: session-ok"}],
            "max_tokens": 16,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
            "session_id": session_id,
            "expected_revision": 0,
        }
        headers = {"Idempotency-Key": idempotency_key}
        first = client.require_json("/v1/chat/completions", payload, headers=headers)
        if first.get("muser_session_revision") != 1:
            raise ParityFailure("stateful generation did not commit revision one")
        replay = client.require_json("/v1/chat/completions", payload, headers=headers)
        replay_fields = ("model", "choices", "usage", "system_fingerprint", "muser_session_revision")
        for field in replay_fields:
            if replay.get(field) != first.get(field):
                raise ParityFailure(f"idempotent replay changed cached field {field}")
        conflict_status, conflict = client.request(
            "/v1/chat/completions",
            {
                **payload,
                "messages": [{"role": "user", "content": "different request"}],
            },
            headers=headers,
        )
        if conflict_status != 409 or "Idempotency-Key" not in json.dumps(conflict):
            raise ParityFailure("idempotency key reuse with a different body did not conflict")
        saved = client.require_json(
            f"/v1/sessions/{session_id}/save", method="POST"
        )
        path = Path(saved.get("path", ""))
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise ParityFailure("saved session bundle is not a private regular file")
        status, body = client.request(
            f"/v1/sessions/{session_id}", method="DELETE"
        )
        if status != 204 or body is not None:
            raise ParityFailure(f"session delete failed: {status} {body}")
        created = False
        restored = client.require_json(
            f"/v1/sessions/{session_id}/restore", method="POST"
        )
        created = True
        if restored.get("revision") != 1 or restored.get("saved") is not True:
            raise ParityFailure("restored session lost revision or durable-save state")
        return {
            "session_id": session_id,
            "revision": restored["revision"],
            "bundle_mode": oct(path.stat().st_mode & 0o777),
        }
    finally:
        if created:
            client.request(f"/v1/sessions/{session_id}", method="DELETE")
def check_cancellation(client: Client) -> dict:
    connection = client.connection()
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Write a long numbered list with detailed explanations.",
            }
        ],
        "max_tokens": 512,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection.putrequest("POST", "/v1/chat/completions")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(len(body)))
    for name, value in client.headers.items():
        connection.putheader(name, value)
    connection.endheaders()
    connection.send(body)
    def busy_slots() -> int:
        status, slots = client.request("/slots")
        if status != 200 or not isinstance(slots, list):
            raise ParityFailure(f"could not observe slots: {status} {slots}")
        return sum(bool(slot.get("is_processing")) for slot in slots)
    began = time.monotonic()
    while busy_slots() == 0 and time.monotonic() - began < 10.0:
        time.sleep(0.05)
    if busy_slots() == 0:
        connection.close()
        raise ParityFailure("cancellation request never acquired a serving slot")
    connection.close()
    disconnected = time.monotonic()
    while busy_slots() != 0 and time.monotonic() - disconnected < 6.0:
        time.sleep(0.05)
    elapsed = time.monotonic() - disconnected
    if busy_slots() != 0:
        raise ParityFailure(f"serving slot remained busy {elapsed:.3f}s after disconnect")
    return {"slot_release_seconds": elapsed}
def check_vision(client: Client, props: dict, image_path: Path | None) -> dict:
    vision = props.get("modalities", {}).get("vision") is True
    if not vision:
        return {"status": "not_advertised"}
    if image_path is None:
        raise ParityFailure("/props advertises vision but --image was not provided")
    suffix = image_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix)
    if mime is None:
        raise ParityFailure("--image must be PNG, JPEG, or WebP")
    data_url = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode()
    value = client.require_json(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "\nDescribe the image in one short sentence."},
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
        },
    )
    choice = one_choice(value, "vision chat")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ParityFailure("vision request returned no visible description")
    return {"status": "passed", "content": content, "prompt_tokens": value.get("usage", {}).get("prompt_tokens")}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--ca-file", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate local arguments and print the check plan without contacting the server",
    )
    return parser.parse_args()
def private_key(path: Path) -> str:
    metadata = path.stat()
    if not path.is_file() or metadata.st_mode & 0o077:
        raise ParityFailure("--api-key-file must be a private regular file")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ParityFailure("--api-key-file is empty")
    return value
def main() -> int:
    args = parse_args()
    try:
        key = private_key(args.api_key_file)
        if args.ca_file is not None and not args.ca_file.is_file():
            raise ParityFailure("--ca-file is not a regular file")
        if args.image is not None and not args.image.is_file():
            raise ParityFailure("--image is not a regular file")
        client = Client(args.base_url, key, args.timeout, args.ca_file)
        names = [
            "chat-stream-equivalence",
            "completion-aliases",
            "ollama-stream-equivalence",
            "grammar-json-schema",
            "tool-reasoning-parser",
            "target-logprobs",
            "embeddings",
            "session-revision-idempotency-save-restore",
            "disconnect-cancellation",
            "vision-if-advertised",
        ]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "schema": "muser.human-engine-smoke.v1",
                        "status": "planned",
                        "base_url": client.base_url,
                        "checks": names,
                        "vision_image": str(args.image) if args.image else None,
                        "output": str(args.output),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        props = client.require_json("/props")
        results: list[dict] = []
        operations = [
            ("chat-stream-equivalence", lambda: check_chat_equivalence(client)),
            ("completion-aliases", lambda: check_completion_aliases(client)),
            ("ollama-stream-equivalence", lambda: check_ollama(client)),
            ("grammar-json-schema", lambda: check_constraints(client)),
            ("tool-reasoning-parser", lambda: check_tools_and_reasoning(client)),
            ("target-logprobs", lambda: check_logprobs(client)),
            ("embeddings", lambda: check_embeddings(client)),
            (
                "session-revision-idempotency-save-restore",
                lambda: check_sessions(client),
            ),
            ("disconnect-cancellation", lambda: check_cancellation(client)),
            (
                "vision-if-advertised",
                lambda: check_vision(client, props, args.image),
            ),
        ]
        for name, operation in operations:
            started = time.monotonic()
            try:
                evidence = operation()
                result = {
                    "name": name,
                    "status": "passed",
                    "elapsed_seconds": time.monotonic() - started,
                    "evidence": evidence,
                }
                print(f"PASS {name}: {json.dumps(evidence, sort_keys=True)}")
            except Exception as error:
                result = {
                    "name": name,
                    "status": "failed",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": str(error),
                }
                print(f"FAIL {name}: {error}", file=sys.stderr)
            results.append(result)
        passed = all(result["status"] == "passed" for result in results)
        report = {
            "schema": "muser.human-engine-smoke.v1",
            "status": "passed" if passed else "failed",
            "seal_eligible": False,
            "base_url": client.base_url,
            "model": MODEL,
            "server": {
                "build_info": props.get("build_info"),
                "slots": props.get("total_slots"),
                "context": props.get("default_generation_settings", {}).get("n_ctx"),
                "modalities": props.get("modalities"),
            },
            "checks": results,
        }
        atomic_json(args.output, report)
        print(f"report: {args.output}")
        return 0 if passed else 1
    except Exception as error:
        report = {
            "schema": "muser.human-engine-smoke.v1",
            "status": "failed",
            "seal_eligible": False,
            "error": str(error),
        }
        atomic_json(args.output, report)
        print(f"FAIL setup: {error}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
