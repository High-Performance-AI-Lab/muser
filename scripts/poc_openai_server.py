#!/usr/bin/env python3
"""Run one cooperative, integrated OpenAI text+vision server proof."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--mtmd-bridge", type=Path, required=True)
    parser.add_argument("--dflash", type=Path, required=True)
    parser.add_argument("--ane-manifest", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--port", type=int, default=4951)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    return parser.parse_args()


def request(port: int, method: str, path: str, body: bytes = b"", timeout: int = 10):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    headers = {"Content-Type": "application/json"} if body else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    status = response.status
    connection.close()
    return status, payload


def wait_ready(port: int, process: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"server exited during startup with {code}")
        try:
            status, _ = request(port, "GET", "/healthz", timeout=2)
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError("server did not become ready before the client deadline")


def chat(port: int, payload: dict[str, object], timeout: int) -> dict[str, object]:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    status, response = request(port, "POST", "/v1/chat/completions", body, timeout)
    if status != 200:
        raise RuntimeError(f"chat returned HTTP {status}: {response[:8192]!r}")
    decoded = json.loads(response)
    choices = decoded.get("choices", [])
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content:
        raise RuntimeError("chat response did not contain nonempty assistant content")
    return {
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_bytes": len(content.encode()),
        "usage": decoded.get("usage"),
        "transport": "json",
    }


def streaming_chat(
    port: int, payload: dict[str, object], timeout: int
) -> dict[str, object]:
    """Consume a complete OpenAI SSE response and validate its framing."""
    request_payload = dict(payload)
    request_payload["stream"] = True
    request_payload["stream_options"] = {"include_usage": True}
    body = json.dumps(
        request_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        failure = response.read()
        connection.close()
        raise RuntimeError(
            f"streaming chat returned HTTP {response.status}: {failure[:8192]!r}"
        )
    pieces: list[str] = []
    usage: object = None
    saw_role = False
    saw_terminal = False
    saw_done = False
    while line := response.readline():
        line = line.strip()
        if not line or not line.startswith(b"data: "):
            continue
        data = line[len(b"data: "):]
        if data == b"[DONE]":
            saw_done = True
            break
        event = json.loads(data)
        choices = event.get("choices", [])
        if not choices:
            usage = event.get("usage")
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        saw_role |= delta.get("role") == "assistant"
        content = delta.get("content")
        if isinstance(content, str):
            pieces.append(content)
        saw_terminal |= choice.get("finish_reason") in {"stop", "length"}
    connection.close()
    content = "".join(pieces)
    if not content or not saw_role or not saw_terminal or not saw_done:
        raise RuntimeError(
            "streaming chat omitted content, role, terminal chunk, or [DONE]"
        )
    if not isinstance(usage, dict):
        raise RuntimeError("streaming chat omitted requested usage chunk")
    return {
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_bytes": len(content.encode()),
        "usage": usage,
        "transport": "sse",
    }


def advertised_model(port: int) -> str:
    status, body = request(port, "GET", "/v1/models", timeout=10)
    if status != 200:
        raise RuntimeError(f"model listing returned HTTP {status}")
    listing = json.loads(body)
    models = listing.get("data", [])
    model_id = models[0].get("id") if models else None
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("model listing did not advertise a model ID")
    return model_id


def main() -> int:
    args = parse_args()
    token = "muser-poc-cooperative-shutdown-20260812"
    command = [
        str(args.server_binary), "serve",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--model", str(args.model), "--backend", "metal",
        "--prefix-cache", "on",
        "--mmproj", str(args.mmproj), "--mtmd-bridge", str(args.mtmd_bridge),
        "--dflash", str(args.dflash), "--dflash-backend", "ane",
        "--ane-manifest", str(args.ane_manifest),
        "--benchmark-shutdown-token", token,
        "--benchmark-deadline-seconds", str(args.timeout_seconds + 60),
    ]
    process = subprocess.Popen(command)
    ready = False
    try:
        wait_ready(args.port, process, args.timeout_seconds)
        ready = True
        model_name = advertised_model(args.port)
        text = streaming_chat(args.port, {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with one short greeting."}],
            "max_tokens": 4,
            "temperature": 0.0,
        }, args.timeout_seconds)
        image = args.image.read_bytes()
        vision = chat(args.port, {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(image).decode()
                    }},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        }, args.timeout_seconds)
        status, health_body = request(args.port, "GET", "/health", timeout=10)
        if status != 200:
            raise RuntimeError(f"health returned HTTP {status}")
        health = json.loads(health_body)
        loaded = health.get("model", {})
        if (
            loaded.get("backend") != "metal"
            or loaded.get("vision_route") != "mtmd-metal:muser-mtmd-muse-vision-v1"
            or loaded.get("dflash_route") != "ane"
        ):
            raise RuntimeError(
                "health did not attest the requested Metal + mtmd + ANE routes"
            )
        summary = {
            "schema": "muser.openai-poc.v1",
            "text": text,
            "vision": vision,
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "health": health,
            "poc_pass": True,
            "seal_eligible": False,
        }
        print(json.dumps(summary, sort_keys=True))
    finally:
        if ready and process.poll() is None:
            status, payload = request(
                args.port, "POST", "/__muser/benchmark/shutdown", token.encode(), 10
            )
            if status != 200:
                raise RuntimeError(
                    f"cooperative shutdown returned HTTP {status}: {payload[:1024]!r}"
                )
        try:
            wait_seconds = 30 if ready else args.timeout_seconds + 120
            code = process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("server ignored cooperative shutdown") from error
        if code != 0:
            raise RuntimeError(f"server exited with {code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
