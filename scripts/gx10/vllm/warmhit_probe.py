#!/usr/bin/env python3
"""Warm-hit TTFT probe: repeated prompt must skip the remote handoff.

Generalized, committed version of the throwaway
``nvfp4-pacing8g-20260818/warmhit-1/warmhit_probe.py`` evidence script.
Reads its prompts from real token-id fixture files (never hand-typed text)
and takes the resident container name as a flag, so it works against
whichever image is currently deployed on the node.

The resident GX10 producer is driven externally over its node-local socket
(request_producer.py), so each request's provenance is unambiguous: cold and
miss requests get a producer drive, the warm repeat gets none -- if it still
answers fast, it was served from the resident radix.

Uses muser_prompt_token_ids so the server and producer see identical tokens.

Timing note (load-bearing): the client request thread is started FIRST and
the producer is driven only after a short, fixed sleep. This mirrors the
only concrete evidence in the repo for how to avoid the resident producer's
900s watchdog wedge -- driving the producer with no server request already
waiting for it wedges the producer until the watchdog fires. Do not remove
or reorder this sleep.

This script drives both a local `muser serve` process (which loads the
model onto Metal) and the remote GX10 producer -- both accelerators are
implicated. The ladder runs it beside a separately accelerator_safe-leased
server process; it never starts, stops, or manages that process itself and
receives its --base-url from the operator.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

VOCAB_SIZE = 202_048


def read_tokens(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_bytes().split()]
    if len(tokens) < 2 or any(not 0 <= value < VOCAB_SIZE for value in tokens):
        raise ValueError(f"invalid token fixture: {path}")
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="running `muser serve` base URL")
    parser.add_argument("--bearer-token-file", required=True, type=Path)
    parser.add_argument(
        "--token-fixture",
        required=True,
        type=Path,
        help="token-id fixture for the cold/warm prompt (whitespace-separated ids)",
    )
    parser.add_argument(
        "--miss-token-fixture",
        required=True,
        type=Path,
        help="a different, shorter token-id fixture, to prove the radix "
        "distinguishes non-matching prompts",
    )
    parser.add_argument("--node", required=True, help="SSH target node alias")
    parser.add_argument(
        "--container", required=True, help="resident producer container name on --node"
    )
    parser.add_argument("--sock", default="/run/muser/work/producer.sock")
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--receiver-port", type=int, default=29590)
    parser.add_argument(
        "--host-work",
        required=True,
        help="node-local host path backing the container's /run/muser/work mount",
    )
    parser.add_argument("--node-work", default="/run/muser/work")
    parser.add_argument(
        "--producer-wait-seconds",
        type=float,
        default=3.0,
        help="delay between starting the client thread and driving the producer "
        "(matches the working precedent; do not lower without new evidence)",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--producer-timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--first-generation",
        type=int,
        required=True,
        help="first replay-safe handoff generation; the miss leg uses the next value",
    )
    parser.add_argument(
        "--request-prefix",
        default="warmhit",
        help="attempt-unique safe filename/request prefix for node-side inputs and receipts",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


class Probe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base = args.base_url.rstrip("/")
        self.auth = {"Authorization": "Bearer " + args.bearer_token_file.read_text().strip()}
        if args.first_generation < 1:
            raise ValueError("first generation must be positive")
        self.next_generation = args.first_generation
        self.request_prefix = getattr(args, "request_prefix", "warmhit")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.request_prefix):
            raise ValueError(
                "request prefix must contain only ASCII letters, digits, '.', '_', or '-'"
            )

    def post(self, tokens: list[int]) -> dict[str, object]:
        payload = {
            "model": "muse-glimmer-30b",
            "messages": [{"role": "user", "content": "unused: raw token ids supplied"}],
            "muser_prompt_token_ids": tokens,
            "max_tokens": self.args.max_tokens,
            "stream": True,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **self.auth},
        )
        t0 = time.monotonic()
        try:
            resp = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:800]
            return {
                "ttft_headers_s": round(time.monotonic() - t0, 4),
                "ttft_first_chunk_s": None,
                "ttft_first_token_s": None,
                "total_s": round(time.monotonic() - t0, 4),
                "text": "",
                "http_status": error.code,
                "error_body": body,
            }
        t_headers = time.monotonic() - t0
        first_chunk = None
        first_token = None
        text = ""
        raw_head = ""
        while True:
            line = resp.readline()
            if not line:
                break
            if len(raw_head) < 800:
                raw_head += line.decode("utf-8", "replace")
            if not line.startswith(b"data:"):
                continue
            if first_chunk is None:
                first_chunk = time.monotonic() - t0
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content") or ""
                if content:
                    if first_token is None:
                        first_token = time.monotonic() - t0
                    text += content
        total = time.monotonic() - t0
        status = getattr(resp, "status", None)
        resp.close()
        return {
            "ttft_headers_s": round(t_headers, 4),
            "ttft_first_chunk_s": round(first_chunk, 4) if first_chunk else None,
            "ttft_first_token_s": round(first_token, 4) if first_token else None,
            "total_s": round(total, 4),
            "text": text,
            "http_status": status,
            # An empty completion on a 200 is the case that looked like a
            # correctness failure; keep the raw stream so it is diagnosable.
            "raw_stream_head": raw_head[:800],
        }

    def snapshot(self) -> dict[str, object]:
        req = urllib.request.Request(f"{self.base}/snapshot", headers=self.auth)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)

    def drive_producer(self, name: str, tokens: list[int]) -> dict[str, object]:
        args = self.args
        generation = self.next_generation
        self.next_generation += 1
        transfer = f"{self.request_prefix}-{name}-g{generation}"
        # These files live on the resident's operational work volume. Keep
        # both generation-scoped so reruns preserve prior O_EXCL receipts and
        # can never consume a stale token input from an earlier request.
        fixture = f"{args.host_work.rstrip('/')}/{transfer}.tokens"
        receipt = f"{args.node_work.rstrip('/')}/{transfer}.json"
        body = "\n".join(str(token) for token in tokens) + "\n"
        subprocess.run(
            ["ssh", args.node, f"cat > {fixture}"], input=body.encode(), check=True
        )
        started = time.monotonic()
        result = subprocess.run(
            [
                "ssh",
                args.node,
                "docker",
                "exec",
                args.container,
                "python3",
                "/opt/muser/scripts/gx10/vllm/request_producer.py",
                "--sock",
                args.sock,
                "--tokens",
                f"{args.node_work.rstrip('/')}/{transfer}.tokens",
                "--request-id",
                transfer,
                "--generation",
                str(generation),
                "--transfer-id",
                transfer,
                "--receiver-host",
                args.receiver_host,
                "--receiver-port",
                str(args.receiver_port),
                "--output",
                receipt,
                "--timeout-seconds",
                str(args.producer_timeout_seconds),
            ],
            capture_output=True,
            text=True,
            timeout=args.producer_timeout_seconds + 40,
        )
        return {
            "generation": generation,
            "transfer": transfer,
            "node_receipt": receipt,
            "elapsed_s": round(time.monotonic() - started, 3),
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }

    def run_leg(self, name: str, tokens: list[int], drive: bool) -> dict[str, object]:
        outcome: dict[str, object] = {}
        leg_warnings: list[str] = []

        def client() -> None:
            try:
                outcome["response"] = self.post(tokens)
            except Exception as error:  # noqa: BLE001 - evidence, not control flow
                outcome["error"] = repr(error)

        try:
            snapshot_before: object = self.snapshot()
        except Exception as error:  # noqa: BLE001 - retained diagnostics
            snapshot_before = {"error": repr(error)}
            leg_warnings.append(f"snapshot_before failed: {error!r}")
        thread = threading.Thread(target=client, daemon=True)
        started = time.monotonic()
        thread.start()
        producer = None
        if drive:
            # Load-bearing: the client's request must already be waiting on
            # the server before the producer is driven, or the resident
            # producer wedges until its 900s watchdog fires.
            time.sleep(self.args.producer_wait_seconds)
            try:
                producer = self.drive_producer(name, tokens)
            except Exception as error:  # noqa: BLE001 - retained diagnostics
                producer = {"error": repr(error)}
        thread.join(timeout=950)
        if thread.is_alive():
            outcome["error"] = "client thread did not finish within 950 seconds"
        try:
            snapshot_after: object = self.snapshot()
        except Exception as error:  # noqa: BLE001 - retained diagnostics
            snapshot_after = {"error": repr(error)}
            leg_warnings.append(f"snapshot_after failed: {error!r}")
        record: dict[str, object] = {
            "tokens": len(tokens),
            "producer_driven": drive,
            "wall_s": round(time.monotonic() - started, 3),
            "snapshot_before": snapshot_before,
            "snapshot_after": snapshot_after,
            "leg_warnings": leg_warnings,
        }
        record.update(outcome)
        if producer is not None:
            record["producer"] = producer
        return record


def leg_error(record: dict[str, object], *, producer_required: bool) -> str | None:
    """Return why a leg is invalid, keeping infrastructure distinct from text."""
    if record.get("error"):
        return str(record["error"])
    warnings = record.get("leg_warnings")
    if isinstance(warnings, list) and warnings:
        return "; ".join(str(warning) for warning in warnings)
    response = record.get("response")
    if not isinstance(response, dict):
        return "response is missing"
    if response.get("http_status") != 200:
        return f"HTTP status is {response.get('http_status')!r}"
    if response.get("text", "") == "":
        detail = response.get("error_body") or response.get("raw_stream_head")
        return f"response contains no generated text ({detail!r})"
    if producer_required:
        producer = record.get("producer")
        if not isinstance(producer, dict):
            return "producer evidence is missing"
        if producer.get("error"):
            return f"producer raised {producer['error']}"
        if producer.get("returncode") != 0:
            return f"producer exited {producer.get('returncode')!r}"
    return None


def print_leg(name: str, drive: bool, tokens: list[int], record: dict[str, object]) -> None:
    response = record.get("response")
    resp = response if isinstance(response, dict) else {}
    print(
        f"{name}: driven={drive} tokens={len(tokens)} "
        f"headers={resp.get('ttft_headers_s')}s "
        f"first_token={resp.get('ttft_first_token_s')}s "
        f"total={resp.get('total_s')}s err={record.get('error')}",
        flush=True,
    )


def run_probe(
    args: argparse.Namespace,
    probe: Probe,
    long_tokens: list[int],
    other_tokens: list[int],
) -> dict[str, object]:
    """Run only legs that can still affect the fail-closed verdict."""
    evidence: dict[str, object] = {
        "prompt_tokens": len(long_tokens),
        "miss_prompt_tokens": len(other_tokens),
        "token_fixture": str(args.token_fixture),
        "miss_token_fixture": str(args.miss_token_fixture),
        "container": args.container,
        "first_generation": args.first_generation,
        "request_prefix": getattr(args, "request_prefix", "warmhit"),
    }

    cold = probe.run_leg("cold", long_tokens, True)
    evidence["cold"] = cold
    print_leg("cold", True, long_tokens, cold)
    cold_error = leg_error(cold, producer_required=True)
    if cold_error is not None:
        reason = f"cold leg invalid: {cold_error}"
        evidence["warm"] = {"skipped": reason}
        evidence["miss"] = {"skipped": reason}
        evidence.update(
            {
                "legs_valid": False,
                "leg_errors": {"cold": cold_error},
                "outputs_match": False,
                "warm_ttft_below_cold": False,
                "miss_control_valid": False,
                "miss_control_error": reason,
            }
        )
        return evidence

    warm = probe.run_leg("warm", long_tokens, False)
    evidence["warm"] = warm
    print_leg("warm", False, long_tokens, warm)
    warm_error = leg_error(warm, producer_required=False)
    if warm_error is not None:
        reason = f"warm leg invalid: {warm_error}"
        evidence["miss"] = {"skipped": reason}
        evidence.update(
            {
                "legs_valid": False,
                "leg_errors": {"warm": warm_error},
                "outputs_match": False,
                "warm_ttft_below_cold": False,
                "miss_control_valid": False,
                "miss_control_error": reason,
            }
        )
        return evidence

    cold_response = cold["response"]
    warm_response = warm["response"]
    assert isinstance(cold_response, dict) and isinstance(warm_response, dict)
    outputs_match = cold_response["text"] == warm_response["text"]
    cold_ttft = cold_response.get("ttft_first_token_s")
    warm_ttft = warm_response.get("ttft_first_token_s")
    evidence.update(
        {
            "legs_valid": True,
            "leg_errors": {},
            "outputs_match": outputs_match,
            "warm_ttft_below_cold": (
                cold_ttft is not None
                and warm_ttft is not None
                and warm_ttft < cold_ttft
            ),
        }
    )
    if not outputs_match:
        reason = "cold and warm generated text differ"
        evidence["miss"] = {"skipped": reason}
        evidence["miss_control_valid"] = False
        evidence["miss_control_error"] = reason
        return evidence

    miss = probe.run_leg("miss", other_tokens, True)
    evidence["miss"] = miss
    print_leg("miss", True, other_tokens, miss)
    miss_error = leg_error(miss, producer_required=True)
    evidence["miss_control_valid"] = miss_error is None
    evidence["miss_control_error"] = miss_error
    return evidence


def write_evidence(path: Path, evidence: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=1)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    for path in (args.bearer_token_file, args.token_fixture, args.miss_token_fixture):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")

    probe = Probe(args)
    long_tokens = read_tokens(args.token_fixture)
    other_tokens = read_tokens(args.miss_token_fixture)

    evidence = run_probe(args, probe, long_tokens, other_tokens)
    write_evidence(args.out, evidence)
    print(f"legs_valid={evidence['legs_valid']} leg_errors={evidence['leg_errors']}")
    print(f"outputs_match={evidence['outputs_match']}")
    print(f"warm_ttft_below_cold={evidence['warm_ttft_below_cold']}")
    print(
        f"miss_control_valid={evidence['miss_control_valid']} "
        f"miss_control_error={evidence['miss_control_error']}"
    )
    return 0 if evidence.get("legs_valid") is True else 1


if __name__ == "__main__":
    sys.exit(main())
