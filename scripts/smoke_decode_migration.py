#!/usr/bin/env python3
"""Human two-server decode migration smoke; never release/notarial evidence.

Both running servers must use identical model/template/layout/DFlash/vision
identities, the same API key, and HTTPS certificates trusted by the source
server's platform trust store. Use --dry-run to validate local wiring only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.parse
import uuid
from api_parity import ParityFailure, atomic_json
from smoke_real_model import Client, MODEL, one_choice, private_key


def require_status(
    client: Client,
    route: str,
    expected: int,
    payload: dict | None = None,
    *,
    method: str | None = None,
    headers: dict[str, str] | None = None,
) -> object:
    status, value = client.request(route, payload, method=method, headers=headers)
    if status != expected:
        raise ParityFailure(f"{route}: expected HTTP {expected}, got {status} {value}")
    return value


def session_view(client: Client, session_id: str) -> tuple[int, object]:
    return client.request(f"/v1/sessions/{session_id}")


def delete_session(client: Client, session_id: str) -> None:
    status, _ = client.request(f"/v1/sessions/{session_id}", method="DELETE")
    if status not in (204, 404):
        raise ParityFailure(f"could not clean session {session_id}: HTTP {status}")


def assistant_message(response: dict) -> dict:
    message = one_choice(response, "stateful generation").get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ParityFailure("stateful generation omitted its assistant message")
    allowed = ("role", "content", "reasoning_content", "tool_calls")
    return {key: message[key] for key in allowed if key in message}


def semantic_response(response: dict) -> dict:
    return {
        "model": response.get("model"),
        "choices": response.get("choices"),
        "usage": response.get("usage"),
        "system_fingerprint": response.get("system_fingerprint"),
        "revision": response.get("muser_session_revision"),
    }


def create_saved_baseline(client: Client, session_id: str) -> tuple[dict, dict]:
    created = client.require_json("/v1/sessions", {"id": session_id}, status=201)
    if created.get("revision") != 0:
        raise ParityFailure("new source session did not start at revision zero")
    user = {
        "role": "user",
        "content": "Reply with exactly this text: migration-baseline-ok",
    }
    payload = {
        "model": MODEL,
        "messages": [user],
        "max_tokens": 24,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "session_id": session_id,
        "expected_revision": 0,
    }
    generated = client.require_json(
        "/v1/chat/completions",
        payload,
        headers={"Idempotency-Key": f"baseline-{session_id}"},
    )
    if generated.get("muser_session_revision") != 1:
        raise ParityFailure("baseline generation did not commit revision one")
    saved = client.require_json(f"/v1/sessions/{session_id}/save", method="POST")
    path = Path(saved.get("path", ""))
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ParityFailure("source save is not a private regular bundle")
    return user, assistant_message(generated)


def continue_session(
    client: Client,
    session_id: str,
    user: dict,
    assistant: dict,
    key_suffix: str,
) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            user,
            assistant,
            {
                "role": "user",
                "content": "Now reply with exactly this text: migration-continued-ok",
            },
        ],
        "max_tokens": 24,
        "temperature": 0,
        "seed": 314159,
        "cache_prompt": False,
        "session_id": session_id,
        "expected_revision": 1,
    }
    response = client.require_json(
        "/v1/chat/completions",
        payload,
        headers={"Idempotency-Key": f"continue-{key_suffix}-{session_id}"},
    )
    if response.get("muser_session_revision") != 2:
        raise ParityFailure("continued generation did not commit revision two")
    return semantic_response(response)


def migration_payload(
    destination: Client, mode: str, transfer_id: str
) -> dict:
    return {
        "destination": destination.base_url,
        "mode": mode,
        "tier": "decode",
        "transfer_id": transfer_id,
    }


def start_migration(
    source: Client,
    destination: Client,
    session_id: str,
    mode: str,
    transfer_id: str,
) -> dict:
    status, value = source.request(
        f"/v1/sessions/{session_id}/migrate",
        migration_payload(destination, mode, transfer_id),
    )
    if status != 202 or not isinstance(value, dict) or value.get("id") != transfer_id:
        raise ParityFailure(f"{mode} migration did not return its accepted ID: {status} {value}")
    return value


def fire_and_forget_migration(
    source: Client,
    destination: Client,
    session_id: str,
    mode: str,
    transfer_id: str,
) -> None:
    body = json.dumps(
        migration_payload(destination, mode, transfer_id), separators=(",", ":")
    ).encode()
    connection = source.connection()
    connection.putrequest("POST", f"/v1/sessions/{session_id}/migrate")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(len(body)))
    for name, value in source.headers.items():
        connection.putheader(name, value)
    connection.endheaders()
    connection.send(body)
    # Simulate an operator/client losing the response after the complete
    # request reached the source. The fixed transfer ID is the retry handle.
    connection.close()


def transfer_view(client: Client, transfer_id: str) -> tuple[int, object]:
    return client.request(f"/v1/session-transfers/{transfer_id}")


def wait_for_transfer(
    source: Client,
    destination: Client,
    session_id: str,
    transfer_id: str,
    mode: str,
    timeout: float,
) -> tuple[dict, dict, list[str]]:
    deadline = time.monotonic() + timeout
    observed: list[str] = []
    last_source: object = None
    last_destination: object = None
    retries = 0
    while time.monotonic() < deadline:
        destination_status, destination_value = transfer_view(destination, transfer_id)
        source_status, source_value = transfer_view(source, transfer_id)
        last_source, last_destination = source_value, destination_value
        if source_status != 200 or not isinstance(source_value, dict):
            time.sleep(0.1)
            continue
        state = str(source_value.get("status"))
        if not observed or observed[-1] != state:
            observed.append(state)
        if state == "ambiguous" and retries < 3:
            start_migration(source, destination, session_id, mode, transfer_id)
            retries += 1
        live_status, _ = session_view(source, session_id)
        if mode == "move" and live_status == 404:
            if destination_status != 200 or not isinstance(destination_value, dict):
                destination_status, destination_value = transfer_view(
                    destination, transfer_id
                )
            if (
                destination_status != 200
                or not isinstance(destination_value, dict)
                or destination_value.get("status") != "committed"
            ):
                raise ParityFailure(
                    "move deleted source before destination durable-commit ACK"
                )
        if (
            state == "completed"
            and destination_status == 200
            and isinstance(destination_value, dict)
            and destination_value.get("status") == "committed"
        ):
            if source_value.get("last_error") is not None:
                raise ParityFailure(f"terminal transfer retained an error: {source_value}")
            return source_value, destination_value, observed
        time.sleep(0.1)
    raise ParityFailure(
        f"{mode} transfer timed out: source={last_source} destination={last_destination}"
    )


def verify_transfer_views(
    source_view: dict,
    destination_view: dict,
    session_id: str,
    transfer_id: str,
    mode: str,
) -> None:
    expected = {
        "id": transfer_id,
        "session_id": session_id,
        "mode": mode,
        "tier": "decode",
    }
    for field, value in expected.items():
        if source_view.get(field) != value or destination_view.get(field) != value:
            raise ParityFailure(f"transfer view mismatch at {field}")
    if source_view.get("direction") != "outgoing":
        raise ParityFailure("source transfer direction is not outgoing")
    if destination_view.get("direction") != "incoming":
        raise ParityFailure("destination transfer direction is not incoming")
    for field in ("bytes", "sha256"):
        if not source_view.get(field) or source_view.get(field) != destination_view.get(field):
            raise ParityFailure(f"transfer payload evidence differs at {field}")
    if source_view.get("source_deleted") is not (mode == "move"):
        raise ParityFailure("source_deleted does not match migration mode")


def check_copy(source: Client, destination: Client, timeout: float) -> tuple[dict, dict]:
    session_id = f"copy-{uuid.uuid4().hex[:16]}"
    transfer_id = f"xfer-copy-{uuid.uuid4().hex[:16]}"
    try:
        user, assistant = create_saved_baseline(source, session_id)
        start_migration(source, destination, session_id, "copy", transfer_id)
        source_view, destination_view, observed = wait_for_transfer(
            source, destination, session_id, transfer_id, "copy", timeout
        )
        verify_transfer_views(
            source_view, destination_view, session_id, transfer_id, "copy"
        )
        source_status, source_session = session_view(source, session_id)
        destination_status, destination_session = session_view(destination, session_id)
        if source_status != 200 or destination_status != 200:
            raise ParityFailure("copy did not retain source and install destination")
        if source_session.get("revision") != 1 or destination_session.get("revision") != 1:
            raise ParityFailure("copy changed the committed session revision")
        source_continued = continue_session(
            source, session_id, user, assistant, "source-copy"
        )
        destination_continued = continue_session(
            destination, session_id, user, assistant, "destination-copy"
        )
        if source_continued != destination_continued:
            raise ParityFailure("copied source/destination continuation differs")
        evidence = {
            "session_id": session_id,
            "transfer_id": transfer_id,
            "bytes": source_view["bytes"],
            "sha256": source_view["sha256"],
            "observed_statuses": observed,
            "continued": source_continued,
            "source_retained": True,
        }
        return evidence, source_continued
    finally:
        delete_session(source, session_id)
        delete_session(destination, session_id)


def check_move_ambiguous_retry(
    source: Client,
    destination: Client,
    expected_continuation: dict,
    timeout: float,
) -> dict:
    session_id = f"move-{uuid.uuid4().hex[:16]}"
    transfer_id = f"xfer-move-{uuid.uuid4().hex[:16]}"
    destination_installed = False
    try:
        user, assistant = create_saved_baseline(source, session_id)
        fire_and_forget_migration(
            source, destination, session_id, "move", transfer_id
        )
        time.sleep(0.1)
        retry = start_migration(
            source, destination, session_id, "move", transfer_id
        )
        source_view, destination_view, observed = wait_for_transfer(
            source, destination, session_id, transfer_id, "move", timeout
        )
        destination_installed = True
        verify_transfer_views(
            source_view, destination_view, session_id, transfer_id, "move"
        )
        source_status, _ = session_view(source, session_id)
        destination_status, destination_session = session_view(destination, session_id)
        if source_status != 404 or destination_status != 200:
            raise ParityFailure("move did not delete source and install destination")
        if destination_session.get("revision") != 1:
            raise ParityFailure("move changed destination revision before continuation")
        continued = continue_session(
            destination, session_id, user, assistant, "destination-move"
        )
        if continued != expected_continuation:
            raise ParityFailure("moved destination continuation differs from copy oracle")
        terminal_before = {
            field: source_view.get(field)
            for field in ("status", "bytes", "sha256", "source_deleted", "last_error")
        }
        # A retry after the destination ACK and source deletion must reconcile
        # to the same terminal journal without recreating or retransmitting.
        terminal_retry = start_migration(
            source, destination, session_id, "move", transfer_id
        )
        deadline = time.monotonic() + min(timeout, 10.0)
        while terminal_retry.get("status") != "completed" and time.monotonic() < deadline:
            time.sleep(0.05)
            status, terminal_retry = transfer_view(source, transfer_id)
            if status != 200 or not isinstance(terminal_retry, dict):
                raise ParityFailure("could not reconcile terminal migration retry")
        terminal_after = {
            field: terminal_retry.get(field)
            for field in ("status", "bytes", "sha256", "source_deleted", "last_error")
        }
        if terminal_after != terminal_before:
            raise ParityFailure("terminal migration retry changed durable journal evidence")
        return {
            "session_id": session_id,
            "transfer_id": transfer_id,
            "initial_retry_status": retry.get("status"),
            "observed_statuses": observed,
            "destination_ack": destination_view["status"],
            "source_deleted": source_view["source_deleted"],
            "continued": continued,
            "terminal_retry_idempotent": True,
        }
    finally:
        delete_session(source, session_id)
        if destination_installed:
            delete_session(destination, session_id)


def compatible_props(source: Client, destination: Client) -> dict:
    left = source.require_json("/props")
    right = destination.require_json("/props")
    fields = (
        "model_ftype",
        "chat_template",
        "chat_template_caps",
        "modalities",
        "build_info",
        "total_slots",
    )
    for field in fields:
        if left.get(field) != right.get(field):
            raise ParityFailure(f"source/destination /props differs at {field}")
    left_ctx = left.get("default_generation_settings", {}).get("n_ctx")
    right_ctx = right.get("default_generation_settings", {}).get("n_ctx")
    if left_ctx != right_ctx:
        raise ParityFailure("source/destination context capacity differs")
    return {
        "build_info": left.get("build_info"),
        "slots": left.get("total_slots"),
        "context": left_ctx,
        "modalities": left.get("modalities"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--destination-url", required=True)
    parser.add_argument("--source-api-key-file", type=Path, required=True)
    parser.add_argument("--destination-api-key-file", type=Path, required=True)
    parser.add_argument("--source-ca-file", type=Path)
    parser.add_argument("--destination-ca-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate arguments and print the plan without contacting either server",
    )
    return parser.parse_args()


def https_origin(value: str, label: str) -> None:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ("127.0.0.1", "::1", "localhost")
        or parsed.port is None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ParityFailure(f"{label} must be a loopback HTTPS origin with explicit port")


def main() -> int:
    args = parse_args()
    results: list[dict] = []
    server: dict | None = None
    try:
        https_origin(args.source_url, "--source-url")
        https_origin(args.destination_url, "--destination-url")
        if args.source_url.rstrip("/") == args.destination_url.rstrip("/"):
            raise ParityFailure("source and destination origins must differ")
        source_key = private_key(args.source_api_key_file)
        destination_key = private_key(args.destination_api_key_file)
        if source_key != destination_key:
            raise ParityFailure(
                "decode migration forwards the source bearer; both servers must use the same API key"
            )
        for label, path in (
            ("--source-ca-file", args.source_ca_file),
            ("--destination-ca-file", args.destination_ca_file),
        ):
            if path is not None and not path.is_file():
                raise ParityFailure(f"{label} must be a regular file")
        source = Client(
            args.source_url, source_key, args.timeout, args.source_ca_file
        )
        destination = Client(
            args.destination_url,
            destination_key,
            args.timeout,
            args.destination_ca_file,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "schema": "muser.human-decode-migration-smoke.v1",
                        "status": "planned",
                        "seal_eligible": False,
                        "source": source.base_url,
                        "destination": destination.base_url,
                        "checks": [
                            "identical-server-contract",
                            "copy-retains-source",
                            "copy-exact-continuation",
                            "move-deletes-after-durable-ack",
                            "move-exact-continuation",
                            "ambiguous-response-retry",
                            "terminal-retry-idempotency",
                        ],
                        "operator_requirement": (
                            "destination certificate must be trusted by the source "
                            "server's platform trust store"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        server = compatible_props(source, destination)
        copy_started = time.monotonic()
        copy_evidence, continuation = check_copy(
            source, destination, args.timeout
        )
        results.append(
            {
                "name": "copy",
                "status": "passed",
                "elapsed_seconds": time.monotonic() - copy_started,
                "evidence": copy_evidence,
            }
        )
        print(f"PASS copy: {json.dumps(copy_evidence, sort_keys=True)}")
        move_started = time.monotonic()
        move_evidence = check_move_ambiguous_retry(
            source, destination, continuation, args.timeout
        )
        results.append(
            {
                "name": "move-ambiguous-retry",
                "status": "passed",
                "elapsed_seconds": time.monotonic() - move_started,
                "evidence": move_evidence,
            }
        )
        print(
            "PASS move-ambiguous-retry: "
            + json.dumps(move_evidence, sort_keys=True)
        )
        report = {
            "schema": "muser.human-decode-migration-smoke.v1",
            "status": "passed",
            "seal_eligible": False,
            "source": source.base_url,
            "destination": destination.base_url,
            "server": server,
            "checks": results,
        }
        atomic_json(args.output, report)
        print(f"report: {args.output}")
        return 0
    except Exception as error:
        report = {
            "schema": "muser.human-decode-migration-smoke.v1",
            "status": "failed",
            "seal_eligible": False,
            "source": args.source_url,
            "destination": args.destination_url,
            "server": server,
            "checks": results,
            "error": str(error),
        }
        atomic_json(args.output, report)
        print(f"FAIL migration smoke: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
