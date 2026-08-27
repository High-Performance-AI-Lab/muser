#!/usr/bin/env python3
"""Build the narrow Muser llama.cpp GX10 exporter and emit its receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts" / "gx10" / "llamacpp"
CUDA_IMAGE = (
    "nvcr.io/nvidia/cuda@"
    "sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92"
)
FILES = (
    "Dockerfile",
    "spark_kv_export.cpp",
    "muser_streaming_kv.patch",
    "muser_logical_swa.patch",
    "muser_cuda_metal_compat.patch",
    "muser_dflash_rope_nco.patch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--llama-dir", default="llama.cpp")
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--image-tag", default="muser-gx10-prefill:0.1.0-beta.1")
    parser.add_argument(
        "--cuda-matmul",
        choices=("default", "force-cublas", "force-mmq"),
        default="default",
        help="Compile-time llama.cpp CUDA quantized-matmul policy",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*command: str, input: str | None = None, merge_stderr: bool = False) -> str:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            input=input,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        stdout = (error.stdout or "").strip()
        stderr = (error.stderr or "").strip()
        detail = "\n".join(value for value in (stdout, stderr) if value)
        raise RuntimeError(
            f"command {command!r} failed with exit {error.returncode}"
            + (f":\n{detail}" if detail else "")
        ) from error


def validate(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.host):
        raise ValueError("--host must be a plain SSH host alias")
    if (
        not re.fullmatch(r"[A-Za-z0-9_./-]+", args.llama_dir)
        or args.llama_dir.startswith("/")
        or ".." in Path(args.llama_dir).parts
    ):
        raise ValueError("--llama-dir must be a safe path relative to remote HOME")
    if not re.fullmatch(r"[0-9a-f]{40}", args.llama_revision):
        raise ValueError("--llama-revision must be an exact lowercase 40-hex commit")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+", args.image_tag):
        raise ValueError("--image-tag is outside the closed local-tag grammar")
    for name in FILES:
        path = ADAPTER / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe build input: {path}")


def adapter_digest(build_input_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256(b"muser-gx10-adapter-v1\0")
    for name in FILES:
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(build_input_hashes[name]))
    return digest.hexdigest()


def write_receipt_atomic(path: Path, receipt: dict[str, object]) -> None:
    """Durably expose either one complete receipt or no receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses to replace an existing
        # receipt, preserving the builder's append-only evidence contract.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def ssh(host: str, script: str, *arguments: str, merge_stderr: bool = False) -> str:
    return run(
        "ssh", "-o", "BatchMode=yes", host, "bash", "-s", "--", *arguments,
        input=script, merge_stderr=merge_stderr,
    )


def main() -> int:
    args = parse_args()
    validate(args)
    build_input_hashes = {name: sha256(ADAPTER / name) for name in FILES}
    digest = adapter_digest(build_input_hashes)
    plan = {
        "schema": "muser.gx10-container.plan.v1",
        "mode": "dry-run" if args.dry_run else "build",
        "host": args.host,
        "llama_dir": args.llama_dir,
        "llama_revision": args.llama_revision,
        "image_tag": args.image_tag,
        "cuda_image": CUDA_IMAGE,
        "cuda_matmul": args.cuda_matmul,
        "adapter_sha256": digest,
        "output": str(args.output),
        "gpu_requested": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"refusing to replace receipt: {args.output}")
    actual = ssh(
        args.host,
        'git -C "$HOME/$1" rev-parse HEAD\n',
        args.llama_dir,
    )
    if actual != args.llama_revision:
        raise RuntimeError(f"remote llama.cpp is {actual}, expected {args.llama_revision}")
    source_tree = ssh(
        args.host,
        'git -C "$HOME/$1" rev-parse HEAD^{tree}\n',
        args.llama_dir,
    )
    archive_sha256 = ssh(
        args.host,
        'git -C "$HOME/$1" archive --format=tar HEAD | sha256sum | cut -d" " -f1\n',
        args.llama_dir,
    )
    context = ssh(
        args.host,
        """set -eu
context=$(mktemp -d /tmp/muser-gx10-container.XXXXXX)
mkdir -p "$context/llama" "$context/adapter"
git -C "$HOME/$1" archive --format=tar HEAD | tar -x -C "$context/llama"
printf '%s\\n' "$context"
""",
        args.llama_dir,
    )
    if not re.fullmatch(r"/tmp/muser-gx10-container\.[A-Za-z0-9]+", context):
        raise RuntimeError(f"remote returned unsafe context path {context!r}")
    try:
        for name in FILES:
            if sha256(ADAPTER / name) != build_input_hashes[name]:
                raise RuntimeError(f"build input changed after planning: {name}")
            run(
                "scp", "-q", "-o", "BatchMode=yes",
                str(ADAPTER / name), f"{args.host}:{context}/adapter/{name}",
            )
        build_output = ssh(
            args.host,
            """set -eu
docker build \
  --build-arg "CUDA_IMAGE=$2" \
  --build-arg "LLAMA_COMMIT=$3" \
  --build-arg "ADAPTER_DIGEST=$4" \
  --build-arg "CUDA_MATMUL=$5" \
  --tag "$6" \
  --file "$1/adapter/Dockerfile" "$1"
""",
            context,
            CUDA_IMAGE,
            args.llama_revision,
            digest,
            args.cuda_matmul,
            args.image_tag,
            merge_stderr=True,
        )
        inspect = json.loads(
            ssh(args.host, 'docker image inspect "$1"\n', args.image_tag)
        )[0]
    finally:
        ssh(
            args.host,
            'case "$1" in /tmp/muser-gx10-container.*) rm -rf -- "$1" ;; *) exit 2 ;; esac\n',
            context,
        )
    image_id = inspect.get("Id")
    architecture = inspect.get("Architecture")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise RuntimeError("built container has no content-addressed image ID")
    if architecture != "arm64":
        raise RuntimeError(f"built container architecture is {architecture!r}, expected arm64")
    labels = (inspect.get("Config") or {}).get("Labels") or {}
    if (
        labels.get("org.opencontainers.image.revision") != args.llama_revision
        or labels.get("io.muser.adapter.sha256") != digest
        or labels.get("io.muser.cuda.matmul") != args.cuda_matmul
    ):
        raise RuntimeError("built container labels do not match its source inputs")
    receipt = {
        "schema": "muser.gx10-container.receipt.v1",
        "status": "built",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": args.llama_revision,
        "source_tree": source_tree,
        "source_archive_sha256": archive_sha256,
        "adapter_sha256": digest,
        # These hashes are captured before any remote work.  A long CUDA build
        # must not accidentally attest newer local files if another campaign
        # lane edits an adapter input while the build is running.
        "build_inputs": build_input_hashes,
        "cuda_image": CUDA_IMAGE,
        "cuda_matmul": args.cuda_matmul,
        "image_tag": args.image_tag,
        "image_id": image_id,
        "architecture": architecture,
        "image_bytes": inspect.get("Size"),
        "entrypoint": (inspect.get("Config") or {}).get("Entrypoint"),
        "gpu_used_during_build": False,
        "build_output_sha256": hashlib.sha256(build_output.encode()).hexdigest(),
        "executed": False,
    }
    write_receipt_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
