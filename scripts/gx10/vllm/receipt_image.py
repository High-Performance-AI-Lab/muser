#!/usr/bin/env python3
"""Create an exclusive reproducibility receipt for the pinned producer image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(
        args, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    files = [
        root / "scripts/gx10/vllm/Dockerfile",
        root / "scripts/gx10/vllm/resident_producer.py",
        root / "scripts/gx10/vllm/benchmark_native_prefill.py",
        root / "scripts/gx10/vllm/request_producer.py",
        root / "scripts/gx10/vllm/score_nvfp4_drift.py",
        root / "scripts/gx10/vllm/receipt_cache_identity.py",
        root / "scripts/gx10/vllm/capture_layer0_qkv.py",
        root / "scripts/gx10/vllm/probe_fp4_gemm.py",
        root / "scripts/gx10/vllm/test_exact_fp4_quant.py",
        root / "scripts/gx10/vllm/test_exact_fp4_mm.py",
        root / "scripts/gx10/vllm/test_exact_rms_norm.py",
        root / "scripts/gx10/vllm/test_exact_swiglu.py",
        root / "scripts/gx10/vllm/test_exact_attention.py",
        root / "scripts/gx10/vllm/test_exact_attention_real.py",
        root / "scripts/gx10/vllm/muser_vllm/__init__.py",
        root / "scripts/gx10/vllm/muser_vllm/connector.py",
        root / "scripts/gx10/vllm/muser_vllm/dflash_capture.py",
        root / "scripts/gx10/vllm/muser_vllm/exact_attention.py",
        root / "scripts/gx10/vllm/muser_vllm/exact_fp4_quant.py",
        root / "scripts/gx10/vllm/muser_vllm/exact_fp4_mm.py",
        root / "scripts/gx10/vllm/muser_vllm/exact_rms_norm.py",
        root / "scripts/gx10/vllm/muser_vllm/exact_rope.py",
        root / "scripts/gx10/vllm/muser_vllm/exact_swiglu.py",
        root / "scripts/gx10/vllm/muser_vllm/native_capture.py",
        root / "scripts/gx10/vllm/muser_vllm/packing.py",
        root / "scripts/gx10/vllm/muser_vllm/receipt.py",
        root / "scripts/gx10/vllm/muser_vllm/rope_cache.py",
        root / "scripts/gx10/llamacpp/muser_v2_send.py",
        root / "scripts/gx10/llamacpp/llamacpp_session_send.py",
        root / "scripts/gx10/llamacpp/protocol.py",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        parser.error(f"receipt sources are missing: {missing}")
    inspect = json.loads(command("docker", "image", "inspect", args.image))[0]
    runtime_script = (
        "import json,subprocess,torch,transformers,vllm;"
        "from muser_vllm.connector import MuserMuseHandoffConnector;"
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'transformers':transformers.__version__,'vllm':vllm.__version__,"
        "'connector':MuserMuseHandoffConnector.__name__,"
        "'pip_freeze':subprocess.check_output(['python3','-m','pip','freeze'],"
        "text=True).splitlines()},sort_keys=True))"
    )
    runtime_output = command(
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        args.image,
        "-c",
        runtime_script,
    )
    runtime = json.loads(runtime_output.splitlines()[-1])
    receipt = {
        "schema": "muser.spark-nvfp4-image-rebuild.v1",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "image": args.image,
        "image_id": inspect["Id"],
        "repo_digests": inspect.get("RepoDigests", []),
        "base_digest": "sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268",
        "vllm_commit": "6adad08767583f52eb4d2122111af0bf638ed5e6",
        "vllm_wheel_sha256": "230d876ef3d90718ce8b42bef3b24c4384a714d623db7a2eb9c27c59d138066c",
        "sources": {
            str(path.relative_to(root)): sha256_file(path) for path in files
        },
        "runtime": runtime,
    }
    output = Path(args.output)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
