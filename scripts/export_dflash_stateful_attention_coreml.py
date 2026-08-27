#!/usr/bin/env python3
"""Export one official DFlash attention layer with public CoreML KV state.

This is an experimental post-release ANE graph pilot. It follows the
stateful transformer construction proven in ``ane-book``: INT8 1x1
convolutions, scale-safe RMSNorm, explicit Q/K head norms and RoPE, matmul KV
writes into ``StateType``, grouped-query attention, and an output projection.
The fixed 16-token DFlash block avoids RangeDim in this first empirical gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COREMLTOOLS_VERSION = "9.0"
NUMPY_VERSION = "2.1.3"
TORCH_VERSION = "2.10.0"
MAX_SHARD_BYTES = 250 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dflash", required=True, type=Path)
    parser.add_argument(
        "--extractor",
        type=Path,
        default=ROOT / "target" / "release" / "muser-dflash-extract",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=384)
    parser.add_argument(
        "--query-size",
        type=int,
        choices=(4, 16),
        default=4,
        help="query/output rows per prediction; KV inputs and state writes remain block-16",
    )
    parser.add_argument(
        "--attention-query-chunk",
        type=int,
        default=4,
        help="static query rows per internal attention subgraph (must divide 16)",
    )
    parser.add_argument(
        "--kv-write-chunk",
        type=int,
        default=16,
        help="static target rows per internal state-write matmul (must divide 16)",
    )
    parser.add_argument(
        "--attention-op", choices=("manual", "sdpa"), default="manual"
    )
    parser.add_argument(
        "--kv-join",
        choices=("split", "concat"),
        default="split",
        help=(
            "manual-GQA score topology; split contracts MLState and noise KV "
            "separately before joining scores"
        ),
    )
    parser.add_argument(
        "--state-group-kv-heads",
        type=int,
        choices=(4, 8),
        default=8,
        help=(
            "KV heads per independent K/V MLState pair; four lowers ANEF "
            "intermediate pressure while preserving one prediction"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--torch-only",
        action="store_true",
        help="extract official weights and validate/trace the graph without CoreML",
    )
    parser.add_argument(
        "--mil-only",
        action="store_true",
        help="convert without loading or compiling and print the optimized MIL op counts",
    )
    parser.add_argument(
        "--runtime-bundle",
        action="store_true",
        help=(
            "emit the v9 runtime primitive: consume CPU-normalized noise rows and "
            "return attention plus exact target K/V for the authoritative shadow"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tree_receipt(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode()
        data = child.read_bytes()
        total += len(data)
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return total, digest.hexdigest()


def describe(extractor: Path, artifact: Path) -> dict:
    result = subprocess.run(
        [str(extractor), "--artifact", str(artifact), "--describe"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def extract(
    extractor: Path,
    artifact: Path,
    name: str,
    output: Path,
    rows: int,
    columns: int,
) -> None:
    result = subprocess.run(
        [
            str(extractor),
            "--artifact",
            str(artifact),
            "--tensor",
            name,
            "--output",
            str(output),
            "--raw-tensor",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    receipt = json.loads(result.stdout)
    expected = rows * columns * 4
    if (
        receipt.get("tensor") != name
        or receipt.get("rows") != rows
        or receipt.get("columns") != columns
        or receipt.get("bytes") != expected
        or output.stat().st_size != expected
    ):
        raise ValueError(f"invalid extractor receipt for {name}: {receipt}")


def load_toolchain():
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    import numpy as np
    import torch
    import torch.nn as nn

    if (
        ct.__version__ != COREMLTOOLS_VERSION
        or np.__version__ != NUMPY_VERSION
        or torch.__version__ != TORCH_VERSION
    ):
        raise RuntimeError(
            "stateful DFlash export requires "
            f"coremltools=={COREMLTOOLS_VERSION}, numpy=={NUMPY_VERSION}, "
            f"torch=={TORCH_VERSION}; got {ct.__version__}, {np.__version__}, "
            f"{torch.__version__}"
        )
    return ct, cto, np, torch, nn


def build_module(
    *,
    np,
    torch,
    nn,
    config: dict,
    weights: dict,
    max_context: int,
    query_size: int,
    attention_query_chunk: int,
    kv_write_chunk: int,
    attention_op: str,
    kv_join: str,
    state_group_kv_heads: int,
    runtime_bundle: bool,
):
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    block = int(config["block_size"])
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    groups = heads // kv_heads
    state_groups = kv_heads // state_group_kv_heads
    epsilon = float(config["rms_norm_eps"])

    def state_name(kind: str, group: int) -> str:
        return kind if state_groups == 1 else f"{kind}_{group}"

    class RmsNorm(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer(
                "weight",
                torch.tensor(weight, dtype=torch.float16).reshape(1, -1, 1, 1),
            )

        def forward(self, value):
            # ane-book's scale-invariant form prevents fp16 square overflow.
            scale = float(hidden) ** 0.5
            reduced = value * (1.0 / scale)
            variance = reduced.square().mean(dim=1, keepdim=True)
            return (
                reduced
                * torch.rsqrt(variance + epsilon / (scale * scale))
                * self.weight
            ).half()

    class HeadNorm(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer(
                "weight",
                torch.tensor(weight, dtype=torch.float16).reshape(1, 1, 1, head_dim),
            )

        def forward(self, value):
            scale = float(head_dim) ** 0.5
            reduced = value * (1.0 / scale)
            variance = reduced.square().mean(dim=-1, keepdim=True)
            return (
                reduced
                * torch.rsqrt(variance + 1.0e-6 / (scale * scale))
                * self.weight
            ).half()

    class DFlashAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = None if runtime_bundle else RmsNorm(weights["input_norm"])
            qkv = np.concatenate(
                [weights["q"], weights["k"], weights["v"]], axis=0
            )
            self.qkv = nn.Conv2d(hidden, q_width + 2 * kv_width, 1, bias=False)
            self.qkv.weight = nn.Parameter(
                torch.tensor(qkv, dtype=torch.float16).reshape(
                    q_width + 2 * kv_width, hidden, 1, 1
                ),
                requires_grad=False,
            )
            self.q_norm = HeadNorm(weights["q_norm"])
            self.k_norm = HeadNorm(weights["k_norm"])
            if not runtime_bundle:
                self.output = nn.Conv2d(q_width, hidden, 1, bias=False)
                self.output.weight = nn.Parameter(
                    torch.tensor(weights["output"], dtype=torch.float16).reshape(
                        hidden, q_width, 1, 1
                    ),
                    requires_grad=False,
                )
            for state_group in range(state_groups):
                self.register_buffer(
                    state_name("k_state", state_group),
                    torch.zeros(
                        1,
                        state_group_kv_heads,
                        max_context,
                        head_dim,
                        dtype=torch.float16,
                    ),
                )
                self.register_buffer(
                    state_name("v_state", state_group),
                    torch.zeros(
                        1,
                        state_group_kv_heads,
                        max_context,
                        head_dim,
                        dtype=torch.float16,
                    ),
                )

        @staticmethod
        def as_heads(value, n_heads):
            tokens = value.shape[2]
            value = value.squeeze(-1).permute(0, 2, 1)
            return value.reshape(1, tokens, n_heads, head_dim).permute(0, 2, 1, 3)

        @staticmethod
        def apply_rope(value, cosine, sine):
            # Headed [1,H,T,D]. Muse/DFlash normalizes Q/K before applying
            # the interleaved-half rotation; those operations do not commute.
            half = head_dim // 2
            low = value[..., :half]
            high = value[..., half:]
            cosine = cosine.reshape(1, 1, block, half)
            sine = sine.reshape(1, 1, block, half)
            rotated = torch.cat(
                [low * cosine - high * sine, high * cosine + low * sine], dim=-1
            )
            return rotated

        def forward(
            self,
            noise_hidden,
            query_selector,
            target_projected,
            target_mask,
            target_rope_cos,
            target_rope_sin,
            noise_rope_cos,
            noise_rope_sin,
            replay_target_key,
            replay_target_value,
            replay_mode,
            attention_mask,
            kv_write_mask,
        ):
            residual = torch.matmul(
                query_selector,
                noise_hidden.squeeze(-1).permute(0, 2, 1).unsqueeze(1),
            ).squeeze(1).permute(0, 2, 1).unsqueeze(-1)
            noise = noise_hidden if runtime_bundle else self.input_norm(noise_hidden)
            combined = torch.cat([noise, target_projected], dim=2)
            projected = self.qkv(combined)
            q_raw = projected[:, :q_width, :block, :]
            k_raw = projected[:, q_width:q_width + kv_width, :, :]
            v_raw = projected[:, q_width + kv_width:, :, :]
            k_noise = k_raw[:, :, :block, :]
            v_noise = v_raw[:, :, :block, :]
            k_target = k_raw[:, :, block:, :] * target_mask
            v_target = v_raw[:, :, block:, :] * target_mask

            q = self.apply_rope(
                self.q_norm(self.as_heads(q_raw, heads)),
                noise_rope_cos,
                noise_rope_sin,
            )
            q = torch.matmul(query_selector, q)
            k_noise = self.apply_rope(
                self.k_norm(self.as_heads(k_noise, kv_heads)),
                noise_rope_cos,
                noise_rope_sin,
            )
            k_target = self.apply_rope(
                self.k_norm(self.as_heads(k_target, kv_heads)),
                target_rope_cos,
                target_rope_sin,
            ) * target_mask.reshape(1, 1, block, 1)

            v_noise = v_noise.squeeze(-1).permute(0, 2, 1)
            v_noise = v_noise.reshape(1, block, kv_heads, head_dim).permute(0, 2, 1, 3)
            v_target = v_target.squeeze(-1).permute(0, 2, 1)
            v_target = v_target.reshape(1, block, kv_heads, head_dim).permute(0, 2, 1, 3)
            if runtime_bundle:
                replay_key = replay_target_key.squeeze(-1).permute(0, 2, 1)
                replay_key = replay_key.reshape(1, block, kv_heads, head_dim).permute(0, 2, 1, 3)
                replay_value = replay_target_value.squeeze(-1).permute(0, 2, 1)
                replay_value = replay_value.reshape(1, block, kv_heads, head_dim).permute(0, 2, 1, 3)
                k_target = k_target * (1.0 - replay_mode) + replay_key * replay_mode
                v_target = v_target * (1.0 - replay_mode) + replay_value * replay_mode

            write_any = kv_write_mask.sum(dim=-1, keepdim=True)
            keep = 1.0 - write_any
            state_group_outputs = []
            for state_group in range(state_groups):
                kv_start = state_group * state_group_kv_heads
                kv_end = kv_start + state_group_kv_heads
                query_start_head = kv_start * groups
                query_end_head = kv_end * groups
                group_k_target = k_target[:, kv_start:kv_end, :, :]
                group_v_target = v_target[:, kv_start:kv_end, :, :]
                group_k_noise = k_noise[:, kv_start:kv_end, :, :]
                group_v_noise = v_noise[:, kv_start:kv_end, :, :]
                group_q = q[:, query_start_head:query_end_head, :, :]
                k_state = getattr(self, state_name("k_state", state_group))
                v_state = getattr(self, state_name("v_state", state_group))

                if kv_write_chunk == block:
                    k_written = torch.matmul(kv_write_mask, group_k_target)
                    v_written = torch.matmul(kv_write_mask, group_v_target)
                else:
                    k_written_parts = []
                    v_written_parts = []
                    for write_start in range(0, block, kv_write_chunk):
                        write_end = write_start + kv_write_chunk
                        write_mask = kv_write_mask[:, :, :, write_start:write_end]
                        k_written_parts.append(
                            torch.matmul(
                                write_mask,
                                group_k_target[:, :, write_start:write_end, :],
                            )
                        )
                        v_written_parts.append(
                            torch.matmul(
                                write_mask,
                                group_v_target[:, :, write_start:write_end, :],
                            )
                        )
                    k_written = torch.stack(k_written_parts, dim=0).sum(dim=0)
                    v_written = torch.stack(v_written_parts, dim=0).sum(dim=0)
                k_updated = k_state * keep + k_written
                v_updated = v_state * keep + v_written
                k_state[:] = k_updated
                v_state[:] = v_updated

                if attention_op == "sdpa" or kv_join == "concat":
                    k_all = torch.cat([k_updated, group_k_noise], dim=2)
                    v_all = torch.cat([v_updated, group_v_noise], dim=2)
                query_outputs = []
                for query_start in range(0, query_size, attention_query_chunk):
                    query_end = query_start + attention_query_chunk
                    if attention_op == "sdpa":
                        k_heads = torch.repeat_interleave(k_all, groups, dim=1)
                        v_heads = torch.repeat_interleave(v_all, groups, dim=1)
                        query_outputs.append(
                            torch.nn.functional.scaled_dot_product_attention(
                                group_q[:, :, query_start:query_end, :],
                                k_heads,
                                v_heads,
                                attn_mask=attention_mask[
                                    :, :, query_start:query_end, :
                                ],
                                dropout_p=0.0,
                                is_causal=False,
                            )
                        )
                    else:
                        head_parts = []
                        scale = head_dim ** -0.5
                        for local_kv_head in range(state_group_kv_heads):
                            q_group = group_q[
                                :,
                                local_kv_head * groups:(local_kv_head + 1) * groups,
                                query_start:query_end,
                                :,
                            ]
                            if kv_join == "concat":
                                score = torch.matmul(
                                    q_group,
                                    k_all[
                                        :, local_kv_head:local_kv_head + 1, :, :
                                    ].transpose(-2, -1),
                                )
                            else:
                                context_score = torch.matmul(
                                    q_group,
                                    k_updated[
                                        :, local_kv_head:local_kv_head + 1, :, :
                                    ].transpose(-2, -1),
                                )
                                noise_score = torch.matmul(
                                    q_group,
                                    group_k_noise[
                                        :, local_kv_head:local_kv_head + 1, :, :
                                    ].transpose(-2, -1),
                                )
                                score = torch.cat(
                                    [context_score, noise_score], dim=-1
                                )
                            score = score * scale + attention_mask[
                                :, :, query_start:query_end, :
                            ]
                            probability = torch.softmax(score.float(), dim=-1).half()
                            if kv_join == "concat":
                                head_parts.append(
                                    torch.matmul(
                                        probability,
                                        v_all[
                                            :, local_kv_head:local_kv_head + 1, :, :
                                        ],
                                    )
                                )
                            else:
                                context_probability = probability[..., :max_context]
                                noise_probability = probability[..., max_context:]
                                head_parts.append(
                                    torch.matmul(
                                        context_probability,
                                        v_updated[
                                            :, local_kv_head:local_kv_head + 1, :, :
                                        ],
                                    )
                                    + torch.matmul(
                                        noise_probability,
                                        group_v_noise[
                                            :, local_kv_head:local_kv_head + 1, :, :
                                        ],
                                    )
                                )
                        query_outputs.append(torch.cat(head_parts, dim=1))
                state_group_outputs.append(torch.cat(query_outputs, dim=2))
            attention = torch.cat(state_group_outputs, dim=1)
            attention = attention.permute(0, 1, 3, 2).reshape(
                1, q_width, query_size, 1
            )
            if runtime_bundle:
                k_bundle = k_target.permute(0, 1, 3, 2).reshape(
                    1, kv_width, block, 1
                )
                v_bundle = v_target.permute(0, 1, 3, 2).reshape(
                    1, kv_width, block, 1
                )
                return torch.cat([attention, k_bundle, v_bundle], dim=1)
            return residual + self.output(attention)

    return DFlashAttention().half().eval()


def main() -> int:
    args = parse_args()
    if args.attention_op == "sdpa" and args.kv_join != "concat":
        raise ValueError("SDPA requires --kv-join concat")
    if sum((args.dry_run, args.torch_only, args.mil_only)) > 1:
        raise ValueError("--dry-run, --torch-only, and --mil-only are mutually exclusive")
    if not args.dflash.is_file() or not args.extractor.is_file():
        raise ValueError("official DFlash GGUF and extractor must exist")
    config = describe(args.extractor, args.dflash)
    if int(config["num_hidden_layers"]) != 5 or int(config["block_size"]) != 16:
        raise ValueError("release DFlash geometry must be five layers with block size 16")
    kv_heads = int(config["num_key_value_heads"])
    if (
        not 0 <= args.layer < 5
        or (args.runtime_bundle and args.query_size != 16)
        or args.max_context < 32
        or args.query_size % args.attention_query_chunk != 0
        or args.attention_query_chunk not in (1, 2, 4, 8, 16)
        or args.kv_write_chunk not in (1, 2, 4, 8, 16)
        or 16 % args.kv_write_chunk != 0
        or kv_heads % args.state_group_kv_heads != 0
    ):
        raise ValueError("invalid layer or max context")
    state_groups = kv_heads // args.state_group_kv_heads
    state_names = [
        (kind if state_groups == 1 else f"{kind}_{state_group}")
        for state_group in range(state_groups)
        for kind in ("k_state", "v_state")
    ]
    plan = {
        "schema": "muser.dflash-stateful-attention-export.v1",
        "mode": (
            "dry-run"
            if args.dry_run
            else "torch-only"
            if args.torch_only
            else "mil-only"
            if args.mil_only
            else "export"
        ),
        "layer": args.layer,
        "block_size": 16,
        "query_size": args.query_size,
        "max_context": args.max_context,
        "attention_query_chunk": args.attention_query_chunk,
        "kv_write_chunk": args.kv_write_chunk,
        "attention_op": args.attention_op,
        "kv_join": args.kv_join,
        "state_group_kv_heads": args.state_group_kv_heads,
        "state": state_names,
        "state_shape": [1, args.state_group_kv_heads, args.max_context, 128],
        "inputs": [
            "noise_hidden", "query_selector", "target_projected", "target_mask",
            "target_rope_cos", "target_rope_sin", "noise_rope_cos",
            "noise_rope_sin", "replay_target_key", "replay_target_value",
            "replay_mode", "attention_mask", "kv_write_mask",
        ],
        "output": (
            "attention_target_kv_bundle"
            if args.runtime_bundle else "post_attention_hidden"
        ),
        "runtime_bundle": args.runtime_bundle,
        "compute_units": "CPU_AND_NE",
        "weight_dtype": "int8",
        "dflash_sha256": sha256(args.dflash),
        "toolchain": {
            "coremltools": COREMLTOOLS_VERSION,
            "numpy": NUMPY_VERSION,
            "torch": TORCH_VERSION,
        },
        "output_created": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"output must be absent: {args.output}")

    ct, cto, np, torch, nn = load_toolchain()
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    block = int(config["block_size"])
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim

    extraction = Path(tempfile.mkdtemp(prefix="muser-dflash-attention-f32-"))
    stage = Path(tempfile.mkdtemp(prefix=".muser-dflash-stateful-", dir=args.output.parent))
    try:
        specs = {
            "q_norm": (f"blk.{args.layer}.attn_q_norm.weight", 1, head_dim),
            "k_norm": (f"blk.{args.layer}.attn_k_norm.weight", 1, head_dim),
            "q": (f"blk.{args.layer}.attn_q.weight", q_width, hidden),
            "k": (f"blk.{args.layer}.attn_k.weight", kv_width, hidden),
            "v": (f"blk.{args.layer}.attn_v.weight", kv_width, hidden),
        }
        if not args.runtime_bundle:
            specs.update(
                {
                    "input_norm": (f"blk.{args.layer}.attn_norm.weight", 1, hidden),
                    "output": (
                        f"blk.{args.layer}.attn_output.weight",
                        hidden,
                        q_width,
                    ),
                }
            )
        weights = {}
        for key, (name, rows, columns) in specs.items():
            raw = extraction / f"{key}.f32"
            extract(args.extractor, args.dflash, name, raw, rows, columns)
            weights[key] = np.memmap(raw, mode="r", dtype="<f4", shape=(rows, columns))

        model = build_module(
            np=np,
            torch=torch,
            nn=nn,
            config=config,
            weights=weights,
            max_context=args.max_context,
            query_size=args.query_size,
            attention_query_chunk=args.attention_query_chunk,
            kv_write_chunk=args.kv_write_chunk,
            attention_op=args.attention_op,
            kv_join=args.kv_join,
            state_group_kv_heads=args.state_group_kv_heads,
            runtime_bundle=args.runtime_bundle,
        )
        half = head_dim // 2
        examples = (
            torch.randn(1, hidden, block, 1, dtype=torch.float16),
            torch.eye(block, dtype=torch.float16)[
                :args.query_size, :
            ].reshape(1, 1, args.query_size, block),
            torch.randn(1, hidden, block, 1, dtype=torch.float16),
            torch.ones(1, 1, block, 1, dtype=torch.float16),
            torch.randn(block, half, dtype=torch.float16),
            torch.randn(block, half, dtype=torch.float16),
            torch.randn(block, half, dtype=torch.float16),
            torch.randn(block, half, dtype=torch.float16),
            torch.zeros(1, kv_width, block, 1, dtype=torch.float16),
            torch.zeros(1, kv_width, block, 1, dtype=torch.float16),
            torch.zeros(1, 1, 1, 1, dtype=torch.float16),
            torch.zeros(
                1,
                1,
                args.query_size,
                args.max_context + block,
                dtype=torch.float16,
            ),
            torch.zeros(1, 1, args.max_context, block, dtype=torch.float16),
        )
        with torch.no_grad():
            torch_output = model(*examples)
        if (
            tuple(torch_output.shape)
            != (
                (1, q_width + 2 * kv_width, block, 1)
                if args.runtime_bundle
                else (1, hidden, args.query_size, 1)
            )
            or not torch.isfinite(torch_output).all()
        ):
            raise ValueError("Torch stateful attention pilot produced invalid output")
        grouped_parity = None
        if args.state_group_kv_heads < kv_heads:
            reference = build_module(
                np=np,
                torch=torch,
                nn=nn,
                config=config,
                weights=weights,
                max_context=args.max_context,
                query_size=args.query_size,
                attention_query_chunk=args.attention_query_chunk,
                kv_write_chunk=args.kv_write_chunk,
                attention_op=args.attention_op,
                kv_join=args.kv_join,
                state_group_kv_heads=kv_heads,
                runtime_bundle=args.runtime_bundle,
            )
            parity_write_mask = examples[-1].clone()
            for token in range(block):
                parity_write_mask[0, 0, token, token] = 1.0
            parity_examples = (*examples[:-1], parity_write_mask)
            with torch.no_grad():
                reference_k = torch.randn_like(reference.k_state)
                reference_v = torch.randn_like(reference.v_state)
                reference.k_state.copy_(reference_k)
                reference.v_state.copy_(reference_v)
                for state_group in range(state_groups):
                    start = state_group * args.state_group_kv_heads
                    end = start + args.state_group_kv_heads
                    getattr(model, f"k_state_{state_group}").copy_(
                        reference_k[:, start:end, :, :]
                    )
                    getattr(model, f"v_state_{state_group}").copy_(
                        reference_v[:, start:end, :, :]
                    )
                grouped_output = model(*parity_examples)
                reference_output = reference(*parity_examples)
                grouped_k = torch.cat(
                    [
                        getattr(model, f"k_state_{group}")
                        for group in range(state_groups)
                    ],
                    dim=1,
                )
                grouped_v = torch.cat(
                    [
                        getattr(model, f"v_state_{group}")
                        for group in range(state_groups)
                    ],
                    dim=1,
                )
                output_max_abs = float(
                    (grouped_output.float() - reference_output.float()).abs().max()
                )
                state_max_abs = max(
                    float(
                        (grouped_k.float() - reference.k_state.float()).abs().max()
                    ),
                    float(
                        (grouped_v.float() - reference.v_state.float()).abs().max()
                    ),
                )
            if output_max_abs != 0.0 or state_max_abs != 0.0:
                raise ValueError(
                    "grouped weighted MLState is not bit-exact with the full-KV-"
                    f"head reference: output_max_abs={output_max_abs}, "
                    f"state_max_abs={state_max_abs}"
                )
            grouped_parity = {
                "reference_state_group_kv_heads": kv_heads,
                "output_max_abs": output_max_abs,
                "state_max_abs": state_max_abs,
            }
        query_chunk_parity = None
        if args.query_size < block:
            full_query_reference = build_module(
                np=np,
                torch=torch,
                nn=nn,
                config=config,
                weights=weights,
                max_context=args.max_context,
                query_size=block,
                attention_query_chunk=args.attention_query_chunk,
                kv_write_chunk=args.kv_write_chunk,
                attention_op=args.attention_op,
                kv_join=args.kv_join,
                state_group_kv_heads=args.state_group_kv_heads,
                runtime_bundle=args.runtime_bundle,
            )
            full_selector = torch.eye(block, dtype=torch.float16).reshape(
                1, 1, block, block
            )
            full_attention_mask = torch.zeros(
                1,
                1,
                block,
                args.max_context + block,
                dtype=torch.float16,
            )
            full_write_mask = examples[-1].clone()
            for token in range(block):
                full_write_mask[0, 0, token, token] = 1.0
            full_examples = (
                examples[0],
                full_selector,
                *examples[2:11],
                full_attention_mask,
                full_write_mask,
            )
            with torch.no_grad():
                for state_name in state_names:
                    initial = torch.randn_like(getattr(model, state_name))
                    getattr(model, state_name).copy_(initial)
                    getattr(full_query_reference, state_name).copy_(initial)
                full_output = full_query_reference(*full_examples)
                no_write = torch.zeros_like(full_write_mask)
                chunk_outputs = []
                for start in range(0, block, args.query_size):
                    selector = full_selector[:, :, start:start + args.query_size, :]
                    chunk_outputs.append(
                        model(
                            examples[0],
                            selector,
                            *examples[2:11],
                            full_attention_mask[
                                :, :, start:start + args.query_size, :
                            ],
                            full_write_mask if start == 0 else no_write,
                        )
                    )
                chunk_output = torch.cat(chunk_outputs, dim=2)
                output_max_abs = float(
                    (chunk_output.float() - full_output.float()).abs().max()
                )
                state_max_abs = max(
                    float(
                        (
                            getattr(model, state_name).float()
                            - getattr(full_query_reference, state_name).float()
                        )
                        .abs()
                        .max()
                    )
                    for state_name in state_names
                )
            if output_max_abs != 0.0 or state_max_abs != 0.0:
                raise ValueError(
                    "four weighted T=4 predictions are not bit-exact with one "
                    f"T=16 prediction: output_max_abs={output_max_abs}, "
                    f"state_max_abs={state_max_abs}"
                )
            query_chunk_parity = {
                "calls": block // args.query_size,
                "reference_query_size": block,
                "output_max_abs": output_max_abs,
                "state_max_abs": state_max_abs,
            }
        with torch.no_grad():
            for state_name in state_names:
                getattr(model, state_name).zero_()
        traced = torch.jit.trace(model, examples)
        if args.torch_only:
            print(
                json.dumps(
                    {
                        **plan,
                        "torch_output_shape": list(torch_output.shape),
                        "torch_output_finite": True,
                        "grouped_state_parity": grouped_parity,
                        "query_chunk_parity": query_chunk_parity,
                        "trace_nodes": sum(1 for _ in traced.graph.nodes()),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        names = plan["inputs"]
        ct_inputs = [
            ct.TensorType(name=names[0], shape=(1, hidden, block, 1), dtype=np.float16),
            ct.TensorType(
                name=names[1],
                shape=(1, 1, args.query_size, block),
                dtype=np.float16,
            ),
            ct.TensorType(name=names[2], shape=(1, hidden, block, 1), dtype=np.float16),
            ct.TensorType(name=names[3], shape=(1, 1, block, 1), dtype=np.float16),
            ct.TensorType(name=names[4], shape=(block, half), dtype=np.float16),
            ct.TensorType(name=names[5], shape=(block, half), dtype=np.float16),
            ct.TensorType(name=names[6], shape=(block, half), dtype=np.float16),
            ct.TensorType(name=names[7], shape=(block, half), dtype=np.float16),
            ct.TensorType(
                name=names[8],
                shape=(1, kv_width, block, 1),
                dtype=np.float16,
            ),
            ct.TensorType(
                name=names[9],
                shape=(1, kv_width, block, 1),
                dtype=np.float16,
            ),
            ct.TensorType(
                name=names[10],
                shape=(1, 1, 1, 1),
                dtype=np.float16,
            ),
            ct.TensorType(
                name=names[11],
                shape=(1, 1, args.query_size, args.max_context + block),
                dtype=np.float16,
            ),
            ct.TensorType(
                name=names[12],
                shape=(1, 1, args.max_context, block),
                dtype=np.float16,
            ),
        ]
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(
                        1,
                        args.state_group_kv_heads,
                        args.max_context,
                        head_dim,
                    ),
                    dtype=np.float16,
                ),
                name=state_name,
            )
            for state_name in state_names
        ]
        converted = ct.convert(
            traced,
            inputs=ct_inputs,
            outputs=[ct.TensorType(name=plan["output"], dtype=np.float16)],
            states=states,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            # Loading here would ask ANEF to compile the unquantized 82 MiB
            # fused QKV convolution before the INT8 rewrite. That intermediate
            # graph is not the shipped artifact and exceeds the single-op
            # compiler envelope. The guarded MLComputePlan qualification below
            # loads only the final quantized package.
            skip_model_load=True,
        )
        if args.mil_only:
            operation_counts = {}
            for operation in converted._mil_program.functions["main"].operations:
                operation_counts[operation.op_type] = (
                    operation_counts.get(operation.op_type, 0) + 1
                )
            print(
                json.dumps(
                    {
                        **plan,
                        "mil_operation_counts": operation_counts,
                        "mil_operation_total": sum(operation_counts.values()),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        quantizer = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_tensor"
        )
        converted = cto.linear_quantize_weights(
            converted, cto.OptimizationConfig(global_config=quantizer)
        )
        package = stage / "dflash-attention.mlpackage"
        converted.save(str(package))
        size, digest = tree_receipt(package)
        if not 0 < size <= MAX_SHARD_BYTES:
            raise ValueError(f"stateful attention package is {size} bytes")
        compile_result = subprocess.run(
            [
                "xcrun", "coremlcompiler", "compile", str(package), str(stage),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        compiled = stage / "dflash-attention.mlmodelc"
        if not compiled.is_dir():
            raise ValueError(f"CoreML compiler did not create {compiled}")
        plan.update(
            {
                "mode": "exported",
                "package": package.name,
                "package_bytes": size,
                "package_sha256": digest,
                "compiled": compiled.name,
                "output_created": True,
            }
        )
        (stage / "manifest.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(args.output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(extraction, ignore_errors=True)
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
