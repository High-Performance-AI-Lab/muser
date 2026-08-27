#!/usr/bin/env python3
"""Export the weight-free stateful attention core of one DFlash layer.

This compiler-isolation artifact preserves the exact 16-token DFlash KV block.
The state update uses ``ane-book``'s single mask×KV contraction; query attention
may remain in T=4 slices. Q/K are already head-normalized and RoPE-rotated by
the caller. One prediction owns the complete block and all grouped public
``MLState`` buffers. A four-row query program is invoked four times against
the same state while retaining all 16 external noise rows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from export_dflash_stateful_attention_coreml import (
    COREMLTOOLS_VERSION,
    NUMPY_VERSION,
    TORCH_VERSION,
    describe,
    load_toolchain,
    sha256,
    tree_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dflash", required=True, type=Path)
    parser.add_argument(
        "--extractor",
        type=Path,
        default=ROOT / "target" / "release" / "muser-dflash-extract",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--block-size",
        type=int,
        choices=(4, 16),
        default=16,
        help="target/noise KV input and state-write width; release runtime is 16",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        choices=(4, 16),
        default=16,
        help=(
            "query rows per public prediction; four matches ane-book's proven "
            "stateful verifier surface while KV inputs/writes remain block-16"
        ),
    )
    parser.add_argument("--max-context", type=int, default=384)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument(
        "--kv-write-chunk",
        type=int,
        choices=(1, 2, 4, 16),
        default=16,
        help=(
            "state-write contraction width; 16 is the ane-book single-"
            "contraction topology and avoids cache-sized stack intermediates"
        ),
    )
    parser.add_argument(
        "--attention-op", choices=("manual", "sdpa"), default="manual"
    )
    parser.add_argument(
        "--kv-join",
        choices=("split", "concat"),
        default="split",
        help=(
            "manual-GQA score topology; split avoids concatenating MLState "
            "with the 16 noise rows before contraction"
        ),
    )
    parser.add_argument(
        "--state-group-kv-heads",
        type=int,
        choices=(4, 8),
        default=4,
        help=(
            "KV heads per independent K/V MLState pair; four halves peak "
            "intermediate pressure while preserving one model prediction"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--torch-only", action="store_true")
    parser.add_argument("--mil-only", action="store_true")
    parser.add_argument(
        "--mil-dump",
        type=Path,
        help="optional textual MIL destination; valid only with --mil-only",
    )
    return parser.parse_args()


def build_module(
    *,
    torch,
    nn,
    config: dict,
    max_context: int,
    block_size: int,
    query_size: int,
    chunk: int,
    kv_write_chunk: int,
    attention_op: str,
    kv_join: str,
    state_group_kv_heads: int,
):
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    block = block_size
    groups = heads // kv_heads
    state_groups = kv_heads // state_group_kv_heads

    def state_name(kind: str, group: int) -> str:
        return kind if state_groups == 1 else f"{kind}_{group}"

    class StatefulAttention(nn.Module):
        def __init__(self):
            super().__init__()
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

        def forward(
            self,
            query,
            noise_key,
            noise_value,
            target_key,
            target_value,
            attention_mask,
            kv_write_mask,
        ):
            write_any = kv_write_mask.sum(dim=-1, keepdim=True)
            keep = 1.0 - write_any
            state_group_outputs = []
            for state_group in range(state_groups):
                kv_start = state_group * state_group_kv_heads
                kv_end = kv_start + state_group_kv_heads
                query_start = kv_start * groups
                query_end = kv_end * groups
                group_target_key = target_key[:, kv_start:kv_end, :, :]
                group_target_value = target_value[:, kv_start:kv_end, :, :]
                group_noise_key = noise_key[:, kv_start:kv_end, :, :]
                group_noise_value = noise_value[:, kv_start:kv_end, :, :]
                group_query = query[:, query_start:query_end, :, :]
                k_state = getattr(self, state_name("k_state", state_group))
                v_state = getattr(self, state_name("v_state", state_group))

                if kv_write_chunk == block:
                    # Match ane-book's single mask×KV state contraction. Each
                    # independent state group completes before the next one,
                    # bounding ANEF intermediate liveness without another
                    # CoreML prediction or any host-visible state transfer.
                    written_key = torch.matmul(kv_write_mask, group_target_key)
                    written_value = torch.matmul(kv_write_mask, group_target_value)
                else:
                    written_key_parts = []
                    written_value_parts = []
                    for write_start in range(0, block, kv_write_chunk):
                        write_end = write_start + kv_write_chunk
                        write = kv_write_mask[:, :, :, write_start:write_end]
                        written_key_parts.append(
                            torch.matmul(
                                write,
                                group_target_key[:, :, write_start:write_end, :],
                            )
                        )
                        written_value_parts.append(
                            torch.matmul(
                                write,
                                group_target_value[:, :, write_start:write_end, :],
                            )
                        )
                    written_key = torch.stack(written_key_parts, dim=0).sum(dim=0)
                    written_value = torch.stack(written_value_parts, dim=0).sum(dim=0)
                key_state = k_state * keep + written_key
                value_state = v_state * keep + written_value
                k_state[:] = key_state
                v_state[:] = value_state

                if attention_op == "sdpa" or kv_join == "concat":
                    key = torch.cat([key_state, group_noise_key], dim=2)
                    value = torch.cat([value_state, group_noise_value], dim=2)
                query_outputs = []
                for start in range(0, query_size, chunk):
                    end = start + chunk
                    if attention_op == "sdpa":
                        repeated_key = torch.repeat_interleave(key, groups, dim=1)
                        repeated_value = torch.repeat_interleave(value, groups, dim=1)
                        query_outputs.append(
                            torch.nn.functional.scaled_dot_product_attention(
                                group_query[:, :, start:end, :],
                                repeated_key,
                                repeated_value,
                                attn_mask=attention_mask[:, :, start:end, :],
                                dropout_p=0.0,
                                is_causal=False,
                            )
                        )
                    else:
                        head_parts = []
                        scale = head_dim ** -0.5
                        for local_kv_head in range(state_group_kv_heads):
                            q_group = group_query[
                                :,
                                local_kv_head * groups:(local_kv_head + 1) * groups,
                                start:end,
                                :,
                            ]
                            if kv_join == "concat":
                                score = torch.matmul(
                                    q_group,
                                    key[
                                        :, local_kv_head:local_kv_head + 1, :, :
                                    ].transpose(-2, -1),
                                )
                            else:
                                context_score = torch.matmul(
                                    q_group,
                                    key_state[
                                        :, local_kv_head:local_kv_head + 1, :, :
                                    ].transpose(-2, -1),
                                )
                                noise_score = torch.matmul(
                                    q_group,
                                    group_noise_key[
                                        :, local_kv_head:local_kv_head + 1, :, :
                                    ].transpose(-2, -1),
                                )
                                score = torch.cat(
                                    [context_score, noise_score], dim=-1
                                )
                            score = (
                                score * scale
                                + attention_mask[:, :, start:end, :]
                            )
                            probability = torch.softmax(score.float(), dim=-1).half()
                            if kv_join == "concat":
                                head_parts.append(
                                    torch.matmul(
                                        probability,
                                        value[
                                            :,
                                            local_kv_head:local_kv_head + 1,
                                            :,
                                            :,
                                        ],
                                    )
                                )
                            else:
                                context_probability = probability[..., :max_context]
                                noise_probability = probability[..., max_context:]
                                head_parts.append(
                                    torch.matmul(
                                        context_probability,
                                        value_state[
                                            :,
                                            local_kv_head:local_kv_head + 1,
                                            :,
                                            :,
                                        ],
                                    )
                                    + torch.matmul(
                                        noise_probability,
                                        group_noise_value[
                                            :,
                                            local_kv_head:local_kv_head + 1,
                                            :,
                                            :,
                                        ],
                                    )
                                )
                        query_outputs.append(torch.cat(head_parts, dim=1))
                state_group_outputs.append(torch.cat(query_outputs, dim=2))
            return torch.cat(state_group_outputs, dim=1)

    return StatefulAttention().half().eval()


def main() -> int:
    args = parse_args()
    if sum((args.dry_run, args.torch_only, args.mil_only)) > 1:
        raise ValueError("qualification modes are mutually exclusive")
    if args.mil_dump is not None and not args.mil_only:
        raise ValueError("--mil-dump requires --mil-only")
    if args.attention_op == "sdpa" and args.kv_join != "concat":
        raise ValueError("SDPA requires --kv-join concat")
    if not args.dflash.is_file() or not args.extractor.is_file():
        raise ValueError("official DFlash GGUF and extractor must exist")
    config = describe(args.extractor, args.dflash)
    kv_heads = int(config["num_key_value_heads"])
    if (
        int(config["block_size"]) != 16
        or args.max_context < 32
        or args.chunk not in (1, 2, 4, 16)
        or args.query_size % args.chunk != 0
        or args.block_size % args.kv_write_chunk != 0
        or kv_heads % args.state_group_kv_heads != 0
    ):
        raise ValueError("invalid stateful attention-only geometry")
    mode = (
        "dry-run"
        if args.dry_run
        else "torch-only"
        if args.torch_only
        else "mil-only"
        if args.mil_only
        else "export"
    )
    state_groups = kv_heads // args.state_group_kv_heads
    state_names = [
        (kind if state_groups == 1 else f"{kind}_{state_group}")
        for state_group in range(state_groups)
        for kind in ("k_state", "v_state")
    ]
    plan = {
        "schema": "muser.dflash-stateful-attention-only-export.v1",
        "mode": mode,
        "block_size": args.block_size,
        "query_size": args.query_size,
        "chunk": args.chunk,
        "kv_write_chunk": args.kv_write_chunk,
        "attention_op": args.attention_op,
        "kv_join": args.kv_join,
        "state_group_kv_heads": args.state_group_kv_heads,
        "max_context": args.max_context,
        "state": state_names,
        "state_shape": [1, args.state_group_kv_heads, args.max_context, 128],
        "inputs": [
            "query",
            "noise_key",
            "noise_value",
            "target_key",
            "target_value",
            "attention_mask",
            "kv_write_mask",
        ],
        "output": "attention",
        "compute_units": "CPU_AND_NE",
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

    ct, _, np, torch, nn = load_toolchain()
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    block = args.block_size
    model = build_module(
        torch=torch,
        nn=nn,
        config=config,
        max_context=args.max_context,
        block_size=args.block_size,
        query_size=args.query_size,
        chunk=args.chunk,
        kv_write_chunk=args.kv_write_chunk,
        attention_op=args.attention_op,
        kv_join=args.kv_join,
        state_group_kv_heads=args.state_group_kv_heads,
    )
    examples = (
        torch.randn(1, heads, args.query_size, head_dim, dtype=torch.float16),
        torch.randn(1, kv_heads, block, head_dim, dtype=torch.float16),
        torch.randn(1, kv_heads, block, head_dim, dtype=torch.float16),
        torch.randn(1, kv_heads, block, head_dim, dtype=torch.float16),
        torch.randn(1, kv_heads, block, head_dim, dtype=torch.float16),
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
        output = model(*examples)
    if (
        tuple(output.shape) != (1, heads, args.query_size, head_dim)
        or not torch.isfinite(output).all()
    ):
        raise ValueError("stateful attention-only Torch output is invalid")
    grouped_parity = None
    if args.state_group_kv_heads < kv_heads:
        reference = build_module(
            torch=torch,
            nn=nn,
            config=config,
            max_context=args.max_context,
            block_size=args.block_size,
            query_size=args.query_size,
            chunk=args.chunk,
            kv_write_chunk=args.kv_write_chunk,
            attention_op=args.attention_op,
            kv_join=args.kv_join,
            state_group_kv_heads=kv_heads,
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
                [getattr(model, f"k_state_{group}") for group in range(state_groups)],
                dim=1,
            )
            grouped_v = torch.cat(
                [getattr(model, f"v_state_{group}") for group in range(state_groups)],
                dim=1,
            )
            output_max_abs = float(
                (grouped_output.float() - reference_output.float()).abs().max()
            )
            state_max_abs = max(
                float((grouped_k.float() - reference.k_state.float()).abs().max()),
                float((grouped_v.float() - reference.v_state.float()).abs().max()),
            )
        if output_max_abs != 0.0 or state_max_abs != 0.0:
            raise ValueError(
                "grouped MLState is not bit-exact with the full-KV-head reference: "
                f"output_max_abs={output_max_abs}, state_max_abs={state_max_abs}"
            )
        grouped_parity = {
            "reference_state_group_kv_heads": kv_heads,
            "output_max_abs": output_max_abs,
            "state_max_abs": state_max_abs,
        }
    query_chunk_parity = None
    if args.query_size < block:
        full_query_reference = build_module(
            torch=torch,
            nn=nn,
            config=config,
            max_context=args.max_context,
            block_size=args.block_size,
            query_size=block,
            chunk=args.chunk,
            kv_write_chunk=args.kv_write_chunk,
            attention_op=args.attention_op,
            kv_join=args.kv_join,
            state_group_kv_heads=args.state_group_kv_heads,
        )
        full_query = torch.randn(1, heads, block, head_dim, dtype=torch.float16)
        full_mask = torch.zeros(
            1,
            1,
            block,
            args.max_context + block,
            dtype=torch.float16,
        )
        full_write_mask = examples[-1].clone()
        for token in range(block):
            full_write_mask[0, 0, token, token] = 1.0
        with torch.no_grad():
            for state_name in state_names:
                initial = torch.randn_like(getattr(model, state_name))
                getattr(model, state_name).copy_(initial)
                getattr(full_query_reference, state_name).copy_(initial)
            full_output = full_query_reference(
                full_query,
                *examples[1:5],
                full_mask,
                full_write_mask,
            )
            chunk_outputs = []
            no_write = torch.zeros_like(full_write_mask)
            for start in range(0, block, args.query_size):
                end = start + args.query_size
                chunk_outputs.append(
                    model(
                        full_query[:, :, start:end, :],
                        *examples[1:5],
                        full_mask[:, :, start:end, :],
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
                "four T=4 predictions are not bit-exact with one T=16 "
                f"prediction: output_max_abs={output_max_abs}, "
                f"state_max_abs={state_max_abs}"
            )
        query_chunk_parity = {
            "calls": block // args.query_size,
            "reference_query_size": block,
            "output_max_abs": output_max_abs,
            "state_max_abs": state_max_abs,
        }
    contraction_parity = None
    if args.query_size == block and args.chunk == block:
        chunked_reference = build_module(
            torch=torch,
            nn=nn,
            config=config,
            max_context=args.max_context,
            block_size=args.block_size,
            query_size=args.query_size,
            chunk=4,
            kv_write_chunk=args.kv_write_chunk,
            attention_op=args.attention_op,
            kv_join=args.kv_join,
            state_group_kv_heads=args.state_group_kv_heads,
        )
        parity_write_mask = examples[-1].clone()
        for token in range(block):
            parity_write_mask[0, 0, token, token] = 1.0
        parity_examples = (*examples[:-1], parity_write_mask)
        with torch.no_grad():
            for state_name in state_names:
                initial = torch.randn_like(getattr(model, state_name))
                getattr(model, state_name).copy_(initial)
                getattr(chunked_reference, state_name).copy_(initial)
            contracted_output = model(*parity_examples)
            chunked_output = chunked_reference(*parity_examples)
            output_max_abs = float(
                (contracted_output.float() - chunked_output.float()).abs().max()
            )
            state_max_abs = max(
                float(
                    (
                        getattr(model, state_name).float()
                        - getattr(chunked_reference, state_name).float()
                    )
                    .abs()
                    .max()
                )
                for state_name in state_names
            )
        if output_max_abs != 0.0 or state_max_abs != 0.0:
            raise ValueError(
                "single T=16 contraction is not bit-exact with four internal "
                f"T=4 contractions: output_max_abs={output_max_abs}, "
                f"state_max_abs={state_max_abs}"
            )
        contraction_parity = {
            "reference_chunk": 4,
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
                    "torch_output_shape": list(output.shape),
                    "torch_output_finite": True,
                    "grouped_state_parity": grouped_parity,
                    "query_chunk_parity": query_chunk_parity,
                    "contraction_parity": contraction_parity,
                    "trace_nodes": sum(1 for _ in traced.graph.nodes()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    names = plan["inputs"]
    inputs = [
        ct.TensorType(
            name=names[0],
            shape=(1, heads, args.query_size, head_dim),
            dtype=np.float16,
        ),
        ct.TensorType(name=names[1], shape=(1, kv_heads, block, head_dim), dtype=np.float16),
        ct.TensorType(name=names[2], shape=(1, kv_heads, block, head_dim), dtype=np.float16),
        ct.TensorType(name=names[3], shape=(1, kv_heads, block, head_dim), dtype=np.float16),
        ct.TensorType(name=names[4], shape=(1, kv_heads, block, head_dim), dtype=np.float16),
        ct.TensorType(
            name=names[5],
            shape=(1, 1, args.query_size, args.max_context + block),
            dtype=np.float16,
        ),
        ct.TensorType(
            name=names[6],
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
        inputs=inputs,
        outputs=[ct.TensorType(name="attention", dtype=np.float16)],
        states=states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    if args.mil_only:
        if args.mil_dump is not None:
            if args.mil_dump.exists() or args.mil_dump.is_symlink():
                raise ValueError(f"MIL dump must be absent: {args.mil_dump}")
            args.mil_dump.parent.mkdir(parents=True, exist_ok=True)
            args.mil_dump.write_text(str(converted._mil_program), encoding="utf-8")
            plan["mil_dump_created"] = True
        counts = {}
        for operation in converted._mil_program.functions["main"].operations:
            counts[operation.op_type] = counts.get(operation.op_type, 0) + 1
        print(
            json.dumps(
                {
                    **plan,
                    "mil_operation_counts": counts,
                    "mil_operation_total": sum(counts.values()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".muser-dflash-attention-only-", dir=args.output.parent))
    try:
        package = stage / "dflash-attention-only.mlpackage"
        converted.save(str(package))
        size, digest = tree_receipt(package)
        compile_result = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(package), str(stage)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        compiled = stage / "dflash-attention-only.mlmodelc"
        if not compiled.is_dir():
            raise ValueError(f"CoreML compiler did not create {compiled}: {compile_result.stderr}")
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
        stage.rename(args.output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
