#!/usr/bin/env python3
"""Estimate SSA tensor pressure from a compiled CoreML ``model.mil``.

This is an offline diagnostic, not a substitute for MLComputePlan.  It reports
declared intermediate sizes and a conservative source-order liveness estimate,
which is useful when ANECompilerService reports::

    RegAlloc: failed to allocate intermediate tensors.

The estimate intentionally reads only the textual MIL sidecar.  It does not
load the model, invoke CoreML, or touch an accelerator.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DECL = re.compile(
    r"tensor<(?P<dtype>[A-Za-z0-9]+),\s*\[(?P<shape>[^]]*)\]>\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<op>[A-Za-z0-9_]+)\("
)
PROGRAM_DECL = re.compile(
    r"%(?P<name>[A-Za-z_][A-Za-z0-9_]*):\s*"
    r"\((?P<signature>[^)]*)\)\(Tensor\)\s*=\s*"
    r"(?P<op>[A-Za-z0-9_]+)\("
)
TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
DTYPE_BYTES = {
    "bool": 1,
    "fp16": 2,
    "fp32": 4,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
}


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    size: int
    op: str
    definition: int


def parse_shape(value: str) -> tuple[int, ...] | None:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if not all(field.isdigit() for field in fields):
        return None
    return tuple(int(field) for field in fields)


def product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def human(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.3f} {suffix}"
        value /= 1024.0
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_mil", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")

    lines = args.model_mil.read_text(encoding="utf-8").splitlines()
    tensors: dict[str, Tensor] = {}
    rhs_tokens: list[set[str]] = []
    for line_number, line in enumerate(lines):
        match = DECL.search(line)
        program_match = PROGRAM_DECL.search(line) if match is None else None
        if match is None and program_match is None:
            rhs_tokens.append(set())
            continue
        if match is not None:
            shape = parse_shape(match.group("shape"))
            dtype = match.group("dtype")
            name = match.group("name")
            op = match.group("op")
            rhs_start = match.end()
        else:
            assert program_match is not None
            signature = [
                field.strip() for field in program_match.group("signature").split(",")
            ]
            shape = parse_shape(",".join(signature[:-1]))
            dtype = signature[-1]
            name = program_match.group("name")
            op = program_match.group("op")
            rhs_start = program_match.end()
        if shape is not None and dtype in DTYPE_BYTES:
            tensors[name] = Tensor(
                name=name,
                dtype=dtype,
                shape=shape,
                size=product(shape) * DTYPE_BYTES[dtype],
                op=op,
                definition=line_number,
            )
        rhs_tokens.append(set(TOKEN.findall(line[rhs_start:])))

    last_use = {name: tensor.definition for name, tensor in tensors.items()}
    for line_number, names in enumerate(rhs_tokens):
        for name in names & tensors.keys():
            last_use[name] = max(last_use[name], line_number)

    peak_line = 0
    peak_bytes = 0
    peak_names: list[str] = []
    for line_number in range(len(lines)):
        active = [
            tensor
            for tensor in tensors.values()
            if tensor.op != "const"
            and tensor.definition <= line_number <= last_use[tensor.name]
        ]
        total = sum(tensor.size for tensor in active)
        if total > peak_bytes:
            peak_bytes = total
            peak_line = line_number
            peak_names = [tensor.name for tensor in active]

    largest = sorted(
        (tensor for tensor in tensors.values() if tensor.op != "const"),
        key=lambda tensor: (-tensor.size, tensor.definition, tensor.name),
    )[: args.top]
    peak = sorted(
        (tensors[name] for name in peak_names),
        key=lambda tensor: (-tensor.size, tensor.definition, tensor.name),
    )
    print(
        json.dumps(
            {
                "schema": "muser.coreml-mil-tensor-pressure.v1",
                "model_mil": str(args.model_mil),
                "line_count": len(lines),
                "declared_tensor_count": len(tensors),
                "nonconst_tensor_count": sum(
                    tensor.op != "const" for tensor in tensors.values()
                ),
                "peak_source_liveness": {
                    "line": peak_line + 1,
                    "bytes": peak_bytes,
                    "human": human(peak_bytes),
                    "tensor_count": len(peak),
                    "largest_live_tensors": [
                        {
                            "name": tensor.name,
                            "op": tensor.op,
                            "shape": tensor.shape,
                            "bytes": tensor.size,
                        }
                        for tensor in peak[: args.top]
                    ],
                },
                "largest_intermediates": [
                    {
                        "name": tensor.name,
                        "op": tensor.op,
                        "shape": tensor.shape,
                        "bytes": tensor.size,
                        "definition_line": tensor.definition + 1,
                        "last_use_line": last_use[tensor.name] + 1,
                    }
                    for tensor in largest
                ],
                "boundary": (
                    "Conservative textual-SSA estimate; ANEF lowering, fusion, "
                    "aliasing, and physical allocation are not observable here."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
