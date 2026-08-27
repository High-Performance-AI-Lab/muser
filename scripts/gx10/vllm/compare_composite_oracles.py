#!/usr/bin/env python3
"""Compare sequential and block GX composite-target oracle captures."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


HIDDEN_ROW_FLOATS = 5 * 6656
LOGIT_ROW_FLOATS = 202_048
SCHEMA = "muser.composite-target-oracle-comparison.v1"


def read_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("mode") != "import":
        raise ValueError(f"not a composite import receipt: {path}")
    capture = value.get("hidden_capture")
    if not isinstance(capture, dict) or capture.get("schema") != "muser.spark-composite-target-oracle.v1":
        raise ValueError(f"receipt has no full target oracle: {path}")
    return value


def read_f32(path: Path, expected_floats: int) -> array.array[float]:
    payload = path.read_bytes()
    if len(payload) != expected_floats * 4:
        raise ValueError(
            f"{path} has {len(payload)} bytes, expected {expected_floats * 4}"
        )
    values = array.array("f")
    values.frombytes(payload)
    if values.itemsize != 4:
        raise RuntimeError("host array('f') is not IEEE-754 binary32")
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path} contains a non-finite value")
    return values


def compare_matrix(
    left_path: Path,
    right_path: Path,
    rows: int,
    row_floats: int,
    *,
    include_argmax: bool,
) -> dict[str, Any]:
    left = read_f32(left_path, rows * row_floats)
    right = read_f32(right_path, rows * row_floats)
    exact_rows = 0
    differing_values = 0
    max_abs_error = 0.0
    sum_abs_error = 0.0
    sum_squared_error = 0.0
    argmax_matches = 0
    row_metrics = []
    for row in range(rows):
        start = row * row_floats
        end = start + row_floats
        left_bytes = left[start:end].tobytes()
        right_bytes = right[start:end].tobytes()
        left_digest = hashlib.sha256(left_bytes).hexdigest()
        right_digest = hashlib.sha256(right_bytes).hexdigest()
        exact = left_digest == right_digest
        exact_rows += int(exact)
        row_max = 0.0
        row_sum = 0.0
        row_squared = 0.0
        row_differing = 0
        for left_value, right_value in zip(left[start:end], right[start:end], strict=True):
            difference = abs(float(left_value) - float(right_value))
            if difference != 0.0:
                row_differing += 1
            row_max = max(row_max, difference)
            row_sum += difference
            row_squared += difference * difference
        differing_values += row_differing
        max_abs_error = max(max_abs_error, row_max)
        sum_abs_error += row_sum
        sum_squared_error += row_squared
        metric: dict[str, Any] = {
            "row": row,
            "bit_exact": exact,
            "left_sha256": left_digest,
            "right_sha256": right_digest,
            "differing_values": row_differing,
            "max_abs_error": row_max,
            "mean_abs_error": row_sum / row_floats,
        }
        if include_argmax:
            left_argmax = max(range(start, end), key=left.__getitem__) - start
            right_argmax = max(range(start, end), key=right.__getitem__) - start
            metric["left_argmax"] = left_argmax
            metric["right_argmax"] = right_argmax
            metric["argmax_match"] = left_argmax == right_argmax
            argmax_matches += int(left_argmax == right_argmax)
        row_metrics.append(metric)
    count = rows * row_floats
    result: dict[str, Any] = {
        "rows": rows,
        "row_floats": row_floats,
        "bit_exact": exact_rows == rows,
        "exact_rows": exact_rows,
        "differing_values": differing_values,
        "max_abs_error": max_abs_error,
        "mean_abs_error": sum_abs_error / count,
        "root_mean_squared_error": math.sqrt(sum_squared_error / count),
        "row_metrics": row_metrics,
    }
    if include_argmax:
        result["argmax_matches"] = argmax_matches
        result["all_argmax_match"] = argmax_matches == rows
    return result


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential", type=Path, required=True)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    sequential = read_receipt(args.sequential)
    block = read_receipt(args.block)
    sequential_capture = sequential["hidden_capture"]
    block_capture = block["hidden_capture"]
    rows = int(sequential_capture["rows"])
    if rows != int(block_capture["rows"]):
        raise ValueError("oracle row counts differ")
    sequential_tokens = [int(token) for token in sequential["generated_tokens"]]
    block_append = [int(token) for token in block["request"]["append_tokens"]]
    block_tokens = [int(token) for token in block["generated_tokens"]]
    relationship_ok = (
        len(sequential_tokens) == rows
        and block_append == sequential_tokens[:-1]
        and block_tokens == sequential_tokens[-1:]
        and sequential["bundle"]["root_sha256"] == block["bundle"]["root_sha256"]
        and sequential["loaded_checkpoint"] == block["loaded_checkpoint"]
    )
    sequential_hidden = args.sequential.with_suffix(".hidden.f32")
    block_hidden = args.block.with_suffix(".hidden.f32")
    sequential_logits = args.sequential.with_suffix(".logits.f32")
    block_logits = args.block.with_suffix(".logits.f32")
    hidden = compare_matrix(
        sequential_hidden,
        block_hidden,
        rows,
        HIDDEN_ROW_FLOATS,
        include_argmax=False,
    )
    logits = compare_matrix(
        sequential_logits,
        block_logits,
        rows,
        LOGIT_ROW_FLOATS,
        include_argmax=True,
    )
    exact = relationship_ok and hidden["bit_exact"] and logits["bit_exact"]
    result = {
        "schema": SCHEMA,
        "sequential_receipt": str(args.sequential),
        "block_receipt": str(args.block),
        "rows": rows,
        "trace_relationship_valid": relationship_ok,
        "generated_tokens": sequential_tokens,
        "hidden": hidden,
        "logits": logits,
        "qualified_bit_exact": exact,
    }
    write_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if args.require_exact and not exact:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
