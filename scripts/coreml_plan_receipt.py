#!/usr/bin/env python3
"""Emit MLComputePlan residency evidence for a Muser DFlash ANE artifact.

This performs CoreML compilation/plan inspection but no predictions. It is an
accelerator-lane qualification command and must be launched by the campaign
driver under the shared lock; receipts belong in ignored result storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def operation_device(plan, operation) -> tuple[str, float]:
    try:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
    except Exception:
        return "unknown", 0.0
    device = type(usage.preferred_compute_device).__name__ if usage else "unknown"
    weight = float(getattr(cost, "weight", 0.0) or 0.0) if cost else 0.0
    return device, weight


def inspect_shard(package: Path) -> dict:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    compiled = Path(ct.utils.compile_model(str(package)))
    # Inspect the same placement policy used by the release runtime.  The
    # CoreML default is ALL, which would permit a silent GPU preference and
    # therefore cannot back an ANE-residency claim.
    plan = MLComputePlan.load_from_path(
        str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    program = plan.model_structure.program
    function = program.functions["main"]
    operations = function.block.operations if hasattr(function, "block") else function.operations
    devices: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    costs: Counter[str] = Counter()
    operator_devices: dict[str, Counter[str]] = {}
    conv_devices = []
    attention_devices = []
    for operation in operations:
        operator = getattr(operation, "operator_name", getattr(operation, "type", "unknown"))
        # Core ML's constexpr nodes materialize compressed weights while the
        # package is compiled. They have no runtime compute-device assignment
        # and zero estimated runtime cost, so treating their expected
        # `unknown` device as fallback would falsely reject quantized ANE
        # packages.
        if operator == "const" or "constexpr_" in operator:
            continue
        device, cost = operation_device(plan, operation)
        operators[operator] += 1
        devices[device] += 1
        costs[device] += cost
        operator_devices.setdefault(operator, Counter())[device] += 1
        if operator.endswith("conv") or operator == "conv":
            conv_devices.append(device)
        if "scaled_dot_product_attention" in operator:
            attention_devices.append(device)
    conv_resident = bool(conv_devices) and all(
        "Neural" in device for device in conv_devices
    )
    attention_resident = bool(attention_devices) and all(
        "Neural" in device for device in attention_devices
    )
    all_compute_resident = bool(devices) and all(
        "Neural" in device for device in devices
    )
    total_cost = sum(costs.values())
    non_neural_cost = sum(
        cost for device, cost in costs.items() if "Neural" not in device
    )
    non_neural_cost_fraction = (
        non_neural_cost / total_cost if total_cost > 0.0 else 1.0
    )
    non_neural_operators = {
        operator: {
            device: count
            for device, count in counts.items()
            if "Neural" not in device
        }
        for operator, counts in operator_devices.items()
        if any("Neural" not in device for device in counts)
    }
    boundary_cast_only = bool(non_neural_operators) and all(
        operator.endswith(".cast") or operator == "cast"
        for operator in non_neural_operators
    )
    ane_compute_qualified = all_compute_resident or (
        boundary_cast_only and non_neural_cost_fraction <= 0.01
    )
    return {
        "operators": dict(sorted(operators.items())),
        "operator_preferred_devices": {
            operator: dict(sorted(counts.items()))
            for operator, counts in sorted(operator_devices.items())
        },
        "preferred_devices": dict(sorted(devices.items())),
        "estimated_cost": {key: round(value, 6) for key, value in sorted(costs.items())},
        "conv_preferred_devices": conv_devices,
        "attention_preferred_devices": attention_devices,
        "conv_resident_on_neural_engine": conv_resident,
        "attention_resident_on_neural_engine": attention_resident,
        "ane_compute_resident": all_compute_resident,
        "ane_compute_qualified": ane_compute_qualified,
        "non_neural_runtime_operators": non_neural_operators,
        "non_neural_estimated_cost_fraction": round(non_neural_cost_fraction, 6),
    }


def main() -> int:
    args = parse_args()
    import coremltools as ct

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("compute_units") != "CPU_AND_NE":
        raise ValueError("manifest does not require CPU_AND_NE")
    root = args.manifest.parent
    shards = []
    for spec in manifest.get("shards", []):
        package = (root / spec["path"]).resolve()
        if not package.is_relative_to(root.resolve()):
            raise ValueError(f"shard escapes artifact root: {spec['path']}")
        plan = inspect_shard(package)
        shards.append(
            {
                "order": spec["order"],
                "projection": spec["projection"],
                "input_offset": spec["input_offset"],
                "input_width": spec["input_width"],
                "output_offset": spec["output_offset"],
                "output_width": spec["output_width"],
                **plan,
            }
        )
    ffn_shards = []
    for spec in manifest.get("ffn_shards", []):
        package = (root / spec["path"]).resolve()
        if not package.is_relative_to(root.resolve()):
            raise ValueError(f"FFN shard escapes artifact root: {spec['path']}")
        plan = inspect_shard(package)
        ffn_shards.append(
            {
                "order": spec["order"],
                "layer": spec["layer"],
                "intermediate_offset": spec["intermediate_offset"],
                "intermediate_width": spec["intermediate_width"],
                **plan,
            }
        )
    tail_shards = []
    for spec in manifest.get("tail_shards", []):
        package = (root / spec["path"]).resolve()
        if not package.is_relative_to(root.resolve()):
            raise ValueError(f"tail shard escapes artifact root: {spec['path']}")
        plan = inspect_shard(package)
        tail_shards.append(
            {
                "order": spec["order"],
                "layer": spec["layer"],
                "head": spec["head"],
                "intermediate_offset": spec["intermediate_offset"],
                "intermediate_width": spec["intermediate_width"],
                **plan,
            }
        )
    target_packages = []
    for spec in manifest.get("packages", []):
        package = (root / spec["path"]).resolve()
        if not package.is_relative_to(root.resolve()):
            raise ValueError(f"target package escapes artifact root: {spec['path']}")
        target_packages.append(
            {
                "order": spec["order"],
                "path": spec["path"],
                **inspect_shard(package),
            }
        )
    stateful_packages = []
    if manifest.get("schema") in {
        "muser.dflash-stateful-attention-export.v1",
        "muser.dflash-stateful-attention-only-export.v1",
    }:
        relative = manifest.get("package")
        if not isinstance(relative, str):
            raise ValueError("stateful DFlash pilot manifest lacks its package")
        package = (root / relative).resolve()
        if not package.is_relative_to(root.resolve()):
            raise ValueError(f"stateful package escapes artifact root: {relative}")
        stateful_packages.append(
            {
                "layer": manifest.get("layer"),
                "path": relative,
                "state": manifest.get("state"),
                "state_shape": manifest.get("state_shape"),
                "query_size": manifest.get("query_size"),
                **inspect_shard(package),
            }
        )
    for spec in manifest.get("attention_shards", []):
        package = (root / spec["path"]).resolve()
        if not package.is_relative_to(root.resolve()):
            raise ValueError(f"attention shard escapes artifact root: {spec['path']}")
        state_group_kv_heads = spec["state_group_kv_heads"]
        if state_group_kv_heads not in (1, 2, 4, 8) or 8 % state_group_kv_heads:
            raise ValueError("invalid attention state group geometry")
        state_groups = 8 // state_group_kv_heads
        state_names = [
            (kind if state_groups == 1 else f"{kind}_{group}")
            for group in range(state_groups)
            for kind in ("k_state", "v_state")
        ]
        stateful_packages.append(
            {
                "layer": spec["layer"],
                "path": spec["path"],
                "state": state_names,
                "state_shape": [1, state_group_kv_heads, spec["max_context"], 128],
                "query_size": spec["query_size"],
                **inspect_shard(package),
            }
        )
    fused_attention_packages = []
    for spec in manifest.get("fused_attention_shards", []):
        package = (root / spec["path"]).resolve()
        if not package.is_relative_to(root.resolve()):
            raise ValueError(f"fused attention shard escapes artifact root: {spec['path']}")
        state_group_kv_heads = spec["state_group_kv_heads"]
        if state_group_kv_heads not in (1, 2, 4, 8) or 8 % state_group_kv_heads:
            raise ValueError("invalid fused attention state group geometry")
        state_groups = 8 // state_group_kv_heads
        state_names = [
            (kind if state_groups == 1 else f"{kind}_{group}")
            for group in range(state_groups)
            for kind in ("k_state", "v_state")
        ]
        fused_attention_packages.append(
            {
                "layer": spec["layer"],
                "path": spec["path"],
                "state": state_names,
                "state_shape": [
                    1,
                    state_group_kv_heads,
                    spec["max_context"],
                    128,
                ],
                "query_size": spec["query_size"],
                "bundle_channels": spec["bundle_channels"],
                **inspect_shard(package),
            }
        )
    receipt = {
        "schema": "muser-coreml-compute-plan-v4",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "coremltools_version": ct.__version__,
        "plan_compute_units": "CPU_AND_NE",
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "manifest_version": manifest.get("version"),
        "shard_count": (
            len(manifest.get("shards", []))
            + len(manifest.get("ffn_shards", []))
            + len(manifest.get("tail_shards", []))
            + len(manifest.get("packages", []))
            + len(stateful_packages)
            + len(fused_attention_packages)
        ),
        "target_identity": manifest.get("target_identity", manifest.get("model_sha256")),
        "dflash_identity": manifest.get("dflash_identity"),
        "all_conv_resident_on_neural_engine": all(
            not shard["conv_preferred_devices"]
            or shard["conv_resident_on_neural_engine"]
            for shard in (
                shards
                + ffn_shards
                + tail_shards
                + target_packages
                + stateful_packages
                + fused_attention_packages
            )
        ),
        "all_ane_compute_resident": all(
            shard["ane_compute_resident"]
            for shard in (
                shards
                + ffn_shards
                + tail_shards
                + target_packages
                + stateful_packages
                + fused_attention_packages
            )
        ),
        "all_ane_compute_qualified": all(
            shard["ane_compute_qualified"]
            for shard in (
                shards
                + ffn_shards
                + tail_shards
                + target_packages
                + stateful_packages
                + fused_attention_packages
            )
        ),
        "shards": shards,
        "ffn_shards": ffn_shards,
        "tail_shards": tail_shards,
        "target_packages": target_packages,
        "stateful_packages": stateful_packages,
        "fused_attention_packages": fused_attention_packages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not receipt["all_ane_compute_qualified"]:
        raise SystemExit(
            "one or more CoreML packages has substantive non-ANE compute or "
            "boundary-cast cost above one percent"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
