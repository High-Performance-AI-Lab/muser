from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_nvfp4_text_matrix as matrix


def valid_records(depth: int, identity: str = "release-identity") -> list[dict]:
    outputs = matrix.output_tokens(depth)
    ttfts = [1_000_000_000, 1_010_000_000, 990_000_000, 1_005_000_000]
    links = [4.0, 4.1, 4.2, 4.3]
    samples = [
        {
            "schema": "muser.remote-qualify.v1",
            "kind": "fast-performance-sample",
            "identity": identity,
            "repetition": repetition,
            "warmup": repetition == 0,
            "prompt_positions": depth,
            "output_tokens": outputs,
            "remote_ttft_ns": ttft,
            "installed_payload_gbps": link,
            "producer_receipt_profile": "enrolled",
            "generated_tokens_sha256": "a" * 64,
            "full_logit_digest": "b" * 64,
            "deterministic_against_first": True,
        }
        for repetition, (ttft, link) in enumerate(zip(ttfts, links))
    ]
    counted_ttfts = ttfts[1:]
    counted_links = links[1:]
    summary = {
        "schema": "muser.remote-qualify.v1",
        "kind": "fast-performance-summary",
        "identity": identity,
        "performance_only": True,
        "reference_comparison": None,
        "prompt_positions": depth,
        "output_tokens": outputs,
        "remote_ttft_raw_ns": counted_ttfts,
        "remote_ttft_warmup_ns": ttfts[0],
        "warmup_repetitions": 1,
        "remote_ttft_median_ns": sorted(counted_ttfts)[1],
        "remote_ttft_cv": matrix.coefficient_of_variation(counted_ttfts),
        "remote_ttft_target_applicable": False,
        "installed_payload_gbps": counted_links,
        "installed_payload_gbps_min": min(counted_links),
        "installed_payload_gbps_minimum": 3.0,
        "producer_receipt_profile": "enrolled",
        "fast_generated_tokens_sha256": "a" * 64,
        "fast_full_logit_digest": "b" * 64,
        "deterministic": True,
        "stable": True,
        "seal_eligible": False,
    }
    return [*samples, summary]


class NativeTextMatrixTests(unittest.TestCase):
    def test_ssh_command_preserves_remote_arguments_with_spaces(self) -> None:
        command = matrix.ssh_command(
            "node-a",
            "docker",
            "inspect",
            "--format",
            "{{json .State}}",
            "resident",
        )
        self.assertEqual(command[:2], ["ssh", "node-a"])
        self.assertEqual(
            command[2], "docker inspect --format '{{json .State}}' resident"
        )

        lease = matrix.ssh_command(
            "node-a", "python3", "-c", "print('LEASE HELD')"
        )
        self.assertEqual(len(lease), 3)
        self.assertIn("python3 -c", lease[2])
        self.assertIn("'\"'\"'LEASE HELD'\"'\"'", lease[2])

    def test_release_geometry_leaves_131008_context_headroom(self) -> None:
        self.assertEqual(matrix.output_tokens(8192), 256)
        self.assertEqual(matrix.output_tokens(131008), 48)
        self.assertLessEqual(131008 + matrix.output_tokens(131008), 131072)

    def test_fast_packet_requires_one_warmup_and_three_counted_samples(self) -> None:
        result = matrix.validate_fast_records(
            valid_records(32768), identity="release-identity", depth=32768
        )
        self.assertEqual(len(result["remote_ttft_raw_ns"]), 3)
        self.assertLessEqual(result["remote_ttft_cv"], 0.02)
        self.assertGreaterEqual(result["installed_payload_gbps_min"], 3.0)

    def test_fast_packet_rejects_determinism_and_link_failures(self) -> None:
        records = valid_records(65536)
        records[2]["deterministic_against_first"] = False
        with self.assertRaisesRegex(RuntimeError, "invalid performance sample"):
            matrix.validate_fast_records(
                records, identity="release-identity", depth=65536
            )

        records = valid_records(65536)
        records[1]["installed_payload_gbps"] = 2.99
        with self.assertRaisesRegex(RuntimeError, "invalid performance sample"):
            matrix.validate_fast_records(
                records, identity="release-identity", depth=65536
            )

    def test_cell_command_is_accelerator_wrapped_and_pins_three_repetitions(self) -> None:
        options = SimpleNamespace(
            model=Path("/model.gguf"),
            cluster_config=Path("/cluster.json"),
            rope_cache=Path("/rope.bin"),
            identity="release-identity",
            resident="resident",
            spark_host="node-a",
            receiver_host="192.0.2.10",
            receiver_port=29590,
            remote_sock="/run/muser/work/producer.sock",
            eee_off_ruling="ledger ruling",
            execute=True,
        )
        command = matrix.cell_command(
            options,
            depth=131008,
            fixture=Path("/prompt.tokens"),
            remote_fixture="/run/muser/work/prompt.tokens",
            first_generation=10,
            cell_dir=Path("/evidence/cell"),
        )
        self.assertEqual(Path(command[1]).name, "accelerator_safe.py")
        self.assertIn("--execute", command)
        separator = command.index("--")
        child = command[separator + 1 :]
        self.assertEqual(child[child.index("--repetitions") + 1], "3")
        self.assertEqual(child[child.index("--output-tokens") + 1], "48")


if __name__ == "__main__":
    unittest.main()
