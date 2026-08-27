from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import campaign  # noqa: E402


class CampaignEvidenceTests(unittest.TestCase):
    def test_retained_accelerator_result_binds_run_command_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20260814T120000Z-" + "a" * 32
            log = root / f"{run_id}.command.log"
            log.write_text("fixture\n")
            receipt = root / "result.json"
            child = ["/usr/bin/true", "--fixture"]
            receipt.write_text(
                json.dumps(
                    {
                        "schema": "muser.accelerator-result.v1",
                        "run_id": run_id,
                        "identity": "fixture-identity",
                        "cell": "fixture-cell",
                        "command": child,
                        "exit_status": 0,
                        "command_log": str(log),
                        "started_at": "2026-08-14T12:00:00+00:00",
                        "finished_at": "2026-08-14T12:00:01+00:00",
                    }
                )
            )
            wrapper = [
                "python3", "scripts/accelerator_safe.py", "--identity",
                "fixture-identity", "--cell", "fixture-cell", "--out-dir",
                str(root), "--", *child,
            ]
            retained, retained_log = campaign.retained_accelerator_result(
                receipt, wrapper, 0,
            )
            self.assertEqual(retained["run_id"], run_id)
            self.assertEqual(retained_log, log)

            bad = json.loads(receipt.read_text())
            bad["cell"] = "different-cell"
            receipt.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "different run"):
                campaign.retained_accelerator_result(
                    receipt, wrapper, 0,
                )

    def test_remote_normalization_binds_combined_components_and_link_samples(self) -> None:
        identity = "sha256:" + "9" * 64
        records = self._remote_records(identity)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "remote.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            normalized = campaign.normalize_evidence(
                "remote", log, "remote-target-plus-dflash-8192", identity
            )
        self.assertEqual(normalized["installed_payload_gbps"], [4.0, 4.0, 4.0])
        self.assertEqual(normalized["installed_payload_gbps_cv"], 0.0)

    def test_remote_normalization_refuses_aggregate_only_dflash_claim(self) -> None:
        identity = "sha256:" + "9" * 64
        records = self._remote_records(identity)
        records[1]["dflash_installed"] = False
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "remote.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(RuntimeError, "DFlash evidence"):
                campaign.normalize_evidence(
                    "remote", log, "remote-target-plus-dflash-8192", identity
                )

    def test_remote_normalization_requires_payload_only_wire_timing(self) -> None:
        identity = "sha256:" + "9" * 64
        records = self._remote_records(identity)
        records[0].pop("producer_payload_wire_ns")
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "remote.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(RuntimeError, "phase timing"):
                campaign.normalize_evidence(
                    "remote", log, "remote-target-plus-dflash-8192", identity
                )

    def test_remote_normalization_refuses_dflash_trace_drift(self) -> None:
        identity = "sha256:" + "9" * 64
        records = self._remote_records(identity)
        records[1]["dflash_draft_trace_sha256"] = "e" * 64
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "remote.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(RuntimeError, "trace changed"):
                campaign.normalize_evidence(
                    "remote", log, "remote-target-plus-dflash-8192", identity
                )

    def test_remote_normalization_refuses_subthreshold_absolute_acceptance(self) -> None:
        identity = "sha256:" + "9" * 64
        records = self._remote_records(identity)
        records[1]["remote_dflash_acceptance"] = 0.94
        records[-1]["remote_dflash_acceptance"] = [0.97, 0.94, 0.97]
        records[-1]["remote_dflash_acceptance_minimum"] = 0.94
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "remote.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(RuntimeError, "below 95%"):
                campaign.normalize_evidence(
                    "remote", log, "remote-target-plus-dflash-8192", identity
                )

    def test_ttft_normalization_projects_one_verified_depth(self) -> None:
        identity = "sha256:" + "1" * 64
        depth = 128
        records = [
            {
                "schema": "muser.server-ttft.v2",
                "kind": "sample",
                "engine": "muser",
                "identity": identity,
                "depth": depth,
                "repetition": repetition,
                "elapsed_ns": 100 + repetition,
            }
            for repetition in range(5)
        ]
        records.append(
            {
                "schema": "muser.server-ttft.v2",
                "kind": "summary",
                "engine": "muser",
                "identity": identity,
                "depth": depth,
                "raw_ns": [100, 101, 102, 103, 104],
                "prompt_sha256": "2" * 64,
                "first_content_digests": ["3" * 64] * 5,
                "reported_prompt_tokens": [depth] * 5,
                "server_lifecycle": "leased-start-ready-exact-requests-cooperative-exit",
                "seal_eligible": True,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "ttft.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            normalized = campaign.normalize_evidence(
                "ttft-muser", log, "ttft-128", identity
            )
        self.assertEqual(normalized["fingerprint"]["reported_prompt_tokens"], depth)
        self.assertEqual(
            normalized["fingerprint"]["server_lifecycle"],
            "leased-start-ready-exact-requests-cooperative-exit",
        )

    def test_ttft_normalization_rejects_wrong_planned_depth(self) -> None:
        identity = "sha256:" + "4" * 64
        records = [
            {
                "schema": "muser.server-ttft.v2",
                "kind": "sample",
                "engine": "muser",
                "identity": identity,
                "depth": 127,
                "repetition": repetition,
                "elapsed_ns": 100,
            }
            for repetition in range(5)
        ]
        records.append(
            {
                "schema": "muser.server-ttft.v2",
                "kind": "summary",
                "engine": "muser",
                "identity": identity,
                "depth": 127,
                "raw_ns": [100, 100, 100, 100, 100],
                "first_content_digests": ["5" * 64] * 5,
                "reported_prompt_tokens": [127] * 5,
                "seal_eligible": True,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "wrong-depth.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(RuntimeError, "mixed engine route"):
                campaign.normalize_evidence("ttft-muser", log, "ttft-128", identity)

    def test_kvpack_normalization_preserves_gate_timings(self) -> None:
        identity = "sha256:" + "6" * 64
        records = self._kvpack_records(identity)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "kvpack.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            normalized = campaign.normalize_evidence(
                "kvpack", log, "kvpack-resident-ancestor-8192-s255", identity
            )
        self.assertEqual(normalized["full_recompute_ns"], 100)
        self.assertEqual(normalized["source_prefill_ns"], 100)
        self.assertEqual(normalized["publication_overhead_ratio"], 0.01)
        self.assertEqual(normalized["miss_overhead_ratio"], 0.01)
        self.assertEqual(normalized["speedup_geomean_cell"], 2.0)

    def test_kvpack_normalization_rejects_mixed_route(self) -> None:
        identity = "sha256:" + "7" * 64
        records = self._kvpack_records(identity)
        records[1]["matched_tokens"] = 8191
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "mixed-kvpack.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(RuntimeError, "mixed or unplanned restore route"):
                campaign.normalize_evidence(
                    "kvpack", log, "kvpack-resident-ancestor-8192-s255", identity
                )

    def test_ane_normalization_preserves_phase_diagnostics(self) -> None:
        identity = "sha256:" + "8" * 64
        samples = []
        for repetition in range(3):
            samples.append(
                {
                    "schema": "muser.ane-qualify.v1",
                    "kind": "sample",
                    "identity": identity,
                    "exact_target_match": True,
                    "target_only_ns": 130,
                    "metal_dflash_ns": 100,
                    "ane_dflash_ns": 80,
                    "target_verification_tax": 0.0,
                    "metal_target_verify_ns": 50,
                    "ane_target_verify_ns": 50,
                    "metal_prefill_ns": 10,
                    "ane_prefill_ns": 12,
                    "metal_draft_ns": 20,
                    "ane_draft_ns": 30,
                    "metal_fallback_target_ns": 0,
                    "ane_fallback_target_ns": 0,
                    "metal_rounds": 2,
                    "ane_rounds": 2,
                    "metal_drafted_tokens": 30,
                    "ane_drafted_tokens": 30,
                    "metal_accepted_draft_tokens": 29,
                    "ane_accepted_draft_tokens": 29,
                    "ane_mirror_capture_fc_ns": 11,
                    "generated_tokens_sha256": "a" * 64,
                }
            )
        samples.append(
            {
                "schema": "muser.ane-qualify.v1",
                "kind": "summary",
                "identity": identity,
                "target_only_raw_ns": [130, 130, 130],
                "metal_dflash_raw_ns": [100, 100, 100],
                "ane_dflash_raw_ns": [80, 80, 80],
                "target_verification_taxes": [0.0, 0.0, 0.0],
                "compute_units": "CPU_AND_NE",
                "exact_target_match": True,
                "prompt_tokens": 512,
                "prompt_file_sha256": "b" * 64,
                "output_tokens": 256,
                "verify_length": 15,
                "target_identity": "target",
                "dflash_identity": "dflash",
                "manifest_sha256": "c" * 64,
                "compute_plan_receipt_sha256": "d" * 64,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "ane.jsonl"
            log.write_text("".join(json.dumps(record) + "\n" for record in samples))
            normalized = campaign.normalize_evidence(
                "ane", log, "ane-512-p1", identity
            )
        self.assertEqual(normalized["phase_timings"]["ane_draft_ns"], [30, 30, 30])
        self.assertEqual(normalized["phase_timings"]["ane_rounds"], [2, 2, 2])

    @staticmethod
    def _kvpack_records(identity: str) -> list[dict[str, object]]:
        tokens = list(range(64))
        token_digest = hashlib.sha256(
            b"".join(token.to_bytes(4, "little") for token in tokens)
        ).hexdigest()
        records: list[dict[str, object]] = [
            {
                "schema": "muser.kvpack-qualify.v1",
                "kind": "sample",
                "identity": identity,
                "source": "resident",
                "lookup": "deepest-ancestor",
                "repetition": repetition,
                "prompt_tokens": 8447,
                "published_cut": 8192,
                "matched_tokens": 8192,
                "suffix_tokens": 255,
                "restore_to_first_logits_ns": 50,
                "full_recompute_ns": 100,
                "token_ids": tokens,
                "full_logit_digest": "c" * 64,
            }
            for repetition in range(3)
        ]
        records.append(
            {
                "schema": "muser.kvpack-qualify.v1",
                "kind": "summary",
                "identity": identity,
                "source": "resident",
                "lookup": "deepest-ancestor",
                "prompt_tokens": 8447,
                "published_cut": 8192,
                "suffix_tokens": 255,
                "raw_restore_ns": [50, 50, 50],
                "restore_cv": 0.0,
                "full_recompute_ns": 100,
                "source_prefill_ns": 100,
                "publication_ns": 1,
                "publication_overhead_ratio": 0.01,
                "miss_lookup_ns": 1,
                "miss_overhead_ratio": 0.01,
                "speedup_geomean_cell": 2.0,
                "generated_tokens_sha256": token_digest,
                "full_logit_digest": "c" * 64,
                "correctness": "exact-64-tokens-and-all-step-full-logit-digest",
                "seal_eligible": True,
            }
        )
        return records

    @staticmethod
    def _remote_records(identity: str) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        orders = [["local", "remote"], ["remote", "local"], ["remote", "local"]]
        for repetition, order in enumerate(orders):
            samples.append(
                {
                    "schema": "muser.remote-qualify.v1",
                    "kind": "sample",
                    "identity": identity,
                    "variant": "target-plus-dflash",
                    "repetition": repetition,
                    "order": order,
                    "prompt_positions": 8192,
                    "output_tokens": 256,
                    "local_ttft_ns": 200,
                    "remote_ttft_ns": 100,
                    "local_first_64_decode_ns": 100,
                    "remote_first_64_decode_ns": 100,
                    "installed_bytes": 1_000_000,
                    "installed_segments": 4,
                    "target_installed_bytes": 750_000,
                    "target_installed_segments": 3,
                    "dflash_installed_bytes": 250_000,
                    "dflash_installed_segments": 1,
                    "target_prepared": True,
                    "target_installed": True,
                    "dflash_prepared": True,
                    "dflash_installed": True,
                    # Includes producer prefill and is retained for TTFT phase
                    # evidence, but must not be used as the link denominator.
                    "receiver_transfer_commit_ns": 400_000_000_000,
                    "producer_payload_wire_ns": 2_000_000,
                    "receiver_segment_drain_ns": 2_000_000,
                    "installed_payload_gbps": 4.0,
                    "producer_payload_bytes": 1_000_000,
                    "producer_export_overhead_ratio": 0.01,
                    "producer_first_tile_prefill_fraction": 0.1,
                    "producer_transfer_hidden_ratio": 0.96,
                    "generated_tokens_sha256": "a" * 64,
                    "full_logit_digest": "b" * 64,
                    "exact_tokens": True,
                    "exact_full_logits": True,
                    "exact_dflash_tokens": True,
                    "exact_dflash_trace": True,
                    "dflash_draft_trace_sha256": "c" * 64,
                    "dflash_accepted_prefix_trace_sha256": "d" * 64,
                    "dflash_accepted_prefix_counts": [7, 7, 6],
                    "local_dflash_acceptance": 0.98,
                    "remote_dflash_acceptance": 0.97,
                    "remote_dflash_acceptance_ratio": 0.99,
                }
            )
        samples.append(
            {
                "schema": "muser.remote-qualify.v1",
                "kind": "summary",
                "identity": identity,
                "variant": "target-plus-dflash",
                "prompt_positions": 8192,
                "output_tokens": 256,
                "local_ttft_raw_ns": [200, 200, 200],
                "remote_ttft_raw_ns": [100, 100, 100],
                "installed_payload_gbps": [4.0, 4.0, 4.0],
                "installed_payload_gbps_median": 4.0,
                "installed_payload_gbps_cv": 0.0,
                "installed_payload_gbps_minimum": 3.0,
                "remote_dflash_acceptance": [0.97, 0.97, 0.97],
                "remote_dflash_acceptance_minimum": 0.97,
                "remote_dflash_acceptance_required": 0.95,
                "generated_tokens_sha256": "a" * 64,
                "full_logit_digest": "b" * 64,
                "exact_remote_local": True,
                "stable": True,
            }
        )
        return samples


if __name__ == "__main__":
    unittest.main()
