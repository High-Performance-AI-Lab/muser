from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import gx_trace_diff as trace_diff


FIXTURE = Path(__file__).parent / "fixtures" / "gx_trace_strict_1x_20260814.json"


def receipt(summary: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "muser.gx10-container.receipt.v1",
        "image_id": summary["image_id"],
    }


class GxTraceDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = trace_diff.load_json(FIXTURE)

    def test_retained_fixture_is_semantic_at_round_one(self) -> None:
        result = trace_diff.compare(self.summary, receipt(self.summary))
        comparison = result["comparison"]
        self.assertEqual(comparison["classification"], "semantic-accepted-prefix-counts")
        self.assertTrue(comparison["semantic_divergence"])
        self.assertFalse(comparison["representation_only"])
        self.assertEqual(comparison["first_count_divergence_round_zero_based"], 1)
        self.assertFalse(result["accelerator_touched"])
        self.assertFalse(result["gx_touched"])

    def test_token_digest_is_canonical_little_endian_u32(self) -> None:
        expected = hashlib.sha256(b"\x01\x00\x00\x00\x00\x01\x00\x00").hexdigest()
        self.assertEqual(trace_diff.token_digest([1, 256]), expected)

    def test_representation_only_json_differences_canonicalize(self) -> None:
        value = copy.deepcopy(self.summary)
        trace = value["trace"]
        trace["draft_remote_sha256"] = trace["draft_local_sha256"]
        trace["accepted_remote_sha256"] = trace["accepted_local_sha256"]
        trace["remote_accepted_prefix_counts"] = trace["local_accepted_prefix_counts"]
        trace["remote_drafted"] = trace["local_drafted"]
        trace["remote_accepted"] = trace["local_accepted"]
        trace["remote_acceptance"] = float(f"{trace['local_acceptance']:.16g}")
        result = trace_diff.compare(value, receipt(value))
        self.assertEqual(result["comparison"]["classification"], "equal-after-canonicalization")
        self.assertFalse(result["comparison"]["semantic_divergence"])
        self.assertTrue(result["comparison"]["representation_only"])

    def test_inconsistent_counts_and_unknown_trace_fields_fail_closed(self) -> None:
        bad_count = copy.deepcopy(self.summary)
        bad_count["trace"]["local_accepted"] += 1
        with self.assertRaisesRegex(trace_diff.TraceError, "sum"):
            trace_diff.compare(bad_count, receipt(bad_count))
        unknown = copy.deepcopy(self.summary)
        unknown["trace"]["ignored"] = True
        with self.assertRaisesRegex(trace_diff.TraceError, "unknown"):
            trace_diff.compare(unknown, receipt(unknown))

    def test_duplicate_json_keys_fail_closed(self) -> None:
        with self.assertRaisesRegex(trace_diff.TraceError, "duplicate JSON key"):
            json.loads('{"schema":1,"schema":2}', object_pairs_hook=trace_diff.reject_duplicate_keys)

    def test_container_identity_mismatch_fails_closed(self) -> None:
        wrong = receipt(self.summary)
        wrong["image_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(trace_diff.TraceError, "image identities"):
            trace_diff.compare(self.summary, wrong)


if __name__ == "__main__":
    unittest.main()
