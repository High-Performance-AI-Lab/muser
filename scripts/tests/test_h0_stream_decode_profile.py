import unittest

from scripts.h0_stream_decode_profile import select_measured_session, summarize


def event(session: str, index: int, sampling: int | None = 12) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "muser.stream-decode-profile.v1",
        "session_id": session,
        "token_index": index,
        "input_token": 1,
        "engine_argmax_token": 2,
        "decode_total_ns": 100,
        "batcher_unaccounted_ns": 10,
        "emit_ns": 3,
        "sampling_after_decode_ns": sampling,
        "model_prepare_ns": 4,
        "model_encode_ns": 20,
        "encoder_end_ns": 2,
        "command_commit_ns": 3,
        "gpu_wait_ns": 50,
        "logits_readback_ns": 4,
        "finite_scan_ns": 1,
        "argmax_ns": 1,
        "result_clone_ns": 2,
    }
    return value


class StreamDecodeProfileTests(unittest.TestCase):
    def test_selects_only_long_session_and_requires_contiguous_indexes(self) -> None:
        events = [event("warmup", 0, None)] + [event("measure", i) for i in range(64)]
        self.assertEqual(len(select_measured_session(events, 64)), 64)
        events[-1]["token_index"] = 65
        with self.assertRaisesRegex(ValueError, "contiguous"):
            select_measured_session(events, 64)

    def test_summary_ranks_serialized_components_and_excludes_gpu_wait(self) -> None:
        events = [event("measure", i, None if i == 63 else 12) for i in range(64)]
        result = summarize(events, 64)
        self.assertEqual(result["tokens"], 64)
        self.assertEqual(result["serialized_rank"][0]["component"], "model_encode_ns")
        self.assertNotIn(
            "gpu_wait_ns", [entry["component"] for entry in result["serialized_rank"]]
        )
        self.assertEqual(
            result["components"]["serialized_host_total_ns"]["median_ms"], 62 / 1_000_000
        )

    def test_rejects_multiple_measured_sessions(self) -> None:
        events = [event("left", i) for i in range(64)] + [event("right", i) for i in range(64)]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_measured_session(events, 64)


if __name__ == "__main__":
    unittest.main()
