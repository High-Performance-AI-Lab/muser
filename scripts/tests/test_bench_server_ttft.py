from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import bench_server_ttft as bench


class CaptureRendezvousTests(unittest.TestCase):
    def test_capture_marker_is_create_once_durable_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "ready.json"
            environment = {
                "MUSER_TTFT_CAPTURE_READY_FILE": str(marker),
                "MUSER_TTFT_CAPTURE_PAUSE_MS": "1000",
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                bench.time, "sleep"
            ) as sleep:
                bench.capture_rendezvous()
                sleep.assert_called_once_with(1.0)
            self.assertEqual(
                marker.read_bytes(),
                b'{"schema":"muser.server-ttft-capture-ready.v1","ready":true}\n',
            )
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                bench.time, "sleep"
            ), self.assertRaises(FileExistsError):
                bench.capture_rendezvous()

    def test_capture_environment_is_closed(self) -> None:
        with mock.patch.dict(
            os.environ, {"MUSER_TTFT_CAPTURE_PAUSE_MS": "1000"}, clear=True
        ), self.assertRaisesRegex(RuntimeError, "requires"):
            bench.capture_rendezvous()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {
                "MUSER_TTFT_CAPTURE_READY_FILE": str(Path(temporary) / "ready.json"),
                "MUSER_TTFT_CAPTURE_PAUSE_MS": "999",
            },
            clear=True,
        ), self.assertRaisesRegex(RuntimeError, "1000"):
            bench.capture_rendezvous()

    def test_trace_prompt_reuse_is_llama_only_and_rendezvous_bound(self) -> None:
        with mock.patch.dict(
            os.environ, {"MUSER_TTFT_CAPTURE_REUSE_PROMPT": "1"}, clear=True
        ), self.assertRaisesRegex(RuntimeError, "only for llama"):
            bench.capture_reuses_prompt("muser")
        with mock.patch.dict(
            os.environ, {"MUSER_TTFT_CAPTURE_REUSE_PROMPT": "1"}, clear=True
        ), self.assertRaisesRegex(RuntimeError, "rendezvous"):
            bench.capture_reuses_prompt("llama")
        with mock.patch.dict(
            os.environ,
            {
                "MUSER_TTFT_CAPTURE_REUSE_PROMPT": "1",
                "MUSER_TTFT_CAPTURE_READY_FILE": "/durable/ready.json",
            },
            clear=True,
        ):
            self.assertTrue(bench.capture_reuses_prompt("llama"))

    def test_llama_trace_request_enables_only_prompt_cache(self) -> None:
        import json

        _, ordinary = bench.request_spec("llama", "muse", [1, 2])
        _, traced = bench.request_spec("llama", "muse", [1, 2], reuse_prompt=True)
        self.assertFalse(json.loads(ordinary)["cache_prompt"])
        self.assertTrue(json.loads(traced)["cache_prompt"])


if __name__ == "__main__":
    unittest.main()
