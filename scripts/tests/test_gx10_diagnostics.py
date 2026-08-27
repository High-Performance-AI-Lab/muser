"""Unit tests for the GX10 operator diagnostic tools in scripts/gx10/."""

from __future__ import annotations

import io
import json
import socket
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gx10"))
sys.path.insert(0, str(ROOT / "scripts" / "gx10" / "vllm"))

import durable_fsync_probe
import handoff_report
import restart_resident_producer
import supervise_resident_producer
import tcp_probe


class DurableFsyncProbeTest(unittest.TestCase):
    def test_probe_collects_samples_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = durable_fsync_probe.probe(Path(tmp), 5, 512)
            self.assertEqual(len(samples), 5)
            self.assertTrue(all(sample >= 0 for sample in samples))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_main_passes_with_generous_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["durable_fsync_probe.py", tmp, "--iterations", "3", "--max-tail-ms", "60000"]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(durable_fsync_probe.main(), 0)


class TcpProbeTest(unittest.TestCase):
    def test_loopback_roundtrip(self) -> None:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.close()
        server = threading.Thread(target=tcp_probe.serve, args=(port,), daemon=True)
        server.start()
        output = io.StringIO()
        with redirect_stdout(output):
            tcp_probe.client("127.0.0.1", port, 0.2, 1, False)
        self.assertIn("streams=1 total=", output.getvalue())


def synthetic_receipt(generation: int, seal_ns: int) -> dict:
    return {
        "response": {
            "producer_receipt": {
                "handoff": {
                    "generation": generation,
                    "payload_bytes": 108998656,
                    "payload_wire_ns": 150_000_000,
                    "payload_pacing_bps": 8_000_000_000,
                    "segments": 16,
                },
                "phase_ns": {
                    "scheduled_to_connector_start": 2_000_000,
                    "first_layer_offset": 90_000_000,
                    "d2h_complete_offset": 1_150_000_000,
                    "host_materialize_hash": 80_000_000,
                    "pack_send": 180_000_000,
                    "seal": seal_ns,
                    "connector_total": 1_500_000_000 + seal_ns,
                },
            }
        }
    }


class HandoffReportTest(unittest.TestCase):
    def test_report_table_from_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for index, seal in enumerate((47_000_000, 872_000_000)):
                Path(tmp, f"f-p4-text-g{600 + index}-client.json").write_text(
                    json.dumps(synthetic_receipt(600 + index, seal))
                )
            argv = ["handoff_report.py", "--out-dir", tmp]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
                self.assertEqual(handoff_report.main(), 0)
            text = output.getvalue()
            self.assertIn("600", text)
            self.assertIn("601", text)
            self.assertIn("872", text)  # the slow seal is visible in the table


class RestartProducerPlanTest(unittest.TestCase):
    INSPECT = [
        {
            "Config": {
                "Cmd": [
                    "--startup-receipt",
                    "/receipts/runtime.json",
                    "--rope-cache-output",
                    "/run/muser/work/rope.bin",
                    "--sock",
                    "/run/muser/work/producer.sock",
                    "--lease-file",
                    "/tmp/ferrite.gpu.lock",
                ]
            },
            "Mounts": [
                {"Source": "/lane/work", "Destination": "/run/muser/work"},
                {"Source": "/lane/receipts", "Destination": "/receipts"},
                {
                    "Source": "/tmp/ferrite.gpu.lock",
                    "Destination": "/tmp/ferrite.gpu.lock",
                },
            ],
        }
    ]

    def test_layout_derives_host_paths(self) -> None:
        with mock.patch.object(
            restart_resident_producer, "docker", return_value=json.dumps(self.INSPECT)
        ):
            layout = restart_resident_producer.container_layout("some-container")
        self.assertEqual(layout["receipt"], Path("/lane/receipts/runtime.json"))
        self.assertEqual(layout["rope_cache"], Path("/lane/work/rope.bin"))
        self.assertEqual(layout["sock"], Path("/lane/work/producer.sock"))
        self.assertEqual(layout["lease_file"], Path("/tmp/ferrite.gpu.lock"))

    def test_layout_rejects_missing_mount(self) -> None:
        broken = [
            {
                "Config": {"Cmd": ["--startup-receipt", "/receipts/x.json"]},
                "Mounts": [],
            }
        ]
        with mock.patch.object(
            restart_resident_producer, "docker", return_value=json.dumps(broken)
        ):
            with self.assertRaises(SystemExit):
                restart_resident_producer.container_layout("broken")

    def test_plan_moves_aside_and_never_deletes_data(self) -> None:
        layout = {
            "work": Path("/lane/work"),
            "receipt": Path("/lane/receipts/runtime.json"),
            "rope_cache": Path("/lane/work/rope.bin"),
            "sock": Path("/lane/work/producer.sock"),
        }
        rows = restart_resident_producer.plan(layout, "20260818T000000Z")
        verbs = [verb for verb, _, _ in rows]
        self.assertEqual(verbs, ["move-aside", "move-aside", "remove"])
        for verb, source, destination in rows:
            if verb == "move-aside":
                self.assertTrue(str(destination).endswith(".stale-20260818T000000Z"))
        self.assertEqual(rows[2][1], Path("/lane/work/producer.sock"))

    def test_naive_dry_run_plan_only_removes_namespaced_socket(self) -> None:
        naive_inspect = json.loads(json.dumps(self.INSPECT))
        command = naive_inspect[0]["Config"]["Cmd"]
        command[command.index("--sock") + 1] = "/run/muser/work/producer-naive.sock"
        with mock.patch.object(
            restart_resident_producer, "docker", return_value=json.dumps(naive_inspect)
        ):
            layout = restart_resident_producer.container_layout("naive-container")

        rows = restart_resident_producer.plan(layout, "20260818T000000Z")
        self.assertEqual(rows[2][0], "remove")
        self.assertEqual(rows[2][1], Path("/lane/work/producer-naive.sock"))
        self.assertNotEqual(rows[2][1], Path("/lane/work/producer.sock"))

    def test_main_refuses_held_lease_before_docker_or_artifact_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "runtime.json"
            rope = root / "rope.bin"
            sock = root / "producer.sock"
            lease = root / "accelerator.lock"
            for path in (receipt, rope, sock, lease):
                path.write_text("old")
            layout = {
                "work": root,
                "receipt": receipt,
                "rope_cache": rope,
                "sock": sock,
                "lease_file": lease,
            }
            output = io.StringIO()
            argv = ["restart_resident_producer.py", "--container", "target"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                restart_resident_producer, "container_layout", return_value=layout
            ), mock.patch.object(
                restart_resident_producer, "accelerator_lease_is_free", return_value=False
            ) as probe, mock.patch.object(
                restart_resident_producer, "lease_holder_hint", return_value="python3 1234"
            ), mock.patch.object(
                restart_resident_producer, "docker"
            ) as docker, redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(restart_resident_producer.main(), 1)

            probe.assert_called_once_with(lease)
            docker.assert_not_called()
            self.assertIn("stop the holding producer first", output.getvalue())
            self.assertIn("python3 1234", output.getvalue())
            self.assertTrue(receipt.exists())
            self.assertTrue(rope.exists())
            self.assertTrue(sock.exists())

    def test_main_proceeds_when_lease_probe_is_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "runtime.json"
            rope = root / "rope.bin"
            sock = root / "producer.sock"
            lease = root / "accelerator.lock"
            rope.write_text("old")
            sock.write_text("old")
            receipt.write_text("old")
            lease.touch()
            layout = {
                "work": root,
                "receipt": receipt,
                "rope_cache": rope,
                "sock": sock,
                "lease_file": lease,
            }
            calls: list[tuple[str, ...]] = []

            def fake_docker(*args: str, **_kwargs: object) -> str:
                calls.append(args)
                if args == ("restart", "target"):
                    receipt.write_text("fresh")
                return ""

            argv = ["restart_resident_producer.py", "--container", "target"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                restart_resident_producer, "container_layout", return_value=layout
            ), mock.patch.object(
                restart_resident_producer, "accelerator_lease_is_free", return_value=True
            ) as probe, mock.patch.object(
                restart_resident_producer, "docker", side_effect=fake_docker
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(restart_resident_producer.main(), 0)

            probe.assert_called_once_with(lease)
            self.assertIn(("restart", "target"), calls)
            self.assertEqual(receipt.read_text(), "fresh")


class SupervisorTest(unittest.TestCase):
    def test_decide_latches_only_at_the_failure_ceiling(self) -> None:
        self.assertEqual(supervise_resident_producer.decide(0, 3), "restart")
        self.assertEqual(supervise_resident_producer.decide(2, 3), "restart")
        self.assertEqual(supervise_resident_producer.decide(3, 3), "latch")
        self.assertEqual(supervise_resident_producer.decide(9, 3), "latch")

    def test_restart_ritual_runs_before_the_restart(self) -> None:
        # A dead container triggers move-aside of the receipt and rope cache,
        # removal of the stale socket, and only then a docker restart.
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            receipts = Path(tmp) / "receipts"
            work.mkdir()
            receipts.mkdir()
            rope = work / "rope.bin"
            sock = work / "producer.sock"
            receipt = receipts / "runtime.json"
            rope.write_bytes(b"old")
            sock.write_bytes(b"old")
            receipt.write_text("{}")
            layout = {"work": work, "receipt": receipt, "rope_cache": rope, "sock": sock}
            calls: list[tuple[str, ...]] = []

            def fake_docker(*args: str) -> str:
                calls.append(args)
                if args[:2] == ("inspect", "some-container") and args[2:] == (
                    "--format",
                    "{{.State.Status}}",
                ):
                    return "exited" if calls.count(args) == 1 else "running"
                return ""

            with mock.patch.object(supervise_resident_producer, "docker", fake_docker), \
                mock.patch.object(
                    supervise_resident_producer, "container_layout", return_value=layout
                ), \
                mock.patch.object(
                    supervise_resident_producer, "await_readiness", return_value=True
                ):
                status = supervise_resident_producer.supervise(
                    "some-container", 3, 1, 60, once=True
                )
            self.assertEqual(status, 0)
            self.assertTrue(receipt.with_name("runtime.json").name.startswith("runtime.json"))
            self.assertFalse(sock.exists())
            self.assertFalse(receipt.exists())
            self.assertTrue(any(p.name.startswith("runtime.json.stale-") for p in receipts.iterdir()))
            self.assertIn(("restart", "some-container"), calls)

    def test_latch_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            receipts = Path(tmp) / "receipts"
            work.mkdir()
            receipts.mkdir()
            layout = {
                "work": work,
                "receipt": receipts / "runtime.json",
                "rope_cache": work / "rope.bin",
                "sock": work / "producer.sock",
            }
            with mock.patch.object(
                supervise_resident_producer, "docker", return_value="exited"
            ), mock.patch.object(
                supervise_resident_producer, "container_layout", return_value=layout
            ), mock.patch.object(
                supervise_resident_producer, "await_readiness", return_value=False
            ), mock.patch.object(
                supervise_resident_producer.time, "sleep", lambda _s: None
            ):
                status = supervise_resident_producer.supervise(
                    "some-container", 2, 1, 60, once=False
                )
            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
