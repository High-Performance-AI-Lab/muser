from __future__ import annotations

import importlib.util
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "accelerator_safe.py"
SPEC = importlib.util.spec_from_file_location("accelerator_safe_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ACCELERATOR_SAFE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCELERATOR_SAFE)


BIN_NAME = re.compile(r'^\s*name\s*=\s*"([^"]+)"', re.MULTILINE)


def workspace_bin_names() -> list[str]:
    names: list[str] = []
    for cargo_toml in ROOT.glob("crates/*/Cargo.toml"):
        text = cargo_toml.read_text()
        for block in text.split("[[bin]]")[1:]:
            match = BIN_NAME.search(block)
            if match is not None:
                names.append(match.group(1))
    return names


class GpuProcessPatternTests(unittest.TestCase):
    def test_profiler_authorization_is_direct_and_closed(self) -> None:
        valid = [
            "/usr/local/bin/gputrace",
            "headless-profile",
            "--attach-launched",
            "--attach-after-file",
            "/evidence/ready.json",
            "--process",
            "muser-metal-phase-diagnostic",
            "--out-dir",
            "/evidence/trace",
            "--",
            "/build/muser-metal-phase-diagnostic",
        ]
        ACCELERATOR_SAFE.validate_command(valid, allow_profiler=True)
        with self.assertRaises(SystemExit):
            ACCELERATOR_SAFE.validate_command(valid)
        with self.assertRaises(SystemExit):
            ACCELERATOR_SAFE.validate_command(
                ["/usr/local/bin/gputrace", "capture", "--", "/build/muser"],
                allow_profiler=True,
            )
        with self.assertRaises(SystemExit):
            ACCELERATOR_SAFE.validate_command(
                valid + ["/usr/bin/xctrace"], allow_profiler=True
            )

    def test_every_workspace_bin_matches_the_gpu_process_pattern(self) -> None:
        names = workspace_bin_names()
        self.assertTrue(names, "expected at least one [[bin]] across the workspace")
        unmatched = [name for name in names if not ACCELERATOR_SAFE.GPU_PROCESS.search(name)]
        self.assertEqual(unmatched, [])

    def test_pattern_still_matches_a_bare_engine_and_comparator_binary(self) -> None:
        for name in ("muser", "llama-bench", "llama-cli", "llama-server"):
            self.assertTrue(ACCELERATOR_SAFE.GPU_PROCESS.search(name), name)

    def test_pattern_matches_python_driven_coreml_and_ane_scripts_by_full_command(self) -> None:
        commands = [
            "python3 /repo/scripts/coreml_plan_receipt.py --model foo",
            "python3 /repo/scripts/coreml_mil_tensor_pressure.py",
            "python3 /repo/scripts/coreml_shard_latency.py",
            "python3 /repo/scripts/export_dflash_coreml.py --out x",
            "python3 /repo/scripts/export_dflash_stateful_attention_coreml.py",
            "python3 /repo/scripts/export_dflash_stateful_attention_only_coreml.py",
            "python3 /repo/scripts/export_muse_target_coreml.py",
            "python3 /repo/scripts/evaluate_ane.py --ledger x",
        ]
        for command in commands:
            self.assertTrue(ACCELERATOR_SAFE.GPU_PROCESS.search(command), command)

    def test_pattern_does_not_flag_unrelated_python_scripts(self) -> None:
        self.assertIsNone(
            ACCELERATOR_SAFE.GPU_PROCESS.search("python3 /repo/scripts/campaign.py --dry-run")
        )

    def test_shell_text_does_not_impersonate_its_accelerator_child(self) -> None:
        command = (
            "/bin/bash -c cd /repo/ferrite-rs && llama-server --model model.gguf"
        )
        self.assertFalse(
            ACCELERATOR_SAFE.process_uses_accelerator("/bin/bash", command)
        )
        self.assertTrue(
            ACCELERATOR_SAFE.process_uses_accelerator(
                "/usr/local/bin/llama-server", "llama-server --model model.gguf"
            )
        )

    def test_cpu_test_in_ferrite_directory_is_not_an_accelerator_process(self) -> None:
        self.assertFalse(
            ACCELERATOR_SAFE.process_uses_accelerator(
                "/repo/target/release/qwen38_cpu_reference",
                "/repo/target/release/qwen38_cpu_reference --nocapture",
            )
        )


class EvidenceHomeSafetyTests(unittest.TestCase):
    def test_default_out_dir_uses_an_explicit_writable_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(ACCELERATOR_SAFE.os.environ, {"MUSER_RESULTS_DIR": temporary}):
                out_dir = ACCELERATOR_SAFE.default_out_dir()
            self.assertEqual(out_dir, Path(temporary))

    def test_default_out_dir_refuses_a_non_writable_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            blocked = Path(temporary) / "blocked"
            blocked.mkdir()
            blocked.chmod(0o500)
            try:
                if ACCELERATOR_SAFE.os.access(blocked, ACCELERATOR_SAFE.os.W_OK):
                    self.skipTest("cannot drop write permission in this environment")
                with mock.patch.dict(
                    ACCELERATOR_SAFE.os.environ, {"MUSER_RESULTS_DIR": str(blocked)}
                ):
                    with self.assertRaises(SystemExit):
                        ACCELERATOR_SAFE.default_out_dir()
            finally:
                blocked.chmod(0o700)

    def test_default_out_dir_falls_back_to_the_system_temp_directory(self) -> None:
        with mock.patch.dict(ACCELERATOR_SAFE.os.environ, {}, clear=False):
            ACCELERATOR_SAFE.os.environ.pop("MUSER_RESULTS_DIR", None)
            out_dir = ACCELERATOR_SAFE.default_out_dir()
        self.assertEqual(out_dir, Path(tempfile.gettempdir()) / "muser-results")

    def test_execute_is_refused_under_the_system_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            under_temp = Path(temporary) / "muser-results"
            self.assertTrue(ACCELERATOR_SAFE.under_temp_dir(under_temp))

    def test_dry_run_out_dir_outside_temp_is_not_flagged(self) -> None:
        self.assertFalse(ACCELERATOR_SAFE.under_temp_dir(ROOT / "results" / "baseline-seal"))

    def test_busy_admission_still_publishes_a_bound_result_and_log(self) -> None:
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as temporary:
            out_dir = Path(temporary)
            receipt = out_dir / "busy.result.json"
            args = argparse.Namespace(
                execute=True,
                identity="fixture-identity",
                cell="fixture-cell",
                out_dir=out_dir,
                result_receipt=receipt,
                quiet_seconds=10,
                allow_profiler=False,
                share_lease=False,
                command=["/usr/bin/true", "--fixture"],
            )
            with mock.patch.object(ACCELERATOR_SAFE, "parse_args", return_value=args), \
                    mock.patch.object(ACCELERATOR_SAFE, "LOCK_PATH", out_dir / "fixture.gpu.lock"), \
                    mock.patch.object(ACCELERATOR_SAFE, "active_gpu_processes", return_value=["123 muser"]):
                self.assertEqual(ACCELERATOR_SAFE.main(), 75)
            retained = json.loads(receipt.read_text())
            self.assertEqual(retained["schema"], "muser.accelerator-result.v1")
            self.assertEqual(retained["command"], args.command)
            self.assertEqual(retained["exit_status"], 75)
            log = Path(retained["command_log"])
            self.assertTrue(log.is_file())
            self.assertIn("another GPU process", log.read_text())


class LeaseInheritanceTests(unittest.TestCase):
    def lease_path(self, directory: str) -> Path:
        return Path(directory) / "fixture.gpu.lock"

    def test_owned_open_description_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = self.lease_path(directory)
            with lock_path.open("a+") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                environment = {
                    "MUSER_ACCELERATOR_LEASE": "1",
                    ACCELERATOR_SAFE.LEASE_FD_ENV: str(owner.fileno()),
                }
                with mock.patch.object(ACCELERATOR_SAFE, "LOCK_PATH", lock_path), \
                        mock.patch.dict(os.environ, environment, clear=True):
                    self.assertEqual(
                        ACCELERATOR_SAFE.inherited_lease_fd(), owner.fileno()
                    )

    def test_unlocked_descriptor_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = self.lease_path(directory)
            with lock_path.open("a+") as candidate, \
                    mock.patch.object(ACCELERATOR_SAFE, "LOCK_PATH", lock_path):
                with self.assertRaisesRegex(RuntimeError, "is not locked"):
                    ACCELERATOR_SAFE.verify_inherited_lease(candidate.fileno())

    def test_separate_descriptor_cannot_borrow_another_owners_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = self.lease_path(directory)
            with lock_path.open("a+") as owner, lock_path.open("a+") as candidate, \
                    mock.patch.object(ACCELERATOR_SAFE, "LOCK_PATH", lock_path):
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "does not own"):
                    ACCELERATOR_SAFE.verify_inherited_lease(candidate.fileno())

    def test_wrong_inode_and_malformed_environment_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = self.lease_path(directory)
            other = Path(directory) / "other.lock"
            with other.open("a+") as candidate, \
                    mock.patch.object(ACCELERATOR_SAFE, "LOCK_PATH", lock_path):
                lock_path.touch()
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "is not .*fixture.gpu.lock"):
                    ACCELERATOR_SAFE.verify_inherited_lease(candidate.fileno())
            with mock.patch.dict(
                os.environ,
                {
                    "MUSER_ACCELERATOR_LEASE": "1",
                    ACCELERATOR_SAFE.LEASE_FD_ENV: "not-an-fd",
                },
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "invalid inherited"):
                ACCELERATOR_SAFE.inherited_lease_fd()

    def test_share_lease_passes_the_owned_descriptor_to_the_child(self) -> None:
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as temporary:
            out_dir = Path(temporary)
            receipt = out_dir / "shared.result.json"
            args = argparse.Namespace(
                execute=True,
                identity="fixture-identity",
                cell="fixture-cell",
                out_dir=out_dir,
                result_receipt=receipt,
                quiet_seconds=10,
                allow_profiler=False,
                share_lease=True,
                command=["/usr/bin/true", "--fixture"],
            )
            completed = subprocess.CompletedProcess(args.command, 0)
            with mock.patch.object(ACCELERATOR_SAFE, "parse_args", return_value=args), \
                    mock.patch.object(ACCELERATOR_SAFE, "LOCK_PATH", out_dir / "fixture.gpu.lock"), \
                    mock.patch.object(ACCELERATOR_SAFE, "active_gpu_processes", return_value=[]), \
                    mock.patch.object(ACCELERATOR_SAFE.time, "sleep"), \
                    mock.patch.object(ACCELERATOR_SAFE.subprocess, "run", return_value=completed) as run:
                self.assertEqual(ACCELERATOR_SAFE.main(), 0)
            child = run.call_args.kwargs
            self.assertEqual(len(child["pass_fds"]), 1)
            self.assertEqual(
                child["env"][ACCELERATOR_SAFE.LEASE_FD_ENV],
                str(child["pass_fds"][0]),
            )
            retained = json.loads(receipt.read_text())
            self.assertEqual(retained["lease_source"], "acquired")
            self.assertTrue(retained["lease_shared_with_child"])


if __name__ == "__main__":
    unittest.main()
