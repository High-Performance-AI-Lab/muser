"""Tests for scripts/gx10/bootstrap_node.sh.

bootstrap_node.sh is meant to run ON the GX10 node (aarch64 Linux, DGX OS);
this machine is a Mac and this repo has a live sealed campaign running
elsewhere tonight, so nothing here may load a model, touch a GPU, or ssh
anywhere. What IS safe to actually execute locally:

  - `bash -n` syntax checking.
  - `--help` / usage-error / unknown-subcommand paths (pure argument
    parsing, no I/O).
  - `probe`: read-only (uname/df/awk/nvidia-smi-if-present/docker-if-
    present); it degrades to null/false fields gracefully when a signal
    isn't available (as it must on the GX10 too, when docker or nvidia-smi
    aren't installed yet), so its JSON *shape* is portable even though the
    *values* are macOS-specific.
  - `model`'s existing-file-matches-hash path: this only stats and hashes
    a fixture file under a temp dir, never mutates anything.
  - every `--dry-run` path (model download/move-aside, daemon install,
    stop): these print a plan and touch no real state by construction.
  - `status`/real `stop`: read-only or best-effort disable of a unit that
    was never installed; guarded so a missing systemctl/tmux on macOS
    degrades to "not running" instead of crashing (verified against this
    literal repo copy of the script, not a stand-in).

Actual downloads, systemd/tmux installation, and port polling are
Linux-only and are exercised here only through --dry-run plan text plus
static checks against the shipped systemd unit template.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gx10" / "bootstrap_node.sh"
UNIT_TEMPLATE = ROOT / "scripts" / "gx10" / "llamacpp" / "muser-prefilld.service"


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=30,
    )


class SyntaxTests(unittest.TestCase):
    def test_bash_dash_n(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_executable_bit(self) -> None:
        self.assertTrue(SCRIPT.stat().st_mode & 0o111, "script must be +x for the orchestrator to run it")

    def test_native_wait_reports_closed_startup_phases_not_raw_logs(self) -> None:
        text = SCRIPT.read_text()
        for milestone in (
            "loading the 23 GB NVFP4 checkpoint into GB10 memory",
            "initializing the 8K chunked-prefill scheduler",
            "allocating 128K-capable KV cache and warming kernels",
            "running the pinned 2K first-request warmup",
        ):
            self.assertIn(milestone, text)
        self.assertIn('logs --tail 200', text)
        self.assertNotIn('emit daemon info "$logs"', text)

    def test_native_wait_emits_structured_milestones_and_heartbeats(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn('readonly NATIVE_STARTUP_HEARTBEAT_SECONDS=15', text)
        self.assertIn('"native_startup"', text)
        for phase in (
            'phase="engine-setup"',
            'phase="weights"',
            'phase="batch-profile"',
            'phase="kv-warmup"',
            'phase="request-warmup"',
            'emit_native_startup ready',
        ):
            self.assertIn(phase, text)
        self.assertIn('"$phase_detail; still working"', text)

    def test_systemd_lifecycle_persists_after_ssh_logout(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("ensure_persistent_user_manager", text)
        self.assertIn('loginctl enable-linger "$user" 2>/dev/null', text)
        self.assertIn('sudo -n loginctl enable-linger "$user"', text)
        self.assertIn('loginctl show-user "$user" -p Linger', text)


class UsageTests(unittest.TestCase):
    def test_help_exits_zero_and_lists_subcommands(self) -> None:
        result = run("--help")
        self.assertEqual(result.returncode, 0)
        for sub in ("probe", "model", "daemon", "stop", "status"):
            self.assertIn(sub, result.stdout)

    def test_missing_subcommand_is_usage_error(self) -> None:
        result = run()
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing subcommand", result.stderr)

    def test_unknown_subcommand_is_usage_error(self) -> None:
        result = run("bogus")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown subcommand", result.stderr)

    def test_unknown_global_flag_is_usage_error(self) -> None:
        result = run("--nope", "probe")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown global option", result.stderr)


class ProbeTests(unittest.TestCase):
    def test_probe_default_is_single_json_line(self) -> None:
        result = run("probe")
        self.assertEqual(result.returncode, 0)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1, result.stdout)
        payload = json.loads(lines[0])
        self.assertEqual(payload["schema"], "muser.node-probe.v1")
        self.assertIn("arch", payload)
        self.assertIn("driver_version", payload)
        self.assertIn(payload["docker_ok"], (True, False))
        # disk is always readable on any POSIX box; memory is /proc/meminfo
        # only, so it degrades to null off Linux - both are valid shapes.
        self.assertTrue(payload["disk_free_gib"] is None or isinstance(payload["disk_free_gib"], (int, float)))
        self.assertTrue(payload["mem_free_gib"] is None or isinstance(payload["mem_free_gib"], (int, float)))

    def test_probe_arch_matches_uname(self) -> None:
        expected = subprocess.run(["uname", "-m"], capture_output=True, text=True).stdout.strip()
        payload = json.loads(run("probe").stdout.strip())
        self.assertEqual(payload["arch"], expected)

    def test_probe_json_mode_wraps_single_progress_line(self) -> None:
        result = run("--json", "probe")
        self.assertEqual(result.returncode, 0)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1, result.stdout)
        event = json.loads(lines[0])
        self.assertEqual(event["schema"], "muser.node-progress.v2")
        self.assertEqual(event["step"], "preflight")
        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["data"]["schema"], "muser.node-probe.v1")

    def test_probe_dir_flag_accepted(self) -> None:
        # --dir only changes which filesystem df measures; must not error.
        result = run("probe", "--dir", "/")
        self.assertEqual(result.returncode, 0)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_requires_dir_name_sha256(self) -> None:
        for args in (
            ["model", "--name", "n", "--bytes", "1", "--sha256", "a" * 64],
            ["model", "--dir", str(self.work), "--bytes", "1", "--sha256", "a" * 64],
            ["model", "--dir", str(self.work), "--name", "n"],
        ):
            result = run(*args)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_rejects_malformed_sha256(self) -> None:
        for bad in ("short", "A" * 64, "g" * 64, "a" * 63):
            result = run("model", "--dir", str(self.work), "--name", "n", "--bytes", "1", "--sha256", bad)
            self.assertEqual(result.returncode, 2, bad)
            self.assertIn("sha256", result.stderr)

    def test_existing_verified_file_is_idempotent_ok(self) -> None:
        target = self.work / "m.gguf"
        target.write_bytes(b"fixture model bytes")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()

        result = run("model", "--dir", str(self.work), "--name", "m.gguf", "--bytes", str(target.stat().st_size), "--sha256", digest)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_bytes(), b"fixture model bytes", "verified file must not be touched")

        # --json mode: a start event then an ok event, nothing else on stdout.
        result = run("--json", "model", "--dir", str(self.work), "--name", "m.gguf", "--bytes", str(target.stat().st_size), "--sha256", digest)
        self.assertEqual(result.returncode, 0)
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual([e["status"] for e in events], ["start", "ok"])
        self.assertEqual(events[-1]["data"]["path"], str(target))

    def test_missing_file_no_source_exits_3(self) -> None:
        result = run("model", "--dir", str(self.work), "--name", "absent.gguf", "--bytes", "1", "--sha256", "a" * 64)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("upload required", result.stderr)

    def test_mismatched_file_is_never_deleted_dry_run(self) -> None:
        target = self.work / "m.gguf"
        target.write_bytes(b"wrong content")
        wrong_digest = "0" * 64
        result = run("--dry-run", "model", "--dir", str(self.work), "--name", "m.gguf", "--bytes", str(target.stat().st_size), "--sha256", wrong_digest)
        self.assertEqual(result.returncode, 3)
        self.assertIn("PLAN: would move aside", result.stdout)
        self.assertIn("never deleted", result.stdout)
        # dry-run must not actually move or delete the file.
        self.assertTrue(target.exists())
        self.assertEqual(list(self.work.iterdir()), [target])

    def test_mismatched_file_real_run_moves_aside_not_delete(self) -> None:
        target = self.work / "m.gguf"
        target.write_bytes(b"wrong content")
        wrong_digest = "0" * 64
        result = run("model", "--dir", str(self.work), "--name", "m.gguf", "--bytes", str(target.stat().st_size), "--sha256", wrong_digest)
        self.assertEqual(result.returncode, 3)
        remaining = list(self.work.iterdir())
        self.assertEqual(len(remaining), 1)
        self.assertNotEqual(remaining[0].name, "m.gguf")
        self.assertIn("mismatch-", remaining[0].name)
        self.assertEqual(remaining[0].read_bytes(), b"wrong content", "content preserved, only renamed")

    def test_dry_run_download_plan_mentions_curl_and_does_not_fetch(self) -> None:
        target_dir = self.work / "models"
        result = run(
            "--dry-run",
            "model",
            "--dir",
            str(target_dir),
            "--name",
            "new.gguf",
            "--bytes",
            "1",
            "--sha256",
            "a" * 64,
            "--source",
            "https://example.invalid/model.gguf",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("would curl -fL -C - --retry 5", result.stdout)
        self.assertIn("https://example.invalid/model.gguf", result.stdout)
        self.assertFalse(target_dir.exists(), "dry-run must not even mkdir the target directory")

    def test_dry_run_download_plan_json_mode_is_pure_json_lines(self) -> None:
        result = run(
            "--json",
            "--dry-run",
            "model",
            "--dir",
            str(self.work / "models"),
            "--name",
            "new.gguf",
            "--bytes",
            "1",
            "--sha256",
            "a" * 64,
            "--source",
            "https://example.invalid/model.gguf",
        )
        self.assertEqual(result.returncode, 0)
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)  # raises if any stdout line isn't valid JSON
            self.assertEqual(event["schema"], "muser.node-progress.v2")

    def test_spaces_in_paths_are_handled(self) -> None:
        space_dir = self.work / "space dir"
        space_dir.mkdir()
        target = space_dir / "my model.gguf"
        target.write_bytes(b"x")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        result = run("model", "--dir", str(space_dir), "--name", "my model.gguf", "--bytes", "1", "--sha256", digest)
        self.assertEqual(result.returncode, 0, result.stderr)


def write_lane_fixture(lane: Path, *, listen_host: str = "0.0.0.0") -> None:
    """Match the real on-node layout: deploy.rs pushes the llamacpp runtime
    (and, once deploy.rs's RUNTIME_FILES grows a line, the unit template)
    into LANE/llamacpp; enroll.rs writes handoff.json at LANE root."""
    payload = lane / "llamacpp"
    payload.mkdir(parents=True, exist_ok=True)
    (payload / "muser_prefilld.py").write_text("# fixture\n")
    launcher = payload / "muser-prefilld"
    launcher.write_text("#!/bin/sh\nexit 0\n")
    launcher.chmod(0o755)
    (payload / "muser-prefilld.service").write_text(UNIT_TEMPLATE.read_text())
    (lane / "handoff.json").write_text(
        json.dumps({"schema_version": 5, "listen_host": listen_host, "listen_port": 29591})
    )


class DaemonTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        self.lane = self.work / "lane"
        self.lane.mkdir()
        write_lane_fixture(self.lane)
        self.model = self.work / "model.gguf"
        self.model.write_bytes(b"m")
        self.dflash = self.work / "dflash.gguf"
        self.dflash.write_bytes(b"d")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_requires_lane_and_model(self) -> None:
        result = run("daemon", "--model", str(self.model))
        self.assertEqual(result.returncode, 2)
        result = run("daemon", "--lane", str(self.lane))
        self.assertEqual(result.returncode, 2)

    def test_nonexistent_lane_is_operational_failure_not_usage(self) -> None:
        result = run("daemon", "--lane", str(self.work / "nope"), "--model", str(self.model))
        self.assertEqual(result.returncode, 1)
        self.assertIn("deploy step first", result.stderr)

    def test_incomplete_lane_payload_fails_with_remediation(self) -> None:
        empty_lane = self.work / "empty_lane"
        empty_lane.mkdir()
        result = run("daemon", "--lane", str(empty_lane), "--model", str(self.model))
        self.assertEqual(result.returncode, 1)
        self.assertIn("deploy step incomplete", result.stderr)

    def test_missing_model_file_fails(self) -> None:
        result = run("daemon", "--lane", str(self.lane), "--model", str(self.work / "nope.gguf"))
        self.assertEqual(result.returncode, 1)

    def test_missing_dflash_file_fails(self) -> None:
        result = run(
            "daemon",
            "--lane",
            str(self.lane),
            "--model",
            str(self.model),
            "--dflash",
            str(self.work / "nope-dflash.gguf"),
        )
        self.assertEqual(result.returncode, 1)

    def test_dry_run_systemd_plan_includes_dflash_when_given(self) -> None:
        result = run(
            "--dry-run",
            "daemon",
            "--lane",
            str(self.lane),
            "--model",
            str(self.model),
            "--dflash",
            str(self.dflash),
            "--systemd",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dflash", result.stdout)
        self.assertIn(str(self.dflash), result.stdout)
        self.assertIn(str(self.model), result.stdout)
        self.assertIn(str(self.lane / "handoff.json"), result.stdout)
        self.assertIn("muser-prefilld.service", result.stdout)
        # never actually installed under --dry-run
        self.assertFalse((self.lane.parent / "etc").exists())

    def test_dry_run_systemd_plan_omits_dflash_when_absent(self) -> None:
        result = run(
            "--dry-run",
            "daemon",
            "--lane",
            str(self.lane),
            "--model",
            str(self.model),
            "--systemd",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--dflash", result.stdout)

    def test_dry_run_tmux_plan_uses_shell_quoting_and_tee(self) -> None:
        result = run(
            "--dry-run",
            "daemon",
            "--lane",
            str(self.lane),
            "--model",
            str(self.model),
            "--tmux",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tee -a", result.stdout)
        self.assertIn("prefilld-console.log", result.stdout)
        self.assertIn("muser-prefilld", result.stdout)

    def test_dry_run_json_mode_emits_only_progress_lines(self) -> None:
        result = run(
            "--json",
            "--dry-run",
            "daemon",
            "--lane",
            str(self.lane),
            "--model",
            str(self.model),
            "--dflash",
            str(self.dflash),
            "--systemd",
        )
        self.assertEqual(result.returncode, 0)
        statuses = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            self.assertEqual(event["schema"], "muser.node-progress.v2")
            self.assertEqual(event["step"], "daemon")
            statuses.append(event["status"])
        self.assertEqual(statuses, ["start", "planned", "planned", "planned"])
        self.assertIn("survives SSH logout", result.stdout)

    def test_lane_with_spaces_dry_run(self) -> None:
        spaced = self.work / "space lane"
        spaced.mkdir()
        write_lane_fixture(spaced)
        result = run(
            "--dry-run",
            "daemon",
            "--lane",
            str(spaced),
            "--model",
            str(self.model),
            "--tmux",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_payload_is_read_from_lane_llamacpp_subdirectory(self) -> None:
        # deploy.rs pushes the runtime into LANE/llamacpp, not flat in LANE
        # (see deploy.rs RUNTIME_FILES / MAKE_LANE) - a lane missing that
        # subdirectory must fail with the same "deploy step incomplete"
        # remediation as a lane missing the files entirely.
        bare_lane = self.work / "bare_lane"
        bare_lane.mkdir()
        (bare_lane / "handoff.json").write_text(
            json.dumps({"listen_host": "0.0.0.0", "listen_port": 29591})
        )
        result = run("daemon", "--lane", str(bare_lane), "--model", str(self.model))
        self.assertEqual(result.returncode, 1)
        self.assertIn("deploy step incomplete", result.stderr)

    def test_dry_run_plan_references_llamacpp_subdirectory_launcher(self) -> None:
        result = run(
            "--dry-run",
            "daemon",
            "--lane",
            str(self.lane),
            "--model",
            str(self.model),
            "--systemd",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.lane / "llamacpp" / "muser-prefilld"), result.stdout)

    def test_native_daemon_uses_the_native_bridge_and_checkpoint_directory(self) -> None:
        native_payload = self.lane / "vllm"
        native_payload.mkdir()
        bridge = native_payload / "muser_native_prefilld.py"
        bridge.write_text("# fixture\n")
        checkpoint = self.work / "checkpoint"
        checkpoint.mkdir()
        result = run(
            "--dry-run",
            "daemon",
            "--native",
            "--lane",
            str(self.lane),
            "--checkpoint",
            str(checkpoint),
            "--systemd",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("python3", result.stdout)
        self.assertIn(str(bridge), result.stdout)
        self.assertIn("--handoff-config", result.stdout)
        self.assertNotIn("--model", result.stdout)

    def test_native_daemon_refuses_llama_only_arguments(self) -> None:
        result = run(
            "daemon",
            "--native",
            "--lane",
            str(self.lane),
            "--checkpoint",
            str(self.work),
            "--model",
            str(self.model),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not accept --model", result.stderr)


class StopStatusTests(unittest.TestCase):
    def test_status_default_is_safe_and_reports_stopped_or_running(self) -> None:
        result = run("status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip().endswith("stopped") or result.stdout.strip().endswith("running"))

    def test_status_tmux_mode(self) -> None:
        result = run("status", "--tmux")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_status_json_mode(self) -> None:
        result = run("--json", "status")
        self.assertEqual(result.returncode, 0)
        event = json.loads(result.stdout.strip())
        self.assertEqual(event["data"]["schema"], "muser.node-daemon-status.v1")
        self.assertIn(event["data"]["running"], (True, False))

    def test_stop_dry_run(self) -> None:
        result = run("--dry-run", "stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PLAN: would stop", result.stdout)

    def test_stop_is_idempotent_when_nothing_installed(self) -> None:
        # Real (non-dry-run) stop against a daemon that was never installed
        # must succeed, not error - idempotency is the whole point.
        result = run("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run("stop", "--tmux")
        self.assertEqual(result.returncode, 0, result.stderr)


class UnitTemplateTests(unittest.TestCase):
    """Static checks on the shipped systemd template bootstrap_node.sh fills in."""

    def test_template_has_exactly_the_expected_placeholders(self) -> None:
        text = UNIT_TEMPLATE.read_text()
        for token in (
            "@@MUSER_WORKING_DIRECTORY@@",
            "@@MUSER_EXEC_START@@",
            "@@MUSER_ENV_FILE@@",
            "@@MUSER_WANTED_BY@@",
        ):
            self.assertEqual(text.count(token), 1, f"{token} must appear exactly once")

    def test_template_no_longer_hardcodes_a_fixed_install_path(self) -> None:
        text = UNIT_TEMPLATE.read_text()
        # Regression guard: the old fixed /opt/muser/llamacpp ExecStart/
        # WorkingDirectory is what made --dflash and user-level installs
        # impossible; only the Documentation= URL may still reference it.
        exec_and_workdir_lines = [
            line
            for line in text.splitlines()
            if line.startswith("ExecStart=") or line.startswith("WorkingDirectory=")
        ]
        for line in exec_and_workdir_lines:
            self.assertNotIn("/opt/muser/llamacpp", line)

    def test_template_environment_file_is_optional(self) -> None:
        text = UNIT_TEMPLATE.read_text()
        self.assertIn("EnvironmentFile=-@@MUSER_ENV_FILE@@", text, "leading '-' keeps a missing env file non-fatal")


if __name__ == "__main__":
    unittest.main()
