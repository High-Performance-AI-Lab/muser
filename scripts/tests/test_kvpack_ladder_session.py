"""Unit tests for the kvpack ladder session driver."""

from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_kvpack_ladder_session as ladder


class DeltaPrefixSlicesTest(unittest.TestCase):
    def test_local_prefix_is_cut_and_node_witness_is_cut_plus_one(self) -> None:
        local_prefix, node_witness = ladder.delta_prefix_slices([10, 11, 12, 13], 3)

        self.assertEqual(local_prefix, [10, 11, 12])
        self.assertEqual(node_witness, [10, 11, 12, 13])
        self.assertEqual(node_witness[:3], local_prefix)

    def test_rejects_fixture_without_witness_token(self) -> None:
        with self.assertRaisesRegex(ladder.SessionAbort, "cannot build a 4-token witness"):
            ladder.delta_prefix_slices([10, 11, 12], 3)

    def test_rejects_mismatched_node_witness(self) -> None:
        with self.assertRaisesRegex(ladder.SessionAbort, "differs from the local"):
            ladder.validate_delta_witness([10, 11, 12], [10, 99, 12, 13], 3)


class ReplayGenerationAllocationTest(unittest.TestCase):
    @staticmethod
    def write_config(root: Path, highest: dict[str, int]) -> Path:
        ledger = root / "replay.json"
        ledger.write_text(
            json.dumps({"highest_generation": highest}), encoding="utf-8"
        )
        ledger.chmod(0o600)
        config = root / "cluster.json"
        config.write_text(json.dumps({"replay_ledger": "replay.json"}), encoding="utf-8")
        return config

    def test_stale_configured_generation_advances_above_live_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.write_config(Path(tmp), {"key:8": 960_205})
            args = SimpleNamespace(execute=True, cluster_config=config)

            first = ladder.resolve_first_generation(
                args, 950_500, handoffs=2, cell="warm-hit"
            )

        self.assertEqual(first, 960_206)

    def test_already_fresh_configured_generation_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.write_config(Path(tmp), {"key:8": 960_205})
            args = SimpleNamespace(execute=True, cluster_config=config)

            first = ladder.resolve_first_generation(
                args, 970_000, handoffs=6, cell="qualifier"
            )

        self.assertEqual(first, 970_000)

    def test_dry_run_does_not_read_or_mutate_a_ledger(self) -> None:
        args = SimpleNamespace(
            execute=False, cluster_config=Path("/definitely/missing/cluster.json")
        )

        first = ladder.resolve_first_generation(
            args, 950_700, handoffs=3, cell="delta"
        )

        self.assertEqual(first, 950_700)

    def test_rejects_a_world_readable_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.write_config(Path(tmp), {"key:8": 1})
            (Path(tmp) / "replay.json").chmod(0o644)

            with self.assertRaisesRegex(ladder.SessionAbort, "0600 or stricter"):
                ladder.replay_high_water(config)

    def test_stage5_threads_a_fresh_range_into_each_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                identity_prefix="attempt-7",
                fixture_65536=root / "65536.tokens",
                fixture_130815=root / "130815.tokens",
                stage5_first_generation=950_500,
                warmhit_base_url="http://127.0.0.1:8080",
                warmhit_bearer_token_file=root / "bearer",
                warmhit_miss_fixture=root / "miss.tokens",
                spark_host="spark",
                resident_container="resident",
                remote_sock="/run/muser/work/producer.sock",
                receiver_host="192.0.2.10",
                receiver_port=29590,
                warmhit_host_work="/host/work",
                execute=False,
            )
            with mock.patch.object(
                ladder,
                "resolve_first_generation",
                side_effect=(960_206, 960_208),
            ) as resolve, mock.patch.object(
                ladder, "_run_probe", return_value={"mode": "dry-run"}
            ) as run_probe:
                for index, (depth_name, fixture) in enumerate(
                    (("65536", args.fixture_65536), ("130815", args.fixture_130815))
                ):
                    ladder._stage5_probe(
                        args,
                        root / "stage5" / depth_name,
                        depth_name,
                        fixture,
                        index,
                    )

        self.assertEqual(
            [call.args[1] for call in resolve.call_args_list], [950_500, 950_600]
        )
        commands = [call.kwargs["command"] for call in run_probe.call_args_list]
        self.assertEqual(
            [command[command.index("--first-generation") + 1] for command in commands],
            ["960206", "960208"],
        )
        self.assertEqual(
            [command[command.index("--request-prefix") + 1] for command in commands],
            [
                "attempt-7-stage5-warmhit-65536",
                "attempt-7-stage5-warmhit-130815",
            ],
        )

    def test_stage5_dispatches_each_depth_to_a_fresh_server_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                fixture_65536=root / "65536.tokens",
                fixture_130815=root / "130815.tokens",
            )
            with mock.patch.object(
                ladder, "_stage5_depth", side_effect=({"depth": 65536}, {"depth": 130815})
            ) as run_depth:
                receipts = ladder.stage5(args, root)

        self.assertEqual(set(receipts), {"65536", "130815"})
        first = run_depth.call_args_list[0].args
        second = run_depth.call_args_list[1].args
        self.assertEqual(first[2:], ("65536", args.fixture_65536, 0))
        self.assertEqual(second[2:], ("130815", args.fixture_130815, 1))
        self.assertNotEqual(first[2], second[2])


class Stage3NativeContextLimitTest(unittest.TestCase):
    def test_preserves_headroom_below_the_checkpoint_limit(self) -> None:
        self.assertEqual(ladder.stage3_native_max_model_len(65_536), 66_560)

    def test_caps_131008_fixture_at_the_checkpoint_limit(self) -> None:
        self.assertEqual(ladder.stage3_native_max_model_len(131_008), 131_072)

    def test_rejects_fixture_without_room_for_the_bookkeeping_token(self) -> None:
        with self.assertRaisesRegex(
            ladder.SessionAbort, "plus 1 bookkeeping token exceeds"
        ):
            ladder.stage3_native_max_model_len(131_072)


class Stage3YardstickGateTest(unittest.TestCase):
    @staticmethod
    def comparison(fixture_id: str, length: int, passed: bool) -> dict[str, object]:
        return {
            "id": fixture_id,
            "regime": "long-context",
            "token_count": length,
            "native_vs_kquant": {
                "top_token_passed": passed,
                "perplexity_passed": True,
                "passed": passed,
            },
        }

    @staticmethod
    def write_report(path: Path, comparisons: list[dict[str, object]]) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": "muser.nvfp4-quant-yardstick.v1",
                    "status": "measured",
                    "comparisons": comparisons,
                }
            ),
            encoding="utf-8",
        )

    def test_accepts_exact_expected_comparisons_when_all_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yardstick.json"
            self.write_report(
                path,
                [
                    self.comparison("e2-rust-65536", 65_536, True),
                    self.comparison("e2-python-65536", 65_536, True),
                ],
            )

            report = ladder.validate_stage3_yardstick(
                path, 65_536, ("rust", "python")
            )

        self.assertEqual(report["status"], "measured")

    def test_retains_a_failed_native_comparison_for_the_aggregate_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yardstick.json"
            self.write_report(
                path,
                [
                    self.comparison("e2-rust-65536", 65_536, True),
                    self.comparison("e2-docs-65536", 65_536, False),
                ],
            )

            report = ladder.validate_stage3_yardstick(
                path, 65_536, ("rust", "docs")
            )

        self.assertFalse(report["comparisons"][1]["native_vs_kquant"]["passed"])

    def test_rejects_a_missing_expected_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yardstick.json"
            self.write_report(
                path,
                [
                    self.comparison("e2-rust-65536", 65_536, True),
                ],
            )

            with self.assertRaisesRegex(ladder.SessionAbort, "comparison set differs"):
                ladder.validate_stage3_yardstick(
                    path, 65_536, ("rust", "python")
                )

    def test_rejects_inconsistent_combined_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yardstick.json"
            comparison = self.comparison("e2-rust-65536", 65_536, True)
            comparison["native_vs_kquant"]["passed"] = False
            self.write_report(path, [comparison])

            with self.assertRaisesRegex(ladder.SessionAbort, "inconsistent"):
                ladder.validate_stage3_yardstick(path, 65_536, ("rust",))


class Stage3AggregateQualityPolicyTest(unittest.TestCase):
    @staticmethod
    def report(length: int, failures: set[str], documents: tuple[str, ...]) -> dict:
        return {
            "comparisons": [
                Stage3YardstickGateTest.comparison(
                    f"e2-{document}-{length}", length, document not in failures
                )
                for document in documents
            ]
        }

    def test_one_document_exceedance_is_published_sensitivity_not_failure(self) -> None:
        reports = {
            65_536: self.report(
                65_536, {"docs"}, ("rust", "python", "docs")
            ),
            131_008: self.report(131_008, set(), ("rust", "python")),
        }

        verdict = ladder.stage3_quality_verdict(
            reports, ("rust", "python", "docs")
        )

        self.assertEqual(verdict["status"], "pass")
        self.assertEqual(verdict["branch"], "content-sensitive-envelope")
        self.assertEqual(verdict["deepest_full_coverage_tokens"], 65_536)
        self.assertEqual(verdict["lengths"][0]["failed_documents"], ["docs"])
        self.assertEqual(verdict["lengths"][1]["missing_documents"], ["docs"])

    def test_replicated_exceedance_at_adjacent_depths_is_blocker(self) -> None:
        documents = ("rust", "python", "docs")
        reports = {
            65_536: self.report(65_536, {"rust", "docs"}, documents),
            131_008: self.report(131_008, {"rust", "python"}, documents),
        }

        verdict = ladder.stage3_quality_verdict(reports, documents)

        self.assertEqual(verdict["status"], "quality-blocker")
        self.assertEqual(verdict["first_persistent_exceedance_tokens"], 65_536)
        self.assertTrue(
            verdict["lengths"][0]["persistent_at_next_measured_length"]
        )


class Stage3CompactScratchTest(unittest.TestCase):
    def test_reference_capture_keeps_raw_logits_off_evidence_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                identity_prefix="attempt-5",
                stage3_scratch_root=root / "internal-scratch",
                execute=False,
                llama_perplexity_receipt=root / "llama-receipt.json",
                llama_perplexity_binary=root / "llama-perplexity",
                e2_source_dir=root / "source",
            )
            out_dir = root / "evidence"
            extended_dir = out_dir / "fixtures"
            manifest_path = extended_dir / "manifest.json"
            with mock.patch.object(
                ladder, "build_or_leased", return_value={"mode": "dry-run"}
            ) as leased:
                ladder._stage3_reference_lane(
                    args,
                    out_dir,
                    extended_dir,
                    manifest_path,
                    65_536,
                    ("rust",),
                    "kquant",
                    root / "model.gguf",
                    "a" * 64,
                )

            command = leased.call_args.kwargs["command"]
            logits = Path(command[command.index("--logits-out") + 1])
            compact = Path(command[command.index("--compact-teacher-output") + 1])

        self.assertTrue(logits.is_relative_to(args.stage3_scratch_root))
        self.assertFalse(logits.is_relative_to(out_dir))
        self.assertTrue(compact.is_relative_to(out_dir))
        self.assertIn("--discard-raw-after-compact", command)


class QuiesceRemoteProducerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(spark_host="spark", execute=True)

    def test_dry_run_prints_every_step_without_executing(self) -> None:
        self.args.execute = False
        with mock.patch.object(ladder.subprocess, "run") as run:
            result = ladder._quiesce_remote_producer(self.args, "resident")

        run.assert_not_called()
        self.assertEqual(result["mode"], "dry-run")
        self.assertIn("docker", result["stop_command"])
        self.assertIn("{{.State.Status}}", result["inspect_command"])
        self.assertIn("python3 -c", result["lease_probe_command"][-1])
        self.assertIn("fuser", result["holder_hint_command"])

    def test_stops_waits_for_exited_and_proves_lease_free(self) -> None:
        responses = [
            mock.Mock(returncode=0, stdout="resident\n", stderr=""),
            mock.Mock(returncode=0, stdout="exited\n", stderr=""),
            mock.Mock(returncode=0, stdout="LEASE FREE\n", stderr=""),
        ]
        with mock.patch.object(ladder.subprocess, "run", side_effect=responses) as run:
            result = ladder._quiesce_remote_producer(self.args, "resident")

        self.assertEqual(result["lease"], "free")
        self.assertEqual(run.call_args_list[0].args[0][-2:], ["stop", "resident"])
        self.assertIn("{{.State.Status}}", run.call_args_list[1].args[0])
        self.assertIn("python3 -c", run.call_args_list[2].args[0][-1])

    def test_held_lease_aborts_with_fuser_holder_hint(self) -> None:
        responses = [
            mock.Mock(returncode=0, stdout="resident\n", stderr=""),
            mock.Mock(returncode=0, stdout="exited\n", stderr=""),
            mock.Mock(returncode=1, stdout="", stderr="held"),
            mock.Mock(returncode=0, stdout="1234 python3\n", stderr=""),
        ]
        with mock.patch.object(ladder.subprocess, "run", side_effect=responses):
            with self.assertRaisesRegex(ladder.SessionAbort, "1234 python3"):
                ladder._quiesce_remote_producer(
                    self.args, "resident", grace_seconds=0
                )

    def test_shutdown_and_lease_release_each_receive_a_full_grace_budget(self) -> None:
        clock = {"now": 0.0}
        responses = iter(
            (
                (0.0, mock.Mock(returncode=0, stdout="resident\n", stderr="")),
                (9.5, mock.Mock(returncode=0, stdout="running\n", stderr="")),
                (9.9, mock.Mock(returncode=0, stdout="running\n", stderr="")),
                (9.9, mock.Mock(returncode=0, stdout="exited\n", stderr="")),
                (10.5, mock.Mock(returncode=1, stdout="", stderr="held")),
                (10.5, mock.Mock(returncode=0, stdout="LEASE FREE\n", stderr="")),
            )
        )

        def run(*_args, **_kwargs):
            timestamp, response = next(responses)
            clock["now"] = timestamp
            return response

        with (
            mock.patch.object(ladder.subprocess, "run", side_effect=run) as invoked,
            mock.patch.object(
                ladder.time, "monotonic", side_effect=lambda: clock["now"]
            ),
            mock.patch.object(ladder.time, "sleep"),
        ):
            result = ladder._quiesce_remote_producer(
                self.args, "resident", grace_seconds=10
            )

        self.assertEqual(result["lease"], "free")
        self.assertEqual(invoked.call_count, 6)


class Stage4SupervisorSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            spark_host="spark",
            resident_container="resident",
            naive_container="naive",
            node_restart_script="/node/restart_resident_producer.py",
            execute=True,
        )

    def test_supervisor_commands_use_stop_and_continue_signals(self) -> None:
        pause = ladder._stage4_supervisor_command(self.args, "STOP")
        resume = ladder._stage4_supervisor_command(self.args, "CONT")

        self.assertEqual(pause[:2], ["ssh", "spark"])
        self.assertIn("pkill -STOP", pause[-1])
        self.assertIn("[s]upervise_resident_producer[.]py", pause[-1])
        self.assertIn("pkill -CONT", resume[-1])
        with self.assertRaisesRegex(ValueError, "unsupported supervisor signal"):
            ladder._stage4_supervisor_command(self.args, "KILL")

    def test_abort_recovery_stops_active_naive_before_restarting_resident(self) -> None:
        with mock.patch.object(
            ladder,
            "_remote_container_status",
            side_effect=("running", "exited", "running"),
        ), mock.patch.object(
            ladder, "_quiesce_remote_producer", return_value={"lease": "free"}
        ) as quiesce, mock.patch.object(
            ladder.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run:
            recovery = ladder._recover_stage4_resident(self.args)

        quiesce.assert_called_once_with(self.args, "naive")
        self.assertTrue(
            any(
                item.endswith("restart_resident_producer.py")
                for item in run.call_args.args[0]
            )
        )
        self.assertTrue(recovery["resident_running"])
        self.assertEqual(recovery["resident_restart_exit"], 0)

    def test_abort_recovery_does_not_restart_an_already_running_resident(self) -> None:
        with mock.patch.object(
            ladder,
            "_remote_container_status",
            side_effect=("created", "running", "running"),
        ), mock.patch.object(ladder.subprocess, "run") as run:
            recovery = ladder._recover_stage4_resident(self.args)

        run.assert_not_called()
        self.assertTrue(recovery["resident_running"])
        self.assertIsNone(recovery["resident_restart_exit"])

    def test_swap_failure_still_recovers_and_resumes_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            ladder, "_stage4_swap_window", side_effect=ladder.SessionAbort("cell failed")
        ), mock.patch.object(
            ladder,
            "_recover_stage4_resident",
            return_value={"resident_running": True, "errors": []},
        ) as recover, mock.patch.object(
            ladder.subprocess,
            "run",
            side_effect=(mock.Mock(returncode=0), mock.Mock(returncode=0)),
        ) as run:
            with self.assertRaisesRegex(ladder.SessionAbort, "cell failed"):
                ladder._stage4_guarded_swap(self.args, Path(tmp))

        recover.assert_called_once_with(self.args)
        self.assertIn("pkill -STOP", run.call_args_list[0].args[0][-1])
        self.assertIn("pkill -CONT", run.call_args_list[1].args[0][-1])

    def test_recovery_exception_still_resumes_supervision_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            ladder, "_stage4_swap_window", return_value={}
        ), mock.patch.object(
            ladder, "_recover_stage4_resident", side_effect=RuntimeError("ssh lost")
        ), mock.patch.object(
            ladder.subprocess,
            "run",
            side_effect=(mock.Mock(returncode=0), mock.Mock(returncode=0)),
        ) as run:
            with self.assertRaisesRegex(ladder.SessionAbort, "operator recovery"):
                ladder._stage4_guarded_swap(self.args, Path(tmp))

        self.assertEqual(len(run.call_args_list), 2)
        self.assertIn("pkill -CONT", run.call_args_list[1].args[0][-1])


class NaiveSocketIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            spark_host="spark",
            resident_container="resident",
            naive_container="naive",
            remote_sock="/run/muser/work/producer.sock",
            naive_remote_sock="/run/muser/work/producer-naive.sock",
            node_checkpoint_dir="/node/checkpoint",
            node_engine_config="/node/config.json",
            node_pki_dir="/node/pki",
            node_work_dir="/node/work",
            node_receipts_dir="/node/receipts",
            naive_image="muser/naive:pinned",
            naive_startup_receipt="runtime-naive.json",
            naive_rope_cache="rope-naive.bin",
        )

    def test_create_and_replacement_guard_require_namespaced_socket(self) -> None:
        command = ladder._ensure_naive_container_command(self.args)

        self.assertEqual(command[:2], ["ssh", "spark"])
        self.assertIn("--user 1000:1000", command[2])
        self.assertIn("--sock /run/muser/work/producer-naive.sock", command[2])
        self.assertIn("grep -Fxq -- /run/muser/work/producer-naive.sock", command[2])
        self.assertIn("{{.Config.User}}", command[2])
        self.assertIn("grep -Fxq -- 1000:1000", command[2])
        self.assertIn("{{.Config.Image}}", command[2])
        self.assertIn("muser/naive:pinned", command[2])
        self.assertIn("{{json .Config.Entrypoint}}", command[2])
        self.assertIn("{{json .Config.Cmd}}", command[2])
        self.assertIn("{{json .HostConfig.Binds}}", command[2])
        self.assertIn("/node/checkpoint:/models/checkpoint:ro", command[2])
        self.assertIn("created|exited", command[2])
        self.assertIn("docker rm naive", command[2])
        self.assertIn("refusing to replace an active naive container", command[2])

    def test_stage4_selects_each_producers_own_socket(self) -> None:
        self.assertEqual(
            ladder._stage4_remote_sock(self.args, "naive"),
            "/run/muser/work/producer-naive.sock",
        )
        self.assertEqual(
            ladder._stage4_remote_sock(self.args, "resident"),
            "/run/muser/work/producer.sock",
        )

    def test_stage4_marks_only_the_naive_arm_as_pre_streaming(self) -> None:
        self.assertEqual(
            ladder._stage4_producer_profile_args(self.args, "naive"),
            ["--pre-streaming-control"],
        )
        self.assertEqual(
            ladder._stage4_producer_profile_args(self.args, "resident"), []
        )
        with self.assertRaisesRegex(ladder.SessionAbort, "unknown producer"):
            ladder._stage4_producer_profile_args(self.args, "foreign")


class LadderOrderingAndAbortReceiptTest(unittest.TestCase):
    def test_selected_order_puts_stage4_last(self) -> None:
        self.assertEqual(ladder.selected_stage_order(3, 6), (3, 5, 6, 4))
        self.assertEqual(ladder.selected_stage_order(4, 4), (4,))

    def test_stage4_requires_explicit_swap_authorization(self) -> None:
        args = SimpleNamespace(allow_producer_swap=False)
        with self.assertRaisesRegex(ladder.SessionAbort, "--allow-producer-swap"):
            ladder.require_stage_authorized(args, 4)
        args.allow_producer_swap = True
        ladder.require_stage_authorized(args, 4)

    def test_abort_receipt_records_containers_sockets_and_lease(self) -> None:
        responses = [
            mock.Mock(returncode=0, stdout="resident Up\n", stderr=""),
            mock.Mock(returncode=0, stdout="producer.sock\n", stderr=""),
            mock.Mock(returncode=1, stdout="LEASE HELD\n", stderr=""),
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            ladder.subprocess, "run", side_effect=responses
        ):
            receipt = Path(tmp) / "finish-stage6.log"
            ladder.append_node_state_receipt(
                receipt,
                spark_host="spark",
                node_work_dir="/node/work",
                context="stage-6-abort",
            )
            text = receipt.read_text()

        self.assertIn("[docker_ps]", text)
        self.assertIn("resident Up", text)
        self.assertIn("[sockets]", text)
        self.assertIn("producer.sock", text)
        self.assertIn("[lease_probe]", text)
        self.assertIn("LEASE HELD", text)
        self.assertIn("context=stage-6-abort", text)


class RerunIsolationTest(unittest.TestCase):
    def test_stage3_node_scorer_directory_is_unique_per_session_and_attempt(self) -> None:
        args = SimpleNamespace(
            identity_prefix="attempt-17/20260822T120000Z",
            node_results_root="/node/results",
        )

        first = ladder.stage3_node_out_dir(args, 65_536, 1)
        second = ladder.stage3_node_out_dir(args, 65_536, 2)

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("/node/results/kvpack-ladder-e2-65536-"))
        self.assertNotIn("/20260822", first)
        self.assertTrue(first.endswith("-a1"))
        self.assertTrue(second.endswith("-a2"))

    def test_stage5_evidence_file_advances_instead_of_reusing_failed_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "warmhit-65536.json"
            first = ladder.fresh_attempt_file(base)
            first.write_text("failed leg", encoding="utf-8")
            second = ladder.fresh_attempt_file(base)

        self.assertEqual(first.name, "warmhit-65536-a1.json")
        self.assertEqual(second.name, "warmhit-65536-a2.json")

    def test_stage2_attribution_arm_uses_a_fresh_qualify_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / ladder.STAGE_NAMES[2] / "eee-active-trace" / "qualify"
            stale.mkdir(parents=True)
            args = SimpleNamespace(
                execute=False,
                identity_prefix="attempt-3",
                stage2_first_generation=950_100,
                stage2_eee_off_first_generation=950_200,
                model=root / "model.gguf",
                fixture_130815=root / "prompt.tokens",
                cluster_config=root / "cluster.json",
                rope_cache=root / "rope.bin",
                resident_container="resident",
                remote_fixture_130815="/node/prompt.tokens",
                spark_host="spark",
                receiver_host="192.0.2.10",
                receiver_port=29590,
                remote_sock="/run/muser/work/producer.sock",
            )
            with mock.patch.object(
                ladder, "run_leased", return_value={"mode": "dry-run"}
            ) as run_leased:
                ladder.stage2(args, root)

        command = run_leased.call_args_list[0].kwargs["command"]
        qualify_out = Path(command[command.index("--out-dir") + 1])
        self.assertEqual(qualify_out.name, "qualify-r2")
        self.assertNotEqual(qualify_out, stale)


class Stage5ManifestEvidenceTest(unittest.TestCase):
    def test_headline_contradiction_is_returned_for_the_session_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                execute=True,
                identity_prefix="attempt-9",
                stage5_first_generation=960_000,
                warmhit_base_url="http://127.0.0.1:8080",
                warmhit_bearer_token_file=root / "bearer",
                warmhit_miss_fixture=root / "miss.tokens",
                spark_host="spark",
                resident_container="resident",
                remote_sock="/run/muser/work/producer.sock",
                receiver_host="192.0.2.10",
                receiver_port=29590,
                warmhit_host_work="/node/work",
            )

            def run_probe(**kwargs):
                command = kwargs["command"]
                evidence_path = Path(command[command.index("--out") + 1])
                evidence_path.write_text(
                    json.dumps(
                        {
                            "legs_valid": True,
                            "outputs_match": True,
                            "miss_control_valid": True,
                            "warm_ttft_below_cold": False,
                        }
                    ),
                    encoding="utf-8",
                )
                return {"exit_status": 0}

            stderr = io.StringIO()
            with (
                mock.patch.object(ladder, "resolve_first_generation", return_value=1),
                mock.patch.object(ladder, "_run_probe", side_effect=run_probe),
                redirect_stderr(stderr),
            ):
                result = ladder._stage5_probe(
                    args, root / "65536", "65536", root / "prompt.tokens", 0
                )

        self.assertFalse(result["warm_ttft_below_cold"])
        self.assertTrue(result["headline_contradiction"])
        self.assertEqual(
            Path(result["warmhit_evidence_path"]).name,
            "warmhit-65536-a1.json",
        )
        self.assertIn("headline-contradicting result", stderr.getvalue())


class Stage2MeritEvidenceTest(unittest.TestCase):
    def test_hardcoded_summary_deterministic_field_is_not_a_gate(self) -> None:
        summary = {
            "remote_ttft_cv": 0.01,
            "installed_payload_gbps_min": 3.2,
            "installed_payload_gbps": [3.2, 3.4],
            "deterministic": False,
        }

        failures = ladder.stage2_merit_failures({"exit_status": 0}, summary)

        self.assertEqual(failures, [])

    def test_load_bearing_cv_and_link_gates_remain(self) -> None:
        failures = ladder.stage2_merit_failures(
            {"exit_status": 1},
            {
                "remote_ttft_cv": 0.03,
                "installed_payload_gbps_min": 2.9,
                "installed_payload_gbps": [2.9],
            },
        )

        self.assertEqual(len(failures), 3)


class SessionManifestDisciplineTest(unittest.TestCase):
    def test_unexpected_stage_exception_records_node_state_then_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                pause_after_stage=[],
                confirm_stage=[],
                from_stage=3,
                to_stage=3,
                out_root=root,
                execute=False,
                identity_prefix="attempt-error",
                spark_host="offline-node",
                node_work_dir="/node/work",
            )
            error = json.JSONDecodeError("broken evidence", "{", 0)
            stage_functions = dict(ladder.STAGE_FUNCS)
            stage_functions[3] = mock.Mock(side_effect=error)
            with (
                mock.patch.object(ladder, "parse_args", return_value=args),
                mock.patch.object(ladder, "check_lock_not_held"),
                mock.patch.object(ladder, "check_qualifier_has_metal"),
                mock.patch.object(ladder, "check_producer_healthy"),
                mock.patch.object(ladder, "STAGE_FUNCS", stage_functions),
                mock.patch.object(ladder, "append_node_state_receipt") as node_state,
                self.assertRaises(json.JSONDecodeError),
            ):
                ladder.main([])

            manifest = json.loads((root / "session-manifest.json").read_text())

        node_state.assert_called_once()
        abort = manifest["stages"]["3"]
        self.assertEqual(abort["error_type"], "JSONDecodeError")
        self.assertTrue(abort["unexpected_error"])
        self.assertIn("node-state-on-abort", abort["node_state_receipt"])

    def test_resume_preserves_the_complete_previous_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session-manifest.json"
            previous = {
                "schema": "muser.kvpack-ladder-session.v1",
                "identity_prefix": "attempt-one",
                "stages": {"5": {"aborted": True, "reason": "timeout"}},
                "aborted_at_stage": 5,
            }
            path.write_text(json.dumps(previous), encoding="utf-8")
            args = SimpleNamespace(execute=False, identity_prefix="attempt-two")

            current = ladder.new_session_manifest(args, path)
            ladder.write_session_manifest(path, current)
            third = ladder.new_session_manifest(
                SimpleNamespace(execute=False, identity_prefix="attempt-three"),
                path,
            )

        self.assertEqual(current["prior_runs"], [previous])
        self.assertEqual(len(third["prior_runs"]), 2)
        self.assertEqual(
            third["prior_runs"][0]["stages"]["5"]["reason"], "timeout"
        )


class LadderCommandSurfaceTest(unittest.TestCase):
    @staticmethod
    def required_args() -> list[str]:
        return [
            "--resident-container", "resident",
            "--cluster-config", "cluster.json",
            "--rope-cache", "rope.bin",
            "--model", "model.gguf",
            "--bench", "muser-bench",
            "--fixture-8192", "8192.tokens",
            "--remote-fixture-8192", "/node/8192.tokens",
            "--fixture-65536", "65536.tokens",
            "--remote-fixture-65536", "/node/65536.tokens",
            "--fixture-130815", "130815.tokens",
            "--remote-fixture-130815", "/node/130815.tokens",
            "--e2-source-dir", "e2",
            "--e2-receipt", "e2.json",
            "--vllm-engine-config", "engine.json",
            "--checkpoint-artifact-sha256", "a" * 64,
            "--kquant-model", "kquant.gguf",
            "--kquant-model-sha256", "b" * 64,
            "--llama-perplexity-binary", "llama-perplexity",
            "--llama-perplexity-receipt", "llama.json",
            "--warmhit-base-url", "http://127.0.0.1:8080",
            "--warmhit-bearer-token-file", "bearer",
            "--warmhit-miss-fixture", "miss.tokens",
            "--warmhit-host-work", "/node/work",
            "--spark-host", "node-a",
            "--receiver-host", "192.0.2.10",
            "--alternate-model", "alt.gguf",
            "--node-checkpoint-dir", "/node/checkpoints",
            "--node-engine-config", "/node/engine.json",
            "--node-work-root", "/node/work-root",
            "--node-results-root", "/node/results",
            "--node-pki-dir", "/node/pki",
            "--node-work-dir", "/node/work",
            "--node-receipts-dir", "/node/receipts",
            "--node-restart-script", "/node/restart.py",
        ]

    def test_dead_e1_yardstick_and_local_lease_args_are_removed(self) -> None:
        args = ladder.parse_args(self.required_args())
        self.assertFalse(hasattr(args, "e1_yardstick"))
        self.assertFalse(hasattr(args, "lease_file"))
        self.assertEqual(args.spark_host, "node-a")
        for removed in ("--e1-yardstick", "--lease-file"):
            with self.subTest(removed=removed), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    ladder.parse_args(self.required_args() + [removed, "unused"])

    def test_planning_docstring_names_live_preflight_and_stage3_restart(self) -> None:
        self.assertIn("not fully\noffline", ladder.__doc__)
        self.assertIn("stage-3 stop/restart scoring windows", ladder.__doc__)
        self.assertNotIn("never kills or force-restarts", ladder.__doc__)

    def test_stage6_note_cites_the_protocol_not_a_stale_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                execute=False,
                fixture_65536=root / "65536.tokens",
                stage6_remote_prefix_fixture="/run/muser/work/prefix.tokens",
                identity_prefix="attempt-10",
                stage6_first_generation=950_700,
                model=root / "model.gguf",
                cluster_config=root / "cluster.json",
                rope_cache=root / "rope.bin",
                resident_container="resident",
                remote_fixture_65536="/node/65536.tokens",
                spark_host="spark",
                receiver_host="192.0.2.10",
                receiver_port=29590,
                remote_sock="/run/muser/work/producer.sock",
            )
            output = io.StringIO()
            with (
                mock.patch.object(
                    ladder, "run_leased", return_value={"mode": "dry-run"}
                ),
                redirect_stdout(output),
            ):
                ladder.stage6(args, root)

        self.assertIn("run_delta_probe's receiver-arming protocol", output.getvalue())
        self.assertNotIn("f7f43a4", output.getvalue())


if __name__ == "__main__":
    unittest.main()
