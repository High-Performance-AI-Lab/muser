from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atomic_seal_campaign
import audit_release
import build_feature_freeze
import evaluate_dflash
import evaluate_baseline
import freeze_dflash_tuning
import release_host_preflight
import release_identity
from release_readiness import (
    MANDATORY,
    bind_lane_report,
    lane_execution_config_sha256,
    substitute_lane_command,
    validate_lane_report,
)
import verify_release_candidate


ROOT = Path(__file__).resolve().parents[2]


class ReleaseToolTests(unittest.TestCase):
    def test_containment_allows_only_the_operator_gated_beta_marker(self) -> None:
        lock = {
            "state": "containment",
            "sealing_enabled": False,
            "candidate_creation_enabled": False,
            "tagging_enabled": True,
            "tagging_policy": copy.deepcopy(
                audit_release.CONTAINMENT_MARKER_TAG_POLICY
            ),
            "publishing_enabled": False,
        }
        self.assertTrue(audit_release.containment_lock_is_safe(lock))

        for mutation in (
            lambda value: value.update(sealing_enabled=True),
            lambda value: value.update(candidate_creation_enabled=True),
            lambda value: value.update(publishing_enabled=True),
            lambda value: value["tagging_policy"]["allowed_tags"].append("v0.1.0"),
            lambda value: value["tagging_policy"].update(operator_go_required=False),
        ):
            widened = copy.deepcopy(lock)
            mutation(widened)
            self.assertFalse(audit_release.containment_lock_is_safe(widened))

    def test_campaign_identity_binds_frozen_nvfp4_runtime(self) -> None:
        self.assertIn(
            "release/nvfp4-runtime-identity-v1.json", release_identity.IDENTITY_FILES
        )

    def test_host_preflight_requires_serial_metal_tests_under_accelerator_safe(
        self,
    ) -> None:
        matrix = {
            "preflight": {
                "serial_metal_tests": dict(
                    release_host_preflight.SERIAL_METAL_TEST_POLICY
                )
            }
        }
        self.assertEqual(
            release_host_preflight.validate_serial_metal_test_policy(matrix),
            release_host_preflight.SERIAL_METAL_TEST_POLICY,
        )
        matrix["preflight"]["serial_metal_tests"] = {
            **release_host_preflight.SERIAL_METAL_TEST_POLICY,
            "argv": ["cargo", "test", "--workspace"],
        }
        with self.assertRaisesRegex(RuntimeError, "test-threads=1"):
            release_host_preflight.validate_serial_metal_test_policy(matrix)

    def test_operational_sources_do_not_reference_the_retired_producer_alias(
        self,
    ) -> None:
        retired = "dgx" + "-spark"
        roots = [
            ROOT / "AGENTS.md",
            ROOT / "docs" / "gx10-return-runbook-2026-08.md",
            ROOT / "scripts",
            ROOT / "crates" / "muser-server" / "src",
            ROOT / "web",
        ]
        for root in roots:
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if not path.is_file() or path.suffix not in {
                    ".html",
                    ".md",
                    ".py",
                    ".rs",
                    ".sh",
                }:
                    continue
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertNotIn(retired, path.read_text(encoding="utf-8"))

    def test_operational_sources_do_not_default_to_the_retired_direct_link(
        self,
    ) -> None:
        retired_hosts = ("10." + "77." + "0." + "1", "10." + "77." + "0." + "2")
        paths = [
            ROOT / "AGENTS.md",
            ROOT / "scripts" / "qualify_nvfp4_fast.py",
            ROOT / "scripts" / "qualify_nvfp4_link_p4.py",
            ROOT / "scripts" / "qualify_nvfp4_p4.py",
            ROOT / "scripts" / "qualify_nvfp4_streaming.py",
            ROOT / "scripts" / "run_kvpack_ladder_session.py",
            ROOT / "scripts" / "gx10" / "README.md",
            ROOT / "scripts" / "gx10" / "tcp_probe.py",
            ROOT / "scripts" / "gx10" / "vllm" / "warmhit_probe.py",
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                for retired in retired_hosts:
                    self.assertNotIn(retired, source)

    def test_feature_freeze_plan_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_feature_freeze.py"),
                    "--matrix-config", "/missing/matrix.json",
                    "--out", str(output),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["mode"], "plan")
            self.assertTrue(report["offline_build"])
            self.assertFalse(report["seals_emitted"])
            self.assertFalse(output.exists())

    def test_feature_freeze_requires_zero_findings_and_v0_1_metal_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix = Path(temporary) / "matrix.json"
            matrix.write_text(
                json.dumps(
                    {
                        "schema": "muser.unsealed-matrix-config.v1",
                        "lanes": {lane: {} for lane in MANDATORY},
                    }
                )
            )
            values = {
                "release/feature-contract-v1.json": {"status": "frozen"},
                "release/findings-v1.json": {
                    "findings": [{"id": "done", "status": "closed"}]
                },
                "release/dflash-tuning-v1.json": {
                    "status": "frozen",
                    "selected_verify_length": 7,
                    "allowed_verify_lengths": [3, 7, 15],
                },
                "release/dflash-route-policy-v1.json": {
                    "schema": "muser.dflash-route-policy.v1",
                    "status": "v0.1-metal-only",
                    "auto_route": "metal",
                    "ane_gate": {
                        "required": False,
                        "passed": False,
                        "same_build_receipt": None,
                    },
                    "policy": "v0.1 auto routing is permanently Metal",
                },
                "release/release-lock.json": {"sealing_enabled": False},
            }
            with mock.patch.object(
                build_feature_freeze, "load", side_effect=lambda name: values[name]
            ):
                build_feature_freeze.validate_frozen_state(matrix)
                values["release/findings-v1.json"] = {
                    "findings": [{"id": "open", "status": "open"}]
                }
                with self.assertRaisesRegex(RuntimeError, "open findings"):
                    build_feature_freeze.validate_frozen_state(matrix)
                values["release/findings-v1.json"] = {
                    "findings": [{"id": "done", "status": "closed"}]
                }
                values["release/dflash-route-policy-v1.json"]["ane_gate"]["passed"] = True
                with self.assertRaisesRegex(RuntimeError, "Metal-only"):
                    build_feature_freeze.validate_frozen_state(matrix)
                values["release/dflash-route-policy-v1.json"]["ane_gate"]["passed"] = False
                values["release/dflash-route-policy-v1.json"]["unexpected"] = "ignored-before"
                with self.assertRaisesRegex(RuntimeError, "Metal-only"):
                    build_feature_freeze.validate_frozen_state(matrix)

    def test_candidate_verifier_accepts_the_complete_atomic_bundle_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = {
                "schema": "muser.campaign-identity.v3",
                "source": {"commit": "1" * 40, "tree": "2" * 40},
                "files": {},
                "cargo_metadata_sha256": "3" * 64,
                "apparatus_sha256": "4" * 64,
                "binaries": {},
            }
            encoded = json.dumps(
                campaign, sort_keys=True, separators=(",", ":")
            ).encode()
            campaign["digest"] = hashlib.sha256(encoded).hexdigest()
            digest = campaign["digest"]
            writer = (
                "import json,sys; "
                "open(sys.argv[1], 'w').write(json.dumps({"
                "'schema':'muser.unsealed-qualification.v1','status':'passed',"
                "'seal_eligible':False,'identity':sys.argv[2],'lane':sys.argv[3]}))"
            )
            lanes = {
                lane: {
                    "check_argv": ["true"],
                    "argv": [
                        sys.executable, "-c", writer, "{output}", "{identity}", lane
                    ],
                }
                for lane in MANDATORY
            }
            config = {"schema": "muser.unsealed-matrix-config.v1", "lanes": lanes}
            config_path = root / "matrix.json"
            config_path.write_text(json.dumps(config))
            readiness = {
                "schema": "muser.release-readiness.v1",
                "status": "passed",
                "identity": digest,
                "matrix_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "lanes": {lane: {} for lane in MANDATORY},
            }
            readiness_path = root / "readiness.json"
            readiness_path.write_text(json.dumps(readiness))
            bundle = root / "built-bundle"
            args = argparse.Namespace(
                out=bundle,
                readiness=readiness_path,
                matrix_config=config_path,
            )
            with mock.patch.object(atomic_seal_campaign, "require_sealing_enabled"):
                result = atomic_seal_campaign.run_campaign(args, readiness, config, campaign)

            candidate = root / "candidate"
            evidence = candidate / "evidence"
            evidence.mkdir(parents=True)
            shutil.copytree(bundle, evidence / "atomic-seal-bundle")
            receipt = {
                "identity": digest,
                "atomic_seal_identity": digest,
                "atomic_seal_result_sha256": hashlib.sha256(
                    (bundle / "RESULT.json").read_bytes()
                ).hexdigest(),
            }
            failures: list[str] = []
            verify_release_candidate.validate_atomic_seal_bundle(
                candidate, digest, receipt, failures
            )
            self.assertEqual(failures, [])
            self.assertTrue(result["fresh_re_evaluation"])

    def test_unsealed_reports_are_bound_to_the_named_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            log = Path(temporary) / "security.log"
            log.write_text("passed\n")
            report.write_text(
                json.dumps(
                    {
                        "schema": "muser.unsealed-qualification.v1",
                        "lane": "security",
                        "status": "passed",
                        "seal_eligible": False,
                        "identity": "f" * 64,
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "execution provenance"):
                validate_lane_report(report, "security", "f" * 64)
            bind_lane_report(
                report,
                "security",
                "f" * 64,
                matrix_config_sha256="e" * 64,
                command_template=["qualify", "{output}"],
                command=["qualify", str(report)],
                log_path=log,
                runner="scripts/run_unsealed_release_matrix.py",
            )
            validate_lane_report(report, "security", "f" * 64)
            with self.assertRaisesRegex(RuntimeError, "different matrix"):
                validate_lane_report(
                    report,
                    "security",
                    "f" * 64,
                    matrix_config_sha256="d" * 64,
                )
            log.write_text("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "log is missing or mismatched"):
                validate_lane_report(
                    report,
                    "security",
                    "f" * 64,
                    log_path=log,
                )
            with self.assertRaisesRegex(RuntimeError, "lane-bound"):
                validate_lane_report(report, "migration", "f" * 64)

    def test_lane_scoped_binding_survives_unrelated_matrix_composition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator_output = root / "original-remote.json"
            report = root / "remote.json"
            log = root / "remote.log"
            payload = json.dumps(
                {
                    "schema": "muser.unsealed-qualification.v1",
                    "lane": "remote",
                    "status": "passed",
                    "seal_eligible": False,
                    "identity": "f" * 64,
                }
            )
            evaluator_output.write_text(payload)
            report.write_text(payload)
            log.write_text("retained packet revalidated\n")
            lane_config = {
                "argv": [
                    "qualify",
                    "--identity={identity}",
                    "--evidence={output_dir}/remote-evidence",
                    "--out={output}",
                ],
                "readiness_runners": [
                    "scripts/run_unsealed_release_matrix.py",
                    "scripts/run_nvfp4_text_matrix.py",
                ],
            }
            command = substitute_lane_command(
                lane_config["argv"], "f" * 64, evaluator_output
            )
            bind_lane_report(
                report,
                "remote",
                "f" * 64,
                matrix_config_sha256=None,
                lane_config=lane_config,
                command_template=lane_config["argv"],
                command=command,
                log_path=log,
                runner="scripts/run_nvfp4_text_matrix.py",
                evaluator_output_path=evaluator_output,
            )
            bound = validate_lane_report(
                report,
                "remote",
                "f" * 64,
                lane_config=lane_config,
                command_template=lane_config["argv"],
                command=command,
                log_path=log,
                runner=lane_config["readiness_runners"],
            )
            self.assertEqual(
                bound["execution_provenance"]["lane_execution_config_sha256"],
                lane_execution_config_sha256(lane_config),
            )
            changed = {**lane_config, "argv": ["different", "{output}"]}
            with self.assertRaisesRegex(RuntimeError, "different lane execution"):
                validate_lane_report(
                    report,
                    "remote",
                    "f" * 64,
                    lane_config=changed,
                )
            evaluator_output.write_text("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "evaluator output"):
                validate_lane_report(report, "remote", "f" * 64)

    def test_atomic_campaign_freshly_reruns_every_lane_before_one_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = "a" * 64
            writer = (
                "import json,sys; "
                "open(sys.argv[1], 'w').write(json.dumps({"
                "'schema':'muser.unsealed-qualification.v1','status':'passed',"
                "'seal_eligible':False,'identity':sys.argv[2],'lane':sys.argv[3]}))"
            )
            lanes = {
                lane: {
                    "check_argv": ["true"],
                    "argv": [
                        sys.executable, "-c", writer, "{output}", "{identity}", lane
                    ],
                }
                for lane in MANDATORY
            }
            config = {"schema": "muser.unsealed-matrix-config.v1", "lanes": lanes}
            config_path = root / "matrix.json"
            readiness_path = root / "readiness.json"
            config_path.write_text(json.dumps(config))
            readiness = {"identity": digest}
            readiness_path.write_text(json.dumps(readiness))
            output = root / "atomic-bundle"
            args = argparse.Namespace(
                out=output,
                readiness=readiness_path,
                matrix_config=config_path,
            )
            with mock.patch.object(atomic_seal_campaign, "require_sealing_enabled"):
                result = atomic_seal_campaign.run_campaign(
                    args, readiness, config, {"digest": digest}
                )
            self.assertTrue(result["fresh_re_evaluation"])
            self.assertEqual(set(result["lanes"]), MANDATORY)
            self.assertTrue((output / "MANIFEST.json").is_file())
            self.assertEqual(len(list((output / "lanes").glob("*.json"))), len(MANDATORY))
            self.assertFalse(any(root.glob(".atomic-bundle.tmp-*")))

    def test_atomic_campaign_failure_exposes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = "b" * 64
            lanes = {
                lane: {"check_argv": ["true"], "argv": ["false", "{output}"]}
                for lane in MANDATORY
            }
            config = {"schema": "muser.unsealed-matrix-config.v1", "lanes": lanes}
            config_path = root / "matrix.json"
            readiness_path = root / "readiness.json"
            config_path.write_text(json.dumps(config))
            readiness = {"identity": digest}
            readiness_path.write_text(json.dumps(readiness))
            output = root / "atomic-bundle"
            args = argparse.Namespace(
                out=output,
                readiness=readiness_path,
                matrix_config=config_path,
            )
            with mock.patch.object(atomic_seal_campaign, "require_sealing_enabled"):
                with self.assertRaises(RuntimeError):
                    atomic_seal_campaign.run_campaign(
                        args, readiness, config, {"digest": digest}
                    )
            self.assertFalse(output.exists())
            self.assertFalse(any(root.glob(".atomic-bundle.tmp-*")))

    def test_atomic_campaign_revalidates_readiness_reports_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = "c" * 64
            lane_config = {
                lane: {
                    "check_argv": ["true"],
                    "argv": ["qualify", lane, "{identity}", "{output}"],
                    **(
                        {
                            "readiness_runners": [
                                "scripts/run_unsealed_release_matrix.py",
                                "scripts/run_nvfp4_text_matrix.py",
                            ]
                        }
                        if lane == "remote"
                        else {}
                    ),
                }
                for lane in MANDATORY
            }
            config = {"schema": "muser.unsealed-matrix-config.v1", "lanes": lane_config}
            config_path = root / "matrix.json"
            config_path.write_text(json.dumps(config))
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            readiness_lanes = {}
            for lane in MANDATORY:
                report = root / f"{lane}.json"
                log = root / f"{lane}.log"
                report.write_text(
                    json.dumps(
                        {
                            "schema": "muser.unsealed-qualification.v1",
                            "lane": lane,
                            "status": "passed",
                            "seal_eligible": False,
                            "identity": digest,
                        }
                    )
                )
                log.write_text(f"{lane} passed\n")
                evaluator_output = report
                if lane == "remote":
                    evaluator_output = root / "remote-evaluator.json"
                    evaluator_output.write_bytes(report.read_bytes())
                command = ["qualify", lane, digest, str(evaluator_output)]
                bind_lane_report(
                    report,
                    lane,
                    digest,
                    matrix_config_sha256=(None if lane == "remote" else config_sha),
                    lane_config=(lane_config[lane] if lane == "remote" else None),
                    command_template=lane_config[lane]["argv"],
                    command=command,
                    log_path=log,
                    runner=(
                        "scripts/run_nvfp4_text_matrix.py"
                        if lane == "remote"
                        else "scripts/run_unsealed_release_matrix.py"
                    ),
                    evaluator_output_path=evaluator_output,
                )
                readiness_lanes[lane] = {
                    "path": str(report),
                    "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                }
            readiness = {
                "schema": "muser.release-readiness.v1",
                "status": "passed",
                "identity": digest,
                "matrix_config_sha256": config_sha,
                "lanes": readiness_lanes,
            }
            readiness_path = root / "readiness.json"
            readiness_path.write_text(json.dumps(readiness))
            args = argparse.Namespace(
                binary=[],
                readiness=readiness_path,
                matrix_config=config_path,
                out=root / "bundle",
            )
            with mock.patch.object(
                atomic_seal_campaign, "identity", return_value={"digest": digest}
            ):
                atomic_seal_campaign.validate(args)
                (root / "security.log").write_text("tampered\n")
                with self.assertRaisesRegex(RuntimeError, "log is missing or mismatched"):
                    atomic_seal_campaign.validate(args)

    def test_repository_release_audit_passes(self) -> None:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "audit_release.py")],
            check=True,
            stdout=subprocess.PIPE,
        )

    def test_compute_plan_is_inspected_under_release_compute_units(self) -> None:
        source = (ROOT / "scripts" / "coreml_plan_receipt.py").read_text()
        self.assertIn("compute_units=ct.ComputeUnit.CPU_AND_NE", source)
        self.assertIn('"plan_compute_units": "CPU_AND_NE"', source)
        self.assertIn('"schema": "muser-coreml-compute-plan-v4"', source)

    def test_accelerator_dry_run_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "accelerator_safe.py"),
                    "--identity",
                    "test-identity",
                    "--cell",
                    "dry-run-cell",
                    "--out-dir",
                    str(output),
                    "--",
                    "true",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertEqual(json.loads(result.stdout)["mode"], "dry-run")
            self.assertFalse(output.exists())

    def test_release_candidate_dry_run_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_release_candidate.py"),
                    "--identity",
                    "sha256:" + "1" * 64,
                    "--output-dir",
                    str(output),
                    "--muser-binary",
                    "/missing/muser",
                    "--mtmd-package",
                    "/missing/mtmd",
                    "--gx10-container-receipt",
                    "/missing/gx10.json",
                    "--llama-comparator-receipt",
                    "/missing/llama.json",
                    "--seal-bundle",
                    "/missing/atomic-seal-bundle",
                    "--dry-run",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertFalse(report["models_bundled"])
            self.assertFalse(report["publishes_or_tags"])
            self.assertEqual(
                report["private_tags"],
                {
                    "muser": "muser-v0.1.0-beta.1",
                    "kvpack": "kvpack-v0.1.0-alpha.2",
                },
            )
            self.assertFalse(any("lib/ane/" in output for output in report["outputs"]))
            self.assertFalse(output.exists())

    def test_cleanroom_candidate_plan_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/verify_release_candidate_cleanroom.py"),
                    "--candidate", "/missing/candidate",
                    "--target", "/missing/target",
                    "--vision", "/missing/vision",
                    "--dflash", "/missing/dflash",
                    "--out", str(output),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["mode"], "plan")
            self.assertTrue(report["offline_cargo"])
            self.assertTrue(report["requires_accelerator_lease"])
            self.assertFalse(output.exists())

    def test_release_demo_verifies_all_three_external_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            arguments = []
            for kind in ("target", "vision", "dflash"):
                path = root / f"{kind}.gguf"
                path.write_bytes(f"pinned-{kind}".encode())
                artifacts[kind] = {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                arguments.extend([f"--{kind}", str(path)])
            manifest = root / "release-artifacts.json"
            manifest.write_text(json.dumps({"artifacts": artifacts}))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "release_demo.py"),
                    "--manifest",
                    str(manifest),
                    *arguments,
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(
                [artifact["status"] for artifact in report["artifacts"]],
                ["verified", "verified", "verified"],
            )

    def test_release_candidate_verifier_fails_closed_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "verify_release_candidate.py"),
                    temporary,
                ],
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["executes_bundled_code"])
            self.assertTrue(report["failures"])

    def test_gx10_container_dry_run_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_gx10_container.py"),
                    "--host",
                    "node-a",
                    "--llama-revision",
                    "1" * 40,
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertFalse(report["gpu_requested"])
            self.assertFalse(output.exists())

    def test_mtmd_bridge_dry_run_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            result = subprocess.run(
                [
                    str(ROOT / "scripts" / "build_mtmd_bridge.sh"),
                    "--llama-dir",
                    "/missing/llama.cpp",
                    "--revision",
                    "2" * 40,
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertFalse(report["accelerator_touched"])
            self.assertFalse(output.exists())

    def test_artifact_validator_rejects_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "target.gguf"
            artifact.write_bytes(b"not the pinned model")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_release_artifacts.py"),
                    "--artifact",
                    f"target={artifact}",
                ],
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stdout)["status"], "size-mismatch")

    def test_matrix_is_complete_and_dry_run_only(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "release_matrix.py"),
                "--identity",
                "test-identity",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        report = json.loads(result.stdout)
        self.assertFalse(report["accelerator_touched"])
        self.assertEqual(len(report["cells"]), 18)
        self.assertEqual(report["cells"][0]["depth"], 128)
        self.assertEqual(report["cells"][-1]["depth"], 131008)
        self.assertEqual(
            set(report["lanes"]),
            {
                "correctness",
                "synthetic_baseline",
                "warm_server_ttft",
                "vision",
                "kvpack",
                "dflash_metal",
                "dflash_ane",
                "remote_prefill",
            },
        )
        self.assertFalse(report["seal"]["eligible"])
        self.assertFalse(report["seal"]["mixed_routes_allowed"])
        dflash = report["lanes"]["dflash_metal"]
        self.assertEqual(len(dflash["tuning_cells"]), 12)
        self.assertEqual(len(dflash["qualification_cells"]), 8)
        self.assertEqual(len(report["lanes"]["dflash_ane"]["qualification_cells"]), 8)
        self.assertFalse(report["lanes"]["dflash_ane"]["required"])
        remote = report["lanes"]["remote_prefill"]
        self.assertEqual(remote["variants"], ["text"])
        self.assertEqual(len(remote["cells"]), 4)
        self.assertEqual(remote["cells"][-1]["output_tokens"], 48)
        vision = report["lanes"]["vision"]["cells"]
        self.assertEqual(len(vision), 4)
        self.assertTrue(all(len(cell["commands"]) == 3 for cell in vision))
        self.assertTrue(
            all(
                "--server-binary" in command
                and "--model-path" in command
                and "--mmproj" in command
                for cell in vision
                for command in cell["commands"][1:]
            )
        )
        self.assertEqual(
            {cell["verify_length"] for cell in dflash["tuning_cells"]},
            {3, 7, 15},
        )

    def test_campaign_dry_run_is_complete_and_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.gguf"
            model.write_bytes(b"dry-run identity only")
            output = root / "must-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "campaign.py"),
                    "--dry-run",
                    "--identity",
                    "test-candidate",
                    "--model",
                    str(model),
                    "--muser-bench",
                    "true",
                    "--muser-ane",
                    "true",
                    "--llama-bench",
                    "true",
                    "--out-dir",
                    str(output),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["cell_count"], 145)
            self.assertEqual(report["commands"], 188)
            self.assertEqual(report["identity"]["schema"], "muser.campaign-identity.v3")
            self.assertTrue(report["identity"].get("preview_only", False))
            self.assertFalse(report["accelerator_touched"])
            blockers = "\n".join(report["blockers"])
            self.assertIn("missing official DFlash artifact", blockers)
            self.assertNotIn("missing ANE qualification artifact", blockers)
            self.assertIn("missing remote cluster config", blockers)
            self.assertIn("--llama-receipt is required", blockers)
            self.assertFalse(output.exists())

    def test_remote_evaluator_requires_complete_overlap_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = "sha256:" + "8" * 64
            remote = root / "remote.jsonl"
            baseline = root / "baseline.jsonl"
            records = []
            for depth in (8192, 32768, 65536, 131008):
                variant = "text"
                records.append(
                    {
                        "identity": identity,
                        "cell": f"remote-{variant}-{depth}",
                        "engine": "remote",
                        "status": "passed",
                        "raw_ns": [100, 100, 100],
                        "local_ttft_raw_ns": [200, 200, 200],
                        "cv": 0.0,
                        "local_ttft_cv": 0.0,
                        "decode_ratios": [1.0, 1.0, 1.0],
                        "producer_export_overhead_ratios": [0.01, 0.01, 0.01],
                        "producer_first_tile_prefill_fractions": [0.10, 0.10, 0.10],
                        "producer_transfer_hidden_ratios": [0.96, 0.96, 0.96],
                        "installed_payload_gbps": [4.0, 4.0, 4.0],
                        "installed_payload_gbps_cv": 0.0,
                        "dflash_acceptance_ratios": None,
                        "dflash_acceptance": None,
                        "fingerprint": {
                            "variant": variant,
                            "prompt_positions": depth,
                            "output_tokens": 48 if depth == 131008 else 256,
                            "generated_tokens_sha256": "a" * 64,
                            "full_logit_digest": "b" * 64,
                        },
                    }
                )
            remote.write_text("".join(json.dumps(record) + "\n" for record in records))
            llama_means = {8192: 120, 32768: 220, 65536: 230, 131008: 260}
            baseline.write_text("".join(
                json.dumps({
                    "identity": identity,
                    "cell": f"ttft-{depth}",
                    "engine": "ttft-llama",
                    "status": "passed",
                    "raw_ns": [value] * 5,
                }) + "\n"
                for depth, value in llama_means.items()
            ))
            output = root / "seal.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_remote.py"),
                    "--ledger", str(remote),
                    "--baseline-ledger", str(baseline),
                    "--identity", identity,
                    "--out", str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            receipt = json.loads(output.read_text())
            self.assertFalse(receipt["seal_eligible"])
            self.assertTrue(receipt["would_be_seal_eligible"])
            self.assertEqual(receipt["schema"], "muser.unsealed-qualification.v1")

            records[0]["installed_payload_gbps"] = [2.9, 2.9, 2.9]
            remote.write_text("".join(json.dumps(record) + "\n" for record in records))
            refused = root / "refused.json"
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "evaluate_remote.py"),
                    "--ledger", str(remote), "--baseline-ledger", str(baseline),
                    "--identity", identity, "--out", str(refused),
                ],
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(
                any(
                    "link median is below 3.0 Gbps" in failure
                    for failure in json.loads(refused.read_text())["failures"]
                )
            )

    def test_baseline_evaluator_seals_only_complete_stable_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run_id = "passing-packet"
            identity = "sha256:" + "1" * 64
            (output / f"identity-{run_id}.json").write_text(
                json.dumps({"digest": identity})
            )
            records = []
            for surface, depths in [
                ("prefill", [128, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131072]),
                ("decode", [0, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131008]),
            ]:
                for depth in depths:
                    fixture = f"{surface}-{depth}"
                    fixture_sha = hashlib.sha256(fixture.encode()).hexdigest()
                    common = {
                        "schema": "muser.campaign.cell.v1",
                        "run_id": run_id,
                        "identity": identity,
                        "cell": fixture,
                        "status": "passed",
                        "cv": 0.0,
                    }
                    records.append(
                        {
                            **common,
                            "engine": "muser",
                            "raw_ns": [100, 100, 100, 100, 100],
                            "fingerprint": {
                                "identity": identity,
                                "backend": "metal-reference",
                                "kv": "f16",
                                "flash_attention_requested": "on",
                                "flash_attention_active": True,
                                "matvec_route": "muser-local-q4k-q5k",
                                "ggml_metallib_sha256": None,
                                "prompt_fixture_sha256": fixture_sha,
                                "prompt_tokens_sha256": fixture_sha,
                                "decode_fixture_sha256": fixture_sha,
                                "decode_tokens_sha256": fixture_sha,
                                "workload_sha256": fixture_sha,
                            },
                        }
                    )
                    records.append(
                        {
                            **common,
                            "engine": "llama",
                            "raw_ns": [110, 110, 110, 110, 110],
                            "fingerprint": {
                                "build_commit": "7347430f4",
                                "build_number": 1,
                                "comparator_upstream_commit": "7347430f4466d4f55cfb841974ee64b80fc18d93",
                                "comparator_patch_sha256": "2" * 64,
                                "n_batch": 2048,
                                "n_ubatch": 512,
                                "n_threads": 20,
                                "n_gpu_layers": 99,
                                "type_k": "f16",
                                "type_v": "f16",
                                "flash_attn": 1,
                                "prompt_fixture_file_sha256": fixture_sha,
                                "prompt_tokens_sha256": fixture_sha,
                                "decode_fixture_file_sha256": fixture_sha,
                                "decode_tokens_sha256": fixture_sha,
                                "workload_sha256": fixture_sha,
                            },
                        }
                    )
            for depth in (128, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131008):
                cell = f"ttft-{depth}"
                prompt_sha = hashlib.sha256(cell.encode()).hexdigest()
                for engine, raw in (
                    ("ttft-muser", [100, 100, 100, 100, 100]),
                    ("ttft-llama", [110, 110, 110, 110, 110]),
                ):
                    records.append(
                        {
                            "schema": "muser.campaign.cell.v1",
                            "run_id": run_id,
                            "identity": identity,
                            "cell": cell,
                            "engine": engine,
                            "status": "passed",
                            "cv": 0.0,
                            "raw_ns": raw,
                            "fingerprint": {
                                "schema": "muser.server-ttft.v2",
                                "engine": engine.removeprefix("ttft-"),
                                "prompt_sha256": prompt_sha,
                                "reported_prompt_tokens": depth,
                                "cache": "disabled",
                            },
                        }
                    )
            baseline_records = records[:36]
            ttft_records = records[36:]
            records = []
            for index in range(18):
                pair = baseline_records[index * 2 : index * 2 + 2]
                records.extend(reversed(pair) if index % 4 in (1, 2) else pair)
            for index in range(9):
                pair = ttft_records[index * 2 : index * 2 + 2]
                records.extend(reversed(pair) if index % 4 in (1, 2) else pair)
            (output / f"campaign-{run_id}.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_baseline.py"),
                    "--out-dir",
                    str(output),
                    "--run-id",
                    run_id,
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            evaluation = json.loads(result.stdout)
            self.assertEqual(evaluation["status"], "passed")

            unstable = [dict(record) for record in records]
            unstable[0] = {
                **unstable[0],
                "cv": 0.0,
                "raw_ns": [1, 100, 100, 100, 199],
            }
            (output / f"campaign-{run_id}.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in unstable)
            )
            failed = evaluate_baseline.evaluate(output, run_id)
            self.assertEqual(failed["status"], "failed")
            self.assertTrue(any("CV exceeds 3%" in item for item in failed["failures"]))

            reordered = list(records)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            (output / f"campaign-{run_id}.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in reordered)
            )
            failed = evaluate_baseline.evaluate(output, run_id)
            self.assertTrue(any("ABBA command order" in item for item in failed["failures"]))
            self.assertEqual(evaluation["status"], "passed")
            self.assertFalse(evaluation["seal_eligible"])
            self.assertTrue(evaluation["would_be_seal_eligible"])
            self.assertFalse((output / f"seal-{run_id}.json").exists())

    def test_dflash_tuning_freezes_best_complete_stable_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "campaign.jsonl"
            identity = "sha256:" + "2" * 64
            records = []
            for depth in (256, 4096):
                for variant in (1, 2):
                    for verify, speedup in ((3, 1.02), (7, 1.20), (15, 1.10)):
                        records.append(
                            {
                                "schema": "muser.campaign.cell.v1",
                                "identity": identity,
                                "engine": "dflash",
                                "cell": f"dflash-tune-{depth}-p{variant}-v{verify}",
                                "status": "passed",
                                "cv": 0.01,
                                "target_only_cv": 0.01,
                                "speedups": [speedup, speedup, speedup],
                                "raw_ns": [1_000_000, 1_000_000, 1_000_000],
                                "target_only_raw_ns": [
                                    round(1_000_000 * speedup),
                                    round(1_000_000 * speedup),
                                    round(1_000_000 * speedup),
                                ],
                                "fingerprint": {
                                    "prompt_tokens": depth,
                                    "output_tokens": 256,
                                    "verify_length": verify,
                                    "target_backend": "metal",
                                    "assistant_backend": "metal",
                                    "prompt_file_sha256": f"{depth + variant:064x}",
                                    "generated_tokens_sha256": (
                                        "sha256:" + f"{depth:064x}"
                                    ),
                                    "sampled_scalar_oracle": (
                                        "muser-engine-scalar-full-distribution-v1"
                                    ),
                                    "sampled_tokens": 32,
                                    "sampled_seed": 1,
                                    "sampled_temperature_milli": 800,
                                    "sampled_top_p_milli": 950,
                                    "sampled_top_k": 50,
                                    "sampled_generated_tokens_sha256": (
                                        "sha256:" + f"{depth + verify:064x}"
                                    ),
                                    "sampled_drafted_tokens": 31,
                                },
                            }
                        )
            ledger.write_text("".join(json.dumps(record) + "\n" for record in records))
            receipt = root / "tuning.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_dflash_tuning.py"),
                    "--ledger",
                    str(ledger),
                    "--identity",
                    identity,
                    "--out",
                    str(receipt),
                ],
                check=True,
                stdout=subprocess.PIPE,
            )
            result = json.loads(receipt.read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["selected_verify_length"], 7)
            self.assertEqual(result["lane"], "dflash-tuning")
            self.assertFalse(result["seal_eligible"])
            self.assertTrue(result["would_be_seal_eligible"])

            frozen = root / "dflash-tuning-v1.json"
            with (
                mock.patch.object(freeze_dflash_tuning, "TARGET", frozen),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "freeze_dflash_tuning.py",
                        "--receipt",
                        str(receipt),
                        "--out",
                        str(frozen),
                    ],
                ),
            ):
                self.assertEqual(freeze_dflash_tuning.main(), 0)
            frozen_value = json.loads(frozen.read_text())
            self.assertEqual(frozen_value["status"], "frozen")
            self.assertEqual(frozen_value["selected_verify_length"], 7)

            # The ledger is append-only. A selected-cell rerun must taint the
            # packet instead of silently replacing the earlier record in a
            # dictionary comprehension.
            with ledger.open("a") as stream:
                stream.write(json.dumps({**records[0], "status": "failed"}) + "\n")
            duplicate_receipt = root / "duplicate-tuning.json"
            duplicate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_dflash_tuning.py"),
                    "--ledger",
                    str(ledger),
                    "--identity",
                    identity,
                    "--out",
                    str(duplicate_receipt),
                ],
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertEqual(duplicate.returncode, 1)
            duplicate_result = json.loads(duplicate_receipt.read_text())
            self.assertFalse(duplicate_result["seal_eligible"])
            self.assertIn("selected-cell reruns", "\n".join(duplicate_result["failures"]))

    def test_kvpack_evaluator_requires_exact_and_ancestor_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = "sha256:" + "9" * 64
            records = []
            for source in ("resident", "durable", "remote"):
                for depth in (8192, 16384, 32768, 65536, 131008):
                    records.append(
                        self._kvpack_record(
                            identity,
                            f"kvpack-{source}-exact-{depth}",
                            source,
                            "exact-final",
                            depth,
                            depth,
                            0,
                        )
                    )
                for cut in (8192, 16384, 32768, 65536, 128768):
                    for suffix in (1, 255, 256, 257, 2047):
                        records.append(
                            self._kvpack_record(
                                identity,
                                f"kvpack-{source}-ancestor-{cut}-s{suffix}",
                                source,
                                "deepest-ancestor",
                                cut + suffix,
                                cut,
                                suffix,
                            )
                        )
            ledger = root / "kvpack.jsonl"
            ledger.write_text("".join(json.dumps(record) + "\n" for record in records))
            seal = root / "seal.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_kvpack.py"),
                    "--ledger", str(ledger),
                    "--identity", identity,
                    "--out", str(seal),
                ],
                check=True,
                stdout=subprocess.PIPE,
            )
            result = json.loads(seal.read_text())
            self.assertFalse(result["seal_eligible"])
            self.assertTrue(result["would_be_seal_eligible"])
            self.assertEqual(len(result["cells"]), 90)

            with ledger.open("a") as stream:
                stream.write(json.dumps(records[0]) + "\n")
            duplicate_seal = root / "duplicate-seal.json"
            duplicate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_kvpack.py"),
                    "--ledger", str(ledger),
                    "--identity", identity,
                    "--out", str(duplicate_seal),
                ],
                stdout=subprocess.PIPE,
            )
            self.assertEqual(duplicate.returncode, 1)
            duplicate_result = json.loads(duplicate_seal.read_text())
            self.assertFalse(duplicate_result["seal_eligible"])
            self.assertIn(
                "selected-cell reruns", "\n".join(duplicate_result["failures"])
            )

    @staticmethod
    def _kvpack_record(identity, cell, source, lookup, prompt, cut, suffix):
        return {
            "identity": identity,
            "cell": cell,
            "engine": "kvpack",
            "status": "passed",
            "raw_ns": [50, 50, 50],
            "full_recompute_ns": 100,
            "source_prefill_ns": 100,
            "publication_ns": 1,
            "cv": 0.0,
            "publication_overhead_ratio": 0.01,
            "miss_lookup_ns": 1,
            "miss_overhead_ratio": 0.01,
            "speedup_geomean_cell": 2.0,
            "fingerprint": {
                "source": source,
                "lookup": lookup,
                "prompt_tokens": prompt,
                "published_cut": cut,
                "suffix_tokens": suffix,
                "generated_tokens_sha256": "a" * 64,
                "full_logit_digest": "b" * 64,
            },
        }

    def test_dflash_evaluator_consumes_frozen_tuning_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = "sha256:" + "d" * 64
            tuning = root / "tuning.json"
            tuning.write_text(
                json.dumps(
                    {
                        "schema": "muser.dflash-tuning-freeze.v1",
                        "status": "frozen",
                        "selected_verify_length": 7,
                    }
                )
            )
            records = []
            for depth in (512, 2048, 8192, 32768):
                for variant in (1, 2):
                    digest = "sha256:" + f"{depth:064x}"
                    prompt_sha = f"{depth + variant:064x}"
                    records.append(
                        {
                            "identity": identity,
                            "cell": f"dflash-{depth}-p{variant}",
                            "engine": "dflash",
                            "status": "passed",
                            "raw_ns": [80, 80, 80, 80, 80],
                            "target_only_raw_ns": [100, 100, 100, 100, 100],
                            "measurement_order": [
                                ["target-only", "dflash"],
                                ["dflash", "target-only"],
                                ["dflash", "target-only"],
                                ["target-only", "dflash"],
                                ["target-only", "dflash"],
                            ],
                            "cv": 0.0,
                            "target_only_cv": 0.0,
                            "fingerprint": {
                                "output_tokens": 256,
                                "verify_length": 7,
                                "target_backend": "metal",
                                "assistant_backend": "metal",
                                "generated_tokens_sha256": digest,
                                "prompt_file_sha256": prompt_sha,
                                "prompt_tokens": depth,
                                "sampled_scalar_oracle": (
                                    "muser-engine-scalar-full-distribution-v1"
                                ),
                                "sampled_tokens": 32,
                                "sampled_seed": 1,
                                "sampled_temperature_milli": 800,
                                "sampled_top_p_milli": 950,
                                "sampled_top_k": 50,
                                "sampled_generated_tokens_sha256": (
                                    "sha256:" + f"{depth + variant + 9:064x}"
                                ),
                                "sampled_drafted_tokens": 31,
                            },
                        }
                    )
                    records.append(
                        {
                            "identity": identity,
                            "cell": f"dflash-{depth}-p{variant}",
                            "engine": "llama-dflash",
                            "status": "passed",
                            "raw_ns": [90, 90, 90, 90, 90],
                            "cv": 0.0,
                            "fingerprint": {
                                "prompt_tokens": depth,
                                "output_tokens": 256,
                                "verify_length": 7,
                                "prompt_file_sha256": prompt_sha,
                                "generated_tokens_sha256": digest,
                                "route": "llama-draft-dflash",
                            },
                        }
                    )
            ledger = root / "dflash.jsonl"
            ledger.write_text("".join(json.dumps(record) + "\n" for record in records))
            seal = root / "seal.json"
            with (
                mock.patch.object(evaluate_dflash, "TRACKED_TUNING_FREEZE", tuning),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_dflash.py",
                        "--ledger",
                        str(ledger),
                        "--tuning-freeze",
                        str(tuning),
                        "--identity",
                        identity,
                        "--out",
                        str(seal),
                    ],
                ),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(evaluate_dflash.main(), 0)
            result = json.loads(seal.read_text())
            self.assertFalse(result["seal_eligible"])
            self.assertTrue(result["would_be_seal_eligible"])
            self.assertGreaterEqual(result["geometric_mean_speedup"], 1.10)
            self.assertTrue(
                all(cell["versus_llama_dflash"] >= 1.0 for cell in result["cells"].values())
            )

            records[0]["raw_ns"] = [80, 80, 80]
            records[0]["target_only_raw_ns"] = [100, 100, 100]
            records[0]["measurement_order"] = records[0]["measurement_order"][:3]
            ledger.write_text("".join(json.dumps(record) + "\n" for record in records))
            refused = root / "refused.json"
            with (
                mock.patch.object(evaluate_dflash, "TRACKED_TUNING_FREEZE", tuning),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_dflash.py", "--ledger", str(ledger),
                        "--tuning-freeze", str(tuning), "--identity", identity,
                        "--out", str(refused),
                    ],
                ),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(evaluate_dflash.main(), 1)
            self.assertTrue(
                any(
                    "five complete ABBA-ordered paired samples" in failure
                    for failure in json.loads(refused.read_text())["failures"]
                )
            )

    def test_vision_evaluator_requires_four_correct_stable_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "campaign.jsonl"
            identity = "sha256:" + "3" * 64
            records = []
            dimensions = {
                "low-square": (224, 224),
                "wide": (1024, 256),
                "tall": (256, 1024),
                "high-resolution": (2048, 1536),
            }
            insertion = hashlib.sha256(
                b"".join(position.to_bytes(8, "little") for position in range(5, 69))
            ).hexdigest()
            for fixture in ("low-square", "wide", "tall", "high-resolution"):
                cell = f"vision-{fixture}"
                digest = fixture.encode().hex().ljust(64, "0")[:64]
                records.extend(
                    [
                        {
                            "identity": identity,
                            "cell": cell,
                            "engine": "vision",
                            "status": "passed",
                            "raw_ns": [100, 100, 100],
                            "cv": 0.0,
                            "fingerprint": {
                                "fixture": fixture,
                                "route": "mtmd-metal:muser-mtmd-muse-vision-v1",
                                "image_sha256": digest,
                                "source_width": dimensions[fixture][0],
                                "source_height": dimensions[fixture][1],
                                "projected_tokens": 64,
                                "insertion_start": 5,
                                "insertion_end": 69,
                                "insertion_count": 64,
                                "insertion_positions_sha256": "sha256:" + insertion,
                                "prefix_tokens": 5,
                                "suffix_tokens": 7,
                                "installed_positions": 76,
                                "max_pixel_error": 0.0,
                                "embedding_cosine": 1.0,
                                "embedding_relative_l2": 0.0,
                                "exact_decoder_tokens": True,
                                "decoder_tokens_sha256": "sha256:" + digest,
                            },
                        },
                        {
                            "identity": identity,
                            "cell": cell,
                            "engine": "vision-ttft-muser",
                            "status": "passed",
                            "raw_ns": [100, 100, 100],
                            "cv": 0.0,
                            "fingerprint": {
                                "fixture": fixture,
                                "image_sha256": digest,
                                "reported_prompt_tokens": [76, 76, 76],
                                "server_lifecycle":
                                    "leased-start-ready-exact-requests-cooperative-exit",
                            },
                        },
                        {
                            "identity": identity,
                            "cell": cell,
                            "engine": "vision-ttft-llama",
                            "status": "passed",
                            "raw_ns": [110, 110, 110, 110, 110],
                            "cv": 0.0,
                            "fingerprint": {
                                "fixture": fixture,
                                "image_sha256": digest,
                                "reported_prompt_tokens": [76, 76, 76, 76, 76],
                                "server_lifecycle":
                                    "leased-start-ready-exact-requests-cooperative-exit",
                            },
                        },
                    ]
                )
            ledger.write_text("".join(json.dumps(record) + "\n" for record in records))
            seal = root / "vision.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_vision.py"),
                    "--ledger",
                    str(ledger),
                    "--identity",
                    identity,
                    "--out",
                    str(seal),
                ],
                check=True,
                stdout=subprocess.PIPE,
            )
            result = json.loads(seal.read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["stable_fixtures"]), 4)
            self.assertGreater(result["geometric_mean_speedup"], 1.0)

            records[0]["fingerprint"]["insertion_positions_sha256"] = "sha256:" + "0" * 64
            ledger.write_text("".join(json.dumps(record) + "\n" for record in records))
            failed_seal = root / "vision-bad-insertion.json"
            failed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_vision.py"),
                    "--ledger",
                    str(ledger),
                    "--identity",
                    identity,
                    "--out",
                    str(failed_seal),
                ],
                stdout=subprocess.PIPE,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(
                "exact insertion-position evidence",
                " ".join(json.loads(failed_seal.read_text())["failures"]),
            )

    def test_ane_evaluator_seals_only_complete_paired_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "campaign.jsonl"
            identity = "sha256:" + "4" * 64
            records = []
            for depth in (512, 2048, 8192, 32768):
                for variant in (1, 2):
                    records.append(
                        {
                            "identity": identity,
                            "cell": f"ane-{depth}-p{variant}",
                            "engine": "ane",
                            "status": "passed",
                            "raw_ns": [80, 80, 80],
                            "metal_dflash_raw_ns": [100, 100, 100],
                            "target_only_raw_ns": [130, 130, 130],
                            "metal_target_verify_ns": [50, 50, 50],
                            "ane_target_verify_ns": [50, 50, 50],
                            "cv": 0.0,
                            "metal_dflash_cv": 0.0,
                            "target_only_cv": 0.0,
                            "speedups": [1.25, 1.25, 1.25],
                            "verification_taxes": [0.0, 0.0, 0.0],
                            "fingerprint": {
                                "target_identity": "target",
                                "dflash_identity": "draft",
                                "manifest_sha256": "a" * 64,
                                "compute_plan_receipt_sha256": "b" * 64,
                                "compute_units": "CPU_AND_NE",
                                "verify_length": 7,
                            },
                        }
                    )
            ledger.write_text("".join(json.dumps(record) + "\n" for record in records))
            seal = root / "ane.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_ane.py"),
                    "--ledger",
                    str(ledger),
                    "--identity",
                    identity,
                    "--out",
                    str(seal),
                ],
                check=True,
                stdout=subprocess.PIPE,
            )
            result = json.loads(seal.read_text())
            self.assertEqual(result["status"], "passed")
            self.assertAlmostEqual(result["geometric_mean_speedup"], 1.25)
            self.assertAlmostEqual(result["mean_target_verification_tax"], 0.0)


if __name__ == "__main__":
    unittest.main()
