from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = ROOT / "scripts" / "gx10" / "vllm"
LLAMACPP_ROOT = ROOT / "scripts" / "gx10" / "llamacpp"
sys.path.insert(0, str(VLLM_ROOT))
sys.path.insert(0, str(LLAMACPP_ROOT))


def load_connector():
    """Load the scheduler seam without requiring a local vLLM install."""

    class KVConnectorBase:
        pass

    class KVConnectorMetadata:
        pass

    class KVConnectorRole:
        pass

    class FlashAttentionMetadata:
        pass

    modules: dict[str, types.ModuleType] = {}
    names = (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.flash_attn",
    )
    for name in names:
        modules[name] = types.ModuleType(name)
    base = modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    base.KVConnectorBase_V1 = KVConnectorBase
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorRole = KVConnectorRole
    flash = modules["vllm.v1.attention.backends.flash_attn"]
    flash.FlashAttentionMetadata = FlashAttentionMetadata

    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(modules)
    module_name = "muser_vllm._connector_scheduler_test"
    spec = importlib.util.spec_from_file_location(
        module_name, VLLM_ROOT / "muser_vllm" / "connector.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


CONNECTOR = load_connector()


def new_request(
    req_id: str,
    token_count: int,
    *,
    computed: int = 0,
    warmup: bool = False,
) -> SimpleNamespace:
    extra_args = (
        {"muser_startup_warmup": True}
        if warmup
        else {
            "kv_transfer_params": {
                "muser_handoff": {
                    "generation": 1,
                    "receiver_host": "192.0.2.10",
                    "receiver_port": 29590,
                    "transfer_id": "fixture",
                }
            }
        }
    )
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(range(token_count)),
        num_computed_tokens=computed,
        block_ids=([0],),
        sampling_params=SimpleNamespace(extra_args=extra_args),
    )


def cached(
    req_id: str | None = None,
    *,
    computed: int = 0,
    output_tokens: int = 0,
    resumed: bool = False,
) -> SimpleNamespace:
    if req_id is None:
        return SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            num_computed_tokens=[],
            num_output_tokens=[],
        )
    return SimpleNamespace(
        req_ids=[req_id],
        resumed_req_ids={req_id} if resumed else set(),
        num_computed_tokens=[computed],
        num_output_tokens=[output_tokens],
    )


def output(
    *,
    new: list[SimpleNamespace] | None = None,
    cached_request: SimpleNamespace | None = None,
    scheduled: dict[str, int] | None = None,
    finished: set[str] | None = None,
    preempted: set[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_new_reqs=new or [],
        scheduled_cached_reqs=cached_request or cached(),
        num_scheduled_tokens=scheduled or {},
        finished_req_ids=finished or set(),
        preempted_req_ids=preempted,
    )


class ChunkedSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connector = object.__new__(CONNECTOR.MuserMuseHandoffConnector)
        self.connector._prefix_caching = False
        self.connector._scheduled_requests = {}

    def build(self, value: SimpleNamespace):
        return self.connector.build_connector_meta(value)

    def test_unchunked_request_preserves_single_step_handoff(self) -> None:
        meta = self.build(
            output(new=[new_request("r", 10)], scheduled={"r": 10})
        )
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests[0].prompt_token_ids, list(range(10)))
        self.assertEqual(meta.requests[0].prefill_chunks, 1)
        self.assertEqual(self.connector._scheduled_requests, {})

    def test_only_final_chunk_activates_full_prompt_handoff(self) -> None:
        first = self.build(
            output(new=[new_request("r", 10)], scheduled={"r": 4})
        )
        self.assertEqual(first.requests, [])
        second = self.build(
            output(
                cached_request=cached("r", computed=4),
                scheduled={"r": 4},
            )
        )
        self.assertEqual(second.requests, [])
        final = self.build(
            output(
                cached_request=cached("r", computed=8),
                scheduled={"r": 2},
            )
        )
        self.assertEqual(len(final.requests), 1)
        self.assertEqual(final.requests[0].prompt_token_ids, list(range(10)))
        self.assertEqual(final.requests[0].prefill_chunks, 3)
        self.assertEqual(self.connector._scheduled_requests, {})

    def test_chunked_startup_warmup_never_exports(self) -> None:
        first = self.build(
            output(new=[new_request("warm", 6, warmup=True)], scheduled={"warm": 3})
        )
        final = self.build(
            output(
                cached_request=cached("warm", computed=3),
                scheduled={"warm": 3},
            )
        )
        self.assertEqual(first.requests, [])
        self.assertEqual(final.requests, [])
        self.assertEqual(self.connector._scheduled_requests, {})

    def test_unknown_context_chunk_fails_closed(self) -> None:
        with self.assertRaisesRegex(CONNECTOR.ProtocolError, "unknown cached"):
            self.build(
                output(
                    cached_request=cached("missing", computed=4),
                    scheduled={"missing": 2},
                )
            )

    def test_preemption_and_resume_fail_closed(self) -> None:
        self.build(output(new=[new_request("r", 10)], scheduled={"r": 4}))
        with self.assertRaisesRegex(CONNECTOR.ProtocolError, "preempted"):
            self.build(output(preempted={"r"}))

        self.connector._scheduled_requests = {}
        self.build(output(new=[new_request("r", 10)], scheduled={"r": 4}))
        with self.assertRaisesRegex(CONNECTOR.ProtocolError, "resumed"):
            self.build(
                output(
                    cached_request=cached("r", computed=4, resumed=True),
                    scheduled={"r": 4},
                )
            )

    def test_finished_request_reclaims_pending_scheduler_state(self) -> None:
        self.build(output(new=[new_request("r", 10)], scheduled={"r": 4}))
        self.build(output(finished={"r"}))
        self.assertEqual(self.connector._scheduled_requests, {})


if __name__ == "__main__":
    unittest.main()
