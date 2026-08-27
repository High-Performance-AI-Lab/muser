"""Read-only target hidden hook for the native Muse Glimmer producer."""

from __future__ import annotations

import re
from typing import Any

from muser_vllm.dflash_capture import TARGET_LAYERS, capture_layer


_ORIGINAL_DECODER_INIT: Any = None
_ORIGINAL_DECODER_FORWARD: Any = None


def native_decoder_layer_init(module: Any, *args: Any, **kwargs: Any) -> None:
    """Bind the stable zero-based layer index without changing construction."""
    if _ORIGINAL_DECODER_INIT is None:
        raise RuntimeError("native capture init was invoked before installation")
    prefix = kwargs.get("prefix")
    if prefix is None and len(args) >= 4:
        prefix = args[3]
    if not isinstance(prefix, str):
        raise RuntimeError("Muse decoder layer has no stable module prefix")
    match = re.search(r"(?:^|\.)layers\.(\d+)$", prefix)
    if match is None:
        raise RuntimeError(f"cannot derive Muse layer index from prefix {prefix!r}")
    _ORIGINAL_DECODER_INIT(module, *args, **kwargs)
    module._muser_layer_index = int(match.group(1))


def native_decoder_layer_forward(
    module: Any,
    positions: Any,
    hidden_states: Any,
    residual: Any,
) -> tuple[Any, Any]:
    """Observe selected completed layer outputs and return the original tuple."""
    if _ORIGINAL_DECODER_FORWARD is None:
        raise RuntimeError("native capture forward was invoked before installation")
    result = _ORIGINAL_DECODER_FORWARD(module, positions, hidden_states, residual)
    layer = module._muser_layer_index
    if layer in TARGET_LAYERS:
        # Stock MuseGlimmerDecoderLayer.forward has already completed both
        # residual additions here. capture_layer only reads result[0] and the
        # exact same tuple object is returned to vLLM.
        capture_layer(layer, result[0])
    return result


def install_native_capture() -> dict[str, Any]:
    """Install the eager-mode observer before vLLM constructs the model."""
    global _ORIGINAL_DECODER_INIT, _ORIGINAL_DECODER_FORWARD
    from vllm.model_executor.models import muse_glimmer

    _ORIGINAL_DECODER_INIT = muse_glimmer.MuseGlimmerDecoderLayer.__init__
    _ORIGINAL_DECODER_FORWARD = muse_glimmer.MuseGlimmerDecoderLayer.forward
    muse_glimmer.MuseGlimmerDecoderLayer.__init__ = native_decoder_layer_init
    muse_glimmer.MuseGlimmerDecoderLayer.forward = native_decoder_layer_forward
    if muse_glimmer.MuseGlimmerDecoderLayer.__init__ is not native_decoder_layer_init:
        raise RuntimeError("failed to install native Muse layer-index binding")
    if muse_glimmer.MuseGlimmerDecoderLayer.forward is not native_decoder_layer_forward:
        raise RuntimeError("failed to install native Muse target-hidden observer")
    return {
        "active": True,
        "producer_mode": "native",
        "selection": "read-only-completed-layer-output",
        "target_layers": list(TARGET_LAYERS),
        "numeric_effect": "none-return-original-tuple",
    }
