"""Pure helpers for the qualified Muse Handoff V2 packed schedule."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Protocol


class PackedIntent(Protocol):
    start: int
    count: int


ROW_ELEMENTS = 2 * 128
ROW_BYTES = ROW_ELEMENTS * 2


def neox_to_interleaved_order(head_dim: int) -> tuple[int, ...]:
    """Canonical producer K order used by the Mac NORM-RoPE artifact.

    vLLM keeps each NeoX pair in the two half-heads. Muser's GGUF conversion
    interleaves that same pair, so exported post-RoPE keys must cross the seam
    as ``[0, half, 1, half+1, ...]``. Values are never permuted.
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("NeoX head dimension must be positive and even")
    half = head_dim // 2
    return tuple(index for pair in zip(range(half), range(half, head_dim)) for index in pair)


def interleaved_to_neox_order(head_dim: int) -> tuple[int, ...]:
    """Inverse of :func:`neox_to_interleaved_order`.

    Handoff V2 carries post-RoPE keys as adjacent real/imaginary pairs while
    stock vLLM FlashAttention stores the same pairs in two half-heads.  An
    imported portable prefix must apply this inverse before touching vLLM's
    paged cache; otherwise the transfer authenticates but the target state is
    semantically corrupt.
    """
    forward = neox_to_interleaved_order(head_dim)
    inverse = [0] * head_dim
    for canonical_index, neox_index in enumerate(forward):
        inverse[neox_index] = canonical_index
    return tuple(inverse)


def token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError("token ID is outside the u32 wire domain")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def pack_intent_payload(
    intent: PackedIntent,
    plane_order: list[tuple[int, str]],
    planes: Mapping[tuple[int, str], bytes],
    cached_token_count: int,
) -> bytes:
    if intent.start < 0 or intent.count < 1:
        raise ValueError("invalid Muse packed range")
    if intent.start + intent.count > cached_token_count:
        raise ValueError("Muse packed range exceeds the cached prefix")
    expected_plane_bytes = cached_token_count * ROW_BYTES
    output = bytearray()
    start = intent.start * ROW_BYTES
    end = (intent.start + intent.count) * ROW_BYTES
    for key in plane_order:
        plane = planes.get(key)
        if plane is None:
            raise ValueError(f"missing Muse KV plane {key!r}")
        if len(plane) != expected_plane_bytes:
            raise ValueError(
                f"Muse KV plane {key!r} has {len(plane)} bytes, "
                f"expected {expected_plane_bytes}"
            )
        output.extend(plane[start:end])
    return bytes(output)
