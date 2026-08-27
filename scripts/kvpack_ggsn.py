"""Parse llama.cpp GGSN session files into kvpack-canonical plane views.

The on-disk layout matches llama.cpp `LLAMA_SESSION_VERSION` 9
(`llama_state_save_file`): magic, version, token prefix, architecture string,
then one or more `llama_kv_cache::state_write` blocks. Hybrid iSWA models
emit the full-attention block first and the sliding-window block second.
Flash-attention sessions store K and V as `[cell][kv_head][head_dim]` rows
(v_trans == 0), which is `canonical-kv-f16-le-v1`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

SESSION_MAGIC = 0x6767736E  # 'ggsn'
SESSION_VERSION = 9
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


class GgsnError(ValueError):
    pass


@dataclass
class PlaneRef:
    offset: int
    length: int
    row_bytes: int
    ggml_type: int


@dataclass
class KvBlock:
    """One llama_kv_cache state_write payload."""

    v_trans: int
    n_layer: int
    cell_count: int
    cell_pos: list[int]
    k_planes: list[PlaneRef] = field(default_factory=list)
    v_planes: list[PlaneRef] = field(default_factory=list)

    def row_slice(self, plane: PlaneRef, start_token: int, end_token: int) -> tuple[int, int]:
        if start_token < 0 or end_token > self.cell_count or start_token > end_token:
            raise GgsnError(
                f"row slice [{start_token}, {end_token}) outside cell_count {self.cell_count}"
            )
        return (
            plane.offset + start_token * plane.row_bytes,
            (end_token - start_token) * plane.row_bytes,
        )


@dataclass
class GgsnSession:
    path: Path
    size: int
    version: int
    tokens: list[int]
    arch: str
    blocks: list[KvBlock]


def _read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise GgsnError(f"truncated session read: wanted {n} bytes, got {len(data)}")
    return data


def _parse_block(fh, size: int) -> KvBlock | None:
    """Parse one state_write block. Returns None at EOF before n_stream."""
    header = fh.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise GgsnError("truncated n_stream")
    (n_stream,) = struct.unpack("<I", header)
    if n_stream == 0 or n_stream > 64:
        raise GgsnError(f"implausible n_stream {n_stream}")
    block = KvBlock(v_trans=0, n_layer=0, cell_count=0, cell_pos=[])
    for _ in range(n_stream):
        (cell_count,) = struct.unpack("<I", _read_exact(fh, 4))
        if cell_count == 0:
            continue
        if block.cell_count != 0:
            raise GgsnError("more than one non-empty stream in a KV block")
        if cell_count > 1_000_000:
            raise GgsnError(f"implausible cell_count {cell_count}")
        block.cell_count = cell_count
        for _cell in range(cell_count):
            pos, n_seq_id = struct.unpack("<iI", _read_exact(fh, 8))
            block.cell_pos.append(pos)
            if n_seq_id:
                if n_seq_id > 1024:
                    raise GgsnError(f"implausible n_seq_id {n_seq_id}")
                _read_exact(fh, 4 * n_seq_id)
        block.v_trans, block.n_layer = struct.unpack("<II", _read_exact(fh, 8))
        if block.n_layer > 256:
            raise GgsnError(f"implausible n_layer {block.n_layer}")
        if block.v_trans not in (0, 1):
            raise GgsnError(f"unexpected v_trans {block.v_trans}")
        for _plane in range(block.n_layer):
            k_type, k_row = struct.unpack("<iQ", _read_exact(fh, 12))
            if k_type not in (GGML_TYPE_F16, GGML_TYPE_F32):
                raise GgsnError(f"unsupported K ggml type {k_type}")
            length = cell_count * k_row
            if fh.tell() + length > size:
                raise GgsnError("K plane overruns the session file")
            block.k_planes.append(PlaneRef(fh.tell(), length, k_row, k_type))
            fh.seek(length, 1)
        if block.v_trans == 0:
            for _plane in range(block.n_layer):
                v_type, v_row = struct.unpack("<iQ", _read_exact(fh, 12))
                if v_type not in (GGML_TYPE_F16, GGML_TYPE_F32):
                    raise GgsnError(f"unsupported V ggml type {v_type}")
                length = cell_count * v_row
                if fh.tell() + length > size:
                    raise GgsnError("V plane overruns the session file")
                block.v_planes.append(PlaneRef(fh.tell(), length, v_row, v_type))
                fh.seek(length, 1)
        else:
            for _plane in range(block.n_layer):
                v_type, v_el, n_embd = struct.unpack("<iII", _read_exact(fh, 12))
                row = n_embd * v_el
                length = cell_count * row
                if fh.tell() + length > size:
                    raise GgsnError("transposed V plane overruns the session file")
                block.v_planes.append(PlaneRef(fh.tell(), length, row, v_type))
                fh.seek(length, 1)
    return block


def parse_ggsn(path: Path) -> GgsnSession:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        magic, version, ntok = struct.unpack("<III", _read_exact(fh, 12))
        if magic != SESSION_MAGIC:
            raise GgsnError(f"session magic {magic:#x} is not GGSN")
        if version != SESSION_VERSION:
            raise GgsnError(f"session version {version} != {SESSION_VERSION}")
        if ntok > 1_000_000:
            raise GgsnError(f"implausible token count {ntok}")
        tokens = list(struct.unpack(f"<{ntok}i", _read_exact(fh, 4 * ntok))) if ntok else []
        (arch_len,) = struct.unpack("<I", _read_exact(fh, 4))
        if arch_len > 4096:
            raise GgsnError(f"implausible arch string length {arch_len}")
        arch = _read_exact(fh, arch_len).decode("utf-8", "replace")
        blocks: list[KvBlock] = []
        while fh.tell() < size:
            block = _parse_block(fh, size)
            if block is None:
                break
            if block.n_layer == 0:
                continue
            blocks.append(block)
        leftover = size - fh.tell()
        if leftover != 0:
            raise GgsnError(f"{leftover} trailing bytes after KV blocks")
    if not blocks:
        raise GgsnError("session contains no KV planes")
    return GgsnSession(
        path=path, size=size, version=version, tokens=tokens, arch=arch, blocks=blocks
    )


def indices_for_positions(block: KvBlock, start_pos: int, end_pos: int) -> list[int]:
    return [i for i, pos in enumerate(block.cell_pos) if start_pos <= pos < end_pos]


def read_rows(path: Path, block: KvBlock, plane: PlaneRef, rows: list[int]) -> bytes:
    if not rows:
        return b""
    consecutive = all(rows[i] + 1 == rows[i + 1] for i in range(len(rows) - 1))
    with path.open("rb") as fh:
        if consecutive:
            offset, length = block.row_slice(plane, rows[0], rows[-1] + 1)
            fh.seek(offset)
            return _read_exact(fh, length)
        chunks = bytearray()
        for row in rows:
            offset, length = block.row_slice(plane, row, row + 1)
            fh.seek(offset)
            chunks.extend(_read_exact(fh, length))
        return bytes(chunks)


def build_synthetic_session(
    tokens: list[int],
    arch: str,
    cell_pos: list[int],
    n_layer: int,
    row_bytes: int,
    fill: bytes,
) -> bytes:
    """Minimal v_trans=0 f16 session for tests. One stream, one block."""
    if len(fill) != row_bytes:
        raise GgsnError("fill must be one row")
    cell_count = len(cell_pos)
    payload = bytearray()
    payload.extend(struct.pack("<III", SESSION_MAGIC, SESSION_VERSION, len(tokens)))
    payload.extend(struct.pack(f"<{len(tokens)}i", *tokens) if tokens else b"")
    arch_bytes = arch.encode("utf-8")
    payload.extend(struct.pack("<I", len(arch_bytes)))
    payload.extend(arch_bytes)
    payload.extend(struct.pack("<I", 1))  # n_stream
    payload.extend(struct.pack("<I", cell_count))
    for pos in cell_pos:
        payload.extend(struct.pack("<iI", pos, 0))
    payload.extend(struct.pack("<II", 0, n_layer))  # v_trans, n_layer
    plane = fill * cell_count
    for _ in range(n_layer):
        payload.extend(struct.pack("<iQ", GGML_TYPE_F16, row_bytes))
        payload.extend(plane)
    for _ in range(n_layer):
        payload.extend(struct.pack("<iQ", GGML_TYPE_F16, row_bytes))
        payload.extend(plane)
    return bytes(payload)
