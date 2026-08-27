from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from kvpack_ggsn import (
    GGML_TYPE_F16,
    GgsnError,
    build_synthetic_session,
    indices_for_positions,
    parse_ggsn,
    read_rows,
)


class GgsnParserTests(unittest.TestCase):
    def test_round_trip_one_block_and_position_slice(self) -> None:
        row = struct.pack("<4e", 1.0, 2.0, 3.0, 4.0)
        blob = build_synthetic_session(
            tokens=[10, 11, 12],
            arch="test-arch",
            cell_pos=[0, 1, 2],
            n_layer=2,
            row_bytes=len(row),
            fill=row,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.bin"
            path.write_bytes(blob)
            session = parse_ggsn(path)
        self.assertEqual(session.arch, "test-arch")
        self.assertEqual(session.tokens, [10, 11, 12])
        self.assertEqual(len(session.blocks), 1)
        block = session.blocks[0]
        self.assertEqual(block.n_layer, 2)
        self.assertEqual(block.v_trans, 0)
        self.assertEqual(block.k_planes[0].ggml_type, GGML_TYPE_F16)
        rows = indices_for_positions(block, 1, 3)
        self.assertEqual(rows, [1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.bin"
            path.write_bytes(blob)
            sliced = read_rows(path, block, block.k_planes[0], rows)
        self.assertEqual(sliced, row + row)

    def test_rejects_bad_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.bin"
            path.write_bytes(b"not a session")
            with self.assertRaises(GgsnError):
                parse_ggsn(path)
