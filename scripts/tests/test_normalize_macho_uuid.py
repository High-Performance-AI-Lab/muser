from __future__ import annotations

import importlib.util
import pathlib
import struct
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "normalize_macho_uuid.py"
SPEC = importlib.util.spec_from_file_location("normalize_macho_uuid_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
normalize_macho_uuid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalize_macho_uuid)


def fixture(uuid: bytes, signature: bytes, payload: bytes = b"release payload") -> bytes:
    assert len(uuid) == 16
    header_size = normalize_macho_uuid.MACH_HEADER_64_SIZE
    uuid_command = struct.pack("<II", normalize_macho_uuid.LC_UUID, 24) + uuid
    signature_offset = header_size + 24 + 16 + len(payload)
    signature_command = struct.pack(
        "<IIII",
        normalize_macho_uuid.LC_CODE_SIGNATURE,
        16,
        signature_offset,
        len(signature),
    )
    header = struct.pack(
        "<IIIIIIII",
        normalize_macho_uuid.MH_MAGIC_64,
        0x0100000C,
        0,
        2,
        2,
        len(uuid_command) + len(signature_command),
        0,
        0,
    )
    return header + uuid_command + signature_command + payload + signature


class NormalizeMachoUuidTests(unittest.TestCase):
    def test_uuid_and_old_signature_do_not_change_the_canonical_value(self) -> None:
        left = fixture(bytes.fromhex("11" * 16), bytes.fromhex("22" * 32))
        right = fixture(bytes.fromhex("33" * 16), bytes.fromhex("44" * 64))
        left_offset, left_uuid = normalize_macho_uuid.canonical_uuid(left)
        right_offset, right_uuid = normalize_macho_uuid.canonical_uuid(right)
        self.assertEqual(left_offset, right_offset)
        self.assertEqual(left_uuid, right_uuid)
        self.assertEqual(left_uuid[6] >> 4, 5)
        self.assertEqual(left_uuid[8] >> 6, 2)

    def test_payload_changes_the_canonical_value(self) -> None:
        left = fixture(bytes(16), bytes(32), b"payload a")
        right = fixture(bytes(16), bytes(32), b"payload b")
        self.assertNotEqual(
            normalize_macho_uuid.canonical_uuid(left)[1],
            normalize_macho_uuid.canonical_uuid(right)[1],
        )

    def test_malformed_or_unsigned_inputs_fail_closed(self) -> None:
        with self.assertRaises(normalize_macho_uuid.MachOError):
            normalize_macho_uuid.canonical_uuid(b"not mach-o")
        unsigned = bytearray(fixture(bytes(16), bytes(32)))
        struct.pack_into("<I", unsigned, 32 + 24, 0)
        with self.assertRaises(normalize_macho_uuid.MachOError):
            normalize_macho_uuid.canonical_uuid(unsigned)


if __name__ == "__main__":
    unittest.main()
