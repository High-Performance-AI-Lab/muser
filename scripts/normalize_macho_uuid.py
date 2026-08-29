#!/usr/bin/env python3
"""Canonicalize a thin 64-bit Mach-O UUID without trusting linker metadata."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import struct


MH_MAGIC_64 = 0xFEEDFACF
MACH_HEADER_64_SIZE = 32
LC_UUID = 0x1B
LC_CODE_SIGNATURE = 0x1D


class MachOError(ValueError):
    """The input is not the exact executable shape the release accepts."""


def _u32(data: bytes | bytearray, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise MachOError("truncated Mach-O integer")
    return struct.unpack_from("<I", data, offset)[0]


def canonical_uuid(data: bytes | bytearray) -> tuple[int, bytes]:
    """Return the UUID field offset and content-derived RFC 4122 bytes."""

    if len(data) < MACH_HEADER_64_SIZE or _u32(data, 0) != MH_MAGIC_64:
        raise MachOError("expected a thin little-endian 64-bit Mach-O executable")
    command_count = _u32(data, 16)
    command_bytes = _u32(data, 20)
    command_end = MACH_HEADER_64_SIZE + command_bytes
    if command_end > len(data):
        raise MachOError("Mach-O load-command table exceeds the file")

    uuid_offsets: list[int] = []
    signature_commands: list[tuple[int, int, int]] = []
    cursor = MACH_HEADER_64_SIZE
    for _ in range(command_count):
        command = _u32(data, cursor)
        command_size = _u32(data, cursor + 4)
        if command_size < 8 or command_size % 4 != 0:
            raise MachOError("Mach-O has an invalid load-command size")
        next_cursor = cursor + command_size
        if next_cursor > command_end:
            raise MachOError("Mach-O load command exceeds its declared table")
        if command == LC_UUID:
            if command_size != 24:
                raise MachOError("LC_UUID has the wrong size")
            uuid_offsets.append(cursor + 8)
        elif command == LC_CODE_SIGNATURE:
            if command_size != 16:
                raise MachOError("LC_CODE_SIGNATURE has the wrong size")
            signature_offset = _u32(data, cursor + 8)
            signature_size = _u32(data, cursor + 12)
            signature_end = signature_offset + signature_size
            if signature_size == 0 or signature_end > len(data):
                raise MachOError("code-signature blob is empty or outside the file")
            signature_commands.append((cursor + 8, signature_offset, signature_end))
        cursor = next_cursor
    if cursor != command_end:
        raise MachOError("Mach-O load-command count and byte size disagree")
    if len(uuid_offsets) != 1:
        raise MachOError("expected exactly one LC_UUID command")
    if len(signature_commands) != 1:
        raise MachOError("expected exactly one LC_CODE_SIGNATURE command")

    signature_fields, signature_start, signature_end = signature_commands[0]
    if signature_end != len(data):
        raise MachOError("code-signature blob is not the final file region")
    # Code signing can change both the blob length and the two link-edit fields
    # that point at it. Neither is executable content, so exclude the terminal
    # blob and clear its offset/size fields before deriving the build UUID.
    canonical = bytearray(data[:signature_start])
    uuid_offset = uuid_offsets[0]
    canonical[uuid_offset : uuid_offset + 16] = bytes(16)
    canonical[signature_fields : signature_fields + 8] = bytes(8)
    value = bytearray(hashlib.sha256(canonical).digest()[:16])
    # Label these deterministic bytes as an RFC 4122 version-5-style UUID.
    value[6] = (value[6] & 0x0F) | 0x50
    value[8] = (value[8] & 0x3F) | 0x80
    return uuid_offset, bytes(value)


def normalize(path: pathlib.Path, *, check: bool) -> None:
    data = bytearray(path.read_bytes())
    offset, expected = canonical_uuid(data)
    actual = bytes(data[offset : offset + 16])
    if check:
        if actual != expected:
            raise MachOError(f"{path} has a non-canonical LC_UUID")
        return
    data[offset : offset + 16] = expected
    path.write_bytes(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("binary", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        normalize(args.binary, check=args.check)
    except (OSError, MachOError) as error:
        raise SystemExit(f"normalize Mach-O UUID: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
