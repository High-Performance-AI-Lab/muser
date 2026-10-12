"""Authenticated portable prefix bundle for a composite Muse target.

The bundle is the disk/CAS-shaped form of the existing Handoff V2 seam: each
layer contains canonical post-RoPE interleaved f16 keys followed by f16 values.
It is deliberately checkpoint-neutral.  A RedHat producer may create the
prefix while a Dudeman target imports it, but the manifest never claims that
the two continuation functions are equal.

Publication is complete-manifest only.  Layer objects are written and fsynced
inside a private staging directory; the authenticated manifest is written
last, then the directory is atomically renamed into place.  Readers reject
unknown fields, wrong geometry, missing/extra layers, payload mutation, and a
wrong HMAC before exposing bytes to a KV-cache importer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

from .packing import token_ids_sha256


SCHEMA = "muser.composite-portable-kv.v1"
HMAC_DOMAIN = b"muser-composite-portable-kv-v1\0"
PORTABLE_KV_ABI = "muser-handoff-v2-post-rope-interleaved-f16le"
PINNED_VLLM_COMMIT = "6adad08767583f52eb4d2122111af0bf638ed5e6"
EXPECTED_LAYERS = 52
EXPECTED_KV_HEADS = 2
EXPECTED_HEAD_DIM = 128
VOCAB_SIZE = 202_048
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9._-]{1,128}")
_MANIFEST_KEYS = {
    "cached_token_count",
    "created_unix_ms",
    "dtype",
    "head_dim",
    "hmac_key_id",
    "hmac_sha256",
    "key_layout",
    "kv_heads",
    "layer_files",
    "layers",
    "portable_kv_abi",
    "schema",
    "source_checkpoint_artifact_sha256",
    "source_checkpoint_revision",
    "source_engine_mode",
    "token_ids",
    "token_ids_sha256",
    "value_layout",
    "vllm_commit",
}
_LAYER_KEYS = {"bytes", "layer", "path", "sha256"}


class CompositeBundleError(RuntimeError):
    """A portable prefix failed a closed validation or publication rule."""


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CompositeBundleError(f"{name} is not a lowercase SHA-256")
    return value


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CompositeBundleError(f"{name} is outside the closed identifier grammar")
    return value


def load_hmac_key(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CompositeBundleError(f"cannot stat composite HMAC key: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise CompositeBundleError("composite HMAC key must be a regular non-symlink")
    if metadata.st_mode & 0o077:
        raise CompositeBundleError("composite HMAC key must not be group/world accessible")
    try:
        encoded = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise CompositeBundleError(f"cannot read composite HMAC key: {error}") from error
    if _DIGEST.fullmatch(encoded) is None:
        raise CompositeBundleError("composite HMAC key must be 32-byte lowercase hex")
    return bytes.fromhex(encoded)


def expected_layer_bytes(cached_token_count: int) -> int:
    if cached_token_count < 1:
        raise CompositeBundleError("portable prefix must contain at least one cached token")
    # key/value * tokens * heads * head-dim * sizeof(f16)
    return 2 * cached_token_count * EXPECTED_KV_HEADS * EXPECTED_HEAD_DIM * 2


def _tag(manifest_without_tag: dict[str, Any], key: bytes) -> str:
    return hmac.new(key, HMAC_DOMAIN + canonical_json(manifest_without_tag), hashlib.sha256).hexdigest()


def bundle_root_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def validate_manifest(
    manifest: object,
    *,
    key: bytes,
    expected_key_id: str,
    expected_source_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        actual = sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__
        raise CompositeBundleError(f"composite manifest keys are not closed: {actual}")
    if manifest["schema"] != SCHEMA or manifest["portable_kv_abi"] != PORTABLE_KV_ABI:
        raise CompositeBundleError("composite manifest schema or portable ABI differs")
    if manifest["vllm_commit"] != PINNED_VLLM_COMMIT:
        raise CompositeBundleError("composite manifest vLLM commit differs")
    if manifest["source_engine_mode"] not in {"native", "exact"}:
        raise CompositeBundleError("composite source engine mode is not closed")
    if manifest["dtype"] != "float16-le" or manifest["key_layout"] != "post-rope-interleaved-pairs":
        raise CompositeBundleError("composite key dtype/layout differs")
    if manifest["value_layout"] != "token-head-dimension":
        raise CompositeBundleError("composite value layout differs")
    if (
        manifest["layers"] != EXPECTED_LAYERS
        or manifest["kv_heads"] != EXPECTED_KV_HEADS
        or manifest["head_dim"] != EXPECTED_HEAD_DIM
    ):
        raise CompositeBundleError("composite Muse geometry differs")
    created = manifest["created_unix_ms"]
    if not isinstance(created, int) or created < 1:
        raise CompositeBundleError("composite creation time is invalid")
    _require_identifier("source checkpoint revision", manifest["source_checkpoint_revision"])
    artifact = _require_digest(
        "source checkpoint artifact", manifest["source_checkpoint_artifact_sha256"]
    )
    if expected_source_artifact_sha256 is not None and artifact != expected_source_artifact_sha256:
        raise CompositeBundleError("composite source checkpoint artifact differs")
    key_id = _require_identifier("HMAC key id", manifest["hmac_key_id"])
    if key_id != expected_key_id:
        raise CompositeBundleError("composite HMAC key id differs")
    supplied_tag = _require_digest("composite HMAC", manifest["hmac_sha256"])

    token_ids = manifest["token_ids"]
    cached = manifest["cached_token_count"]
    if (
        not isinstance(token_ids, list)
        or not isinstance(cached, int)
        or cached < 1
        or len(token_ids) != cached + 1
        or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in token_ids)
    ):
        raise CompositeBundleError("composite token transcript/cut is invalid")
    if manifest["token_ids_sha256"] != token_ids_sha256(token_ids):
        raise CompositeBundleError("composite token transcript digest differs")

    layer_files = manifest["layer_files"]
    expected_bytes = expected_layer_bytes(cached)
    if not isinstance(layer_files, list) or len(layer_files) != EXPECTED_LAYERS:
        raise CompositeBundleError("composite layer manifest is incomplete")
    seen: set[int] = set()
    for ordinal, layer in enumerate(layer_files):
        if not isinstance(layer, dict) or set(layer) != _LAYER_KEYS:
            raise CompositeBundleError("composite layer entry keys are not closed")
        index = layer["layer"]
        expected_path = f"layer-{ordinal:02d}.kv.f16le"
        if (
            type(index) is not int
            or index != ordinal
            or index in seen
            or layer["path"] != expected_path
            or layer["bytes"] != expected_bytes
        ):
            raise CompositeBundleError("composite layer order/path/geometry differs")
        seen.add(index)
        _require_digest("composite layer payload", layer["sha256"])

    unsigned = dict(manifest)
    del unsigned["hmac_sha256"]
    expected_tag = _tag(unsigned, key)
    if not hmac.compare_digest(supplied_tag, expected_tag):
        raise CompositeBundleError("composite manifest HMAC rejected")
    return manifest


class CompositeBundleWriter:
    """Exclusive atomic writer for one portable prefix bundle."""

    def __init__(
        self,
        destination: Path,
        *,
        token_ids: list[int],
        source_checkpoint_revision: str,
        source_checkpoint_artifact_sha256: str,
        source_engine_mode: str,
        hmac_key_id: str,
        hmac_key: bytes,
    ) -> None:
        if not destination.is_absolute() or not destination.parent.is_dir():
            raise CompositeBundleError("composite destination must have an existing absolute parent")
        if destination.exists() or destination.is_symlink():
            raise CompositeBundleError("composite destination already exists")
        _require_identifier("source checkpoint revision", source_checkpoint_revision)
        _require_digest("source checkpoint artifact", source_checkpoint_artifact_sha256)
        _require_identifier("HMAC key id", hmac_key_id)
        if source_engine_mode not in {"native", "exact"}:
            raise CompositeBundleError("composite source engine mode is not closed")
        if len(hmac_key) != 32:
            raise CompositeBundleError("composite HMAC key is not 32 bytes")
        if len(token_ids) < 2:
            raise CompositeBundleError("composite transcript needs a held boundary token")
        if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in token_ids):
            raise CompositeBundleError("composite transcript contains an invalid token")

        self.destination = destination
        self._key = hmac_key
        self._stage = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
        )
        os.chmod(self._stage, 0o700)
        self._lock_path = destination.parent / f".{destination.name}.publish.lock"
        try:
            self._lock_fd = os.open(self._lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as error:
            shutil.rmtree(self._stage)
            raise CompositeBundleError(f"composite publication is already reserved: {error}") from error
        self._core: dict[str, Any] = {
            "cached_token_count": len(token_ids) - 1,
            "created_unix_ms": time.time_ns() // 1_000_000,
            "dtype": "float16-le",
            "head_dim": EXPECTED_HEAD_DIM,
            "hmac_key_id": hmac_key_id,
            "key_layout": "post-rope-interleaved-pairs",
            "kv_heads": EXPECTED_KV_HEADS,
            "layer_files": [],
            "layers": EXPECTED_LAYERS,
            "portable_kv_abi": PORTABLE_KV_ABI,
            "schema": SCHEMA,
            "source_checkpoint_artifact_sha256": source_checkpoint_artifact_sha256,
            "source_checkpoint_revision": source_checkpoint_revision,
            "source_engine_mode": source_engine_mode,
            "token_ids": list(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "value_layout": "token-head-dimension",
            "vllm_commit": PINNED_VLLM_COMMIT,
        }
        self._closed = False

    @property
    def cached_token_count(self) -> int:
        return int(self._core["cached_token_count"])

    def write_layer(self, layer: int, payload: bytes) -> None:
        if self._closed or layer != len(self._core["layer_files"]):
            raise CompositeBundleError("composite layers must be written once in order")
        if len(payload) != expected_layer_bytes(self.cached_token_count):
            raise CompositeBundleError("composite layer payload byte geometry differs")
        name = f"layer-{layer:02d}.kv.f16le"
        path = self._stage / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        self._core["layer_files"].append(
            {
                "bytes": len(payload),
                "layer": layer,
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    def commit(self) -> dict[str, Any]:
        if self._closed or len(self._core["layer_files"]) != EXPECTED_LAYERS:
            raise CompositeBundleError("cannot publish an incomplete composite layer set")
        manifest = dict(self._core)
        manifest["hmac_sha256"] = _tag(self._core, self._key)
        validate_manifest(
            manifest,
            key=self._key,
            expected_key_id=str(self._core["hmac_key_id"]),
            expected_source_artifact_sha256=str(
                self._core["source_checkpoint_artifact_sha256"]
            ),
        )
        manifest_path = self._stage / "manifest.json"
        descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        stage_fd = os.open(self._stage, os.O_RDONLY)
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        if self.destination.exists() or self.destination.is_symlink():
            raise CompositeBundleError("composite destination appeared before publication")
        os.rename(self._stage, self.destination)
        parent_fd = os.open(self.destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self._closed = True
        self._release_lock()
        return manifest

    def abort(self) -> None:
        if self._closed:
            return
        shutil.rmtree(self._stage, ignore_errors=True)
        self._closed = True
        self._release_lock()

    def _release_lock(self) -> None:
        descriptor = getattr(self, "_lock_fd", None)
        if descriptor is not None:
            os.close(descriptor)
            self._lock_fd = None
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def __del__(self) -> None:
        try:
            self.abort()
        except BaseException:
            pass


def read_bundle_manifest(
    bundle: Path,
    *,
    key: bytes,
    expected_key_id: str,
    expected_source_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    if not bundle.is_absolute() or not bundle.is_dir() or bundle.is_symlink():
        raise CompositeBundleError("composite bundle must be an absolute non-symlink directory")
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CompositeBundleError("composite bundle has no regular manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompositeBundleError(f"cannot decode composite manifest: {error}") from error
    validated = validate_manifest(
        manifest,
        key=key,
        expected_key_id=expected_key_id,
        expected_source_artifact_sha256=expected_source_artifact_sha256,
    )
    allowed = {"manifest.json"} | {entry["path"] for entry in validated["layer_files"]}
    actual = {entry.name for entry in bundle.iterdir()}
    if actual != allowed:
        raise CompositeBundleError("composite bundle contains missing or unexpected objects")
    return validated


def read_layer_payload(bundle: Path, manifest: dict[str, Any], layer: int) -> bytes:
    if not 0 <= layer < EXPECTED_LAYERS:
        raise CompositeBundleError("composite layer index is out of range")
    entry = manifest["layer_files"][layer]
    path = bundle / entry["path"]
    if not path.is_file() or path.is_symlink():
        raise CompositeBundleError("composite layer object is not a regular file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CompositeBundleError(f"cannot read composite layer object: {error}") from error
    if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise CompositeBundleError("composite layer object digest/length rejected")
    return payload
