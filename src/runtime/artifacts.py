"""Atomic publication helpers for cross-repository runtime artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
from dataclasses import dataclass
import errno
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import MappingProxyType
from typing import Any

from runtime.provenance import (
    DigestIdentity,
    strict_canonical_json_bytes,
    strict_sha256_json,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_INVENTORY_DIGEST_ALGORITHM = (
    "rotating-shield-runtime.artifact-inventory-v1+canonical-json-sha256"
)


@dataclass(frozen=True, slots=True)
class ArtifactInventory:
    """Store an immutable path-to-digest inventory for ordinary files."""

    sha256_by_path: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate, sort, and freeze the portable relative-path mapping."""
        normalized: dict[str, str] = {}
        for relative_path, digest in self.sha256_by_path.items():
            if not isinstance(relative_path, str) or not relative_path:
                raise TypeError("Artifact inventory paths must be nonempty strings.")
            path = Path(relative_path)
            if path.is_absolute() or relative_path in {".", ".."} or ".." in path.parts:
                raise ValueError(
                    "Artifact inventory paths must be root-relative without '..'."
                )
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(
                    "Artifact inventory digests must be lowercase SHA-256 strings."
                )
            normalized[relative_path] = digest
        object.__setattr__(
            self,
            "sha256_by_path",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    @property
    def sha256(self) -> str:
        """Return the canonical digest of the complete inventory mapping."""
        return strict_sha256_json(dict(self.sha256_by_path))

    @property
    def digest(self) -> DigestIdentity:
        """Return the inventory hash together with its stable algorithm ID."""
        return DigestIdentity(
            algorithm=ARTIFACT_INVENTORY_DIGEST_ALGORITHM,
            sha256=self.sha256,
        )

    @property
    def file_count(self) -> int:
        """Return the number of regular files represented by the inventory."""
        return len(self.sha256_by_path)

    @classmethod
    def from_allowlist(
        cls,
        root: str | Path,
        relative_paths: tuple[str, ...] | list[str] | frozenset[str],
    ) -> "ArtifactInventory":
        """Hash exactly one declared allowlist of root-relative regular files."""
        supplied_root = Path(root)
        if supplied_root.is_symlink():
            raise ValueError("Artifact inventory root must not be a symlink.")
        root_path = supplied_root.resolve()
        if not root_path.is_dir():
            raise NotADirectoryError(
                f"Artifact inventory root is not a directory: {root_path}"
            )
        names = tuple(relative_paths)
        if len(names) != len(set(names)):
            raise ValueError("Artifact inventory allowlist contains duplicate paths.")
        digests: dict[str, str] = {}
        for relative in sorted(names):
            normalized = _relative_artifact_path(relative)
            candidate = root_path / normalized
            if candidate.is_symlink() or not candidate.is_file():
                raise FileNotFoundError(
                    f"Allowlisted artifact is not a regular file: {relative}"
                )
            digest = sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[normalized.as_posix()] = digest.hexdigest()
        return cls(digests)


def _relative_artifact_path(value: str | Path) -> Path:
    """Return one portable root-relative artifact member path."""
    if not isinstance(value, (str, Path)):
        raise TypeError("Artifact member path must be text or Path.")
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or path in {Path("."), Path("..")}
        or ".." in path.parts
    ):
        raise ValueError("Artifact member path must be root-relative without '..'.")
    return path


def build_artifact_inventory(root: str | Path) -> ArtifactInventory:
    """Hash every root-contained regular file without applying schema policy."""
    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise ValueError("Artifact inventory root must not be a symlink.")
    root_path = supplied_root.resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Artifact inventory root is not a directory: {root_path}")
    digests: dict[str, str] = {}
    for candidate in sorted(root_path.rglob("*")):
        relative = candidate.relative_to(root_path).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"Artifact inventory must not contain symlink {relative}.")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"Artifact inventory entry is not a regular file: {relative}")
        digest = sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[relative] = digest.hexdigest()
    return ArtifactInventory(digests)


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically replace one file with durable serialized bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace one text file using the requested encoding."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Serialize strict canonical JSON and publish it as one atomic file."""
    return atomic_write_bytes(path, strict_canonical_json_bytes(payload))


def atomic_copy_file(source: str | Path, target: str | Path) -> Path:
    """Copy a completed file and atomically replace its publication target."""
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Artifact source does not exist: {source_path}")
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source_path.open("rb") as source_handle, temporary.open(
            "wb"
        ) as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target_path)
        _fsync_directory(target_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target_path


def _linux_renameat2(source: Path, target: Path, *, flags: int) -> None:
    """Invoke Linux renameat2 for no-replace or atomic directory exchange."""
    if sys.platform != "linux":
        raise OSError(errno.ENOTSUP, "Atomic directory exchange requires Linux renameat2.")
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable on this platform.")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        flags,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


class AtomicBundlePublisher:
    """Stage and atomically publish one immutable artifact directory bundle.

    ``create`` installs a new directory without replacing an existing target.
    ``replace_known`` clones an existing directory, permits changes only to a
    declared member allowlist, then atomically exchanges the two directories.
    The latter intentionally fails on platforms without Linux ``renameat2``
    instead of silently degrading to a partially visible multi-file update.
    """

    def __init__(
        self,
        target: str | Path,
        *,
        policy: str = "create",
        known_members: tuple[str, ...] | list[str] | frozenset[str] = (),
    ) -> None:
        """Create one same-filesystem staging directory for a bundle update."""
        if policy not in {"create", "replace_known"}:
            raise ValueError("Bundle policy must be 'create' or 'replace_known'.")
        self.target = Path(target).expanduser().absolute()
        if self.target == Path(self.target.anchor):
            raise ValueError("Artifact bundle target must not be a filesystem root.")
        self.policy = policy
        self._known_members = frozenset(
            _relative_artifact_path(value).as_posix() for value in known_members
        )
        if policy == "replace_known" and not self._known_members:
            raise ValueError("replace_known requires a nonempty member allowlist.")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._staging = Path(
            tempfile.mkdtemp(
                dir=self.target.parent,
                prefix=f".{self.target.name}.bundle-",
            )
        )
        self._published = False
        self._baseline_inventory: ArtifactInventory | None = None
        try:
            if policy == "create":
                if self.target.exists() or self.target.is_symlink():
                    raise FileExistsError(
                        f"Refusing to replace existing artifact bundle: {self.target}"
                    )
            else:
                if self.target.is_symlink() or not self.target.is_dir():
                    raise NotADirectoryError(
                        f"replace_known target is not a regular directory: {self.target}"
                    )
                existing = build_artifact_inventory(self.target)
                self._baseline_inventory = existing
                shutil.copytree(
                    self.target,
                    self._staging,
                    dirs_exist_ok=True,
                    symlinks=False,
                )
        except BaseException:
            shutil.rmtree(self._staging, ignore_errors=True)
            raise

    @property
    def staging_path(self) -> Path:
        """Return the private staging root for specialized byte encoders."""
        return self._staging

    def _member(self, relative_path: str | Path) -> Path:
        """Resolve and authorize one staging member path."""
        relative = _relative_artifact_path(relative_path)
        name = relative.as_posix()
        if self.policy == "replace_known" and name not in self._known_members:
            raise ValueError(f"Bundle member is outside the replacement allowlist: {name}")
        destination = self._staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def write_bytes(self, relative_path: str | Path, payload: bytes) -> Path:
        """Write and synchronize one private staging member."""
        if not isinstance(payload, bytes):
            raise TypeError("Bundle member payload must be bytes.")
        destination = self._member(relative_path)
        if destination.is_symlink() or destination.is_dir():
            raise ValueError("Bundle member destination must be a regular file path.")
        with destination.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return destination

    def write_json(self, relative_path: str | Path, payload: object) -> Path:
        """Write one strict canonical JSON member into the private stage."""
        return self.write_bytes(relative_path, strict_canonical_json_bytes(payload))

    def copy_file(self, source: str | Path, relative_path: str | Path) -> Path:
        """Copy and synchronize one completed regular file into the stage."""
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(f"Bundle source is not a regular file: {source_path}")
        destination = self._member(relative_path)
        with source_path.open("rb") as source_handle, destination.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        return destination

    def remove(self, relative_path: str | Path) -> None:
        """Remove one allowlisted staging member from a replacement generation."""
        destination = self._member(relative_path)
        if destination.is_dir():
            raise IsADirectoryError(destination)
        destination.unlink(missing_ok=True)

    def inventory(self) -> ArtifactInventory:
        """Return the authenticated inventory of the staged generation."""
        return build_artifact_inventory(self._staging)

    def publish(self) -> ArtifactInventory:
        """Atomically install the staged directory and return its inventory."""
        if self._published:
            raise RuntimeError("Artifact bundle has already been published.")
        inventory = self.inventory()
        _fsync_tree_directories(self._staging)
        if self.policy == "create":
            _linux_renameat2(self._staging, self.target, flags=1)
            self._published = True
        else:
            staged_names = set(inventory.sha256_by_path)
            baseline = self._baseline_inventory
            if baseline is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("Replacement bundle has no baseline inventory.")
            baseline_names = set(baseline.sha256_by_path)
            undeclared_changes = {
                name
                for name in staged_names | baseline_names
                if name not in self._known_members
                and inventory.sha256_by_path.get(name)
                != baseline.sha256_by_path.get(name)
            }
            if undeclared_changes:
                raise ValueError(
                    "Replacement stage changed undeclared members: "
                    f"{sorted(undeclared_changes)}"
                )
            _linux_renameat2(self._staging, self.target, flags=2)
            self._published = True
            # The exchange is already committed.  The old generation now at
            # the staging name is cleanup-only and must not turn a successful
            # publication into a false failure signal.
            shutil.rmtree(self._staging, ignore_errors=True)
        _fsync_directory(self.target.parent)
        return inventory

    def close(self) -> None:
        """Discard an unpublished stage without changing the public target."""
        if not self._published:
            shutil.rmtree(self._staging, ignore_errors=True)

    def __enter__(self) -> "AtomicBundlePublisher":
        """Return this publisher for a context-managed transaction."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Discard the private stage unless publication already succeeded."""
        del exc_type, exc, traceback
        self.close()


class DurableJSONLWriter:
    """Append strict one-line JSON records with locking and durable flushes."""

    def __init__(self, path: str | Path, *, mode: int = 0o600) -> None:
        """Open one append-only non-symlink JSONL file descriptor."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        self._descriptor = os.open(self.path, flags, mode)
        _fsync_directory(self.path.parent)
        self._closed = False

    def append(self, payload: object) -> int:
        """Append one canonical JSON object and return the encoded byte count."""
        if self._closed:
            raise ValueError("Cannot append to a closed JSONL writer.")
        normalized = json.loads(strict_canonical_json_bytes(payload))
        if not isinstance(normalized, dict):
            raise TypeError("Durable JSONL records must be JSON objects.")
        encoded = (
            json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        fcntl.flock(self._descriptor, fcntl.LOCK_EX)
        try:
            written = 0
            while written < len(encoded):
                count = os.write(self._descriptor, encoded[written:])
                if count <= 0:
                    raise OSError("JSONL append made no forward progress.")
                written += count
            os.fsync(self._descriptor)
        finally:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        return len(encoded)

    def close(self) -> None:
        """Close the append descriptor exactly once."""
        if not self._closed:
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> "DurableJSONLWriter":
        """Return this writer for a context-managed append session."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the descriptor when leaving a managed append session."""
        del exc_type, exc, traceback
        self.close()


def publish_artifact_manifest(
    path: str | Path,
    inventory: ArtifactInventory,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically publish a versioned manifest for an authenticated inventory."""
    if not isinstance(inventory, ArtifactInventory):
        raise TypeError("inventory must be ArtifactInventory.")
    payload = {
        "schema_version": 1,
        "inventory_digest": inventory.digest.to_payload(),
        "artifacts": dict(inventory.sha256_by_path),
        "metadata": {} if metadata is None else dict(metadata),
    }
    return atomic_write_json(path, payload)


def _fsync_directory(path: Path) -> None:
    """Synchronize one directory after an atomic entry replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    """Synchronize staged directories from leaves through the root."""
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


__all__ = [
    "ARTIFACT_INVENTORY_DIGEST_ALGORITHM",
    "AtomicBundlePublisher",
    "ArtifactInventory",
    "DurableJSONLWriter",
    "atomic_copy_file",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "build_artifact_inventory",
    "publish_artifact_manifest",
]
