"""Provide legacy provenance and strict JSON contracts for runtime artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_ALGORITHM_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.+_-]{2,127}$")


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"Strict JSON cannot contain {value}.")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one object while rejecting duplicate JSON member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Strict JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class DigestIdentity:
    """Bind a digest value to the exact algorithm that produced it."""

    algorithm: str
    sha256: str

    def __post_init__(self) -> None:
        """Reject ambiguous algorithm names and malformed SHA-256 values."""
        if (
            not isinstance(self.algorithm, str)
            or _DIGEST_ALGORITHM_PATTERN.fullmatch(self.algorithm) is None
        ):
            raise ValueError("Digest algorithm must be a stable lowercase identifier.")
        if (
            not isinstance(self.sha256, str)
            or _SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise ValueError("Digest value must be a lowercase SHA-256 string.")

    def to_payload(self) -> dict[str, str]:
        """Serialize the versioned digest identity to strict JSON data."""
        return {"algorithm": self.algorithm, "sha256": self.sha256}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DigestIdentity":
        """Parse one exact versioned digest identity payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("Digest identity must be an object.")
        if set(payload) != {"algorithm", "sha256"}:
            raise ValueError("Digest identity must contain exactly algorithm and sha256.")
        algorithm = payload["algorithm"]
        digest = payload["sha256"]
        if not isinstance(algorithm, str) or not isinstance(digest, str):
            raise TypeError("Digest identity algorithm and sha256 must be strings.")
        return cls(algorithm=algorithm, sha256=digest)


@dataclass(frozen=True, slots=True)
class ConfigIdentity:
    """Bind source configuration bytes to their resolved effective payload."""

    source_sha256: str
    effective_sha256: str

    def __post_init__(self) -> None:
        """Reject malformed source and effective SHA-256 values."""
        for name, value in (
            ("source_sha256", self.source_sha256),
            ("effective_sha256", self.effective_sha256),
        ):
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 string.")

    @classmethod
    def from_source_bytes(
        cls,
        source: bytes,
        effective_config: object,
    ) -> "ConfigIdentity":
        """Hash exact source bytes and one strict resolved configuration value."""
        if not isinstance(source, bytes):
            raise TypeError("Configuration source must be bytes.")
        return cls(
            source_sha256=hashlib.sha256(source).hexdigest(),
            effective_sha256=hashlib.sha256(
                strict_canonical_json_bytes(effective_config)
            ).hexdigest(),
        )

    @classmethod
    def from_path(
        cls,
        source: str | Path,
        effective_config: object,
    ) -> "ConfigIdentity":
        """Hash one source file and its separately resolved configuration value."""
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Configuration source must be a regular non-symlink file.")
        return cls.from_source_bytes(path.read_bytes(), effective_config)

    def to_payload(self) -> dict[str, object]:
        """Serialize this configuration identity as strict versioned JSON data."""
        return {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "effective_sha256": self.effective_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ConfigIdentity":
        """Parse one exact configuration-identity payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("Configuration identity must be an object.")
        expected = {"schema_version", "source_sha256", "effective_sha256"}
        if set(payload) != expected or payload.get("schema_version") != 1:
            raise ValueError("Configuration identity must use the exact v1 fields.")
        source = payload["source_sha256"]
        effective = payload["effective_sha256"]
        if not isinstance(source, str) or not isinstance(effective, str):
            raise TypeError("Configuration identity digests must be strings.")
        return cls(source_sha256=source, effective_sha256=effective)


def strict_json_loads(payload: str | bytes | bytearray) -> object:
    """Parse strict JSON while rejecting duplicate keys and non-finite values."""
    if not isinstance(payload, (str, bytes, bytearray)):
        raise TypeError("Strict JSON input must be text or bytes.")
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("Strict JSON bytes must be valid UTF-8.") from exc


def load_strict_json(path: str | Path) -> object:
    """Read and parse one strict UTF-8 JSON document from a regular file."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Strict JSON source must be a regular non-symlink file.")
    return strict_json_loads(source.read_bytes())


def json_safe(value: Any) -> Any:
    """Convert values under the legacy schema-v1 provenance policy.

    This compatibility function stringifies unsupported values and mapping
    keys. New artifact schemas must use :func:`strict_canonical_json_bytes`.
    """
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(
                "Canonical provenance JSON cannot contain NaN or infinity."
            )
        return value
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize legacy schema-v1 JSON without changing existing digests."""
    text = json.dumps(
        json_safe(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the legacy schema-v1 canonical JSON digest."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json_value(value: Any, *, location: str) -> Any:
    """Return JSON-native data while rejecting lossy or unstable coercions."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} keys must be JSON strings.")
            normalized[key] = _strict_json_value(
                nested,
                location=f"{location}.{key}",
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} contains unsupported JSON value {type(value).__name__}."
    )


def strict_canonical_json_bytes(value: Any) -> bytes:
    """Serialize a new-schema JSON value without implicit type coercion."""
    normalized = _strict_json_value(value, location="payload")
    text = json.dumps(
        normalized,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{text}\n".encode("utf-8")


def strict_sha256_json(value: Any) -> str:
    """Return the strict new-schema canonical JSON digest."""
    return hashlib.sha256(strict_canonical_json_bytes(value)).hexdigest()


def repository_commit(repository_root: Path | None = None) -> str:
    """Return the checked-out Git commit or an explicit unavailable marker."""
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    commit = completed.stdout.strip()
    return commit if commit else "unavailable"


def repository_source_snapshot_sha256(
    repository_root: Path | None = None,
) -> str:
    """Hash the actual runtime source/config snapshot, including dirty files."""
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Cannot enumerate the repository source snapshot."
        ) from exc
    prefixes = ("src/", "scripts/", "configs/", "native/", "tests/")
    root_files = frozenset(
        {"AGENTS.md", "main.py", "pyproject.toml", "uv.lock"}
    )
    paths = sorted(
        {
            Path(raw.decode("utf-8"))
            for raw in completed.stdout.split(b"\0")
            if raw
            and (
                raw.decode("utf-8") in root_files
                or raw.decode("utf-8").startswith(prefixes)
            )
        },
        key=lambda value: value.as_posix(),
    )
    digest = hashlib.sha256(b"repository_source_snapshot_v1\0")
    for relative in paths:
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(path.readlink().as_posix().encode("utf-8"))
        elif path.is_file():
            digest.update(b"file\0")
            digest.update(path.read_bytes())
        else:
            digest.update(b"missing\0")
        digest.update(b"\0")
    return digest.hexdigest()
