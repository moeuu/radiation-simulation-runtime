"""Authenticate native Geant4 libraries, data trees, and process settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess


NATIVE_EXECUTION_ENVIRONMENT_SCHEMA_VERSION = 2
_DYNAMIC_LOADER_ENVIRONMENT_KEYS = frozenset(
    {
        "GLIBC_TUNABLES",
        "LD_ASSUME_KERNEL",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    }
)
_NUMERIC_LOCALE_ENVIRONMENT_KEYS = frozenset(
    {"LANG", "LC_ALL", "LC_NUMERIC"}
)


@dataclass(frozen=True)
class _DirectoryTreeDigest:
    """Store one stable directory-tree content digest."""

    sha256: str
    file_count: int
    total_size_bytes: int


def _canonical_json_sha256(payload: object) -> str:
    """Return the SHA-256 digest of one canonical strict JSON payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_regular_file_sha256(path: Path) -> tuple[str, int]:
    """Hash one regular file and reject replacement during the read."""
    if path.is_symlink():
        raise ValueError(
            f"Native execution-environment input cannot be a symlink: {path}."
        )
    status_before = path.stat()
    if not stat.S_ISREG(status_before.st_mode):
        raise ValueError(
            "Native execution-environment input must be a regular file: "
            f"{path}."
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    status_after = path.stat()
    identity_before = (
        status_before.st_dev,
        status_before.st_ino,
        status_before.st_size,
        status_before.st_mtime_ns,
    )
    identity_after = (
        status_after.st_dev,
        status_after.st_ino,
        status_after.st_size,
        status_after.st_mtime_ns,
    )
    if identity_after != identity_before:
        raise RuntimeError(
            "Native execution-environment input changed while it was hashed: "
            f"{path}."
        )
    return digest.hexdigest(), int(status_before.st_size)


def _directory_inventory(root: Path) -> tuple[tuple[str, str], ...]:
    """Return the exact sorted directory-tree entry inventory."""
    inventory: list[tuple[str, str]] = []
    for current_root, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(
                    "Geant4 data trees cannot contain symlinked directories: "
                    f"{path}."
                )
            if not path.is_dir():
                raise ValueError(
                    "Geant4 data-tree entry is not a directory: "
                    f"{path}."
                )
            inventory.append((relative, "directory"))
        for name in file_names:
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(
                    "Geant4 data trees cannot contain symlinked files: "
                    f"{path}."
                )
            if not path.is_file():
                raise ValueError(
                    "Geant4 data-tree entry is not a regular file: "
                    f"{path}."
                )
            inventory.append((relative, "file"))
    return tuple(sorted(inventory))


def _stable_directory_tree_sha256(root: Path) -> _DirectoryTreeDigest:
    """Hash every path and file byte in one stable Geant4 data tree."""
    inventory_before = _directory_inventory(root)
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for relative_path, entry_type in inventory_before:
        encoded_path = relative_path.encode("utf-8")
        encoded_type = entry_type.encode("ascii")
        digest.update(len(encoded_type).to_bytes(8, "big"))
        digest.update(encoded_type)
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if entry_type == "file":
            file_hash, file_size = _stable_regular_file_sha256(
                root / relative_path
            )
            digest.update(bytes.fromhex(file_hash))
            digest.update(file_size.to_bytes(8, "big"))
            file_count += 1
            total_size += file_size
    inventory_after = _directory_inventory(root)
    if inventory_after != inventory_before:
        raise RuntimeError(
            f"Geant4 data tree changed while it was hashed: {root}."
        )
    return _DirectoryTreeDigest(
        sha256=digest.hexdigest(),
        file_count=file_count,
        total_size_bytes=total_size,
    )


def _is_geant4_data_environment_key(name: str) -> bool:
    """Return whether an environment variable selects Geant4 physics data."""
    return name == "GEANT4_DATA_DIR" or (
        name.startswith("G4") and name.endswith("DATA")
    )


def _canonical_library_search_path(raw_value: str) -> str:
    """Return one strict ordered search path with duplicate entries removed."""
    raw_entries = raw_value.split(os.pathsep)
    if not raw_entries or any(not entry for entry in raw_entries):
        raise ValueError(
            "LD_LIBRARY_PATH cannot contain empty entries or an implicit "
            "working-directory search."
        )
    canonical_entries: list[str] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        candidate = Path(raw_entry)
        if not candidate.is_absolute():
            raise ValueError("LD_LIBRARY_PATH entries must be absolute directories.")
        absolute = candidate.absolute()
        if absolute.is_symlink():
            raise ValueError(
                f"LD_LIBRARY_PATH entries cannot be symlinks: {absolute}."
            )
        try:
            resolved = absolute.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"LD_LIBRARY_PATH directory is missing: {absolute}."
            ) from exc
        if resolved != absolute or not absolute.is_dir():
            raise ValueError(
                "LD_LIBRARY_PATH entries must name exact nonsymlinked "
                f"directories: {absolute}."
            )
        normalized = absolute.as_posix()
        if normalized not in seen:
            seen.add(normalized)
            canonical_entries.append(normalized)
    return os.pathsep.join(canonical_entries)


def _relevant_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Return canonical settings that can affect native physics execution."""
    relevant: dict[str, str] = {}
    for name, value in environment.items():
        if type(name) is not str or type(value) is not str:
            raise TypeError("Native process environment must map strings to strings.")
        if (
            name.startswith("G4")
            or name.startswith("GEANT4")
            or name in _DYNAMIC_LOADER_ENVIRONMENT_KEYS
            or name in _NUMERIC_LOCALE_ENVIRONMENT_KEYS
        ):
            relevant[name] = (
                _canonical_library_search_path(value)
                if name == "LD_LIBRARY_PATH"
                else value
            )
    effective_numeric_locale = (
        relevant.get("LC_ALL")
        or relevant.get("LC_NUMERIC")
        or relevant.get("LANG")
    )
    if effective_numeric_locale is not None:
        for shadowed_name in ("LANG", "LC_NUMERIC"):
            if shadowed_name in relevant:
                relevant[shadowed_name] = effective_numeric_locale
    return dict(sorted(relevant.items()))


def _strict_data_root(raw_path: str, *, variable_name: str) -> Path:
    """Resolve one explicit absolute nonsymlinked Geant4 data directory."""
    if not raw_path:
        raise ValueError(f"{variable_name} must not be empty.")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise ValueError(f"{variable_name} must name an absolute directory.")
    candidate = candidate.absolute()
    if candidate.is_symlink():
        raise ValueError(f"{variable_name} cannot name a symlink: {candidate}.")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{variable_name} Geant4 data directory is missing: {candidate}."
        ) from exc
    if resolved != candidate or not candidate.is_dir():
        raise ValueError(
            f"{variable_name} must name one exact nonsymlinked directory: "
            f"{candidate}."
        )
    return candidate


def _dynamic_library_paths(
    executable: Path,
    *,
    environment: Mapping[str, str],
) -> tuple[tuple[str, Path], ...]:
    """Resolve the complete dynamic-library closure selected for an executable."""
    ldd_path = shutil.which("ldd")
    if ldd_path is None:
        raise FileNotFoundError(
            "ldd is required to authenticate native dynamic libraries."
        )
    result = subprocess.run(
        [ldd_path, executable.as_posix()],
        env=dict(environment),
        text=True,
        capture_output=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to resolve native Geant4 dynamic libraries: "
            f"returncode={result.returncode}, stderr={result.stderr.strip()!r}."
        )
    resolved: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=> not found" in line:
            raise RuntimeError(
                f"Native Geant4 dynamic library is unresolved: {line}."
            )
        if line.startswith("linux-vdso.so"):
            continue
        if "=>" in line:
            library_name, remainder = line.split("=>", maxsplit=1)
            path_text = remainder.rsplit(" (", maxsplit=1)[0].strip()
            name = library_name.strip()
        else:
            path_text = line.rsplit(" (", maxsplit=1)[0].strip()
            name = Path(path_text).name
        if not path_text.startswith("/") or not name:
            raise RuntimeError(
                f"Unrecognized native dynamic-library resolution line: {line}."
            )
        if name in seen_names:
            raise RuntimeError(
                f"Native dynamic-library closure repeats {name!r}."
            )
        try:
            path = Path(path_text).resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Resolved native dynamic library is missing: {path_text}."
            ) from exc
        if not path.is_file():
            raise ValueError(
                f"Resolved native dynamic library is not a file: {path}."
            )
        seen_names.add(name)
        resolved.append((name, path))
    if not resolved:
        raise RuntimeError(
            "Native Geant4 executable has no authenticated dynamic libraries."
        )
    return tuple(resolved)


def _strict_native_executable(executable_path: str | Path) -> Path:
    """Return one absolute regular executable without symlink traversal."""
    executable = Path(executable_path).absolute()
    if executable.is_symlink():
        raise ValueError(
            f"Native Geant4 executable cannot be a symlink: {executable}."
        )
    try:
        resolved_executable = executable.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Native Geant4 executable is missing: {executable}."
        ) from exc
    if resolved_executable != executable or not executable.is_file():
        raise ValueError(
            "Native Geant4 executable must be one exact regular file without "
            f"symlink traversal: {executable}."
        )
    if executable.stat().st_mode & 0o111 == 0:
        raise PermissionError(
            f"Native Geant4 executable is not executable: {executable}."
        )
    return executable


def native_executable_sha256(executable_path: str | Path) -> str:
    """Hash one strict native Geant4 executable without following symlinks."""
    executable = _strict_native_executable(executable_path)
    digest, _ = _stable_regular_file_sha256(executable)
    return digest


def native_execution_environment_bundle_payload(
    executable_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the exact native library, Geant4-data, and environment payload."""
    executable = _strict_native_executable(executable_path)
    process_environment = (
        dict(os.environ) if environment is None else dict(environment)
    )
    relevant_environment = _relevant_environment(process_environment)
    data_variables = {
        name: value
        for name, value in relevant_environment.items()
        if _is_geant4_data_environment_key(name)
    }
    if not data_variables:
        raise RuntimeError(
            "Native Geant4 execution requires an explicit GEANT4_DATA_DIR or "
            "G4*DATA environment contract."
        )
    data_tree_cache: dict[Path, _DirectoryTreeDigest] = {}
    data_trees: list[dict[str, object]] = []
    for variable_name, raw_path in sorted(data_variables.items()):
        root = _strict_data_root(raw_path, variable_name=variable_name)
        tree = data_tree_cache.get(root)
        if tree is None:
            tree = _stable_directory_tree_sha256(root)
            data_tree_cache[root] = tree
        data_trees.append(
            {
                "environment_variable": variable_name,
                "resolved_path": root.as_posix(),
                "tree_sha256": tree.sha256,
                "file_count": tree.file_count,
                "total_size_bytes": tree.total_size_bytes,
            }
        )
    dynamic_libraries: list[dict[str, object]] = []
    for library_name, path in _dynamic_library_paths(
        executable,
        environment=process_environment,
    ):
        library_sha256, library_size = _stable_regular_file_sha256(path)
        dynamic_libraries.append(
            {
                "loader_name": library_name,
                "resolved_path": path.as_posix(),
                "file_sha256": library_sha256,
                "size_bytes": library_size,
            }
        )
    return {
        "schema_version": NATIVE_EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
        "relevant_environment": relevant_environment,
        "dynamic_libraries": dynamic_libraries,
        "geant4_data_trees": data_trees,
    }


def native_execution_environment_bundle_sha256(
    executable_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Hash the exact native Geant4 execution-environment bundle."""
    return _canonical_json_sha256(
        native_execution_environment_bundle_payload(
            executable_path,
            environment=environment,
        )
    )


def require_native_execution_bundle(
    executable_path: str | Path,
    *,
    expected_executable_sha256: str,
    expected_environment_sha256: str,
) -> None:
    """Require exact approved native inputs immediately before process launch."""
    actual_executable = native_executable_sha256(executable_path)
    if actual_executable != expected_executable_sha256:
        raise RuntimeError(
            "Native Geant4 executable changed after provenance validation."
        )
    actual_environment = native_execution_environment_bundle_sha256(
        executable_path
    )
    if actual_environment != expected_environment_sha256:
        raise RuntimeError(
            "Native Geant4 libraries, physics data, or environment changed "
            "after provenance validation."
        )


__all__ = [
    "NATIVE_EXECUTION_ENVIRONMENT_SCHEMA_VERSION",
    "native_executable_sha256",
    "native_execution_environment_bundle_payload",
    "native_execution_environment_bundle_sha256",
    "require_native_execution_bundle",
]
