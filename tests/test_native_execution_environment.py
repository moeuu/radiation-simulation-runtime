"""Tests for strict native Geant4 execution-environment provenance."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from sim.geant4_app.execution_environment import (
    native_execution_environment_bundle_payload,
    native_execution_environment_bundle_sha256,
)


def _native_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create one executable, shared library, and tiny Geant4 data tree."""
    executable = tmp_path / "geant4_sidecar"
    executable.write_bytes(b"native executable")
    executable.chmod(0o755)
    library = tmp_path / "libG4physicslists.so"
    library.write_bytes(b"geant4 library v1")
    data_root = tmp_path / "geant4-data"
    dataset = data_root / "G4EMLOW-test"
    dataset.mkdir(parents=True)
    (dataset / "physics.dat").write_bytes(b"cross sections v1")
    return executable, library, data_root


def _mock_ldd(
    monkeypatch: pytest.MonkeyPatch,
    *,
    library: Path,
) -> None:
    """Return one deterministic dynamic-library resolution."""

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return a successful ldd result for the fake native executable."""
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["ldd"],
            returncode=0,
            stdout=f"libG4physicslists.so => {library} (0x1)\n",
            stderr="",
        )

    monkeypatch.setattr(
        "sim.geant4_app.execution_environment.subprocess.run",
        run,
    )


def test_bundle_binds_libraries_data_and_geant4_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every mutable native physics input must change the bundle digest."""
    executable, library, data_root = _native_fixture(tmp_path)
    _mock_ldd(monkeypatch, library=library)
    environment = {
        "GEANT4_DATA_DIR": data_root.as_posix(),
        "G4NEUTRONHP_SKIP_MISSING_ISOTOPES": "0",
        "LANG": "C",
    }

    original = native_execution_environment_bundle_sha256(
        executable,
        environment=environment,
    )
    library.write_bytes(b"geant4 library v2")
    changed_library = native_execution_environment_bundle_sha256(
        executable,
        environment=environment,
    )
    library.write_bytes(b"geant4 library v1")
    (data_root / "G4EMLOW-test" / "physics.dat").write_bytes(
        b"cross sections v2"
    )
    changed_data = native_execution_environment_bundle_sha256(
        executable,
        environment=environment,
    )
    changed_environment = native_execution_environment_bundle_sha256(
        executable,
        environment={
            **environment,
            "G4NEUTRONHP_SKIP_MISSING_ISOTOPES": "1",
        },
    )

    assert len({original, changed_library, changed_data}) == 3
    assert changed_environment != changed_data


def test_bundle_payload_records_complete_tiny_data_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provenance payload must record the selected tree and file count."""
    executable, library, data_root = _native_fixture(tmp_path)
    _mock_ldd(monkeypatch, library=library)

    payload = native_execution_environment_bundle_payload(
        executable,
        environment={"GEANT4_DATA_DIR": data_root.as_posix()},
    )

    assert payload["schema_version"] == 1
    assert payload["dynamic_libraries"][0]["loader_name"] == (
        "libG4physicslists.so"
    )
    assert payload["geant4_data_trees"] == [
        {
            "environment_variable": "GEANT4_DATA_DIR",
            "resolved_path": data_root.as_posix(),
            "tree_sha256": payload["geant4_data_trees"][0]["tree_sha256"],
            "file_count": 1,
            "total_size_bytes": len(b"cross sections v1"),
        }
    ]


def test_bundle_rejects_implicit_or_unresolved_native_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing data selection and unresolved shared libraries must fail closed."""
    executable, library, data_root = _native_fixture(tmp_path)
    _mock_ldd(monkeypatch, library=library)
    with pytest.raises(RuntimeError, match="explicit GEANT4_DATA_DIR"):
        native_execution_environment_bundle_sha256(
            executable,
            environment={},
        )

    monkeypatch.setattr(
        "sim.geant4_app.execution_environment.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["ldd"],
            returncode=0,
            stdout="libG4run.so => not found\n",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="unresolved"):
        native_execution_environment_bundle_sha256(
            executable,
            environment={"GEANT4_DATA_DIR": data_root.as_posix()},
        )


def test_bundle_rejects_symlinked_geant4_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physics-data aliases cannot hide replacement of an approved tree."""
    executable, library, data_root = _native_fixture(tmp_path)
    _mock_ldd(monkeypatch, library=library)
    linked_root = tmp_path / "linked-data"
    linked_root.symlink_to(data_root, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot name a symlink"):
        native_execution_environment_bundle_sha256(
            executable,
            environment={"GEANT4_DATA_DIR": linked_root.as_posix()},
        )
