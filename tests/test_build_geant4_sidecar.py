"""Tests for the native Geant4 sidecar build profiles."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import build_geant4_sidecar


def test_native_profile_enables_safe_host_optimizations() -> None:
    """The default local profile must optimize without unsafe math flags."""
    command = build_geant4_sidecar._build_command(
        source_path=Path("/tmp/sidecar.cpp"),
        output_path=Path("/tmp/sidecar"),
        standard="c++17",
        profile="native",
        cflags="-pthread -I/tmp/geant4",
        libs="-lG4run -lG4event",
    )

    assert "-O3" in command
    assert "-march=native" in command
    assert "-flto" in command
    assert "-ffast-math" not in command


def test_portable_profile_omits_host_specific_instructions() -> None:
    """The portable profile must retain optimization without ``-march``."""
    command = build_geant4_sidecar._build_command(
        source_path=Path("/tmp/sidecar.cpp"),
        output_path=Path("/tmp/sidecar"),
        standard="c++17",
        profile="portable",
        cflags="",
        libs="",
    )

    assert "-O3" in command
    assert "-flto" in command
    assert not any(flag.startswith("-march=") for flag in command)
    assert "-ffast-math" not in command


def test_build_command_rejects_unknown_profile() -> None:
    """An unknown release profile must fail before invoking the compiler."""
    with pytest.raises(ValueError, match="Unsupported build profile"):
        build_geant4_sidecar._build_command(
            source_path=Path("/tmp/sidecar.cpp"),
            output_path=Path("/tmp/sidecar"),
            standard="c++17",
            profile="unknown",
            cflags="",
            libs="",
        )


def test_pgo_build_flags_are_explicit_and_safe(tmp_path: Path) -> None:
    """PGO generation and use must retain the same safe release semantics."""
    pgo_dir = tmp_path / "profiles"
    generate = build_geant4_sidecar._build_command(
        source_path=Path("/tmp/sidecar.cpp"),
        output_path=Path("/tmp/sidecar"),
        standard="c++17",
        profile="native",
        cflags="",
        libs="",
        pgo_mode="generate",
        pgo_dir=pgo_dir,
    )
    use = build_geant4_sidecar._build_command(
        source_path=Path("/tmp/sidecar.cpp"),
        output_path=Path("/tmp/sidecar"),
        standard="c++17",
        profile="native",
        cflags="",
        libs="",
        pgo_mode="use",
        pgo_dir=pgo_dir,
    )

    assert f"-fprofile-generate={pgo_dir}" in generate
    assert f"-fprofile-use={pgo_dir}" in use
    assert "-fprofile-correction" in use
    assert "-ffast-math" not in generate
    assert "-ffast-math" not in use


def test_pgo_mode_requires_profile_directory() -> None:
    """PGO must not silently write profile data to an implicit directory."""
    with pytest.raises(ValueError, match="requires an explicit pgo_dir"):
        build_geant4_sidecar._build_command(
            source_path=Path("/tmp/sidecar.cpp"),
            output_path=Path("/tmp/sidecar"),
            standard="c++17",
            profile="native",
            cflags="",
            libs="",
            pgo_mode="generate",
        )


def test_cli_defaults_to_native_and_records_build_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The compatible CLI default must select and record the native profile."""
    source = tmp_path / "sidecar.cpp"
    output = tmp_path / "sidecar"
    metadata = tmp_path / "sidecar-build.json"
    calls: list[list[str]] = []

    def _fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Return deterministic Geant4 configuration and compiler results."""
        calls.append(command)
        if command == ["geant4-config", "--version"]:
            return subprocess.CompletedProcess(command, 0, "11.3.2\n", "")
        if command == ["geant4-config", "--cflags"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "-pthread -I/example/geant4\n",
                "",
            )
        if command == ["geant4-config", "--libs"]:
            return subprocess.CompletedProcess(command, 0, "-lG4run\n", "")
        assert command[0] == "g++"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_geant4_sidecar.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_geant4_sidecar.py",
            "--source",
            source.as_posix(),
            "--output",
            output.as_posix(),
            "--metadata-output",
            metadata.as_posix(),
        ],
    )

    build_geant4_sidecar.main()

    compiler_command = calls[-1]
    assert "-O3" in compiler_command
    assert "-march=native" in compiler_command
    assert "-flto" in compiler_command
    assert "-ffast-math" not in compiler_command
    captured = capsys.readouterr().out
    assert "build profile: native" in captured
    assert "compiler command:" in captured
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["build_profile"] == "native"
    assert payload["optimization_flags"] == [
        "-O3",
        "-march=native",
        "-flto",
    ]
    assert payload["geant4_version"] == "11.3.2"
    assert payload["pgo_mode"] == "off"
    assert payload["pgo_profile_directory"] is None
