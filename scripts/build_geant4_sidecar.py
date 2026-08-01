"""Build the native Geant4 sidecar executable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Final


ROOT = Path(__file__).resolve().parents[1]
BUILD_PROFILE_FLAGS: Final[dict[str, tuple[str, ...]]] = {
    "native": ("-O3", "-march=native", "-flto"),
    "portable": ("-O3", "-flto"),
}
PGO_MODES: Final[tuple[str, ...]] = ("off", "generate", "use")


def _split_flags(raw_flags: str) -> list[str]:
    """Split shell-style compiler flags into a list."""
    return shlex.split(raw_flags.strip()) if raw_flags.strip() else []


def _build_command(
    *,
    source_path: Path,
    output_path: Path,
    standard: str,
    profile: str,
    cflags: str,
    libs: str,
    pgo_mode: str = "off",
    pgo_dir: Path | None = None,
) -> list[str]:
    """Return the compiler command for one supported release profile."""
    try:
        profile_flags = BUILD_PROFILE_FLAGS[profile]
    except KeyError as exc:
        supported = ", ".join(sorted(BUILD_PROFILE_FLAGS))
        raise ValueError(
            f"Unsupported build profile {profile!r}; expected one of: "
            f"{supported}."
        ) from exc
    if pgo_mode not in PGO_MODES:
        raise ValueError(
            f"Unsupported PGO mode {pgo_mode!r}; expected one of: "
            f"{', '.join(PGO_MODES)}."
        )
    pgo_flags: tuple[str, ...] = ()
    if pgo_mode != "off":
        if pgo_dir is None:
            raise ValueError("PGO generate/use mode requires an explicit pgo_dir.")
        pgo_option = (
            "-fprofile-generate"
            if pgo_mode == "generate"
            else "-fprofile-use"
        )
        pgo_flags = (f"{pgo_option}={pgo_dir.as_posix()}",)
        if pgo_mode == "use":
            pgo_flags += ("-fprofile-correction",)
    return [
        "g++",
        *_split_flags(cflags),
        f"-std={standard}",
        *profile_flags,
        *pgo_flags,
        source_path.as_posix(),
        *_split_flags(libs),
        "-o",
        output_path.as_posix(),
    ]


def _write_build_metadata(
    *,
    metadata_path: Path,
    geant4_version: str,
    profile: str,
    standard: str,
    command: list[str],
    pgo_mode: str,
    pgo_dir: Path | None,
) -> None:
    """Write a deterministic record of the successful compiler invocation."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "build_profile": profile,
        "command": command,
        "compiler": command[0],
        "cxx_standard": standard,
        "geant4_version": geant4_version.strip(),
        "optimization_flags": list(BUILD_PROFILE_FLAGS[profile]),
        "pgo_mode": pgo_mode,
        "pgo_profile_directory": (
            None if pgo_dir is None else pgo_dir.as_posix()
        ),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Resolve Geant4 flags and compile the native sidecar executable."""
    parser = argparse.ArgumentParser(
        description="Build the native Geant4 sidecar executable."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=(ROOT / "native" / "geant4_sidecar" / "geant4_sidecar.cpp").as_posix(),
        help="Path to the C++ source file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=(ROOT / "build" / "geant4_sidecar").as_posix(),
        help="Output executable path.",
    )
    parser.add_argument(
        "--std",
        type=str,
        default="c++17",
        help="C++ language standard to use for the build.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(BUILD_PROFILE_FLAGS),
        default="native",
        help=(
            "Release optimization profile. 'native' targets the local CPU; "
            "'portable' omits host-specific instructions."
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default=None,
        help=(
            "Optional path for a JSON record of the successful compiler "
            "invocation."
        ),
    )
    parser.add_argument(
        "--pgo",
        choices=PGO_MODES,
        default="off",
        help=(
            "Profile-guided optimization phase. Build with 'generate', run "
            "representative native workloads, then rebuild with 'use' and "
            "the same --pgo-dir."
        ),
    )
    parser.add_argument(
        "--pgo-dir",
        type=str,
        default=(ROOT / "build" / "geant4_sidecar.pgo").as_posix(),
        help="Directory holding GCC profile-guided optimization data.",
    )
    args = parser.parse_args()

    geant4_config = subprocess.run(
        ["geant4-config", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if geant4_config.returncode != 0:
        raise SystemExit(
            "geant4-config was not found. Install Geant4 and ensure "
            "geant4-config is on PATH before building."
        )
    cflags = subprocess.run(
        ["geant4-config", "--cflags"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    libs = subprocess.run(
        ["geant4-config", "--libs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    pgo_dir = (
        None
        if args.pgo == "off"
        else Path(args.pgo_dir).expanduser().resolve()
    )
    if args.pgo == "generate":
        assert pgo_dir is not None
        pgo_dir.mkdir(parents=True, exist_ok=True)
    elif args.pgo == "use" and (
        pgo_dir is None
        or not pgo_dir.is_dir()
        or not any(pgo_dir.iterdir())
    ):
        raise SystemExit(
            "PGO use mode requires a nonempty --pgo-dir generated by a "
            "representative instrumented run."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(
        source_path=source_path,
        output_path=output_path,
        standard=args.std,
        profile=args.profile,
        cflags=cflags,
        libs=libs,
        pgo_mode=args.pgo,
        pgo_dir=pgo_dir,
    )
    print(f"Geant4 sidecar build profile: {args.profile}")
    print(
        "Geant4 sidecar optimization flags: "
        f"{shlex.join(BUILD_PROFILE_FLAGS[args.profile])}"
    )
    print(f"Geant4 sidecar compiler command: {shlex.join(command)}")
    print(f"Geant4 sidecar PGO mode: {args.pgo}")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if args.metadata_output is not None:
        metadata_path = Path(args.metadata_output).expanduser().resolve()
        _write_build_metadata(
            metadata_path=metadata_path,
            geant4_version=geant4_config.stdout,
            profile=args.profile,
            standard=args.std,
            command=command,
            pgo_mode=args.pgo,
            pgo_dir=pgo_dir,
        )
        print(f"Recorded Geant4 sidecar build metadata: {metadata_path}")
    print(f"Built Geant4 sidecar: {output_path}")


if __name__ == "__main__":
    main()
