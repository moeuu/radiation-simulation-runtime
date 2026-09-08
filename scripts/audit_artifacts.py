"""Audit output placement and accidental Git tracking without changing files."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESULT_CATEGORIES = frozenset({
    "runs", "diagnostics", "benchmarks", "previews", "blender_environments",
    "detector_green_construction", "detector_green_validation",
    "full_spectrum_all64_acceptance", "mean_calibration",
    "decay_cascade_comparison", "ral_isaac_figures", "ral_supplementary_video",
})
# Existing acceptance evidence and figure tools own these fixed bundle layouts.
FIXED_BUNDLES = frozenset({
    "full_spectrum_all64_acceptance", "ral_isaac_figures",
    "ral_supplementary_video",
})


def audit(root: Path) -> tuple[list[str], list[str]]:
    """Return the artifact inventory and placement/index violations."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "-ci", "--exclude-standard"], cwd=root,
    ).decode().split("\0")
    problems = [f"Tracked generated/local file: {p}" for p in tracked if p]
    for relative in ("build/lib", "sim", "configs/geant4/legacy"):
        path = root / relative
        if path.exists() or path.is_symlink():
            problems.append(f"Retired directory: {relative}")
    inventory = []
    for name in ("results", "private_runs", "logs", "tmp"):
        base = root / name
        if not base.exists() and not base.is_symlink():
            continue
        if base.is_symlink() or not base.is_dir():
            problems.append(f"Expected a local directory: {name}")
            continue
        for entry in sorted(base.iterdir()):
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                problems.append(f"Output root must not be a symlink: {relative}")
                continue
            files = [entry] if entry.is_file() else entry.rglob("*")
            size = sum(p.stat().st_size for p in files
                       if not p.is_symlink() and p.is_file())
            inventory.append(f"{size / 1024**2:9.2f} MiB  {relative}")
            if name != "tmp" and not entry.is_dir():
                problems.append(f"Loose output; use a run directory: {relative}")
            if name != "results":
                continue
            if entry.name not in RESULT_CATEGORIES:
                problems.append(f"Unexpected results category: {relative}")
            elif entry.is_dir() and entry.name not in FIXED_BUNDLES:
                for child in sorted(entry.iterdir()):
                    if child.is_symlink() or not child.is_dir():
                        problems.append(
                            "Expected a run directory: "
                            + child.relative_to(root).as_posix()
                        )
    return inventory, problems


def main() -> int:
    """Print an inventory and optionally fail on hygiene problems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    inventory, problems = audit(ROOT)
    for line in inventory:
        print(line)
    for problem in problems:
        print(f"ERROR: {problem}")
    print(f"{len(problems)} hygiene violation(s); no files changed.")
    return int(arguments.check and bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
