#!/usr/bin/env python3
"""Validate a detector Green operator on independent monoenergetic photons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from spectrum.detector_green_validation import (
    DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY,
    load_detector_green_validation_manifest,
)
from spectrum.detector_green_validation_runner import (
    run_detector_green_validation,
)
from spectrum.detector_green_operator import DetectorGreenOperator


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _REPOSITORY_ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)


def _parser() -> argparse.ArgumentParser:
    """Return the fail-closed validation command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--operator-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--histories-per-energy",
        type=int,
        default=DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY,
    )
    parser.add_argument("--design-seed", type=int)
    parser.add_argument("--transport-seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Acquire validation evidence and return zero only on acceptance."""
    arguments = _parser().parse_args(argv)
    _, manifest_path = run_detector_green_validation(
        runtime_config_path=arguments.runtime_config,
        operator_manifest_path=arguments.operator_manifest,
        output_root=arguments.output_root,
        repository_root=_REPOSITORY_ROOT,
        histories_per_energy=arguments.histories_per_energy,
        design_seed=arguments.design_seed,
        transport_seed=arguments.transport_seed,
    )
    operator = DetectorGreenOperator.from_artifact(arguments.operator_manifest)
    manifest = load_detector_green_validation_manifest(
        manifest_path,
        operator=operator,
        require_passed=False,
    )
    print(manifest_path.resolve(), flush=True)
    print(
        json.dumps(
            {
                "all_passed": manifest["all_passed"],
                "design_seed": manifest["design_seed"],
                "transport_seed": manifest["transport_seed"],
                "metrics": manifest["metrics"],
            },
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )
    return 0 if manifest["all_passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
