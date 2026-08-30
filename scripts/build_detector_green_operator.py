#!/usr/bin/env python3
"""Build an isotope-independent full-detector Green-operator artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectrum.detector_green_construction_runner import (
    DETECTOR_GREEN_MINIMUM_HISTORIES_PER_ENERGY,
    run_detector_green_construction,
)


def _parser() -> argparse.ArgumentParser:
    """Return the strict command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--histories-per-energy",
        type=int,
        default=DETECTOR_GREEN_MINIMUM_HISTORIES_PER_ENERGY,
    )
    parser.add_argument("--impact-strata", type=int, default=8)
    parser.add_argument("--transport-seed", type=int)
    return parser


def main() -> None:
    """Run construction and print the two immutable artifact paths."""
    arguments = _parser().parse_args()
    raw_path, manifest_path = run_detector_green_construction(
        runtime_config_path=arguments.runtime_config,
        output_root=arguments.output_root,
        repository_root=arguments.repository_root,
        histories_per_energy=arguments.histories_per_energy,
        impact_strata=arguments.impact_strata,
        transport_seed=arguments.transport_seed,
    )
    print(f"raw_corpus={raw_path}")
    print(f"operator_manifest={manifest_path}")


if __name__ == "__main__":
    main()
