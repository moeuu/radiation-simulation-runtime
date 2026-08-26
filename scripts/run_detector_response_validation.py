"""Run the formal shield-free full-detector response validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from spectrum.detector_response_validation import (
    DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY,
    load_detector_response_validation_manifest,
)
from spectrum.detector_response_validation_runner import (
    run_detector_response_validation,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _REPOSITORY_ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)


def _parser() -> argparse.ArgumentParser:
    """Return the strict formal detector-response command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare the analytic incident-gamma response against native "
            "full-detector energy deposition for every formal gamma line."
        )
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--histories-per-energy",
        type=int,
        default=DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY,
    )
    parser.add_argument(
        "--transport-seed",
        type=int,
        help=(
            "Optional explicit diagnostic replay seed. Omit it for a fresh "
            "formal acquisition seed."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Acquire formal response evidence and return zero only on acceptance."""
    arguments = _parser().parse_args(argv)
    _, manifest_path = run_detector_response_validation(
        runtime_config_path=arguments.config,
        output_root=arguments.output_root,
        repository_root=_REPOSITORY_ROOT,
        histories_per_energy=arguments.histories_per_energy,
        transport_seed=arguments.transport_seed,
    )
    manifest = load_detector_response_validation_manifest(
        manifest_path,
        require_passed=False,
    )
    print(manifest_path.resolve(), flush=True)
    print(
        json.dumps(
            {
                "all_passed": manifest["all_passed"],
                "transport_seed": manifest["transport_seed"],
                "histories_per_energy": manifest["histories_per_energy"],
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
