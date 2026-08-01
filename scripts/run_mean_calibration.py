"""Acquire and fit the dedicated fixed-quota Geant4 mean calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sim.runtime import load_runtime_config
from spectrum.additive_scatter import (
    AdditiveNoncollidedTransportResponse,
)
from spectrum.mean_calibration_runner import (
    ExternalGeant4MeanCalibrationBackend,
    MeanCalibrationLayout,
    build_predeclared_mean_calibration_design,
    fit_additive_scatter_from_complete_mean_calibration,
    freeze_mean_calibration_completion_manifest,
    freeze_mean_calibration_scene_manifest,
    freeze_runtime_ready_model,
    initialize_mean_calibration_layout,
)
from spectrum.transport_spectral import (
    DESIGNATED_TRAINING_SCENE_SEEDS,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
)
from spectrum.full_spectrum_acceptance_runner import ACCEPTANCE_PAIR_IDS


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _REPOSITORY_ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
_DEFAULT_OUTPUT = _REPOSITORY_ROOT / "results" / "mean_calibration"


def _add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared immutable-design arguments to one subparser."""
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Calibration artifact root (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--histories-per-source-line",
        type=int,
        required=True,
        help="Fixed primary-history quota for every source and gamma line.",
    )
    parser.add_argument("--angle-strata-mu", type=int, default=8)
    parser.add_argument("--angle-strata-phi", type=int, default=16)
    parser.add_argument(
        "--forced-collision",
        action="store_true",
        help=(
            "Enable unbiased forced collision only for this immutable "
            "calibration design."
        ),
    )
    parser.add_argument(
        "--design-scene-seed",
        action="append",
        type=int,
        help=(
            "Predeclare one training scene seed; repeat as needed. The "
            "default is the full legacy training set."
        ),
    )
    parser.add_argument(
        "--design-scenario-id",
        action="append",
        choices=VALIDATION_SCENARIO_IDS,
        help=(
            "Predeclare one implemented scenario; repeat as needed. The "
            "default is every implemented training scenario."
        ),
    )
    parser.add_argument(
        "--design-pair-id",
        action="append",
        type=int,
        choices=range(64),
        help=(
            "Predeclare one shield pair; repeat as needed. The default is "
            "all 64 pairs."
        ),
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments needed only by native acquisition."""
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Geometry/runtime source config (default: {_DEFAULT_CONFIG}).",
    )


def _parser() -> argparse.ArgumentParser:
    """Return the phase-oriented fixed-quota calibration parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Acquire transport means with fixed source-line quotas. This does "
            "not alter the standard full-simulation runtime."
        )
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    initialize = subparsers.add_parser(
        "init",
        help="Freeze the training design before any native acquisition.",
    )
    _add_layout_arguments(initialize)

    acquire = subparsers.add_parser(
        "acquire",
        help="Acquire selected predeclared training pairs with external Geant4.",
    )
    _add_layout_arguments(acquire)
    _add_runtime_arguments(acquire)
    acquire.add_argument(
        "--scene-seed",
        type=int,
        required=True,
    )
    acquire.add_argument(
        "--scenario-id",
        choices=VALIDATION_SCENARIO_IDS,
        required=True,
    )
    acquire.add_argument(
        "--pair-id",
        action="append",
        type=int,
        choices=range(64),
        required=True,
        help="Repeat to acquire multiple pair IDs in one cached scene.",
    )

    seal_scene = subparsers.add_parser(
        "seal-scene",
        help="Require and authenticate every scenario and pair for one scene.",
    )
    _add_layout_arguments(seal_scene)
    seal_scene.add_argument(
        "--scene-seed",
        type=int,
        required=True,
    )

    seal_training = subparsers.add_parser(
        "seal-training",
        help="Require every predeclared training scene and freeze completion.",
    )
    _add_layout_arguments(seal_training)

    fit_additive = subparsers.add_parser(
        "fit-additive",
        help="Fit only the physical additive transport-mean component.",
    )
    _add_layout_arguments(fit_additive)

    freeze_runtime = subparsers.add_parser(
        "freeze-runtime",
        help=(
            "Freeze the exact physical statistical model or an authenticated "
            "empirical-discrepancy candidate."
        ),
    )
    _add_layout_arguments(freeze_runtime)
    _add_runtime_arguments(freeze_runtime)
    freeze_runtime.add_argument(
        "--candidate",
        type=Path,
        help=(
            "Optional candidate carrying an independently trained discrepancy "
            "contract. When omitted, use exact physical statistics."
        ),
    )
    freeze_runtime.add_argument("--output", type=Path, required=True)
    return parser


def _design(arguments: argparse.Namespace) -> dict[str, object]:
    """Build the exact design selected on the command line."""
    return build_predeclared_mean_calibration_design(
        histories_per_source_line=arguments.histories_per_source_line,
        angle_strata_mu=arguments.angle_strata_mu,
        angle_strata_phi=arguments.angle_strata_phi,
        forced_collision=bool(arguments.forced_collision),
        training_scene_seeds=(
            tuple(arguments.design_scene_seed)
            if arguments.design_scene_seed
            else DESIGNATED_TRAINING_SCENE_SEEDS
        ),
        scenario_ids=(
            tuple(arguments.design_scenario_id)
            if arguments.design_scenario_id
            else VALIDATION_SCENARIO_IDS
        ),
        shield_pair_ids=(
            tuple(arguments.design_pair_id)
            if arguments.design_pair_id
            else ACCEPTANCE_PAIR_IDS
        ),
    )


def _layout(arguments: argparse.Namespace) -> MeanCalibrationLayout:
    """Return the selected immutable artifact layout."""
    return MeanCalibrationLayout(Path(arguments.output_root).resolve())


def _load_additive(layout: MeanCalibrationLayout) -> (
    AdditiveNoncollidedTransportResponse
):
    """Load the fitted additive response from its immutable JSON."""
    payload = json.loads(
        layout.additive_model_path.read_text(encoding="utf-8")
    )
    return AdditiveNoncollidedTransportResponse.from_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit calibration phase and return a process status."""
    arguments = _parser().parse_args(argv)
    design = _design(arguments)
    layout = _layout(arguments)
    initialize_mean_calibration_layout(layout=layout, design=design)

    if arguments.phase == "init":
        print(layout.design_path.resolve())
        return 0

    if arguments.phase == "acquire":
        backend = ExternalGeant4MeanCalibrationBackend(
            runtime_config_path=arguments.config,
            repository_root=_REPOSITORY_ROOT,
            design=design,
        )
        paths = backend.acquire_scenario(
            layout=layout,
            scene_seed=arguments.scene_seed,
            scenario_id=arguments.scenario_id,
            pair_ids=tuple(dict.fromkeys(arguments.pair_id)),
        )
        for path in paths:
            print(path)
        return 0

    if arguments.phase == "seal-scene":
        path = freeze_mean_calibration_scene_manifest(
            layout=layout,
            design=design,
            scene_seed=arguments.scene_seed,
        )
        print(path)
        return 0

    if arguments.phase == "seal-training":
        path = freeze_mean_calibration_completion_manifest(
            layout=layout,
            design=design,
        )
        print(path)
        return 0

    if arguments.phase == "fit-additive":
        base_model = GeometryConditionedSpectralModel.standard_native(
            ("Co-60", "Cs-137", "Eu-154"),
            dead_time_tau_s=0.0,
            background_rate_cps=0.0,
        )
        response = fit_additive_scatter_from_complete_mean_calibration(
            layout=layout,
            design=design,
            model=base_model,
        )
        print(response.contract_hash_sha256)
        return 0

    if arguments.phase == "freeze-runtime":
        additive_response = _load_additive(layout)
        if arguments.candidate is None:
            runtime_config = load_runtime_config(arguments.config)
            candidate = GeometryConditionedSpectralModel.standard_native(
                ("Co-60", "Cs-137", "Eu-154"),
                dead_time_tau_s=float(runtime_config["dead_time_tau_s"]),
                background_rate_cps=float(
                    runtime_config["background_rate_cps"]
                ),
                additive_scatter_response=additive_response,
            )
        else:
            candidate_payload = json.loads(
                arguments.candidate.read_text(encoding="utf-8")
            )
            candidate = GeometryConditionedSpectralModel.from_manifest_payload(
                candidate_payload
            )
        path = freeze_runtime_ready_model(
            output_path=arguments.output,
            model=candidate,
            additive_response=additive_response,
            layout=layout,
            design=design,
        )
        print(path)
        return 0

    raise AssertionError(f"Unhandled phase: {arguments.phase}.")


if __name__ == "__main__":
    raise SystemExit(main())
