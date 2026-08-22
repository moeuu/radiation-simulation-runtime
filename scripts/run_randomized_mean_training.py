"""Acquire and freeze the randomized-family physical transport mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from sim.runtime import load_runtime_config
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
from spectrum.additive_scatter import EXACT_SINGLE_SCATTER_BASIS_SEMANTICS
from spectrum.transport_spectral import (
    DESIGNATED_TRAINING_SCENE_SEEDS,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
)

from build_physical_component_full_spectrum_candidate import (
    COMPONENT_TRAINING_PAIR_IDS,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
_DEFAULT_OUTPUT_ROOT = (
    _ROOT / "results" / "mean_calibration" / "randomized_geometry_family"
)
_DEFAULT_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_randomized_mean.json"
)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the immutable training design."""
    parser = argparse.ArgumentParser(
        description=(
            "Acquire the predeclared three-scene randomized-family mean "
            "using fixed-quota, angle-stratified analog Geant4 transport."
        )
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--histories-per-source-line", type=int, default=16384)
    parser.add_argument("--angle-strata-mu", type=int, default=8)
    parser.add_argument("--angle-strata-phi", type=int, default=16)
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help=(
            "Authenticate and refit an already complete training corpus "
            "without launching Geant4."
        ),
    )
    return parser


def _load_additive(layout: MeanCalibrationLayout) -> object:
    """Load the fitted additive response through its public payload parser."""
    from spectrum.additive_scatter import (
        AdditiveNoncollidedTransportResponse,
    )

    payload = json.loads(
        layout.additive_model_path.read_text(encoding="utf-8")
    )
    return AdditiveNoncollidedTransportResponse.from_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the resumable randomized mean-training pipeline."""
    arguments = _parser().parse_args(argv)
    if arguments.histories_per_source_line <= 0:
        raise ValueError("histories-per-source-line must be positive.")
    design = build_predeclared_mean_calibration_design(
        histories_per_source_line=arguments.histories_per_source_line,
        angle_strata_mu=arguments.angle_strata_mu,
        angle_strata_phi=arguments.angle_strata_phi,
        forced_collision=False,
        training_scene_seeds=DESIGNATED_TRAINING_SCENE_SEEDS,
        scenario_ids=VALIDATION_SCENARIO_IDS,
        shield_pair_ids=COMPONENT_TRAINING_PAIR_IDS,
    )
    layout = MeanCalibrationLayout(arguments.output_root.resolve())
    initialize_mean_calibration_layout(layout=layout, design=design)
    if not arguments.fit_only:
        backend = ExternalGeant4MeanCalibrationBackend(
            runtime_config_path=arguments.config.resolve(),
            repository_root=_ROOT,
            design=design,
        )
        for seed in DESIGNATED_TRAINING_SCENE_SEEDS:
            for scenario in VALIDATION_SCENARIO_IDS:
                print(
                    f"[randomized-mean] seed={seed} scenario={scenario}",
                    flush=True,
                )
                backend.acquire_scenario(
                    layout=layout,
                    scene_seed=int(seed),
                    scenario_id=str(scenario),
                    pair_ids=COMPONENT_TRAINING_PAIR_IDS,
                )
            freeze_mean_calibration_scene_manifest(
                layout=layout,
                design=design,
                scene_seed=int(seed),
            )
        freeze_mean_calibration_completion_manifest(
            layout=layout,
            design=design,
        )
    base_model = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
    )
    fit_additive_scatter_from_complete_mean_calibration(
        layout=layout,
        design=design,
        model=base_model,
        feature_basis_semantics=EXACT_SINGLE_SCATTER_BASIS_SEMANTICS,
    )
    runtime_config = load_runtime_config(arguments.config.resolve())
    candidate = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=float(runtime_config["dead_time_tau_s"]),
        background_rate_cps=float(runtime_config["background_rate_cps"]),
        additive_scatter_response=_load_additive(layout),
    )
    output_path = freeze_runtime_ready_model(
        output_path=arguments.output_model.resolve(),
        model=candidate,
        additive_response=candidate.additive_scatter_response,
        layout=layout,
        design=design,
    )
    print(output_path.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
