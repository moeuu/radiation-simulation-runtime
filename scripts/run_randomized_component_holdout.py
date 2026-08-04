"""Evaluate a frozen component model on one unused randomized environment.

This diagnostic acquires only the predeclared 16-pair design for all five
acceptance scenarios.  It never fits, selects, or mutates the candidate model,
so holdout observations cannot leak into training.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from build_physical_component_full_spectrum_candidate import (
    COMPONENT_TRAINING_PAIR_IDS,
    _group_arrays,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    AcceptanceScenarioSession,
    canonical_json_bytes,
    file_sha256,
    line_identity_contract_sha256,
    load_acceptance_pair,
)
from spectrum.geant4_acceptance_backend import ExternalGeant4AcceptanceBackend
from spectrum.transport_spectral import (
    DESIGNATED_HOLDOUT_SCENE_SEEDS,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
_DEFAULT_CANDIDATE = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_ral_eu154_component_v4.json"
)
_DEFAULT_OUTPUT = _ROOT / "results" / "randomized_component_holdout_v4"


def _parser() -> argparse.ArgumentParser:
    """Return the immutable holdout command-line contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--candidate-model", type=Path, default=_DEFAULT_CANDIDATE)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help=(
            "Optional immutable report path. Acquisition artifacts remain "
            "under --output-root so a newly frozen candidate can be "
            "evaluated without repeating Geant4 transport."
        ),
    )
    parser.add_argument(
        "--scene-seed",
        type=int,
        choices=DESIGNATED_HOLDOUT_SCENE_SEEDS,
        default=DESIGNATED_HOLDOUT_SCENE_SEEDS[1],
    )
    return parser


def _write_immutable(path: Path, payload: object) -> None:
    """Write one checkpoint once and reject content-changing replacement."""
    encoded = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable holdout artifact changed: {path}.")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load_candidate(path: Path) -> GeometryConditionedSpectralModel:
    """Load and authenticate the frozen runtime candidate before acquisition."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = GeometryConditionedSpectralModel.from_manifest_payload(payload)
    model.require_runtime_ready()
    if tuple(sorted(set(row["isotope"] for row in model.line_identity))) != tuple(
        sorted(ACCEPTANCE_ISOTOPES)
    ):
        raise ValueError("Candidate isotope set differs from acceptance design.")
    return model


def _evaluate(
    *,
    root: Path,
    seed: int,
    candidate_path: Path,
    model: GeometryConditionedSpectralModel,
) -> dict[str, object]:
    """Evaluate fixed likelihood diagnostics without selecting parameters."""
    line_hash = line_identity_contract_sha256(model)
    scenarios: dict[str, object] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    for scenario in VALIDATION_SCENARIO_IDS:
        paths = tuple(
            root
            / "holdout"
            / f"scene_{seed}"
            / scenario
            / f"pair_{pair_id:02d}.json"
            for pair_id in COMPONENT_TRAINING_PAIR_IDS
        )
        records = tuple(
            load_acceptance_pair(
                path,
                expected_line_identity_sha256=line_hash,
            )
            for path in paths
        )
        for record in records:
            if record.geometry_family is None:
                raise RuntimeError("Holdout record lacks geometry-family identity.")
            model.require_environment_applicable(
                {"geometry_family": record.geometry_family}
            )
        observed, total, uncollided, features = _group_arrays(
            records,
            model=model,
        )
        live_times = np.full(observed.shape[0], 30.0, dtype=np.float64)
        total_x = total[np.newaxis, ...]
        uncollided_x = uncollided[np.newaxis, ...]
        features_x = features[np.newaxis, ...]
        predicted = model.predict_mean_numpy(
            total_x,
            uncollided_x,
            features_x,
            live_times,
        )[0]
        innovation = model.posterior_predictive_innovation_numpy(
            observed,
            total_x,
            uncollided_x,
            features_x,
            live_times,
            np.ones(1, dtype=np.float64),
            confidence=0.99,
        )
        mark_tail = innovation["conditional_mark_tail_probability"]
        scenarios[scenario] = {
            "view_count": int(observed.shape[0]),
            "log_likelihood_per_view": float(
                model.log_likelihood_numpy(
                    observed,
                    total_x,
                    uncollided_x,
                    features_x,
                    live_times,
                )[0]
                / observed.shape[0]
            ),
            "count_log_likelihood_per_view": float(
                model.count_log_likelihood_numpy(
                    observed,
                    total_x,
                    uncollided_x,
                    features_x,
                    live_times,
                )[0]
                / observed.shape[0]
            ),
            "predicted_to_observed_total_ratio": float(
                np.sum(predicted) / max(float(np.sum(observed)), 1.0)
            ),
            "renewal_total_max_abs_z": float(
                innovation["renewal_total_max_abs_z"]
            ),
            "conditional_mark_tail_probability": (
                None if mark_tail is None else float(mark_tail)
            ),
        }
        artifact_hashes[scenario] = {
            str(record.shield_pair_id): file_sha256(record.path)
            for record in records
        }
    return {
        "schema_version": 1,
        "evaluation": "fixed_randomized_component_short_holdout_v1",
        "fit_or_selection_performed": False,
        "candidate_model_path": str(candidate_path.resolve()),
        "candidate_model_file_sha256": file_sha256(candidate_path),
        "candidate_model_contract_hash_sha256": model.contract_hash_sha256,
        "scene_seed": int(seed),
        "training_scene_seed": False,
        "shield_pair_ids": list(COMPONENT_TRAINING_PAIR_IDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "artifact_sha256_by_scenario": artifact_hashes,
        "scenarios": scenarios,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Acquire missing holdout pairs and evaluate the frozen candidate."""
    arguments = _parser().parse_args(argv)
    candidate_path = arguments.candidate_model.resolve()
    model = _load_candidate(candidate_path)
    line_hash = line_identity_contract_sha256(model)
    seed = int(arguments.scene_seed)
    root = arguments.output_root.resolve()
    backend = ExternalGeant4AcceptanceBackend(
        runtime_config_path=arguments.config.resolve(),
        repository_root=_ROOT,
    )
    for scenario in VALIDATION_SCENARIO_IDS:
        paths = {
            pair_id: (
                root
                / "holdout"
                / f"scene_{seed}"
                / scenario
                / f"pair_{pair_id:02d}.json"
            )
            for pair_id in COMPONENT_TRAINING_PAIR_IDS
        }
        missing = [pair_id for pair_id, path in paths.items() if not path.exists()]
        if not missing:
            continue
        context: AbstractContextManager[AcceptanceScenarioSession] = (
            backend.open_scenario(
                scene_seed=seed,
                split="holdout",
                scenario_id=scenario,
                line_identity_sha256=line_hash,
            )
        )
        with context as session:
            for pair_id, path in paths.items():
                if path.exists():
                    continue
                print(
                    f"[component-holdout] acquire scenario={scenario} "
                    f"pair={pair_id}",
                    flush=True,
                )
                _write_immutable(path, session.acquire_pair(pair_id))
    report = _evaluate(
        root=root,
        seed=seed,
        candidate_path=candidate_path,
        model=model,
    )
    report_path = (
        (root / f"scene_{seed}_report.json")
        if arguments.report_output is None
        else arguments.report_output.resolve()
    )
    _write_immutable(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
