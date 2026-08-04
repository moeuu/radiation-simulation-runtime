"""Acquire the predeclared randomized component-discrepancy corpus."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Sequence

from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    AcceptanceScenarioSession,
    canonical_json_bytes,
    line_identity_contract_sha256,
    load_acceptance_pair,
)
from spectrum.geant4_acceptance_backend import (
    ExternalGeant4AcceptanceBackend,
)
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
_DEFAULT_OUTPUT = _ROOT / "results" / "randomized_component_training_v2"


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Acquire a fixed 16-pair randomized-geometry training design."
        )
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--scene-seed",
        type=int,
        action="append",
        choices=DESIGNATED_TRAINING_SCENE_SEEDS,
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        choices=VALIDATION_SCENARIO_IDS,
    )
    parser.add_argument(
        "--pair-id",
        type=int,
        action="append",
        choices=COMPONENT_TRAINING_PAIR_IDS,
    )
    return parser


def _write_immutable(path: Path, payload: object) -> None:
    """Write a checkpoint once, rejecting content-changing replacement."""
    encoded = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable training artifact changed: {path}.")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Acquire missing randomized training checkpoints with external Geant4."""
    arguments = _parser().parse_args(argv)
    seeds = tuple(arguments.scene_seed or DESIGNATED_TRAINING_SCENE_SEEDS)
    scenarios = tuple(arguments.scenario_id or VALIDATION_SCENARIO_IDS)
    pair_ids = tuple(arguments.pair_id or COMPONENT_TRAINING_PAIR_IDS)
    model = GeometryConditionedSpectralModel.standard_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
    )
    line_hash = line_identity_contract_sha256(model)
    backend = ExternalGeant4AcceptanceBackend(
        runtime_config_path=arguments.config.resolve(),
        repository_root=_ROOT,
    )
    root = arguments.output_root.resolve()
    for seed in seeds:
        for scenario in scenarios:
            paths = {
                pair_id: (
                    root
                    / "training"
                    / f"scene_{seed}"
                    / scenario
                    / f"pair_{pair_id:02d}.json"
                )
                for pair_id in pair_ids
            }
            missing = [
                pair_id
                for pair_id, path in paths.items()
                if not path.exists()
            ]
            if not missing:
                for path in paths.values():
                    load_acceptance_pair(
                        path,
                        expected_line_identity_sha256=line_hash,
                    )
                print(f"[component-training] reuse seed={seed} scenario={scenario}")
                continue
            context: AbstractContextManager[AcceptanceScenarioSession] = (
                backend.open_scenario(
                    scene_seed=int(seed),
                    split="training",
                    scenario_id=str(scenario),
                    line_identity_sha256=line_hash,
                )
            )
            with context as session:
                for pair_id in pair_ids:
                    path = paths[pair_id]
                    if path.exists():
                        load_acceptance_pair(
                            path,
                            expected_line_identity_sha256=line_hash,
                        )
                        continue
                    print(
                        "[component-training] acquire "
                        f"seed={seed} scenario={scenario} pair={pair_id}",
                        flush=True,
                    )
                    _write_immutable(path, session.acquire_pair(int(pair_id)))
                    load_acceptance_pair(
                        path,
                        expected_line_identity_sha256=line_hash,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
