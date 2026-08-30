"""Run the resumable external-Geant4 all-64 acceptance acquisition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from runtime.experiment_profiles import STANDARD_ACQUISITION_LIVE_TIME_S
from spectrum.full_spectrum_acceptance import (
    write_independent_validation_manifest,
)
from spectrum.detector_green_validation import (
    load_detector_green_validation_manifest,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    ACCEPTANCE_PAIR_IDS,
    AcceptanceRunLayout,
    acquire_designated_split,
    build_acceptance_run_contract,
    canonical_json_bytes,
    freeze_candidate_model,
    line_identity_contract_sha256,
    load_acceptance_pair,
    canonical_detector_green_operator,
)
from spectrum.full_spectrum_acceptance_evaluator import (
    approve_frozen_candidate,
    evaluate_all_designated_scenes,
)
from spectrum.geant4_acceptance_backend import (
    ExternalGeant4AcceptanceBackend,
)
from spectrum.transport_spectral import (
    DESIGNATED_VALIDATION_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    GeometryConditionedSpectralModel,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = (
    _REPOSITORY_ROOT
    / "configs"
    / "geant4"
    / "variance_reduction_external_no_isaac_32threads.json"
)
_DEFAULT_OUTPUT = (
    _REPOSITORY_ROOT
    / "results"
    / "full_spectrum_all64_acceptance"
    / FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
)


def _common_parser(parser: argparse.ArgumentParser) -> None:
    """Add the shared native acquisition arguments to one subparser."""
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=(f"Standard external-Geant4 runtime config (default: {_DEFAULT_CONFIG})."),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(f"Immutable/resumable artifact root (default: {_DEFAULT_OUTPUT})."),
    )


def _detector_green_validation_argument(
    parser: argparse.ArgumentParser,
) -> None:
    """Require the independent full-detector response gate artifact."""
    parser.add_argument(
        "--detector-green-validation-manifest",
        type=Path,
        required=True,
        help=(
            "Independent catalog-excluded monoenergetic validation for the "
            "exact detector Green operator."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    """Return the fail-closed phase-oriented command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate one predeclared physics-only Cs/Co model with five new "
            "real-Geant4 environments. Pair files are immutable checkpoints, so "
            "an interrupted phase can be rerun safely."
        )
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    status = subparsers.add_parser(
        "status",
        help="Report immutable pair/corpus/model checkpoint counts.",
    )
    _common_parser(status)

    smoke = subparsers.add_parser(
        "smoke",
        help=(
            f"Acquire one real {STANDARD_ACQUISITION_LIVE_TIME_S:g} s "
            "background pair plus the native signed-"
            "epsilon gate; the checkpoint is reusable by validation."
        ),
    )
    _common_parser(smoke)
    smoke.add_argument(
        "--scene-seed",
        type=int,
        choices=DESIGNATED_VALIDATION_SCENE_SEEDS,
        default=DESIGNATED_VALIDATION_SCENE_SEEDS[0],
    )
    smoke.add_argument(
        "--pair-id",
        type=int,
        choices=ACCEPTANCE_PAIR_IDS,
        default=0,
    )

    freeze = subparsers.add_parser(
        "freeze",
        help="Freeze the predeclared physics-only candidate before validation.",
    )
    _common_parser(freeze)

    validation = subparsers.add_parser(
        "validation",
        help="Acquire all five new Cs/Co validation scenes and all 64 pairs.",
    )
    _common_parser(validation)

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Evaluate the same frozen candidate on all five immutable raw "
            "scene corpora and write one fixed-metric artifact per seed."
        ),
    )
    _common_parser(evaluate)

    approve = subparsers.add_parser(
        "approve",
        help=(
            "Conservatively aggregate validation-only metrics and create a "
            "production model only when every fixed threshold passes."
        ),
    )
    _common_parser(approve)
    _detector_green_validation_argument(approve)

    all_phases = subparsers.add_parser(
        "all",
        help=(
            "Freeze the physics-only model, acquire all validation scenes, "
            "evaluate fixed metrics, and approve in order."
        ),
    )
    _common_parser(all_phases)
    _detector_green_validation_argument(all_phases)

    aggregate = subparsers.add_parser(
        "aggregate",
        help=(
            "Aggregate exactly five already evaluated scene-acceptance "
            "artifacts into the independent validation manifest."
        ),
    )
    aggregate.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=Path,
        help=("One evaluated scene artifact; pass all five validation artifacts."),
    )
    _detector_green_validation_argument(aggregate)
    aggregate.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination for the deterministic validation manifest.",
    )
    return parser


def _write_immutable_json(path: Path, payload: object) -> Path:
    """Atomically write one canonical artifact or verify exact resumption."""
    destination = path.resolve()
    encoded = canonical_json_bytes(payload)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise RuntimeError(
                f"Refusing to overwrite incompatible artifact: {destination}."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    return destination


def _progress(message: str) -> None:
    """Print one timestamp-free line suitable for a persistent log."""
    print(message, flush=True)


def _native_context(
    arguments: argparse.Namespace,
) -> tuple[
    AcceptanceRunLayout,
    ExternalGeant4AcceptanceBackend,
    GeometryConditionedSpectralModel,
    str,
]:
    """Build and authenticate the shared immutable acquisition context."""
    layout = AcceptanceRunLayout(Path(arguments.output_root).resolve())
    backend = ExternalGeant4AcceptanceBackend(
        runtime_config_path=Path(arguments.config),
        repository_root=_REPOSITORY_ROOT,
    )
    run_contract = build_acceptance_run_contract(
        runtime_config_sha256=backend.runtime_config_sha256,
        native_executable_sha256=backend.native_executable_sha256,
        native_execution_environment_sha256=(
            backend.native_execution_environment_sha256
        ),
        implementation_bundle_sha256=backend.implementation_bundle_sha256,
    )
    _write_immutable_json(layout.run_contract_path, run_contract)
    base_model = GeometryConditionedSpectralModel.physics_only_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=float(backend.app_config.dead_time_tau_s),
        background_rate_cps=float(backend.app_config.background_cps),
        detector_green_operator=backend.detector_green_operator,
    )
    base_model.require_runtime_ready()
    line_hash = line_identity_contract_sha256(base_model)
    return layout, backend, base_model, line_hash


def _freeze_physics_candidate(
    *,
    layout: AcceptanceRunLayout,
    base_model: GeometryConditionedSpectralModel,
) -> None:
    """Freeze the generic no-scene-fit candidate before any validation."""
    freeze_candidate_model(layout=layout, model=base_model)
    _progress(
        "physics-only candidate frozen before validation: "
        f"{base_model.contract_hash_sha256}"
    )


def _run_validation(
    *,
    layout: AcceptanceRunLayout,
    backend: ExternalGeant4AcceptanceBackend,
    line_hash: str,
) -> None:
    """Acquire all new application validation scenes after candidate freeze."""
    acquire_designated_split(
        layout=layout,
        backend=backend,
        seeds=DESIGNATED_VALIDATION_SCENE_SEEDS,
        split="validation",
        line_identity_sha256=line_hash,
        progress=_progress,
    )


def _status_payload(layout: AcceptanceRunLayout) -> Mapping[str, object]:
    """Return a compact, deterministic checkpoint summary."""
    split_payload: dict[str, object] = {}
    for split, seeds in (("validation", DESIGNATED_VALIDATION_SCENE_SEEDS),):
        pair_count = len(tuple((layout.root / split).glob("scene_*/*/pair_*.json")))
        corpus_count = sum(
            layout.scene_corpus_path(split=split, scene_seed=seed).is_file()
            for seed in seeds
        )
        split_payload[split] = {
            "pair_checkpoint_count": pair_count,
            "expected_pair_checkpoint_count": len(seeds) * 5 * 64,
            "scene_corpus_manifest_count": corpus_count,
            "expected_scene_corpus_manifest_count": len(seeds),
        }
    return {
        "output_root": layout.root.as_posix(),
        "run_contract": layout.run_contract_path.is_file(),
        "candidate_model": layout.candidate_model_path.is_file(),
        "independent_validation": layout.validation_manifest_path.is_file(),
        "production_model": layout.production_model_path.is_file(),
        "splits": split_payload,
    }


def _run_smoke(
    *,
    layout: AcceptanceRunLayout,
    backend: ExternalGeant4AcceptanceBackend,
    line_hash: str,
    scene_seed: int,
    pair_id: int,
) -> None:
    """Acquire one resumable real-native background observation."""
    destination = layout.pair_path(
        split="validation",
        scene_seed=scene_seed,
        scenario_id="background_only",
        shield_pair_id=pair_id,
    )
    if not destination.exists():
        with backend.open_scenario(
            scene_seed=scene_seed,
            split="validation",
            scenario_id="background_only",
            line_identity_sha256=line_hash,
        ) as session:
            payload = session.acquire_pair(pair_id)
        _write_immutable_json(destination, payload)
    record = load_acceptance_pair(
        destination,
        expected_line_identity_sha256=line_hash,
    )
    _progress(
        "real-native smoke strict-loader passed: "
        f"{destination} total_counts="
        f"{int(np.sum(record.observed_spectrum_counts, dtype=np.int64))}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one explicitly ordered resumable acceptance phase."""
    arguments = _parser().parse_args(argv)
    if arguments.phase == "aggregate":
        operator = canonical_detector_green_operator()
        green_validation_payload = load_detector_green_validation_manifest(
            arguments.detector_green_validation_manifest,
            operator=operator,
        )
        write_independent_validation_manifest(
            arguments.artifact,
            arguments.output,
            detector_green_validation_manifest=(green_validation_payload),
        )
        return 0

    layout, backend, base_model, line_hash = _native_context(arguments)
    if arguments.phase == "status":
        print(
            json.dumps(
                _status_payload(layout),
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return 0
    if arguments.phase == "smoke":
        _freeze_physics_candidate(layout=layout, base_model=base_model)
        _run_smoke(
            layout=layout,
            backend=backend,
            line_hash=line_hash,
            scene_seed=int(arguments.scene_seed),
            pair_id=int(arguments.pair_id),
        )
        return 0
    if arguments.phase in {"freeze", "all"}:
        _freeze_physics_candidate(layout=layout, base_model=base_model)
    if arguments.phase in {"validation", "all"}:
        _run_validation(
            layout=layout,
            backend=backend,
            line_hash=line_hash,
        )
    if arguments.phase in {"evaluate", "all"}:
        for path in evaluate_all_designated_scenes(layout=layout):
            _progress(f"scene acceptance: {path}")
    if arguments.phase in {"approve", "all"}:
        validation_path, production_path = approve_frozen_candidate(
            layout=layout,
            detector_green_validation_manifest_path=(
                arguments.detector_green_validation_manifest
            ),
        )
        _progress(f"independent validation: {validation_path}")
        _progress(f"approved production model: {production_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
