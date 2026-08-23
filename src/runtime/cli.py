"""Command-line interface for the shared simulation runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from runtime.adaptive import serve_adaptive_session
from runtime.discrepancy_calibrator import calibrate_discrepancy
from runtime.experiment_profiles import (
    DEFAULT_EXPERIMENT_PROFILE_ID,
    available_experiment_profiles,
)
from runtime.measurement_log import load_measurement_log
from runtime.scenarios import (
    build_private_truth_manifest,
    build_random_surface_scenario,
    generate_fresh_scene_seed,
    write_private_scenario,
    write_private_truth_manifest,
)
from runtime.session import run_acquisition_plan
from sim.geant4_app.bridge_server import Geant4BridgeServerConfig, serve_forever
from sim.runtime import load_runtime_config


def _build_parser() -> argparse.ArgumentParser:
    """Build the public command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-log")
    validate.add_argument("path", type=Path)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    run_plan = subparsers.add_parser("run-plan")
    run_plan.add_argument("plan", type=Path)
    run_adaptive = subparsers.add_parser("run-adaptive-session")
    run_adaptive.add_argument("scenario", type=Path)
    serve_adaptive_socket = subparsers.add_parser("serve-adaptive-session-socket")
    serve_adaptive_socket.add_argument("scenario", type=Path)
    serve_adaptive_socket.add_argument("--socket-path", type=Path, required=True)
    serve_adaptive_socket.add_argument("--resume-stage", type=Path, default=None)
    serve_adaptive_socket.add_argument(
        "--resume-compatibility",
        type=Path,
        default=None,
    )
    run_adaptive.add_argument(
        "--resume-stage",
        type=Path,
        default=None,
        help="Resume from a verified adaptive MeasurementLog stream stage.",
    )
    run_adaptive.add_argument(
        "--resume-compatibility",
        type=Path,
        default=None,
        help=(
            "Explicit compatibility provenance required when resuming under a "
            "different runtime commit."
        ),
    )
    generate_scenario = subparsers.add_parser("generate-scenario")
    generate_scenario.add_argument("output", type=Path)
    generate_scenario.add_argument(
        "--truth-manifest-output",
        type=Path,
        required=True,
    )
    generate_scenario.add_argument(
        "--measurement-log-output",
        type=Path,
        required=True,
    )
    generate_scenario.add_argument("--run-id", required=True)
    generate_scenario.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help="Explicit runtime-physics override for a controlled experiment variant.",
    )
    generate_scenario.add_argument("--scene-seed", type=int, default=None)
    generate_scenario.add_argument(
        "--experiment-profile",
        choices=available_experiment_profiles(),
        default=DEFAULT_EXPERIMENT_PROFILE_ID,
    )
    generate_scenario.add_argument("--scene-variant", default=None)
    calibrate = subparsers.add_parser("calibrate-discrepancy")
    calibrate.add_argument("input", type=Path)
    calibrate.add_argument("output", type=Path)
    calibrate.add_argument("--calibration-id", required=True)
    calibrate.add_argument("--residual-rank", type=int, default=3)
    return parser


def _load_resume_compatibility(path: Path | None) -> dict[str, object] | None:
    """Load one JSON compatibility object supplied for adaptive resume."""
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Adaptive resume compatibility must be a JSON object.")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run one simulation-runtime command."""
    args = _build_parser().parse_args(argv)
    if args.command == "validate-log":
        log = load_measurement_log(args.path)
        print(
            f"valid MeasurementLog v{log.schema_version}: "
            f"run_id={log.run_id} records={len(log.records)}"
        )
        return 0
    if args.command == "run-plan":
        log = run_acquisition_plan(args.plan)
        print(f"published {log.path} records={len(log.records)}")
        return 0
    if args.command == "run-adaptive-session":
        import sys

        if args.resume_compatibility is not None and args.resume_stage is None:
            raise ValueError("--resume-compatibility requires --resume-stage.")
        return serve_adaptive_session(
            args.scenario,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
            resume_stage_dir=args.resume_stage,
            resume_compatibility=_load_resume_compatibility(args.resume_compatibility),
        )
    if args.command == "serve-adaptive-session-socket":
        from runtime.adaptive import serve_adaptive_session_socket

        if args.resume_compatibility is not None and args.resume_stage is None:
            raise ValueError("--resume-compatibility requires --resume-stage.")
        return serve_adaptive_session_socket(
            args.scenario,
            socket_path=args.socket_path,
            resume_stage_dir=args.resume_stage,
            resume_compatibility=_load_resume_compatibility(args.resume_compatibility),
        )
    if args.command == "generate-scenario":
        scene_seed = (
            generate_fresh_scene_seed()
            if args.scene_seed is None
            else int(args.scene_seed)
        )
        scenario = build_random_surface_scenario(
            scene_seed=scene_seed,
            measurement_log_output_dir=args.measurement_log_output,
            run_id=str(args.run_id),
            runtime_config_path=args.runtime_config,
            experiment_profile_id=str(args.experiment_profile),
            scene_variant_id=args.scene_variant,
        )
        output = write_private_scenario(args.output, scenario)
        truth_manifest = write_private_truth_manifest(
            args.truth_manifest_output,
            build_private_truth_manifest(scenario),
        )
        print(
            "published private scenario "
            f"{output} truth_manifest={truth_manifest} scene_seed={scene_seed}"
        )
        return 0
    if args.command == "calibrate-discrepancy":
        calibration = calibrate_discrepancy(
            args.input,
            args.output,
            calibration_id=args.calibration_id,
            residual_rank=args.residual_rank,
        )
        print(
            f"published calibration {calibration.calibration_id}: "
            f"environments={len(calibration.independent_environment_ids)}"
        )
        return 0
    config = load_runtime_config(args.config)
    serve_forever(
        Geant4BridgeServerConfig(
            host=str(config.get("host", "127.0.0.1")),
            port=int(config.get("port", 5556)),
            app_config=config,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
