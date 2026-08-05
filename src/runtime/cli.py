"""Command-line interface for the shared simulation runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from runtime.adaptive import serve_adaptive_session
from runtime.discrepancy_calibrator import calibrate_discrepancy
from runtime.measurement_log import load_measurement_log
from runtime.scenarios import (
    RAL_PRIVATE_SOURCE_PROFILES,
    build_random_ral_mix9_scenario,
    generate_fresh_scene_seed,
    write_private_scenario,
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
    run_adaptive.add_argument(
        "--private-scene-profile",
        choices=tuple(RAL_PRIVATE_SOURCE_PROFILES),
        default=None,
    )
    generate_scenario = subparsers.add_parser("generate-ral-scenario")
    generate_scenario.add_argument("output", type=Path)
    generate_scenario.add_argument(
        "--measurement-log-output",
        type=Path,
        required=True,
    )
    generate_scenario.add_argument("--run-id", required=True)
    generate_scenario.add_argument("--runtime-config", type=Path, required=True)
    generate_scenario.add_argument("--scene-seed", type=int, default=None)
    generate_scenario.add_argument("--candidate-count", type=int, default=256)
    generate_scenario.add_argument(
        "--source-profile",
        choices=tuple(RAL_PRIVATE_SOURCE_PROFILES),
        default="ral-mix9",
    )
    calibrate = subparsers.add_parser("calibrate-discrepancy")
    calibrate.add_argument("input", type=Path)
    calibrate.add_argument("output", type=Path)
    calibrate.add_argument("--calibration-id", required=True)
    calibrate.add_argument("--residual-rank", type=int, default=3)
    return parser


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

        return serve_adaptive_session(
            args.scenario,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
            private_scene_profile=args.private_scene_profile,
        )
    if args.command == "generate-ral-scenario":
        scene_seed = (
            generate_fresh_scene_seed()
            if args.scene_seed is None
            else int(args.scene_seed)
        )
        scenario = build_random_ral_mix9_scenario(
            scene_seed=scene_seed,
            runtime_config_path=args.runtime_config,
            measurement_log_output_dir=args.measurement_log_output,
            run_id=str(args.run_id),
            candidate_count=int(args.candidate_count),
            source_profile=str(args.source_profile),
        )
        output = write_private_scenario(args.output, scenario)
        print(f"published private scenario {output} scene_seed={scene_seed}")
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
