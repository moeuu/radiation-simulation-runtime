"""Command-line interface for the shared simulation runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from runtime.adaptive import serve_adaptive_session
from runtime.discrepancy_calibrator import calibrate_discrepancy
from runtime.experiment_profiles import available_experiment_profiles
from runtime.measurement_log import load_measurement_log
from runtime.scenarios import (
    build_private_truth_manifest,
    build_random_surface_scenario,
    generate_fresh_scene_seed,
    write_private_scenario,
    write_private_truth_manifest,
)
from runtime.session import require_production_runtime_preflight
from sim.geant4_app.bridge_server import Geant4BridgeServerConfig, serve_forever
from sim.runtime import (
    load_production_runtime_config,
    production_runtime_config_sha256,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the public command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-log")
    validate.add_argument("path", type=Path)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    run_adaptive = subparsers.add_parser("run-adaptive-session")
    run_adaptive.add_argument("scenario", type=Path)
    serve_adaptive_socket = subparsers.add_parser("serve-adaptive-session-socket")
    serve_adaptive_socket.add_argument("scenario", type=Path)
    serve_adaptive_socket.add_argument("--socket-path", type=Path, required=True)
    serve_adaptive_socket.add_argument(
        "--cui-truth-overlay-socket-path",
        type=Path,
        default=None,
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
        required=True,
    )
    generate_scenario.add_argument("--scene-variant", required=True)
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
    if args.command == "run-adaptive-session":
        import sys

        return serve_adaptive_session(
            args.scenario,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
        )
    if args.command == "serve-adaptive-session-socket":
        from runtime.adaptive import serve_adaptive_session_socket

        return serve_adaptive_session_socket(
            args.scenario,
            socket_path=args.socket_path,
            cui_truth_overlay_socket_path=args.cui_truth_overlay_socket_path,
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
    config = load_production_runtime_config(args.config)
    require_production_runtime_preflight(
        config,
        requested_backend=config["backend"],
    )
    serve_forever(
        Geant4BridgeServerConfig(
            host=config["host"],
            port=config["port"],
            app_config=config,
            production_runtime_config_sha256=(production_runtime_config_sha256(config)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
