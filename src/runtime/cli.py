"""Command-line interface for the shared simulation runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from runtime.measurement_log import load_measurement_log
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

