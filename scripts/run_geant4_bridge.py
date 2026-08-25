"""Run the Geant4 bridge sidecar."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.geant4_app.bridge_server import Geant4BridgeServerConfig, serve_forever
from runtime.session import require_production_runtime_preflight
from sim.runtime import (
    load_production_runtime_config,
    production_runtime_config_sha256,
)


def main() -> None:
    """Parse CLI arguments and start the Geant4 bridge server."""
    parser = argparse.ArgumentParser(description="Run the Geant4 bridge sidecar.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the bridge configuration JSON file.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_production_runtime_config(config_path)
    require_production_runtime_preflight(
        config,
        requested_backend=config["backend"],
    )
    if not Path(config["usd_path"]).is_absolute():
        raise RuntimeError(
            "Production config loader did not canonicalize usd_path."
        )
    server_config = Geant4BridgeServerConfig(
        host=config["host"],
        port=config["port"],
        app_config=config,
        production_runtime_config_sha256=(
            production_runtime_config_sha256(config)
        ),
    )
    serve_forever(server_config)


if __name__ == "__main__":
    main()
