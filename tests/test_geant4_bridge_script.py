"""Production-boundary tests for the Geant4 bridge launcher script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from sim.runtime import production_runtime_config_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_geant4_bridge.py"
STANDARD_CONFIG = (
    ROOT / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
)


def _load_script_module() -> ModuleType:
    """Load the launcher without executing its ``__main__`` branch."""
    spec = importlib.util.spec_from_file_location(
        "test_run_geant4_bridge",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the Geant4 bridge launcher script.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bridge_script_rejects_unknown_config_before_server_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The auto-start child must not retain the permissive legacy loader."""
    module = _load_script_module()
    payload = json.loads(STANDARD_CONFIG.read_text(encoding="utf-8"))
    payload["unknown_legacy_key"] = True
    config_path = tmp_path / "unknown.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    started = False

    def start_server(*args: object, **kwargs: object) -> None:
        """Record any forbidden server start after invalid config input."""
        nonlocal started
        started = True

    monkeypatch.setattr(module, "serve_forever", start_server)
    monkeypatch.setattr(sys, "argv", [SCRIPT.name, "--config", str(config_path)])

    with pytest.raises(ValueError, match="unknown_or_retired"):
        module.main()

    assert started is False


def test_bridge_script_rejects_unapproved_model_before_server_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The auto-start child must repeat production model approval."""
    module = _load_script_module()
    started = False

    def start_server(*args: object, **kwargs: object) -> None:
        """Record any forbidden server start after failed model approval."""
        nonlocal started
        started = True

    monkeypatch.setattr(module, "serve_forever", start_server)
    monkeypatch.setattr(sys, "argv", [SCRIPT.name, "--config", str(STANDARD_CONFIG)])

    with pytest.raises(RuntimeError, match="independent all-64 validation"):
        module.main()

    assert started is False


def test_bridge_script_rejects_invalid_registry_before_server_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The auto-start child must resolve authenticated model assets itself."""
    module = _load_script_module()
    payload = json.loads(STANDARD_CONFIG.read_text(encoding="utf-8"))
    payload["full_spectrum_model_registry_path"] = "missing-registry.json"
    payload["full_spectrum_model_registry_file_sha256"] = "0" * 64
    config_path = tmp_path / "invalid-registry.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    started = False

    def start_server(*args: object, **kwargs: object) -> None:
        """Record any forbidden server start after invalid model identity."""
        nonlocal started
        started = True

    monkeypatch.setattr(module, "serve_forever", start_server)
    monkeypatch.setattr(sys, "argv", [SCRIPT.name, "--config", str(config_path)])

    with pytest.raises(FileNotFoundError, match="missing-registry"):
        module.main()

    assert started is False


def test_bridge_script_rejects_removed_mock_stage_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production startup must not override canonical stage fidelity by CLI."""
    module = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [SCRIPT.name, "--config", str(STANDARD_CONFIG), "--mock-stage"],
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2


def test_bridge_script_uses_canonical_usd_path_after_config_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temp config must not make the child resolve USD assets below /tmp."""
    module = _load_script_module()
    frozen = tmp_path / "frozen.json"
    frozen.write_bytes(STANDARD_CONFIG.read_bytes())
    captured: list[object] = []

    monkeypatch.setattr(
        module,
        "require_production_runtime_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(module, "serve_forever", captured.append)
    monkeypatch.setattr(sys, "argv", [SCRIPT.name, "--config", str(frozen)])

    module.main()

    assert len(captured) == 1
    server_config = captured[0]
    assert Path(server_config.app_config["usd_path"]) == (
        ROOT / "configs/isaacsim/demo_room.usda"
    ).resolve()
    assert server_config.production_runtime_config_sha256 == (
        production_runtime_config_sha256(server_config.app_config)
    )
