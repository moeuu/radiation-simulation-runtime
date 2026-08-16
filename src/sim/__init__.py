"""Simulation runtime interfaces and protocol helpers."""

from __future__ import annotations

from typing import Any

from sim.protocol import (
    SimulationCommand,
    SimulationObservation,
    normalize_json_payload,
)


_RUNTIME_EXPORTS = frozenset(
    {
        "AnalyticSimulationRuntime",
        "Geant4WithIsaacSimRuntime",
        "Geant4TCPClientRuntime",
        "IsaacSimTCPClientRuntime",
        "ManagedIsaacSimTCPClientRuntime",
        "ManagedGeant4TCPClientRuntime",
        "SimulationRuntime",
        "TCPSidecarClientRuntime",
        "create_simulation_runtime",
        "load_runtime_config",
    }
)


def __getattr__(name: str) -> Any:
    """Load runtime exports lazily to keep submodule imports acyclic."""
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from sim import runtime

    value = getattr(runtime, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager and lazy public package attributes."""
    return sorted(set(globals()) | _RUNTIME_EXPORTS)

__all__ = [
    "AnalyticSimulationRuntime",
    "Geant4WithIsaacSimRuntime",
    "Geant4TCPClientRuntime",
    "IsaacSimTCPClientRuntime",
    "ManagedIsaacSimTCPClientRuntime",
    "ManagedGeant4TCPClientRuntime",
    "SimulationCommand",
    "SimulationObservation",
    "SimulationRuntime",
    "TCPSidecarClientRuntime",
    "create_simulation_runtime",
    "load_runtime_config",
    "normalize_json_payload",
]
