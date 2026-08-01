"""Contract tests for the estimator-neutral acquisition boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from runtime.contracts import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from runtime.session import AcquisitionAction, ObservationSession
from sim.protocol import SimulationCommand, SimulationObservation
from sim.runtime import SimulationRuntime


class _FakeRuntime(SimulationRuntime):
    """Return one deterministic raw observation for boundary tests."""

    def __init__(self) -> None:
        """Initialize observable fake state."""
        self.reset_payload: dict[str, Any] | None = None
        self.closed = False

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Store the reset payload."""
        self.reset_payload = dict(payload or {})

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Return a unit-weight two-bin spectrum."""
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=command.target_pose_xyz,
            detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=[2, 3],
            energy_bin_edges_keV=[0.0, 1.0, 2.0],
            metadata={
                "detector_response_sampling_mode": (
                    "multinomial_marking_with_nonparalyzable_event_time"
                ),
                "physics_profile": "em_option4",
            },
        )

    def close(self) -> None:
        """Mark the fake runtime closed."""
        self.closed = True


class _FakeWriter:
    """Expose the durable-writer surface used by ObservationSession."""

    def __init__(self) -> None:
        """Initialize staged records and call order."""
        self.records: list[Any] = []
        self.events: list[str] = []

    def append_before_update(self, record: Any) -> int:
        """Stage one record and report its index."""
        self.events.append("append")
        self.records.append(record)
        return len(self.records) - 1

    def mark_station_complete_before_update(self, station_id: int) -> int:
        """Record the causal station boundary."""
        self.events.append(f"complete:{station_id}")
        return len(self.records) - 1

    def finalize(self) -> str:
        """Return a sentinel final artifact."""
        self.events.append("finalize")
        return "published"


def _action(*, step_id: int = 0) -> AcquisitionAction:
    """Build one deterministic action."""
    return AcquisitionAction(
        station_id=0,
        station_complete=True,
        command=SimulationCommand(
            step_id=step_id,
            target_pose_xyz=(1.0, 2.0, 3.0),
            target_base_yaw_rad=0.0,
            fe_orientation_index=1,
            pb_orientation_index=2,
            dwell_time_s=30.0,
        ),
    )


def test_session_persists_raw_observation_before_returning() -> None:
    """A controller must never see an observation that is not staged."""
    runtime = _FakeRuntime()
    writer = _FakeWriter()
    session = ObservationSession(
        simulation_runtime=runtime,
        writer=writer,  # type: ignore[arg-type]
        full_spectrum_contract_hash_sha256="a" * 64,
    )

    observation = session.step(_action())

    assert observation.spectrum_counts == [2, 3]
    assert writer.events == ["append", "complete:0"]
    record = writer.records[0]
    assert record.spectrum_counts.dtype == np.int64
    np.testing.assert_array_equal(record.spectrum_counts, [2, 3])
    assert record.metadata[FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY] == "a" * 64


def test_session_rejects_noncausal_step_id() -> None:
    """Action numbering cannot skip or replay an observation."""
    session = ObservationSession(
        simulation_runtime=_FakeRuntime(),
        writer=_FakeWriter(),  # type: ignore[arg-type]
        full_spectrum_contract_hash_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="causal action index"):
        session.step(_action(step_id=2))


def test_action_parser_rejects_unknown_fields() -> None:
    """The external controller cannot smuggle unversioned fields."""
    with pytest.raises(ValueError, match="unknown"):
        AcquisitionAction.from_mapping(
            {
                "station_id": 0,
                "station_complete": True,
                "command": _action().command.to_dict(),
                "estimator_state": {},
            }
        )

