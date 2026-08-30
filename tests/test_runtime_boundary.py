"""Contract tests for the estimator-neutral acquisition boundary."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from runtime.contracts import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from runtime.session import (
    AcquisitionAction,
    ObservationSession,
    _open_owned_observation_session,
)
from sim.protocol import SimulationCommand, SimulationObservation
from sim.runtime import SimulationRuntime
from spectrum.detector_green_operator import DETECTOR_GREEN_SAMPLING_MODE


def test_observation_model_import_is_independent_of_sim_import_order() -> None:
    """The observation model must import in a fresh interpreter."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from measurement.observation_model import "
                "build_runtime_observation_model"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


class _FakeRuntime(SimulationRuntime):
    """Return one deterministic raw observation for boundary tests."""

    def __init__(self, response_override: str | None = None) -> None:
        """Initialize observable fake state."""
        self.reset_payload: dict[str, Any] | None = None
        self.closed = False
        self.response_override = response_override

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Store the reset payload."""
        self.reset_payload = dict(payload or {})

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Return a unit-weight two-bin spectrum."""
        payload: dict[str, Any] = {
            "step_id": command.step_id,
            "detector_pose_xyz": command.target_pose_xyz,
            "detector_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
            "fe_orientation_index": command.fe_orientation_index,
            "pb_orientation_index": command.pb_orientation_index,
            "spectrum_counts": [2, 3],
            "energy_bin_edges_keV": [0.0, 1.0, 2.0],
            "metadata": {
                "detector_response_sampling_mode": DETECTOR_GREEN_SAMPLING_MODE,
                "physics_profile": "em_option4",
                "dwell_time_s": command.dwell_time_s,
            },
        }
        if self.response_override == "step_id":
            payload["step_id"] = command.step_id + 1
        elif self.response_override == "detector_pose_xyz":
            payload["detector_pose_xyz"] = (9.0, 2.0, 3.0)
        elif self.response_override == "detector_quat_wxyz":
            payload["detector_quat_wxyz"] = (0.0, 0.0, 0.0, 1.0)
        elif self.response_override == "fe_orientation_index":
            payload["fe_orientation_index"] = 7
        elif self.response_override == "pb_orientation_index":
            payload["pb_orientation_index"] = 7
        elif self.response_override == "dwell_time_s":
            payload["metadata"]["dwell_time_s"] = command.dwell_time_s + 1.0
        elif self.response_override == "energy_bin_edges_keV":
            payload["energy_bin_edges_keV"] = [0.0, 1.5, 2.0]
        return SimulationObservation(
            **payload,
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

    def abort(self) -> None:
        """Record removal of the incomplete test WAL."""
        self.events.append("abort")


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
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    )

    observation = session.step(_action())

    assert observation.spectrum_counts == [2, 3]
    assert writer.events == ["append", "complete:0"]
    record = writer.records[0]
    assert record.spectrum_counts.dtype == np.int64
    np.testing.assert_array_equal(record.spectrum_counts, [2, 3])
    assert record.metadata[FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY] == "a" * 64


def test_finalize_requires_transport_shutdown_before_publication() -> None:
    """A public log may exist only after clean transport shutdown."""
    events: list[str] = []

    class OrderedRuntime(_FakeRuntime):
        """Record the exact shutdown position in the finalize sequence."""

        def close(self) -> None:
            """Record successful transport shutdown."""
            events.append("runtime.close")
            super().close()

    class OrderedWriter(_FakeWriter):
        """Record the exact publication position in the finalize sequence."""

        def finalize(self) -> str:
            """Record public artifact publication."""
            events.append("writer.finalize")
            return super().finalize()

    session = ObservationSession(
        simulation_runtime=OrderedRuntime(),
        writer=OrderedWriter(),  # type: ignore[arg-type]
        full_spectrum_contract_hash_sha256="d" * 64,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    )

    assert session.finalize() == "published"
    session.close()
    assert events == ["runtime.close", "writer.finalize"]
    assert session.writer.events == ["finalize"]


def test_finalize_aborts_wal_when_transport_shutdown_fails() -> None:
    """A shutdown failure must leave neither a published log nor a private WAL."""

    class FailingRuntime(_FakeRuntime):
        """Inject one fatal transport shutdown error."""

        def close(self) -> None:
            """Fail the required graceful shutdown acknowledgement."""
            self.closed = True
            raise RuntimeError("injected sidecar shutdown failure")

    writer = _FakeWriter()
    session = ObservationSession(
        simulation_runtime=FailingRuntime(),
        writer=writer,  # type: ignore[arg-type]
        full_spectrum_contract_hash_sha256="e" * 64,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    )

    with pytest.raises(RuntimeError, match="injected sidecar shutdown failure"):
        session.finalize()

    assert writer.events == ["abort"]


def test_writer_startup_failure_closes_the_already_started_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer construction failure must release the earlier runtime owner."""
    runtime = _FakeRuntime()

    def fail_writer(*args: object, **kwargs: object) -> object:
        """Inject failure while constructing the private WAL writer."""
        del args, kwargs
        raise OSError("injected writer startup failure")

    monkeypatch.setattr("runtime.session.MeasurementLogStreamWriter", fail_writer)

    with pytest.raises(OSError, match="injected writer startup failure"):
        _open_owned_observation_session(
            simulation_runtime=runtime,
            output_dir="unused",
            writer_arguments={},
            full_spectrum_contract_hash_sha256="f" * 64,
            energy_bin_edges_keV=np.asarray(
                [0.0, 1.0, 2.0],
                dtype=np.float64,
            ),
        )

    assert runtime.closed is True


def test_session_rejects_noncausal_step_id() -> None:
    """Action numbering cannot skip or duplicate an observation."""
    session = ObservationSession(
        simulation_runtime=_FakeRuntime(),
        writer=_FakeWriter(),  # type: ignore[arg-type]
        full_spectrum_contract_hash_sha256="b" * 64,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="causal action index"):
        session.step(_action(step_id=2))


@pytest.mark.parametrize(
    ("response_override", "message"),
    [
        ("step_id", "step_id"),
        ("detector_pose_xyz", "detector pose"),
        ("detector_quat_wxyz", "detector orientation"),
        ("fe_orientation_index", "shield orientations"),
        ("pb_orientation_index", "shield orientations"),
        ("dwell_time_s", "dwell_time_s"),
        ("energy_bin_edges_keV", "energy axis"),
    ],
)
def test_session_rejects_response_that_differs_from_exact_action(
    response_override: str,
    message: str,
) -> None:
    """No mismatched simulator response may reach the durable writer."""
    writer = _FakeWriter()
    session = ObservationSession(
        simulation_runtime=_FakeRuntime(response_override),
        writer=writer,  # type: ignore[arg-type]
        full_spectrum_contract_hash_sha256="c" * 64,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    )

    with pytest.raises(RuntimeError, match=message):
        session.step(_action())

    assert writer.records == []
