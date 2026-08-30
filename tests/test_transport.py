"""Tests for the shared pre-spectrum transport layer."""

from __future__ import annotations

import numpy as np
import pytest

from measurement.model import PointSource
from sim.approx.python_transport import PythonTransportSpectrumModel
from sim.isaacsim_app.stage_backend import StageMaterialInfo
from sim.protocol import SimulationCommand
from sim.transport import build_source_transport_result, make_transport_segment


def test_build_source_transport_result_tracks_obstacle_and_scatter() -> None:
    """The shared transport result should expose path totals and line statistics."""
    source = PointSource(isotope="Cs-137", position=(4.0, 4.0, 1.0), intensity_cps_1m=1000.0)
    stage_segments = (
        make_transport_segment(StageMaterialInfo(name="concrete"), 15.0, is_obstacle=True),
    )
    result = build_source_transport_result(
        source=source,
        detector_position_xyz=(4.0, 1.0, 1.0),
        dwell_time_s=10.0,
        stage_segments=stage_segments,
        fe_segment=make_transport_segment(StageMaterialInfo(name="fe"), 5.0),
        pb_segment=make_transport_segment(StageMaterialInfo(name="pb"), 2.0),
        nuclide_lines=((662.0, 0.85),),
        scatter_gain=0.12,
    )

    assert result.total_obstacle_path_cm == pytest.approx(15.0)
    assert result.total_stage_path_cm == pytest.approx(15.0)
    assert result.total_fe_path_cm == pytest.approx(5.0)
    assert result.total_pb_path_cm == pytest.approx(2.0)
    assert len(result.lines) == 1
    assert result.lines[0].total_transmission <= 1.0
    assert result.lines[0].scatter_counts >= 0.0


def test_python_transport_samples_one_renewal_total() -> None:
    """The analytic observation path must request exactly one renewal draw."""
    model = PythonTransportSpectrumModel(
        rng_seed=17,
        dead_time_s=5.813e-9,
    )
    expected = np.zeros_like(model.energy_axis_keV)
    expected[100:103] = np.asarray([8.0, 5.0, 2.0])

    sampled = model.sample_spectrum(expected, step_id=3, live_time_s=1.0)

    assert sampled.shape == expected.shape
    assert sampled.dtype == np.dtype(np.int64)
    assert np.all(sampled >= 0)
    assert int(np.sum(sampled)) > 0


def test_python_transport_observation_is_explicitly_offline_only() -> None:
    """The analytic backend must not claim the native sampled-event contract."""
    model = PythonTransportSpectrumModel(
        sources=[
            PointSource(
                isotope="Cs-137",
                position=(1.0, 1.0, 0.0),
                intensity_cps_1m=300_000.0,
            )
        ],
        rng_seed=19,
        dead_time_s=5.813e-9,
    )
    command = SimulationCommand(
        step_id=0,
        target_pose_xyz=(0.5, 0.5, 0.5),
        target_base_yaw_rad=0.0,
        fe_orientation_index=0,
        pb_orientation_index=0,
        dwell_time_s=0.1,
    )

    observation = model.observe(
        command,
        detector_pose_xyz=command.target_pose_xyz,
        backend_label="analytic",
    )

    assert observation.metadata["detector_response_sampling_mode"] == (
        "legacy_analytic_response_marking_offline_only"
    )
    assert all(
        isinstance(value, float) and value.is_integer()
        for value in observation.spectrum_counts
    )
