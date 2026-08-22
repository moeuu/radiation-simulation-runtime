"""Tests for the estimator-neutral resolved physical context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runtime import ResolvedForwardContext
from runtime.forward_model_manifest import build_forward_model_manifest
from runtime.measurement_log import (
    MeasurementLogValidationError,
    load_measurement_log,
    write_measurement_log,
)
from runtime.provenance import sha256_json
from runtime.records import RunContext
import runtime.forward_context as forward_context_module

from tests.runtime_test_support import (
    TEST_COMMIT,
    TEST_ISOTOPES,
    environment,
    make_measurement_log,
    records,
    runtime_config,
)


def test_resolved_forward_context_from_log_builds_shared_kernel(
    tmp_path: Path,
) -> None:
    """A validated log resolves immutable geometry and one physical kernel."""
    log = load_measurement_log(make_measurement_log(tmp_path / "run"))

    context = ResolvedForwardContext.from_log(log)

    assert context.environment.size_x == 2.0
    assert np.array_equal(context.bounds_xyz[0], np.zeros(3))
    assert np.array_equal(context.bounds_xyz[1], [2.0, 2.0, 1.5])
    assert not context.bounds_xyz[0].flags.writeable
    assert not context.bounds_xyz[1].flags.writeable
    assert context.obstacle_grid is None
    assert context.resolved_obstacle_path is None
    assert context.asset_identities == {}
    assert set(context.model_identifiers) == {
        "detector",
        "shield",
        "environment",
        "obstacle",
        "transport",
        "spectrum",
    }
    assert context.spectral_model.runtime_ready
    assert (
        context.observation_model.additive_scatter_response
        is context.spectral_model.additive_scatter_response
    )

    kernel = context.build_continuous_kernel(
        use_gpu=False,
        gpu_device="cpu",
        gpu_dtype="float64",
    )
    assert kernel.use_gpu is False
    assert kernel.gpu_device == "cpu"
    assert kernel.gpu_dtype == "float64"
    assert kernel.obstacle_grid is None
    assert (
        kernel.additive_scatter_response
        is context.spectral_model.additive_scatter_response
    )


def test_live_context_resolves_spectral_model_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live construction must not rebuild the authenticated spectral model."""
    log = load_measurement_log(make_measurement_log(tmp_path / "run"))
    original = (
        forward_context_module.geometry_conditioned_model_from_runtime_config
    )
    call_count = 0

    def counted_resolver(*args: object, **kwargs: object) -> object:
        """Count and delegate one spectrum-contract resolution."""
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        forward_context_module,
        "geometry_conditioned_model_from_runtime_config",
        counted_resolver,
    )

    resolved = ResolvedForwardContext.from_run_context(
        log.context,
        run_root=log.path,
    )

    assert call_count == 1
    assert resolved.run_root == log.path


def test_live_context_rejects_relative_asset_root(tmp_path: Path) -> None:
    """Live file resolution must use an explicit absolute directory."""
    log = load_measurement_log(make_measurement_log(tmp_path / "run"))

    with pytest.raises(ValueError, match="absolute path"):
        ResolvedForwardContext.from_run_context(
            log.context,
            run_root=Path("relative-run-root"),
        )


def test_live_context_rejects_runtime_config_identity_mismatch(
    tmp_path: Path,
) -> None:
    """A live context cannot change physical config behind its declared hash."""
    log = load_measurement_log(make_measurement_log(tmp_path / "run"))
    payload = log.context.to_payload()
    runtime_payload = dict(payload["runtime_config"])
    runtime_payload["detector_model_id"] = "tampered-detector"
    payload["runtime_config"] = runtime_payload
    tampered = RunContext.from_payload(payload)

    with pytest.raises(
        MeasurementLogValidationError,
        match="runtime_config_sha256",
    ):
        ResolvedForwardContext.from_run_context(
            tampered,
            run_root=log.path,
        )


def test_file_backed_obstacle_identity_and_resolution(tmp_path: Path) -> None:
    """Portable obstacle assets remain hash-bound and root-confined."""
    config = runtime_config()
    env = environment()
    env.update(
        {
            "size_x": 10.0,
            "size_y": 20.0,
            "detector_position": [0.25, 0.25, 0.4],
        }
    )
    obstacle_path = "obstacle_layouts/no_obstacles.json"
    config_hash = sha256_json(config)
    forward = build_forward_model_manifest(
        runtime_config=config,
        environment=env,
        obstacle_layout_path=obstacle_path,
        isotopes=TEST_ISOTOPES,
        repository_commit=TEST_COMMIT,
        resolved_config_sha256=config_hash,
    )
    log = write_measurement_log(
        tmp_path / "run",
        run_id="file-backed-obstacle",
        repository_commit=TEST_COMMIT,
        runtime_config=config,
        environment=env,
        forward_model_manifest=forward,
        isotopes=TEST_ISOTOPES,
        records=records(),
        obstacle_layout_path=obstacle_path,
    )

    context = ResolvedForwardContext.from_log(log)

    assert context.obstacle_grid is not None
    assert context.obstacle_grid.grid_shape == (10, 20)
    assert context.resolved_obstacle_path == (
        Path(__file__).resolve().parents[1] / obstacle_path
    )
    identity = context.asset_identities["obstacle_layout_path"]
    assert identity["component"] == "obstacle"
    assert identity["path"] == obstacle_path
    assert len(identity["sha256"]) == 64


@pytest.mark.parametrize("gpu_dtype", ["float16", "double", ""])
def test_kernel_builder_rejects_unsupported_dtype(
    tmp_path: Path,
    gpu_dtype: str,
) -> None:
    """Kernel construction must not silently reinterpret precision settings."""
    log = load_measurement_log(make_measurement_log(tmp_path / "run"))
    context = ResolvedForwardContext.from_log(log)

    with pytest.raises(ValueError, match="gpu_dtype"):
        context.build_continuous_kernel(
            use_gpu=False,
            gpu_device="cpu",
            gpu_dtype=gpu_dtype,
        )
