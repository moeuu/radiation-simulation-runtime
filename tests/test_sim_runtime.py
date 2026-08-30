"""Focused runtime tests for native full-spectrum fidelity handshakes."""

from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess

import pytest

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
    surface_source_runtime_contract_sha256,
)
from sim.runtime import (
    Geant4TCPClientRuntime,
    ManagedGeant4TCPClientRuntime,
    SimulationRuntime,
    TCPSidecarClientRuntime,
    _config_bool,
    _config_integer,
    _config_number,
    _config_string,
    _finish_managed_sidecar_close,
    _resolve_geant4_sidecar_config_path,
    _start_sidecar_process,
    create_simulation_runtime,
    load_production_runtime_config,
    load_production_runtime_config_with_digest,
    load_runtime_config,
    production_runtime_config_sha256,
)
from sim.protocol import decode_message, encode_message
from spectrum.detector_green_operator import (
    DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
    DETECTOR_GREEN_SAMPLING_MODE,
)
from spectrum.response_matrix import (
    NATIVE_GEANT4_BACKGROUND_MODEL_ID,
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_ENERGY_MAX_KEV,
    NATIVE_GEANT4_ENERGY_MIN_KEV,
)


TEST_GREEN_CONTRACT_SHA256 = "1" * 64
TEST_GREEN_BINARY_SHA256 = "2" * 64


def _full_spectrum_handshake() -> dict[str, object]:
    """Return a unit-history native reset handshake."""
    return {
        "runtime_fidelity": {
            "primary_sampling_fraction": 1.0,
            "requested_primary_sampling_fraction": 1.0,
            "primary_history_weight": 1.0,
            "target_sampled_primaries": 0,
            "primary_sampling_budget_enabled": False,
            "primary_sampling_fraction_resolution": "fixed_fraction",
            "accelerated_weighted_transport_enable": False,
            "sample_detector_response": True,
            "detector_response_sampling_contract_sha256": (TEST_GREEN_CONTRACT_SHA256),
            "detector_response_operator_binary_sha256": (TEST_GREEN_BINARY_SHA256),
            "detector_response_sampling_model": (
                "isotope_independent_full_detector_green_operator_v3"
            ),
            "detector_response_sampling_mode": DETECTOR_GREEN_SAMPLING_MODE,
            "detector_response_boundary_state": (
                "normalized_impact_parameter_at_detector_housing_entry_v1"
            ),
            "detector_response_conditioning": (
                "registered_pulse_subprobability_given_housing_incident_gamma_v1"
            ),
            "detector_response_coincidence_semantics": (
                DETECTOR_GREEN_COINCIDENCE_SEMANTICS
            ),
            "detector_cps_green_reference_normalization": (
                "catalog_branching_weighted_absolute_detection_efficiency_at_1m_v1"
            ),
            "background_spectrum_model_id": NATIVE_GEANT4_BACKGROUND_MODEL_ID,
            "spectrum_energy_min_keV": NATIVE_GEANT4_ENERGY_MIN_KEV,
            "spectrum_energy_max_keV": NATIVE_GEANT4_ENERGY_MAX_KEV,
            "spectrum_bin_width_keV": NATIVE_GEANT4_BIN_WIDTH_KEV,
            "spectrum_bin_count": NATIVE_GEANT4_BIN_COUNT,
            "requested_threads": 32,
            "source_position_semantics": "air_side_native_emission_xyz",
            "source_anchor_semantics": ("exact_surface_chart_uv_evaluation_truth"),
            "all_sources_surface_bound": True,
            "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
            "surface_emission_policy_sha256": (surface_emission_policy_sha256()),
            "surface_source_contract_sha256": "b" * 64,
            "scene_hash": "c" * 64,
            "native_executable_sha256": "d" * 64,
            "native_execution_environment_sha256": "e" * 64,
            "implementation_bundle_sha256": "f" * 64,
            "intensity_cps_1m_definition": ("pre_dead_time_detector_pulse_rate_at_1m"),
        }
    }


def _client(
    *,
    expected_thread_count: int | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_native_executable_sha256: str | None = None,
    expected_native_execution_environment_sha256: str | None = None,
    expected_implementation_bundle_sha256: str | None = None,
) -> Geant4TCPClientRuntime:
    """Return an unconnected client configured for production response sampling."""
    return Geant4TCPClientRuntime(
        "127.0.0.1",
        65530,
        expected_detector_response_sampling=True,
        expected_detector_green_operator_contract_sha256=(TEST_GREEN_CONTRACT_SHA256),
        expected_detector_green_operator_binary_sha256=(TEST_GREEN_BINARY_SHA256),
        expected_thread_count=expected_thread_count,
        expected_runtime_config_sha256=expected_runtime_config_sha256,
        expected_native_executable_sha256=(expected_native_executable_sha256),
        expected_native_execution_environment_sha256=(
            expected_native_execution_environment_sha256
        ),
        expected_implementation_bundle_sha256=(expected_implementation_bundle_sha256),
    )


def test_tcp_runtime_normalizes_geometry_tuples_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public runtime boundary must encode internal tuples as JSON arrays."""
    sent: list[bytes] = []
    response = encode_message("ok", {"accepted": True})

    class FakeConnection:
        """Capture one request and return one deterministic sidecar response."""

        def __init__(self) -> None:
            """Initialize the one-response receive cursor."""
            self.response_pending = True

        def __enter__(self) -> "FakeConnection":
            """Return this fake socket from its context manager."""
            return self

        def __exit__(self, *args: object) -> None:
            """Close the fake socket without suppressing errors."""
            del args

        def sendall(self, payload: bytes) -> None:
            """Capture the complete encoded request."""
            sent.append(payload)

        def shutdown(self, how: int) -> None:
            """Accept the client write-shutdown notification."""
            del how

        def recv(self, size: int) -> bytes:
            """Return the response once and EOF thereafter."""
            del size
            if not self.response_pending:
                return b""
            self.response_pending = False
            return response

    fake = FakeConnection()
    monkeypatch.setattr(
        "sim.runtime.socket.create_connection",
        lambda *args, **kwargs: fake,
    )
    client = TCPSidecarClientRuntime("127.0.0.1", 5556)

    result = client._round_trip(
        "reset",
        {"obstacle_cells": [(1, 2)], "transport_boxes_m": [(0.0,) * 6]},
    )

    assert result == {"accepted": True}
    assert len(sent) == 1
    assert decode_message(sent[0].strip()) == (
        "reset",
        {
            "obstacle_cells": [[1, 2]],
            "transport_boxes_m": [[0.0] * 6],
        },
    )


@pytest.mark.parametrize("value", (0, 1, "false", "true", None))
def test_sidecar_switches_require_exact_json_booleans(value: object) -> None:
    """A truthy string must never enable a mock or alternate sidecar path."""
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        _config_bool({"sidecar_mock_stage": value}, "sidecar_mock_stage", False)


def test_managed_geant4_transport_has_no_restart_override() -> None:
    """A managed production sidecar must propagate its first socket failure."""
    assert (
        ManagedGeant4TCPClientRuntime._round_trip is Geant4TCPClientRuntime._round_trip
    )
    assert ManagedGeant4TCPClientRuntime.reset is Geant4TCPClientRuntime.reset


def test_tcp_sidecar_close_propagates_shutdown_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing shutdown acknowledgement must remain a fatal close error."""
    client = TCPSidecarClientRuntime("127.0.0.1", 65530)

    def fail_shutdown(*args: object, **kwargs: object) -> dict[str, object]:
        """Inject loss of the shutdown acknowledgement."""
        del args, kwargs
        raise OSError("injected shutdown transport failure")

    monkeypatch.setattr(client, "_round_trip", fail_shutdown)

    with pytest.raises(OSError, match="injected shutdown transport failure"):
        client.close()


def test_persistent_geant4_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native persistent-process failure must execute exactly one attempt."""
    from sim.geant4_app.engine import ExternalCommandGeant4Engine

    engine = object.__new__(ExternalCommandGeant4Engine)
    engine.scene = object()
    attempts = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        """Raise the first native-process failure without a restart path."""
        nonlocal attempts
        del args, kwargs
        attempts += 1
        raise RuntimeError("Persistent Geant4 executable exited unexpectedly")

    monkeypatch.setattr(
        ExternalCommandGeant4Engine,
        "_simulate_persistent_once",
        fail_once,
    )
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        engine._simulate_persistent(object())
    assert attempts == 1


def test_dead_persistent_geant4_child_cannot_be_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later request must fail closed after the native child has exited."""
    from sim.geant4_app.engine import ExternalCommandGeant4Engine

    class DeadProcess:
        """Represent one already-exited persistent child."""

        def poll(self) -> int:
            """Return the child's fatal exit status."""
            return 17

    engine = object.__new__(ExternalCommandGeant4Engine)
    engine._persistent_process = DeadProcess()
    launches = 0

    def forbidden_popen(*args: object, **kwargs: object) -> object:
        """Record any forbidden replacement child launch."""
        nonlocal launches
        del args, kwargs
        launches += 1
        return object()

    monkeypatch.setattr(
        "sim.geant4_app.engine.subprocess.Popen",
        forbidden_popen,
    )

    with pytest.raises(RuntimeError, match="cannot be restarted.*returncode=17"):
        engine._ensure_persistent_process()

    assert launches == 0
    assert engine._persistent_process is not None


def test_persistent_geant4_forced_shutdown_is_fatal() -> None:
    """Forced native termination must not authenticate acquisition completion."""
    from sim.geant4_app.engine import ExternalCommandGeant4Engine

    class StubbornProcess:
        """Ignore graceful shutdown until the owner terminates the process."""

        def __init__(self) -> None:
            """Initialize running process state and observable streams."""
            self.returncode: int | None = None
            self.stdin = StringIO()
            self.stdout = StringIO()
            self.terminated = False

        def poll(self) -> int | None:
            """Return the current child exit status."""
            return self.returncode

        def wait(self, timeout: float) -> int:
            """Time out until forced termination occurs."""
            del timeout
            if self.returncode is None:
                raise subprocess.TimeoutExpired("geant4", 5.0)
            return self.returncode

        def terminate(self) -> None:
            """Record forced termination and expose its signal status."""
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            """Expose an unexpected hard-kill status if reached."""
            self.returncode = -9

    class FakeTemporaryDirectory:
        """Record cleanup of the persistent process directory."""

        def __init__(self) -> None:
            """Initialize cleanup state."""
            self.cleaned = False

        def cleanup(self) -> None:
            """Record removal of temporary native inputs."""
            self.cleaned = True

    process = StubbornProcess()
    temporary = FakeTemporaryDirectory()
    engine = object.__new__(ExternalCommandGeant4Engine)
    engine._persistent_process = process
    engine._persistent_tmpdir = temporary
    engine._persistent_scene_path = Path("scene.txt")
    engine._persistent_scene_hash = "a" * 64

    with pytest.raises(RuntimeError, match="required forced termination"):
        engine._close_persistent_process()

    assert process.terminated is True
    assert temporary.cleaned is True
    assert engine._persistent_process is None


def test_persistent_native_launch_rechecks_bundle_before_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-handshake native input change must abort before process launch."""
    from sim.geant4_app.engine import (
        ExternalCommandGeant4Engine,
        Geant4EngineConfig,
    )

    engine = ExternalCommandGeant4Engine(
        Geant4EngineConfig(
            executable_path="/approved/geant4_sidecar",
            persistent_process=True,
            expected_native_executable_sha256="d" * 64,
            expected_native_execution_environment_sha256="e" * 64,
            expected_implementation_bundle_sha256="f" * 64,
        )
    )
    launches = 0

    def changed_bundle(*args: object, **kwargs: object) -> None:
        """Represent a library or data change after reset validation."""
        del args, kwargs
        raise RuntimeError("physics data changed after provenance validation")

    def forbidden_popen(*args: object, **kwargs: object) -> object:
        """Record any forbidden process launch after a provenance mismatch."""
        nonlocal launches
        del args, kwargs
        launches += 1
        return object()

    monkeypatch.setattr(
        "sim.geant4_app.engine.require_native_execution_bundle",
        changed_bundle,
    )
    monkeypatch.setattr(
        "sim.geant4_app.engine.subprocess.Popen",
        forbidden_popen,
    )

    with pytest.raises(RuntimeError, match="physics data changed"):
        engine._ensure_persistent_process()
    assert launches == 0


def test_persistent_native_launch_rechecks_python_before_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-handshake Python source change must abort before native launch."""
    from sim.geant4_app.engine import (
        ExternalCommandGeant4Engine,
        Geant4EngineConfig,
    )

    engine = ExternalCommandGeant4Engine(
        Geant4EngineConfig(
            executable_path="/approved/geant4_sidecar",
            persistent_process=True,
            expected_native_executable_sha256="d" * 64,
            expected_native_execution_environment_sha256="e" * 64,
            expected_implementation_bundle_sha256="f" * 64,
        )
    )
    launches = 0

    def forbidden_popen(*args: object, **kwargs: object) -> object:
        """Record any forbidden native launch after Python source drift."""
        nonlocal launches
        del args, kwargs
        launches += 1
        return object()

    monkeypatch.setattr(
        "sim.geant4_app.engine.require_native_execution_bundle",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "spectrum.full_spectrum_acceptance_runner."
        "acceptance_implementation_bundle_sha256",
        lambda _root: "0" * 64,
    )
    monkeypatch.setattr(
        "sim.geant4_app.engine.subprocess.Popen",
        forbidden_popen,
    )

    with pytest.raises(RuntimeError, match="Python implementation bundle"):
        engine._ensure_persistent_process()
    assert launches == 0


@pytest.mark.parametrize("value", (True, 5556.0, "5556", None))
def test_sidecar_ports_require_exact_json_integers(value: object) -> None:
    """Sidecar port values must not be silently coerced."""
    with pytest.raises(ValueError, match="must be a JSON integer"):
        _config_integer(
            {"port": value},
            "port",
            5556,
            minimum=1,
            maximum=65535,
        )


@pytest.mark.parametrize("value", (True, "120", None, float("nan")))
def test_sidecar_timeouts_require_finite_json_numbers(value: object) -> None:
    """Invalid timeouts must stop before starting an unmonitored transport."""
    with pytest.raises(ValueError, match="must be|finite"):
        _config_number(
            {"timeout_s": value},
            "timeout_s",
            120.0,
            minimum=0.0,
            strict_minimum=True,
        )


@pytest.mark.parametrize("value", (True, 127, None, " "))
def test_sidecar_strings_reject_stringification(value: object) -> None:
    """Hosts, backends, and material IDs must be exact nonempty strings."""
    with pytest.raises(ValueError, match="JSON string"):
        _config_string({"host": value}, "host", "127.0.0.1")


def test_sidecar_popen_failure_closes_its_log_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A child launch error must not leak the newly opened sidecar log."""
    log_handle = StringIO()

    def open_log(*args: object, **kwargs: object) -> StringIO:
        """Return one observable in-memory sidecar log handle."""
        del args, kwargs
        return log_handle

    def fail_popen(*args: object, **kwargs: object) -> object:
        """Inject an operating-system child launch failure."""
        del args, kwargs
        raise OSError("injected Popen failure")

    monkeypatch.setattr(Path, "open", open_log)
    monkeypatch.setattr(
        "sim.runtime._resolve_sidecar_python",
        lambda *_args: "/usr/bin/python3",
    )
    monkeypatch.setattr("sim.runtime.subprocess.Popen", fail_popen)

    with pytest.raises(OSError, match="injected Popen failure"):
        _start_sidecar_process(
            script_path=tmp_path / "sidecar.py",
            config_path=tmp_path / "config.json",
            config={},
            host="127.0.0.1",
            port=65530,
            timeout_s=1.0,
            log_path=tmp_path / "sidecar.log",
            sidecar_name="test",
        )

    assert log_handle.closed is True


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("forced_kill", "forced termination"),
        ("nonzero", "nonzero status"),
    ),
)
def test_managed_sidecar_rejects_unclean_child_exit(
    mode: str,
    message: str,
    tmp_path: Path,
) -> None:
    """Forced kill and nonzero child exit must both fail finalization."""

    class FakeProcess:
        """Expose deterministic forced-kill or nonzero-exit behavior."""

        def __init__(self) -> None:
            """Initialize the requested shutdown outcome."""
            self.returncode: int | None = None
            self.wait_calls = 0
            self.terminated = False
            self.killed = False

        def poll(self) -> int | None:
            """Return the current fake child status."""
            return self.returncode

        def wait(self, timeout: float) -> int:
            """Produce the selected child shutdown outcome."""
            del timeout
            self.wait_calls += 1
            if mode == "nonzero":
                self.returncode = 7
                return 7
            if not self.killed:
                raise subprocess.TimeoutExpired("sidecar", 5.0)
            return -9

        def terminate(self) -> None:
            """Record a soft forced-stop attempt."""
            self.terminated = True

        def kill(self) -> None:
            """Record the required hard kill."""
            self.killed = True
            self.returncode = -9

    process = FakeProcess()
    log_handle = StringIO()
    temp_config = tmp_path / "sidecar-config.json"
    temp_config.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        _finish_managed_sidecar_close(
            process=process,  # type: ignore[arg-type]
            log_handle=log_handle,
            temp_config_path=temp_config,
            shutdown_failure=None,
        )

    assert log_handle.closed is True
    assert not temp_config.exists()
    if mode == "forced_kill":
        assert process.terminated is True
        assert process.killed is True


def test_runtime_config_rejects_duplicate_json_keys(tmp_path) -> None:
    """Duplicate physics settings must not be resolved by last-key wins."""
    config_path = tmp_path / "duplicate.json"
    config_path.write_text(
        '{"background_cps": 12.0, "background_cps": 0.0}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key 'background_cps'"):
        load_runtime_config(config_path)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_runtime_config_rejects_nonfinite_json_numbers(
    tmp_path,
    constant: str,
) -> None:
    """Non-standard non-finite JSON cannot alter a scientific runtime."""
    config_path = tmp_path / "nonfinite.json"
    config_path.write_text(
        f'{{"background_cps": {constant}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite standard-JSON numbers"):
        load_runtime_config(config_path)


@pytest.mark.parametrize("parent_ref", ("", 1, False, ["parent.json"]))
def test_runtime_config_extends_requires_nonempty_string(
    tmp_path,
    parent_ref: object,
) -> None:
    """Inheritance must not stringify ambiguous JSON values into paths."""
    config_path = tmp_path / "invalid_extends.json"
    config_path.write_text(
        json.dumps({"extends": parent_ref}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nonempty JSON string"):
        load_runtime_config(config_path)


@pytest.mark.parametrize(
    "config_name",
    (
        "diagnostic_external_no_isaac_1thread.json",
        "variance_reduction_external_gui_32threads.json",
        "variance_reduction_external_no_isaac_32threads.json",
        "variance_reduction_external_no_isaac_32threads_cpu_guarded.json",
    ),
)
def test_standard_production_runtime_config_matches_the_exact_schema(
    config_name: str,
) -> None:
    """The canonical Geant4 production config must be self-contained."""
    root = Path(__file__).resolve().parents[1]
    payload = load_production_runtime_config(root / "configs/geant4" / config_name)
    registry_path = root / payload["full_spectrum_model_registry_path"]

    assert payload["simulation_runtime_schema_version"] == 1
    assert (
        payload["full_spectrum_model_registry_file_sha256"]
        == hashlib.sha256(registry_path.read_bytes()).hexdigest()
    )
    assert payload["obstacle_attenuation_enabled"] is True
    assert payload["backend"] == "geant4"
    assert (
        payload["use_mock_stage"] is not payload["start_isaacsim_sidecar_with_geant4"]
    )
    if payload["start_isaacsim_sidecar_with_geant4"]:
        assert payload["isaacsim_keep_sidecar_alive"] is False


def test_production_gui_rejects_keep_alive_sidecar_ownership(
    tmp_path: Path,
) -> None:
    """A production GUI sidecar cannot outlive its exact acquisition owner."""
    root = Path(__file__).resolve().parents[1]
    payload = load_production_runtime_config(
        root / "configs/geant4/variance_reduction_external_gui_32threads.json"
    )
    payload["isaacsim_keep_sidecar_alive"] = True
    config_path = tmp_path / "unowned-gui.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="keep_sidecar_alive=false"):
        load_production_runtime_config(config_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("unknown", "unknown_or_retired"),
        ("missing", "missing"),
        ("retired", "unknown_or_retired"),
        ("string_boolean", "exact JSON booleans"),
        ("analytic_backend", "backend='geant4'"),
        ("mock_stage", "Production use_mock_stage"),
        ("ignored_cui", "unknown_or_retired"),
    ),
)
def test_production_runtime_config_rejects_fail_open_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Typos, omissions, retired controls, and coercion must fail closed."""
    root = Path(__file__).resolve().parents[1]
    payload = load_production_runtime_config(
        root / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    if mutation == "unknown":
        payload["num_particels"] = 2000
    elif mutation == "missing":
        del payload["thread_count"]
    elif mutation == "retired":
        payload["num_particles"] = 2000
    elif mutation == "string_boolean":
        payload["obstacle_attenuation_enabled"] = "false"
    elif mutation == "analytic_backend":
        payload["backend"] = "analytic"
    elif mutation == "mock_stage":
        payload["use_mock_stage"] = not payload["use_mock_stage"]
    else:
        payload["cui_split_view"] = True
    config_path = tmp_path / f"{mutation}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_production_runtime_config(config_path)


def test_production_runtime_config_rejects_extends(tmp_path: Path) -> None:
    """Production must not inherit an implicit legacy parent configuration."""
    config_path = tmp_path / "production.json"
    config_path.write_text(
        json.dumps({"extends": "parent.json"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="retired 'extends'"):
        load_production_runtime_config(config_path)


@pytest.mark.parametrize(
    "retired_field",
    (
        "detector_height_pair_xy_tolerance_m",
        "detector_height_pair_z_tolerance_m",
        "detector_height_sampling_mode",
        "detector_pose_consistency_tolerance_m",
        "detector_transport_height_m",
        "background_rate_cps",
        "source_bias_cone_half_angle_deg",
        "headless_visualizer_defer",
        "min_rotations_per_pose",
        "random_environment_base_usd_path",
        "random_source_intensity_min_cps_1m",
        "random_source_intensity_max_cps_1m",
        "random_source_same_isotope_min_distance_m",
        "random_source_surface_sampling_measure",
        "spectrum_plot_save_every",
    ),
)
def test_production_runtime_config_rejects_removed_noop_fields(
    tmp_path: Path,
    retired_field: str,
) -> None:
    """Removed no-op controls cannot return as accepted production settings."""
    root = Path(__file__).resolve().parents[1]
    payload = load_production_runtime_config(
        root / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    payload[retired_field] = 1
    config_path = tmp_path / f"retired-{retired_field}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_or_retired"):
        load_production_runtime_config(config_path)


def test_geant4_auto_start_freezes_validated_mapping_in_private_config() -> None:
    """The child must never reread a mutable source config after preflight."""
    root = Path(__file__).resolve().parents[1]
    source = root / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    payload = load_production_runtime_config(source)

    frozen, temporary = _resolve_geant4_sidecar_config_path(payload, source)
    try:
        assert temporary == frozen
        assert frozen != source
        assert (
            Path(payload["usd_path"])
            == (root / "configs/isaacsim/demo_room.usda").resolve()
        )
        assert load_production_runtime_config(frozen) == payload
    finally:
        frozen.unlink(missing_ok=True)


def test_moved_production_config_keeps_canonical_usd_asset_root(
    tmp_path: Path,
) -> None:
    """Moving a config must not silently relocate its repository assets."""
    root = Path(__file__).resolve().parents[1]
    source = root / "configs/geant4/diagnostic_external_no_isaac_1thread.json"
    moved = tmp_path / "copied-config.json"
    moved.write_bytes(source.read_bytes())

    payload = load_production_runtime_config(moved)

    assert (
        Path(payload["usd_path"])
        == (root / "configs/isaacsim/demo_room.usda").resolve()
    )


def test_production_loader_runs_geant4_value_validation(tmp_path: Path) -> None:
    """Exact keys cannot bypass the application's numeric value checks."""
    root = Path(__file__).resolve().parents[1]
    payload = load_production_runtime_config(
        root / "configs/geant4/diagnostic_external_no_isaac_1thread.json"
    )
    payload["thread_count"] = 0
    invalid = tmp_path / "invalid-thread-count.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="thread_count must be at least 1"):
        load_production_runtime_config(invalid)


def test_production_runtime_digest_binds_detector_geometry() -> None:
    """Changing detector geometry must identify a different sidecar config."""
    root = Path(__file__).resolve().parents[1]
    config = load_production_runtime_config(
        root / "configs/geant4/diagnostic_external_no_isaac_1thread.json"
    )
    modified = json.loads(json.dumps(config))
    modified["detector_model"]["crystal_radius_m"] += 0.001

    assert production_runtime_config_sha256(modified) != (
        production_runtime_config_sha256(config)
    )


def test_production_runtime_loader_returns_the_canonical_digest() -> None:
    """All validation and acceptance phases must share one config identity."""
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"

    config, digest = load_production_runtime_config_with_digest(path)

    assert digest == production_runtime_config_sha256(config)


def test_reset_handshake_accepts_exact_native_full_spectrum_contract() -> None:
    """The reset boundary authenticates response sampling before any action."""
    _client()._validate_fidelity_handshake(_full_spectrum_handshake())


def test_production_reset_accepts_matching_runtime_config_digest() -> None:
    """The strict client accepts the exact canonical config identity."""
    digest = "d" * 64
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["production_runtime_config_sha256"] = digest

    _client(expected_runtime_config_sha256=digest)._validate_fidelity_handshake(
        handshake
    )


@pytest.mark.parametrize(
    ("field_name", "expected", "actual"),
    (
        ("native_executable_sha256", "d" * 64, None),
        ("native_executable_sha256", "d" * 64, "0" * 64),
        ("native_execution_environment_sha256", "e" * 64, None),
        ("native_execution_environment_sha256", "e" * 64, "0" * 64),
        ("implementation_bundle_sha256", "f" * 64, None),
        ("implementation_bundle_sha256", "f" * 64, "0" * 64),
    ),
)
def test_production_reset_rejects_mismatched_native_execution_provenance(
    field_name: str,
    expected: str,
    actual: str | None,
) -> None:
    """The reset handshake must bind the actual sidecar binary and environment."""
    handshake = _full_spectrum_handshake()
    if actual is None:
        del handshake["runtime_fidelity"][field_name]
    else:
        handshake["runtime_fidelity"][field_name] = actual
    arguments = {
        "expected_native_executable_sha256": None,
        "expected_native_execution_environment_sha256": None,
        "expected_implementation_bundle_sha256": None,
    }
    arguments[f"expected_{field_name}"] = expected

    with pytest.raises(RuntimeError, match="native execution provenance"):
        _client(**arguments)._validate_fidelity_handshake(handshake)


def test_production_create_rejects_existing_tcp_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production must never attach to a stale or foreign sidecar process."""
    root = Path(__file__).resolve().parents[1]
    config = load_production_runtime_config(
        root / "configs/geant4/diagnostic_external_no_isaac_1thread.json"
    )
    monkeypatch.setattr("sim.runtime._tcp_server_available", lambda *_args: True)
    monkeypatch.setattr(
        "sim.runtime._configured_detector_green_hashes",
        lambda *_args: (
            TEST_GREEN_CONTRACT_SHA256,
            TEST_GREEN_BINARY_SHA256,
        ),
    )

    with pytest.raises(RuntimeError, match="refuses to attach"):
        create_simulation_runtime(
            "geant4",
            sources=[],
            mu_by_isotope={},
            shield_params=None,
            runtime_config=config,
            expected_native_executable_sha256="d" * 64,
            expected_native_execution_environment_sha256="e" * 64,
            expected_implementation_bundle_sha256="f" * 64,
        )


def test_production_gui_rejects_existing_isaac_endpoint_and_closes_geant4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production GUI pairing must never attach to an unauthenticated Isaac server."""
    root = Path(__file__).resolve().parents[1]
    config = load_production_runtime_config(
        root / "configs/geant4/variance_reduction_external_gui_32threads.json"
    )

    class FakeGeant4Runtime(SimulationRuntime):
        """Record cleanup of a freshly started Geant4 companion."""

        def __init__(self) -> None:
            """Initialize close state."""
            self.closed = False

        def reset(self, payload: dict[str, object] | None = None) -> None:
            """Accept an unused reset payload."""
            del payload

        def step(self, command: object) -> object:
            """Reject steps because startup must fail before acquisition."""
            del command
            raise AssertionError("step must not be called")

        def close(self) -> None:
            """Record cleanup after the Isaac pairing failure."""
            self.closed = True

    geant4_runtime = FakeGeant4Runtime()
    endpoint_checks = iter((False, True))
    monkeypatch.setattr(
        "sim.runtime._tcp_server_available",
        lambda *_args: next(endpoint_checks),
    )
    monkeypatch.setattr(
        "sim.runtime._start_geant4_sidecar",
        lambda *_args, **_kwargs: geant4_runtime,
    )
    monkeypatch.setattr(
        "sim.runtime._configured_detector_green_hashes",
        lambda *_args: (
            TEST_GREEN_CONTRACT_SHA256,
            TEST_GREEN_BINARY_SHA256,
        ),
    )
    monkeypatch.setattr(
        "sim.runtime._resolve_isaacsim_sidecar_config_path",
        lambda *_args, **_kwargs: (
            root / "configs/isaacsim/demo_room_gui.json",
            None,
            {"host": "127.0.0.1", "port": 5555, "timeout_s": 10.0},
        ),
    )

    with pytest.raises(RuntimeError, match="Isaac Sim refuses to attach"):
        create_simulation_runtime(
            "geant4",
            sources=[],
            mu_by_isotope={},
            shield_params=None,
            runtime_config=config,
            expected_native_executable_sha256="d" * 64,
            expected_native_execution_environment_sha256="e" * 64,
            expected_implementation_bundle_sha256="f" * 64,
        )

    assert geant4_runtime.closed is True


def test_production_create_rejects_incorrect_declared_config_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller cannot weaken exact-config binding with an arbitrary digest."""
    root = Path(__file__).resolve().parents[1]
    config = load_production_runtime_config(
        root / "configs/geant4/diagnostic_external_no_isaac_1thread.json"
    )
    monkeypatch.setattr("sim.runtime._tcp_server_available", lambda *_args: True)

    with pytest.raises(ValueError, match="does not authenticate"):
        create_simulation_runtime(
            "geant4",
            sources=[],
            mu_by_isotope={},
            shield_params=None,
            runtime_config=config,
            expected_runtime_config_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    "key",
    (
        "sample_detector_response",
        "detector_response_sampling_contract_sha256",
        "detector_response_operator_binary_sha256",
        "detector_response_sampling_model",
        "detector_response_sampling_mode",
        "detector_response_boundary_state",
        "detector_response_conditioning",
        "detector_response_coincidence_semantics",
        "detector_cps_green_reference_normalization",
        "background_spectrum_model_id",
        "spectrum_energy_min_keV",
        "spectrum_energy_max_keV",
        "spectrum_bin_width_keV",
        "spectrum_bin_count",
        "source_position_semantics",
        "source_anchor_semantics",
        "all_sources_surface_bound",
        "surface_emission_epsilon_m",
        "surface_emission_policy_sha256",
        "surface_source_contract_sha256",
        "scene_hash",
        "intensity_cps_1m_definition",
    ),
)
def test_reset_handshake_rejects_missing_response_contract_field(key: str) -> None:
    """A stale native process cannot survive reset compatibility checks."""
    handshake = _full_spectrum_handshake()
    del handshake["runtime_fidelity"][key]

    with pytest.raises(RuntimeError, match="handshake|response sampling"):
        _client()._validate_fidelity_handshake(handshake)


def test_reset_handshake_rejects_disabled_response_sampling() -> None:
    """Incident-energy histograms without native detector marking are invalid."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["sample_detector_response"] = False

    with pytest.raises(RuntimeError, match="response sampling mismatch"):
        _client()._validate_fidelity_handshake(handshake)


@pytest.mark.parametrize(
    "value",
    (
        float(NATIVE_GEANT4_BIN_COUNT),
        NATIVE_GEANT4_BIN_COUNT + 0.5,
        str(NATIVE_GEANT4_BIN_COUNT),
    ),
)
def test_reset_handshake_rejects_coerced_spectrum_bin_count(
    value: object,
) -> None:
    """A fractional-capable or string bin count cannot authenticate the axis."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["spectrum_bin_count"] = value

    with pytest.raises(RuntimeError, match="spectrum_bin_count"):
        _client()._validate_fidelity_handshake(handshake)


@pytest.mark.parametrize("value", (32.0, 32.5, "32"))
def test_reset_handshake_rejects_coerced_thread_count(value: object) -> None:
    """Native thread-count provenance must remain an exact JSON integer."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["requested_threads"] = value

    with pytest.raises(RuntimeError, match="requested_threads"):
        _client(expected_thread_count=32)._validate_fidelity_handshake(handshake)


@pytest.mark.parametrize("value", (True, 0, 32.0, "32"))
def test_client_rejects_coerced_expected_thread_count(value: object) -> None:
    """The local thread contract cannot be weakened before reset validation."""
    with pytest.raises(ValueError, match="expected_thread_count"):
        _client(expected_thread_count=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("primary_sampling_fraction", "1.0"),
        ("primary_sampling_fraction", True),
        (
            "surface_emission_epsilon_m",
            str(SURFACE_EMISSION_EPSILON_M),
        ),
        (
            "spectrum_bin_width_keV",
            str(NATIVE_GEANT4_BIN_WIDTH_KEV),
        ),
    ),
)
def test_reset_handshake_rejects_coerced_numeric_provenance(
    key: str,
    value: object,
) -> None:
    """Wire numeric provenance cannot be accepted through float coercion."""
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"][key] = value

    with pytest.raises(RuntimeError, match=key):
        _client()._validate_fidelity_handshake(handshake)


def test_reset_handshake_binds_source_strength_and_transport_sha() -> None:
    """The native reset must authenticate the exact source payload it received."""
    source = {
        "isotope": "Cs-137",
        "position": [0.0, 0.5, 0.5],
        "transport_position": [
            SURFACE_EMISSION_EPSILON_M,
            0.5,
            0.5,
        ],
        "intensity_cps_1m": 300_000.0,
        "surface_chart_id": 0,
        "surface_uv": [0.5, 0.5],
        "surface_normal": [1.0, 0.0, 0.0],
        "surface_emission_policy_sha256": surface_emission_policy_sha256(),
    }
    expected_hash = surface_source_runtime_contract_sha256([source])
    handshake = _full_spectrum_handshake()
    handshake["runtime_fidelity"]["surface_source_contract_sha256"] = expected_hash
    client = _client()
    client._round_trip = lambda *_args, **_kwargs: handshake  # type: ignore[method-assign]

    client.reset({"sources": [source]})

    assert client.expected_surface_source_contract_sha256 == expected_hash
    assert client.expected_scene_hash == "c" * 64

    stale = _full_spectrum_handshake()
    stale["runtime_fidelity"]["surface_source_contract_sha256"] = "d" * 64
    client = _client()
    client._round_trip = lambda *_args, **_kwargs: stale  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="contract hash differs"):
        client.reset({"sources": [source]})
