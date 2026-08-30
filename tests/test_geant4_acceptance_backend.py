"""Tests for the real-Geant4 acceptance backend contracts."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import spectrum.geant4_acceptance_backend as acceptance_backend

from measurement.model import PointSource
from measurement.geometry_family import randomized_training_geometry_parameters
from measurement.obstacles import ObstacleGrid
from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    NATIVE_ACCEPTANCE_FIDELITY,
)
from spectrum.geant4_acceptance_backend import (
    ExternalGeant4AcceptanceBackend,
    _boundary_probe_evidence_sha256,
    _build_environment,
    _generate_sources,
    _geometry_batch,
    _mutated_surface_scene,
    _native_fidelity,
    _perturbed_sources,
    _validation_labels,
)
from spectrum.native_metadata import (
    native_source_line_token,
    sanitize_native_metadata_token,
)
from spectrum.geant4_physics import GEANT4_VERSION_TAG
from spectrum.detector_green_operator import (
    DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
    DETECTOR_GREEN_SAMPLING_MODE,
)
from spectrum.response_matrix import NATIVE_GEANT4_BIN_COUNT
from spectrum.transport_spectral import (
    ACCEPTANCE_GEOMETRY_DEVICE,
    ACCEPTANCE_GEOMETRY_DTYPE,
    ACCEPTANCE_GEOMETRY_USE_GPU,
    ACCEPTANCE_PERTURBATION_MINIMUM_BEARING_ANGLE_RAD,
    ACCEPTANCE_PERTURBATION_MINIMUM_DISPLACEMENT_M,
    ACCEPTANCE_PERTURBATION_MINIMUM_LOG_RATE_SEPARATION,
    DETECTOR_IMPACT_PHASE_COUNT,
    GeometryConditionedSpectralModel,
)
from tests.green_test_support import synthetic_detector_green_operator


class _FakeKernel:
    """Record isotope-batched geometry calls and return physical tensors."""

    def __init__(self) -> None:
        """Initialize an empty call trace."""
        self.calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def line_transport_components_selected_pairs_for_detectors(
        self,
        isotope: str,
        detectors: np.ndarray,
        sources: np.ndarray,
        fe_indices: np.ndarray,
        pb_indices: np.ndarray,
        local_indices: np.ndarray,
        impact_parameter_edges_fraction: np.ndarray,
    ) -> SimpleNamespace:
        """Return one finite attenuation batch for all requested pairs."""
        del fe_indices, pb_indices
        assert impact_parameter_edges_fraction.shape == (
            DETECTOR_IMPACT_PHASE_COUNT + 1,
        )
        self.calls.append((isotope, sources.shape, local_indices.shape))
        shape = (detectors.shape[0], sources.shape[0], local_indices.size)
        unattenuated = np.full(shape, 2.0, dtype=np.float64)
        return SimpleNamespace(
            unattenuated_kernel=unattenuated,
            uncollided_kernel=0.75 * unattenuated,
            tau_fe=np.full(shape, 0.1, dtype=np.float64),
            tau_pb=np.full(shape, 0.2, dtype=np.float64),
            tau_obstacle=np.full(shape, 0.3, dtype=np.float64),
            tau_obstacle_compton=np.full(shape, 0.15, dtype=np.float64),
            distance_m=np.full(shape, 3.0, dtype=np.float64),
            uncollided_impact_fractions=np.full(
                shape + (DETECTOR_IMPACT_PHASE_COUNT,),
                1.0 / DETECTOR_IMPACT_PHASE_COUNT,
                dtype=np.float64,
            ),
        )

    def line_branching_weights(
        self,
        isotope: str,
        local_indices: np.ndarray,
    ) -> np.ndarray:
        """Return normalized positive weights for one isotope subset."""
        del isotope
        return np.full(
            local_indices.shape,
            1.0 / max(local_indices.size, 1),
            dtype=np.float64,
        )


class _Source:
    """Minimal source object consumed by the batched geometry constructor."""

    def __init__(self, isotope: str, position: tuple[float, float, float]) -> None:
        """Store one isotope, transport position, and physical strength."""
        self.isotope = isotope
        self._position = np.asarray(position, dtype=np.float64)
        self.intensity_cps_1m = 500_000.0

    def transport_position_array(self) -> np.ndarray:
        """Return the continuous source transport coordinate."""
        return self._position.copy()


def test_surface_perturbation_is_predeclared_and_geometry_separable() -> None:
    """Acceptance alternatives must differ before any response is observed."""
    scene_seed = 991_337
    parameters = randomized_training_geometry_parameters(
        scene_seed,
        room_size_xyz=(10.0, 20.0, 10.0),
    )
    obstacle_height = float(parameters["obstacle_height_m"])
    environment, grid, _ = _build_environment(
        scene_seed=scene_seed,
        obstacle_height_m=obstacle_height,
        author_room_boundaries=True,
        room_boundary_thickness_m=0.1,
    )
    expected_isotopes = set(ACCEPTANCE_ISOTOPES)
    assert set(grid.transport_mu_by_isotope) == expected_isotopes
    assert set(grid.transport_line_mu_by_isotope) == expected_isotopes
    assert set(grid.transport_line_compton_mu_by_isotope) == expected_isotopes
    truth = _generate_sources(
        environment=environment,
        grid=grid,
        scene_seed=scene_seed,
        scenario_id="continuous_surface_perturbation_ranking",
        obstacle_height_m=obstacle_height,
    )[0]
    alternative = _perturbed_sources(
        environment=environment,
        grid=grid,
        sources=(truth,),
        obstacle_height_m=obstacle_height,
    )[0]
    detector = environment.detector()
    truth_position = np.asarray(truth.position, dtype=np.float64)
    alternative_position = np.asarray(alternative.position, dtype=np.float64)
    truth_vector = truth_position - detector
    alternative_vector = alternative_position - detector
    truth_distance = float(np.linalg.norm(truth_vector))
    alternative_distance = float(np.linalg.norm(alternative_vector))
    log_rate_separation = abs(2.0 * np.log(alternative_distance / truth_distance))
    bearing_angle = float(
        np.arccos(
            np.clip(
                np.dot(truth_vector, alternative_vector)
                / (truth_distance * alternative_distance),
                -1.0,
                1.0,
            )
        )
    )

    assert np.linalg.norm(alternative_position - truth_position) >= (
        ACCEPTANCE_PERTURBATION_MINIMUM_DISPLACEMENT_M
    )
    assert (
        log_rate_separation
        >= ACCEPTANCE_PERTURBATION_MINIMUM_LOG_RATE_SEPARATION
        or bearing_angle >= ACCEPTANCE_PERTURBATION_MINIMUM_BEARING_ANGLE_RAD
    )
    assert alternative.isotope == truth.isotope
    assert alternative.intensity_cps_1m == truth.intensity_cps_1m


def test_acceptance_kernel_uses_predeclared_cpu_float64_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime config must not select acceptance geometry compute semantics."""
    backend = object.__new__(ExternalGeant4AcceptanceBackend)
    backend.runtime_config = {
        "benign_observation_field": "retained",
        "full_spectrum_generative_model": {"candidate": True},
        "full_spectrum_generative_model_path": "candidate.json",
        "full_spectrum_generative_model_file_sha256": "a" * 64,
        "full_spectrum_contract_hash_sha256": "b" * 64,
        "full_spectrum_model_registry_file_sha256": "c" * 64,
        "full_spectrum_model_registry_path": "registry.json",
        "isotope_experiment_profile": "unapproved_candidate",
    }
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
    )
    observation = SimpleNamespace(additive_scatter_response=None)
    captured: dict[str, object] = {}

    def _build_observation(
        payload: object,
        *,
        isotopes: object,
    ) -> SimpleNamespace:
        """Return one model-free observation and record its exact inputs."""
        captured["payload"] = payload
        captured["isotopes"] = isotopes
        return observation

    def _build_kernel(
        actual_observation: object,
        *,
        obstacle_grid: object,
        use_gpu: object,
    ) -> SimpleNamespace:
        """Return one mutable fake kernel and record compute selection."""
        captured["observation"] = actual_observation
        captured["grid"] = obstacle_grid
        captured["use_gpu"] = use_gpu
        return SimpleNamespace()

    monkeypatch.setattr(
        acceptance_backend,
        "build_nonproduction_observation_model",
        _build_observation,
    )
    monkeypatch.setattr(
        acceptance_backend,
        "continuous_kernel_from_observation_model",
        _build_kernel,
    )

    kernel = backend._kernel(grid)

    assert captured["payload"] == {
        "benign_observation_field": "retained",
    }
    assert captured["use_gpu"] is ACCEPTANCE_GEOMETRY_USE_GPU
    assert kernel.gpu_device == ACCEPTANCE_GEOMETRY_DEVICE
    assert kernel.gpu_dtype == ACCEPTANCE_GEOMETRY_DTYPE


def _config() -> SimpleNamespace:
    """Return a minimal exact native-fidelity application config."""
    return SimpleNamespace(
        engine_mode="external",
        physics_profile="balanced",
        thread_count=32,
        source_rate_model="detector_cps_1m",
        primary_emission_model="independent_gamma_lines",
        source_bias_mode="detector_cone",
        source_bias_cone_policy="detector_covering",
        source_bias_isotropic_fraction=1.0,
        detector_model=SimpleNamespace(coincidence_window_s=1.0e-6),
        detector_scoring_mode="incident_gamma_energy",
        secondary_transport_mode="full_transport",
        primary_sampling_fraction=1.0,
        target_sampled_primaries=None,
        accelerated_weighted_transport_enable=False,
        sample_detector_response=True,
        validation_entry_class_spectra=True,
        absorbing_transport_groups=("wall",),
    )


def _metadata(*, source_count: int) -> dict[str, object]:
    """Return exact postconditions emitted by a native unit-history run."""
    operator = synthetic_detector_green_operator()
    payload: dict[str, object] = {
        "backend": "geant4",
        "engine_mode": "external",
        "physics_profile": "balanced",
        "source_rate_model": "detector_cps_1m",
        "primary_emission_model": "independent_gamma_lines",
        "emission_model": "detector_equivalent_cone",
        "source_strength_field": "intensity_cps_1m",
        "intensity_cps_1m_definition": (
            "pre_dead_time_detector_pulse_rate_at_1m"
        ),
        "true_coincidence_summing": "disabled",
        "radioactive_decay_time_window": "disabled",
        "source_bias_mode": "detector_cone",
        "source_bias_cone_policy": "detector_covering",
        "detector_scoring_mode": "incident_gamma_energy",
        "secondary_transport_mode": "full_transport",
        "transport_history_mode": "full_unit_weight",
        "validation_entry_spectrum_space": ("pre_dead_time_raw_incident_gamma"),
        "validation_entry_spectrum_grouping": (
            "source_token_initial_gamma_line_entry_class"
        ),
        "absorbing_transport_groups": "wall",
        "geant4_version_number": 1132,
        "geant4_version_tag": GEANT4_VERSION_TAG,
        "reference_physics_list": "FTFP_BERT",
        "electromagnetic_physics_constructor": ("G4EmStandardPhysics_option4"),
        "production_cut_range_mm": 0.7,
        "source_bias_isotropic_fraction": 1.0,
        "source_bias_effective_cone_half_angle_deg_min": (
            0.1 if source_count else 0.0
        ),
        "source_bias_effective_cone_half_angle_deg_max": (
            0.1 if source_count else 0.0
        ),
        "detector_coincidence_window_s": 1.0e-6,
        "gamma_process_names": "GammaGeneralProc,Transportation",
        "gamma_em_subprocess_names": "Rayl,compt,conv,phot",
        "geant4_physics_contract_id": NATIVE_ACCEPTANCE_FIDELITY[
            "geant4_physics_contract_id"
        ],
        "geant4_physics_contract_sha256": NATIVE_ACCEPTANCE_FIDELITY[
            "geant4_physics_contract_sha256"
        ],
        "material_resolution_contract_id": NATIVE_ACCEPTANCE_FIDELITY[
            "material_resolution_contract_id"
        ],
        "process_count_compton": 0,
        "process_count_rayleigh": 0,
        "process_count_photoelectric": 0,
        "transport_process_counts": (
            "Rayl:0,Transportation:1,compt:0,phot:0" if source_count else "-"
        ),
        "requested_threads": 32,
        "primary_sampling_fraction": 1.0,
        "primary_history_weight": 1.0,
        "target_sampled_primaries": 0,
        "spectrum_bin_count": NATIVE_GEANT4_BIN_COUNT,
        "multithreaded_run_manager": True,
        "primary_sampling_budget_enabled": False,
        "history_thinning_enabled": False,
        "transport_tally_weighted": False,
        "weighted_transport": False,
        "theory_tvl_attenuation": False,
        "detector_response_applied_in_native": True,
        "validation_entry_class_spectra": True,
        "source_bias_weighted_transport": False,
        "line_intensities_normalized": True,
        "prompt_decay_cascade_transport": False,
        "delayed_decay_pulse_separation": False,
        "detector_response_sampling_contract_sha256": (operator.contract_hash_sha256),
        "detector_response_operator_binary_sha256": operator.binary_sha256,
        "detector_response_sampling_model": (
            "isotope_independent_full_detector_green_operator_v3"
        ),
        "detector_response_boundary_state": (
            "normalized_impact_parameter_at_detector_housing_entry_v1"
        ),
        "detector_response_conditioning": (
            "registered_pulse_subprobability_given_housing_incident_gamma_v1"
        ),
        "detector_cps_green_reference_normalization": (
            "catalog_branching_weighted_absolute_detection_efficiency_at_1m_v1"
        ),
        "detector_response_sampling_mode": DETECTOR_GREEN_SAMPLING_MODE,
        "detector_response_coincidence_semantics": (
            DETECTOR_GREEN_COINCIDENCE_SEMANTICS
        ),
        "detector_response_incident_entry_count": 0,
        "detector_response_registered_entry_count": 0,
        "detector_response_coincidence_pulse_count": 0,
        "detector_response_multi_entry_pulse_count": 0,
        "pre_dead_time_total_spectrum_counts": 0,
        "weighted_spectrum_sumw2": 0,
        "dead_time_observed_scale": 1.0,
        "all_sources_surface_bound": source_count > 0,
        "surface_emission_policy_sha256": (
            surface_emission_policy_sha256() if source_count else ""
        ),
        "surface_emission_epsilon_m": (
            SURFACE_EMISSION_EPSILON_M if source_count else 0.0
        ),
    }
    return payload


def test_geometry_batch_uses_one_batched_call_per_present_isotope() -> None:
    """All 64 pairs must not be evaluated through scalar pair/source loops."""
    model = GeometryConditionedSpectralModel.physics_only_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=synthetic_detector_green_operator(),
    )
    sources = (
        _Source("Cs-137", (1.0, 2.0, 3.0)),
        _Source("Co-60", (2.0, 3.0, 4.0)),
        _Source("Cs-137", (3.0, 4.0, 5.0)),
    )
    kernel = _FakeKernel()

    batch = _geometry_batch(
        kernel=kernel,
        model=model,
        detector_pose_xyz=(1.0, 1.0, 0.5),
        sources=sources,
    )

    assert len(kernel.calls) == 2
    assert {call[0] for call in kernel.calls} == {"Cs-137", "Co-60"}
    assert batch.unattenuated_vsl.shape == (64, 3, len(model.line_identity))
    assert batch.features_vslf.shape == (
        64,
        3,
        len(model.line_identity),
        len(model.transport_feature_order),
    )
    for source_index, source in enumerate(sources):
        wrong_lines = np.asarray(
            [line["isotope"] != source.isotope for line in model.line_identity]
        )
        assert np.all(batch.unattenuated_vsl[:, source_index, wrong_lines] == 0)


@pytest.mark.parametrize("source_count", (0, 1))
def test_native_fidelity_accepts_exact_background_and_surface_runs(
    source_count: int,
) -> None:
    """Background and source scenes have distinct strict surface provenance."""
    assert (
        _native_fidelity(
            _metadata(source_count=source_count),
            config=_config(),
            source_count=source_count,
            operator=synthetic_detector_green_operator(),
        )
        == NATIVE_ACCEPTANCE_FIDELITY
    )


def test_native_fidelity_rejects_truthy_integer_boolean() -> None:
    """Native postconditions must not accept integers as booleans."""
    metadata = _metadata(source_count=1)
    metadata["weighted_transport"] = 0

    with pytest.raises(RuntimeError, match="must be boolean"):
        _native_fidelity(
            metadata,
            config=_config(),
            source_count=1,
            operator=synthetic_detector_green_operator(),
        )


def test_native_fidelity_rejects_transport_processes_without_sources() -> None:
    """Background-only acquisition cannot conceal transported primaries."""
    metadata = _metadata(source_count=0)
    metadata["transport_process_counts"] = "Transportation:1"

    with pytest.raises(RuntimeError, match="Background-only"):
        _native_fidelity(
            metadata,
            config=_config(),
            source_count=0,
            operator=synthetic_detector_green_operator(),
        )


def test_native_fidelity_rejects_empty_processes_with_sources() -> None:
    """A source-bearing acquisition must prove native photon transport."""
    metadata = _metadata(source_count=1)
    metadata["transport_process_counts"] = "-"

    with pytest.raises(RuntimeError, match="must be nonempty"):
        _native_fidelity(
            metadata,
            config=_config(),
            source_count=1,
            operator=synthetic_detector_green_operator(),
        )


def test_native_fidelity_rejects_retired_response_sampling_mode() -> None:
    """Production acceptance must not adapt the retired response marker."""
    metadata = _metadata(source_count=1)
    metadata["detector_response_sampling_mode"] = (
        "multinomial_marking_with_nonparalyzable_event_time"
    )

    with pytest.raises(RuntimeError, match="marking is incompatible"):
        _native_fidelity(
            metadata,
            config=_config(),
            source_count=1,
            operator=synthetic_detector_green_operator(),
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("primary_emission_model", "geant4_radioactive_decay"),
        ("emission_model", "isotropic_parent_radioactive_decay"),
        ("true_coincidence_summing", "global_time_window_energy_deposit_sum"),
        ("source_bias_mode", "analog"),
        ("source_bias_cone_policy", "fixed_angle"),
        ("line_intensities_normalized", False),
        ("source_bias_isotropic_fraction", 0.5),
        ("detector_coincidence_window_s", 2.0e-6),
    ),
)
def test_native_fidelity_rejects_changed_catalog_emission_semantics(
    field: str,
    invalid: object,
) -> None:
    """Acceptance must bind independent catalog lines and detector timing."""
    metadata = _metadata(source_count=1)
    metadata[field] = invalid

    with pytest.raises(RuntimeError, match="metadata|fidelity"):
        _native_fidelity(
            metadata,
            config=_config(),
            source_count=1,
            operator=synthetic_detector_green_operator(),
        )


def test_native_fidelity_rejects_inconsistent_coincidence_counters() -> None:
    """One multi-entry pulse must account for at least one merged entry."""
    metadata = _metadata(source_count=1)
    metadata["detector_response_incident_entry_count"] = 2
    metadata["detector_response_registered_entry_count"] = 2
    metadata["detector_response_coincidence_pulse_count"] = 2
    metadata["detector_response_multi_entry_pulse_count"] = 1

    with pytest.raises(RuntimeError, match="counters are inconsistent"):
        _native_fidelity(
            metadata,
            config=_config(),
            source_count=1,
            operator=synthetic_detector_green_operator(),
        )


def test_native_metadata_token_matches_cpp_character_contract() -> None:
    """Metadata tokens must preserve isotope hyphens and sanitize separators."""
    assert sanitize_native_metadata_token("Cs-137") == "Cs-137"
    assert sanitize_native_metadata_token("Co 60,\t=x") == "Co_60___x"
    assert sanitize_native_metadata_token("") == "unknown"
    assert (
        native_source_line_token(
            source_index=2,
            isotope="Eu-154",
            energy_keV=123.4,
        )
        == "src2_Eu-154_e123p4"
    )


def _multi_source_validation_metadata(
    model: GeometryConditionedSpectralModel,
) -> dict[str, object]:
    """Return literal C++-style labels for two hyphenated isotopes."""
    sources = ("Cs-137", "Co-60")
    spectrum = ",".join(
        "1" if index == 10 else "0" for index in range(NATIVE_GEANT4_BIN_COUNT)
    )
    metadata: dict[str, object] = {
        "validation_only_background_analysis_spectrum": ",".join(
            "0" for _ in range(NATIVE_GEANT4_BIN_COUNT)
        )
    }
    for source_index, isotope in enumerate(sources):
        metadata[f"source_equivalent_counts_src{source_index}_{isotope}"] = 1.0
        for line in model.line_identity:
            if line["isotope"] != isotope:
                continue
            token = (
                f"src{source_index}_{isotope}_e{float(line['energy_keV']):.1f}"
            ).replace(".", "p")
            metadata[f"scheduled_incident_gamma_counts_{token}"] = 1.0
            metadata[f"validation_only_entry_spectrum_{token}_uncollided_primary"] = (
                spectrum
            )
    return metadata


def test_validation_labels_accept_literal_cpp_multi_source_tokens() -> None:
    """A real multi-source response must retain native isotope hyphens."""
    model = GeometryConditionedSpectralModel.physics_only_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=synthetic_detector_green_operator(),
    )
    sources = (
        PointSource("Cs-137", (1.0, 1.0, 1.0), 800_000.0),
        PointSource("Co-60", (2.0, 2.0, 2.0), 600_000.0),
    )

    labels = _validation_labels(
        _multi_source_validation_metadata(model),
        sources=sources,
        model=model,
    )

    totals = labels["entry_class_totals_by_source_line"]
    assert isinstance(totals, dict)
    assert "src0_Cs-137_e662p0" in totals
    assert "src1_Co-60_e1173p0" in totals
    assert all("Cs_137" not in token for token in totals)
    assert all(row["uncollided_primary"] == 1 for row in totals.values())


def test_validation_labels_require_every_scheduled_line_count_key() -> None:
    """Zero detector hits must not hide native source-line token drift."""
    model = GeometryConditionedSpectralModel.physics_only_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=synthetic_detector_green_operator(),
    )
    sources = (
        PointSource("Cs-137", (1.0, 1.0, 1.0), 800_000.0),
        PointSource("Co-60", (2.0, 2.0, 2.0), 600_000.0),
    )
    metadata = _multi_source_validation_metadata(model)
    missing_key = "scheduled_incident_gamma_counts_src0_Cs-137_e662p0"
    del metadata[missing_key]

    with pytest.raises(RuntimeError, match="every catalog line"):
        _validation_labels(metadata, sources=sources, model=model)


def test_validation_labels_reject_unexpected_line_count_token() -> None:
    """A Python-only underscore isotope token must fail before checkpointing."""
    model = GeometryConditionedSpectralModel.physics_only_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        detector_green_operator=synthetic_detector_green_operator(),
    )
    sources = (
        PointSource("Cs-137", (1.0, 1.0, 1.0), 800_000.0),
        PointSource("Co-60", (2.0, 2.0, 2.0), 600_000.0),
    )
    metadata = _multi_source_validation_metadata(model)
    metadata["scheduled_incident_gamma_counts_src0_Cs_137_e662p0"] = 1.0

    with pytest.raises(RuntimeError, match="unexpected source-resolved"):
        _validation_labels(metadata, sources=sources, model=model)


def test_signed_surface_scene_variants_use_exact_normal_offsets() -> None:
    """The native gate must probe anchor, air+epsilon, and solid-epsilon."""
    scene = (
        "SOURCE isotope=Cs-137 x=1 y=2 z=3 "
        "anchor_x=1 anchor_y=2 anchor_z=3 "
        "surface_normal_x=0 surface_normal_y=0 surface_normal_z=1\n"
    )

    exact = _mutated_surface_scene(scene, variant="exact_surface_anchor")
    air = _mutated_surface_scene(scene, variant="air_plus_epsilon")
    solid = _mutated_surface_scene(scene, variant="solid_minus_epsilon")

    assert "z=3 " in exact
    assert f"z={format(3.0 + SURFACE_EMISSION_EPSILON_M, '.17g')} " in air
    assert f"z={format(3.0 - SURFACE_EMISSION_EPSILON_M, '.17g')} " in solid


def test_boundary_evidence_ignores_stochastic_success_output() -> None:
    """Equivalent successful probes must have stable resumable evidence."""
    common = {
        "variant": "air_plus_epsilon",
        "scene_sha256": "a" * 64,
        "request_sha256": "b" * 64,
        "response_contract_valid": True,
        "native_executable_sha256": "c" * 64,
        "native_execution_environment_sha256": "d" * 64,
        "implementation_bundle_sha256": "e" * 64,
    }
    first = _boundary_probe_evidence_sha256(
        result=subprocess.CompletedProcess(
            args=("geant4",),
            returncode=0,
            stdout="sampled total=127\n",
            stderr="worker schedule A\n",
        ),
        **common,
    )
    second = _boundary_probe_evidence_sha256(
        result=subprocess.CompletedProcess(
            args=("geant4",),
            returncode=0,
            stdout="sampled total=143\n",
            stderr="worker schedule B\n",
        ),
        **common,
    )
    invalid = _boundary_probe_evidence_sha256(
        result=subprocess.CompletedProcess(
            args=("geant4",),
            returncode=0,
            stdout="sampled total=143\n",
            stderr="worker schedule B\n",
        ),
        **{**common, "response_contract_valid": False},
    )

    assert first == second
    assert invalid != first
