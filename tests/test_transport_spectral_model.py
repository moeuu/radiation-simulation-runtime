"""Tests for the shared geometry-conditioned full-spectrum model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy import stats

import spectrum.transport_spectral as transport_spectral
from runtime.experiment_profiles import (
    MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE,
    STANDARD_ACQUISITION_LIVE_TIME_S,
)
from spectrum.detector_green_operator import (
    DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
    DETECTOR_GREEN_SAMPLING_MODE,
    DetectorGreenOperator,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_DETECTOR_POSE_XYZ,
    ACCEPTANCE_GEOMETRY_DEVICE,
    ACCEPTANCE_GEOMETRY_DTYPE,
    ACCEPTANCE_GEOMETRY_USE_GPU,
    ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION,
    ACCEPTANCE_PASSAGE_WIDTH_M,
    ACCEPTANCE_PERTURBATION_MINIMUM_BEARING_ANGLE_RAD,
    ACCEPTANCE_PERTURBATION_MINIMUM_DISPLACEMENT_M,
    ACCEPTANCE_PERTURBATION_MINIMUM_LOG_RATE_SEPARATION,
    ACCEPTANCE_PERTURBATION_TANGENT_DIRECTIONS_UV,
    ACCEPTANCE_PERTURBATION_TANGENT_MAGNITUDES_M,
    ACCEPTANCE_ROOM_SIZE_XYZ,
    ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M,
    continuous_rate_scale_quadrature_for_half_width,
    DESIGNATED_VALIDATION_SCENE_SEEDS,
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    DETECTOR_IMPACT_PHASE_COUNT,
    GeometryConditionedSpectralModel,
    LowRankSpectralMeanCorrection,
    PhysicalComponentDiscrepancy,
    full_spectrum_acceptance_contract_payload,
    geometry_conditioned_model_from_runtime_config,
    nonparalyzable_count_log_probability_numpy,
    nonparalyzable_count_log_probability_torch,
    rate_scale_mixture_for_half_width,
    sample_nonparalyzable_counts_numpy,
    station_shared_gamma_poisson_count_log_increments_numpy,
    station_shared_gamma_poisson_count_log_increments_torch,
    with_catalog_independent_production_approval,
)
from tests.runtime_test_support import (
    approved_full_spectrum_model,
    runtime_config as approved_runtime_config,
)
from tests.green_test_support import write_synthetic_detector_green_artifact


@pytest.fixture(scope="module")
def _detector_green_manifest(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Publish one immutable synthetic Green artifact for this test module."""
    return write_synthetic_detector_green_artifact(
        tmp_path_factory.mktemp("detector-green-model") / "operator"
    )


@pytest.fixture(autouse=True)
def _use_explicit_detector_green(
    _detector_green_manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route unit models through one authenticated synthetic Green artifact."""
    monkeypatch.setattr(
        transport_spectral,
        "DEFAULT_DETECTOR_GREEN_OPERATOR_MANIFEST",
        _detector_green_manifest,
    )


def _valid_transport_features(values: np.ndarray) -> np.ndarray:
    """Return valid aggregate and detector-impact transport features."""
    base = np.asarray(values, dtype=np.float64).copy()
    if base.shape[-1] != 4:
        raise ValueError("Unit-test base transport features must have four columns.")
    base[..., 3] = np.maximum(base[..., 3], 1.25)
    base = np.concatenate(
        (
            base[..., :3],
            base[..., 2:3],
            base[..., 3:4],
        ),
        axis=-1,
    )
    impact = np.full(
        base.shape[:-1] + (DETECTOR_IMPACT_PHASE_COUNT,),
        1.0 / DETECTOR_IMPACT_PHASE_COUNT,
        dtype=np.float64,
    )
    return np.concatenate((base, impact), axis=-1)


@pytest.mark.parametrize(
    "kernel_environment",
    (
        {"OPENBLAS_CORETYPE": "Haswell"},
        {"OPENBLAS_CORETYPE": "SkylakeX"},
        {"NPY_ENABLE_CPU_FEATURES": "X86_V2"},
    ),
    ids=("openblas-haswell", "openblas-skylakex", "numpy-x86-v2"),
)
def test_profile_contract_hash_is_portable_across_cpu_kernels(
    kernel_environment: dict[str, str],
) -> None:
    """Derived-array identity must not depend on the runner CPU kernel."""
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        pytest.skip("OpenBLAS x86 kernel selection is unavailable.")
    repository_root = Path(__file__).resolve().parents[1]
    model_path = (
        repository_root
        / "configs/geant4/models/profiles/unconditioned_eu154_physics_only.json"
    )
    expected_hash = json.loads(model_path.read_text(encoding="utf-8"))[
        "contract_hash_sha256"
    ]
    child_code = "\n".join(
        (
            "import json",
            "from pathlib import Path",
            (
                "from spectrum.transport_spectral import "
                "GeometryConditionedSpectralModel"
            ),
            (
                "from spectrum.full_spectrum_acceptance_runner import "
                "canonical_detector_green_operator"
            ),
            f"path = Path({str(model_path)!r})",
            "payload = json.loads(path.read_text(encoding='utf-8'))",
            (
                "model = GeometryConditionedSpectralModel.from_manifest_payload("
                "payload, detector_green_operator="
                "canonical_detector_green_operator())"
            ),
            "print(model.contract_hash_sha256)",
        )
    )
    child_environment = os.environ.copy()
    child_environment.update(kernel_environment)

    completed = subprocess.run(
        (sys.executable, "-c", child_code),
        cwd=repository_root,
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected_hash


def test_portable_derived_array_digest_ignores_only_roundoff_noise() -> None:
    """Portable identity must retain material changes and reject nonfinite data."""
    baseline = np.asarray((0.25, -0.0), dtype=np.float64)
    roundoff = baseline.copy()
    roundoff[0] = np.nextafter(roundoff[0], np.inf)
    positive_zero = np.asarray((0.25, 0.0), dtype=np.float64)
    material_change = baseline.copy()
    material_change[0] += 2.0e-12

    baseline_digest = transport_spectral._portable_derived_array_digest(baseline)

    assert (
        transport_spectral._portable_derived_array_digest(roundoff) == baseline_digest
    )
    assert (
        transport_spectral._portable_derived_array_digest(positive_zero)
        == baseline_digest
    )
    assert (
        transport_spectral._portable_derived_array_digest(material_change)
        != baseline_digest
    )
    assert (
        transport_spectral._portable_derived_array_digest(baseline.reshape((1, 2)))
        != baseline_digest
    )
    with pytest.raises(ValueError, match="finite"):
        transport_spectral._portable_derived_array_digest(
            np.asarray((np.nan,), dtype=np.float64)
        )


def test_continuous_rate_scale_quadrature_integrates_uniform_moments() -> None:
    """Nine-node quadrature must represent a continuous mean-one uniform."""
    width = 0.20
    nodes, weights = continuous_rate_scale_quadrature_for_half_width(width)
    node_array = np.asarray(nodes, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)

    assert node_array.size == 9
    assert np.sum(weight_array) == pytest.approx(1.0)
    assert np.sum(weight_array * node_array) == pytest.approx(1.0)
    assert np.sum(weight_array * np.square(node_array)) == pytest.approx(
        1.0 + width**2 / 3.0
    )


def test_shared_gamma_count_prefixes_match_closed_form_and_torch() -> None:
    """Shared-Gamma count increments must telescope to the exact joint PMF."""
    torch = pytest.importorskip("torch")
    observed = np.asarray([[2.0, 3.0]], dtype=np.float64)
    expected = np.asarray([[[4.0, 5.0]]], dtype=np.float64)
    concentration = 7.0

    increments = station_shared_gamma_poisson_count_log_increments_numpy(
        observed,
        expected,
        concentration=concentration,
    )
    total_count = 5.0
    total_mean = 9.0
    expected_log = (
        transport_spectral.special.gammaln(concentration + total_count)
        - transport_spectral.special.gammaln(concentration)
        + concentration * np.log(concentration)
        - (concentration + total_count) * np.log(concentration + total_mean)
        + 2.0 * np.log(4.0)
        + 3.0 * np.log(5.0)
        - transport_spectral.special.gammaln(3.0)
        - transport_spectral.special.gammaln(4.0)
    )
    torch_increments = (
        station_shared_gamma_poisson_count_log_increments_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(expected, dtype=torch.float64),
            concentration=concentration,
        )
        .detach()
        .cpu()
        .numpy()
    )

    assert float(np.sum(increments)) == pytest.approx(expected_log)
    assert np.allclose(increments, torch_increments, rtol=1.0e-12, atol=1.0e-12)


def _model(
    *,
    dead_time_tau_s: float = 5.813e-9,
    background_rate_cps: float = 5.0,
) -> GeometryConditionedSpectralModel:
    """Return an unapproved physical model for deterministic unit tests."""
    return GeometryConditionedSpectralModel.nonproduction_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=float(dead_time_tau_s),
        background_rate_cps=float(background_rate_cps),
    )


def test_model_response_cache_requires_exact_authenticated_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated model reductions reuse only the exact operator and energy axis."""
    first = _model()
    expected_response = first.response_operator_br.copy()

    def fail_uncached_request(*args: object, **kwargs: object) -> object:
        """Make any uncached full-axis Green reduction observable."""
        del args, kwargs
        raise AssertionError("uncached model response request")

    monkeypatch.setattr(
        DetectorGreenOperator,
        "marginal_absolute_response_for_axis",
        fail_uncached_request,
    )
    monkeypatch.setattr(
        DetectorGreenOperator,
        "marginal_response_for_axis",
        fail_uncached_request,
    )
    repeated = _model()

    np.testing.assert_array_equal(repeated.response_operator_br, expected_response)
    assert repeated.response_operator_br is not first.response_operator_br
    changed_axis = first.energy_axis_keV
    changed_axis[1] = np.nextafter(changed_axis[1], changed_axis[2])
    with pytest.raises(AssertionError, match="uncached model response request"):
        transport_spectral._detector_green_model_response_bundle(
            first.detector_green_operator,
            changed_axis,
        )


def _subset_likelihood_branch_model(
    branch: str,
) -> GeometryConditionedSpectralModel:
    """Return a compact model exercising one subset-likelihood branch."""
    keywords: dict[str, object] = {}
    if branch == "rate_nodes":
        keywords.update(
            rate_scale_nodes_j=(0.8, 1.0, 1.2),
            rate_scale_weights_j=(0.25, 0.5, 0.25),
        )
    elif branch in ("station_shared_count", "view_independent_count"):
        keywords.update(
            count_discrepancy_concentration=40.0,
            count_discrepancy_scope=(
                "station_shared"
                if branch == "station_shared_count"
                else "view_independent"
            ),
            mark_concentration_source=80.0,
        )
    elif branch == "component_tree":
        keywords["physical_component_discrepancy"] = PhysicalComponentDiscrepancy(
            count_uncollided_concentration=60.0,
            count_scatter_concentration=12.0,
            mark_uncollided_concentration=90.0,
            mark_scatter_concentration=15.0,
            mark_background_group_concentration=15.0,
            mark_background_within_concentration=90.0,
        )
    elif branch != "fixed_renewal":
        raise ValueError(f"Unknown subset likelihood branch: {branch!r}.")
    return GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137",),
        dead_time_tau_s=1.0e-5,
        background_rate_cps=1.0,
        **keywords,
    )


def _subset_likelihood_inputs(
    model: GeometryConditionedSpectralModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic action/sample/state/view subset test tensors."""
    rng = np.random.default_rng(20260824)
    action_count = 2
    sample_count = 2
    state_count = 3
    view_count = 5
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.1,
        4.0,
        (action_count, state_count, view_count, 1, line_count),
    )
    uncollided = total * rng.uniform(0.2, 1.0, total.shape)
    features = _valid_transport_features(rng.uniform(0.0, 1.0, total.shape + (4,)))
    live = np.linspace(1.0, 3.0, view_count, dtype=np.float64)
    predictive_mean = model.predict_mean_numpy(
        total[:, 0],
        uncollided[:, 0],
        features[:, 0],
        live,
    )
    observed = np.repeat(
        np.rint(predictive_mean)[:, np.newaxis, :, :],
        sample_count,
        axis=1,
    ).astype(np.float64)
    observed[:, 1, :, 0] += 1.0
    return observed, total, uncollided, features, live


def _runtime_ready_candidate() -> GeometryConditionedSpectralModel:
    """Return a physics-only model without application approval."""
    approved = approved_full_spectrum_model()
    return GeometryConditionedSpectralModel.nonproduction_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=approved.dead_time_tau_s,
        background_rate_cps=approved.background_rate_cps,
        physical_component_discrepancy=(approved.physical_component_discrepancy),
        additive_scatter_response=approved.additive_scatter_response,
        detector_green_operator=approved.detector_green_operator,
    )


def _physical_component_candidate() -> GeometryConditionedSpectralModel:
    """Return an offline component-latent numerical test model."""
    component = PhysicalComponentDiscrepancy(
        count_uncollided_concentration=100_000.0,
        count_scatter_concentration=300.0,
        mark_uncollided_concentration=100_000.0,
        mark_scatter_concentration=300.0,
        mark_background_group_concentration=300.0,
        mark_background_within_concentration=100_000.0,
    )
    return GeometryConditionedSpectralModel.nonproduction_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        physical_component_discrepancy=component,
    )


def test_physical_component_concentration_propagates_detector_information() -> None:
    """Count, tree, and leaf uncertainty must retain physical components."""
    model = _physical_component_candidate()
    total = np.ones((2, 1, 1, len(model.line_identity)), dtype=np.float64)
    uncollided = total.copy()
    uncollided[1] *= 0.0
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))

    count = model._component_count_concentration_numpy(
        total,
        uncollided,
        features,
    )
    component_means = np.full(
        (2, 1, 1, 3, model.energy_axis_keV.size),
        0.01,
        dtype=np.float64,
    )
    component_means[0, ..., 0, :] = 1.0
    component_means[1, ..., 1, :] = 1.0
    tree, leaf = model._component_tree_mark_concentrations_numpy(
        total,
        uncollided,
        component_means,
    )

    assert model.runtime_ready is False
    assert model.discrepancy_training_ready is False
    assert count[0, 0] > count[1, 0] > 0.0
    assert np.median(tree[0]) > np.median(tree[1]) > 0.0
    assert np.all(leaf > 0.0)
    assert not np.allclose(leaf[0], leaf[1], rtol=0.0, atol=1.0e-12)
    assert count[0, 0] < 100_000.0
    assert np.all(np.isfinite(tree))
    assert np.all(np.isfinite(leaf))


def test_physical_component_concentrations_match_torch() -> None:
    """CPU and Torch component concentration paths must be identical."""
    torch = pytest.importorskip("torch")
    model = _physical_component_candidate()
    rng = np.random.default_rng(71)
    total = rng.uniform(0.1, 4.0, size=(3, 2, 2, len(model.line_identity)))
    uncollided = total * rng.uniform(0.0, 1.0, size=total.shape)
    features = _valid_transport_features(rng.uniform(0.0, 1.0, size=total.shape + (4,)))

    count_numpy = model._component_count_concentration_numpy(
        total,
        uncollided,
        features,
    )
    component_means = rng.uniform(
        0.01,
        2.0,
        size=(
            3,
            1,
            2,
            3,
            model.energy_axis_keV.size,
        ),
    )
    tree_numpy, leaf_numpy = model._component_tree_mark_concentrations_numpy(
        total,
        uncollided,
        component_means,
    )
    count_torch = (
        model._component_count_concentration_torch(
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    tree_torch, leaf_torch = model._component_tree_mark_concentrations_torch(
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(component_means, dtype=torch.float64),
    )

    assert np.allclose(count_numpy, count_torch, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        tree_numpy,
        tree_torch.detach().cpu().numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        leaf_numpy,
        leaf_torch.detach().cpu().numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_fused_torch_count_concentration_matches_standalone_transport() -> None:
    """Fused hierarchical transport must preserve count concentrations."""
    torch = pytest.importorskip("torch")
    model = approved_full_spectrum_model()
    rng = np.random.default_rng(73191)
    line_count = len(model.line_identity)
    total = rng.uniform(0.1, 4.0, size=(3, 2, 2, line_count))
    uncollided = total * rng.uniform(0.1, 1.0, size=total.shape)
    features = _valid_transport_features(
        rng.uniform(0.0, 1.0, size=total.shape + (4,))
    )
    live_times = np.asarray([7.0, 20.0], dtype=np.float64)
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        total_torch = torch.as_tensor(total, device=device, dtype=torch.float64)
        uncollided_torch = torch.as_tensor(
            uncollided,
            device=device,
            dtype=torch.float64,
        )
        features_torch = torch.as_tensor(
            features,
            device=device,
            dtype=torch.float64,
        )
        live_torch = torch.as_tensor(
            live_times,
            device=device,
            dtype=torch.float64,
        )
        *_, fused = model._pre_dead_time_mean_torch(
            total_torch,
            uncollided_torch,
            features_torch,
            live_torch,
            return_physical_components=True,
            return_component_count_concentration=True,
        )
        standalone = model._component_count_concentration_torch(
            total_torch,
            uncollided_torch,
            features_torch,
        )

        assert torch.allclose(fused, standalone, rtol=1.0e-12, atol=1.0e-12)


def test_physics_only_green_uncertainty_round_trip_and_matches_torch() -> None:
    """Physics-only Green-aware marks must match on CPU and Torch."""
    torch = pytest.importorskip("torch")
    payload = PhysicalComponentDiscrepancy.physics_only_budget().to_payload()
    component = PhysicalComponentDiscrepancy.from_payload(payload)

    assert payload["schema_version"] == 5
    assert component.mark_latent_model == "component_dirichlet_tree_hierarchical"
    model = approved_full_spectrum_model()
    rng = np.random.default_rng(7391)
    line_count = len(model.line_identity)
    total = rng.uniform(0.1, 3.0, size=(3, 2, 2, line_count))
    uncollided = total * rng.uniform(0.2, 1.0, size=total.shape)
    features = _valid_transport_features(rng.uniform(0.0, 1.0, size=total.shape + (4,)))
    live = np.asarray([2.0, 3.0], dtype=np.float64)
    observed = model.sample_predictive_numpy(
        total[:1],
        uncollided[:1],
        features[:1],
        live,
        sample_count=1,
        rng=rng,
    )[0, 0].astype(np.float64)
    numpy_log = model.log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_log = (
        model.log_likelihood_torch(
            torch.as_tensor(observed, device=device, dtype=torch.float64),
            torch.as_tensor(total, device=device, dtype=torch.float64),
            torch.as_tensor(uncollided, device=device, dtype=torch.float64),
            torch.as_tensor(features, device=device, dtype=torch.float64),
            torch.as_tensor(live, device=device, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(numpy_log, torch_log, rtol=0.0, atol=2.0e-10)
    assert model.manifest_payload()["mark_model"] == (
        "component_background_source_dirichlet_tree_hierarchical"
    )


def test_canonical_log_gamma_matches_numpy_torch_and_cuda() -> None:
    """Every likelihood backend must execute one float64 log-gamma formula."""
    torch = pytest.importorskip("torch")
    values = np.asarray(
        [0.125, 0.5, 1.0, 2.0, 17.0, 1_001.0, 835_001.0],
        dtype=np.float64,
    )
    expected = transport_spectral.canonical_log_gamma_numpy(values)
    np.testing.assert_allclose(
        expected,
        transport_spectral.special.gammaln(values),
        rtol=0.0,
        atol=5.0e-9,
    )
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        actual = transport_spectral.canonical_log_gamma_torch(
            torch.as_tensor(values, device=device, dtype=torch.float64)
        )
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected,
            rtol=0.0,
            atol=5.0e-10,
        )


def test_physics_only_uncertainty_budget_cannot_be_scene_tuned() -> None:
    """Physics-only provenance must reject altered uncertainty values."""
    with pytest.raises(ValueError, match="immutable"):
        PhysicalComponentDiscrepancy(
            count_uncollided_concentration=2501.0,
            count_scatter_concentration=4.0,
            mark_uncollided_concentration=9999.0,
            mark_scatter_concentration=23.999999999999996,
            mark_background_group_concentration=23.999999999999996,
            mark_background_within_concentration=9999.0,
            provenance="physics_only_uncertainty_budget_v1",
        )


def test_component_tree_absent_co_false_positive_calibration() -> None:
    """Null Cs observations must calibrate weak absent-Co alternatives."""
    model = approved_full_spectrum_model()
    view_count = 4
    line_count = len(model.line_identity)
    isotopes = np.asarray(
        [str(row["isotope"]) for row in model.line_identity],
        dtype=object,
    )
    cs_lines = isotopes == "Cs-137"
    co_lines = isotopes == "Co-60"
    attenuation = np.linspace(0.6, 1.0, view_count, dtype=np.float64)
    null_total = np.zeros((view_count, 1, line_count), dtype=np.float64)
    null_total[:, 0, cs_lines] = 10.0 * attenuation[:, np.newaxis]
    null_uncollided = 0.8 * null_total
    base_features = _valid_transport_features(
        np.zeros(null_total.shape + (4,), dtype=np.float64)
    )
    live_times = np.full(
        view_count,
        STANDARD_ACQUISITION_LIVE_TIME_S,
        dtype=np.float64,
    )
    rng = np.random.default_rng(80315)
    observed = model.sample_predictive_numpy(
        null_total[np.newaxis, ...],
        null_uncollided[np.newaxis, ...],
        base_features[np.newaxis, ...],
        live_times,
        sample_count=256,
        rng=rng,
    )[0]
    candidates = []
    for relative_co_rate in (0.0, 0.002, 0.005, 0.01, 0.02):
        candidate = null_total.copy()
        candidate[:, 0, co_lines] = relative_co_rate * 10.0 * attenuation[:, np.newaxis]
        candidates.append(candidate)
    total = np.stack(candidates, axis=0)
    uncollided = 0.8 * total
    features = np.broadcast_to(
        base_features,
        total.shape + (base_features.shape[-1],),
    ).copy()
    log_likelihood = model.cross_log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
        action_chunk_size=1,
        sample_chunk_size=64,
        state_chunk_size=len(candidates),
    )
    null_log = np.log(0.95) + log_likelihood[:, 0]
    alternative_log = (
        np.log(0.05)
        + transport_spectral.special.logsumexp(log_likelihood[:, 1:], axis=1)
        - np.log(float(len(candidates) - 1))
    )
    posterior_positive = np.exp(
        alternative_log - np.logaddexp(null_log, alternative_log)
    )
    false_positive_rate = float(np.mean(posterior_positive > 0.5))

    diagnostic_index = int(np.argmax(posterior_positive))
    decomposition = model.decompose_log_likelihood_numpy(
        observed[diagnostic_index],
        total,
        uncollided,
        features,
        live_times,
    )

    assert false_positive_rate <= 0.05
    np.testing.assert_allclose(
        decomposition.total_log_likelihood_n,
        log_likelihood[diagnostic_index],
        rtol=0.0,
        atol=1.0e-8,
    )
    assert decomposition.total_count_nv.shape == (len(candidates), view_count)
    assert decomposition.background_mark_nv.shape == (len(candidates), view_count)
    assert decomposition.source_mark_nv.shape == (len(candidates), view_count)


def test_physics_only_background_bypasses_source_component_gamma_limit() -> None:
    """Source-free likelihoods must use exact renewal counts on CPU and Torch."""
    torch = pytest.importorskip("torch")
    model = approved_full_spectrum_model()
    view_count = 5
    line_count = len(model.line_identity)
    total = np.zeros((1, view_count, 0, line_count), dtype=np.float64)
    uncollided = total.copy()
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    live = np.full(view_count, 20.0, dtype=np.float64)
    observed = model.sample_predictive_numpy(
        total,
        uncollided,
        features,
        live,
        sample_count=1,
        rng=np.random.default_rng(8701),
    )[0, 0].astype(np.float64)

    numpy_log = model.log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
    )
    torch_log = (
        model.log_likelihood_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(numpy_log, torch_log, rtol=0.0, atol=2.0e-9)


def _training_ready_mean_correction(
    model: GeometryConditionedSpectralModel,
) -> LowRankSpectralMeanCorrection:
    """Return one synthetic authenticated correction for equivalence tests."""
    descriptor_count = 2 + len(model.line_identity) + 4
    basis = np.zeros((1, model.energy_axis_keV.size), dtype=np.float64)
    basis[0, :2] = (0.2, -0.1)
    regression = np.zeros((descriptor_count + 1, 1), dtype=np.float64)
    regression[0, 0] = 0.5
    training = {
        "schema_version": 1,
        "training_policy": ("fixed_quota_loso_training_only_low_rank_log_mean_v1"),
        "training_scene_seeds": [2026072701, 2026072702],
        "scenario_ids": ["multi_isotope_superposition"],
        "pair_ids_by_scene": {
            "2026072701": [0],
            "2026072702": [0],
        },
        "artifact_sha256_by_scene": {
            "2026072701": "1" * 64,
            "2026072702": "2" * 64,
        },
        "rank_grid": [1],
        "ridge_lambda_grid": [1.0],
        "selected_rank": 1,
        "selected_ridge_lambda": 1.0,
        "selection_objective": (
            "leave_one_scene_out_target_probability_weighted_log_mse"
        ),
        "selected_validation_score": 0.1,
        "selection_completed": True,
        "holdout_artifacts_consumed": False,
    }
    return LowRankSpectralMeanCorrection(
        descriptor_order=tuple(
            f"descriptor_{index}" for index in range(descriptor_count)
        ),
        descriptor_center_d=np.zeros(descriptor_count),
        descriptor_scale_d=np.ones(descriptor_count),
        regression_qk=regression,
        basis_kb=basis,
        maximum_abs_log_correction=2.0,
        training_manifest=training,
    )


def test_total_below_uncollided_fails_closed_numpy_and_torch() -> None:
    """An impossible incident-count decomposition must never be normalized."""
    model = _model(dead_time_tau_s=0.0, background_rate_cps=0.0)
    line_count = len(model.line_identity)
    total = np.zeros((2, 1, 3, line_count), dtype=np.float64)
    total[0, 0, 0, 0] = 7.0
    total[0, 0, 1, 4] = 11.0
    total[1, 0, 2, 8] = 13.0
    uncollided = 3.0 * total
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    live_times = np.array([30.0], dtype=np.float64)
    with pytest.raises(ValueError, match="cannot exceed total"):
        model._pre_dead_time_mean_numpy(
            total,
            uncollided,
            features,
            live_times,
        )
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="cannot exceed total"):
        model._pre_dead_time_mean_torch(
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )


def test_low_rank_mean_correction_is_rejected_by_schema_four() -> None:
    """Scene-fitted spectral corrections cannot enter runtime schema four."""
    approved = approved_full_spectrum_model()
    payload = json.loads(json.dumps(approved.manifest_payload()))
    payload["low_rank_spectral_mean_correction"] = {"schema_version": 1}

    with pytest.raises(ValueError, match="low-rank"):
        GeometryConditionedSpectralModel.from_manifest_payload(
            payload,
            detector_green_operator=approved.detector_green_operator,
        )


def test_spectrum_mean_and_cross_likelihood_match_torch() -> None:
    """CPU and Torch must agree for batched source-resolved spectra."""
    torch = pytest.importorskip("torch")
    model = _model()
    rng = np.random.default_rng(42)
    particle_count = 4
    view_count = 3
    source_count = 5
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.0,
        100.0,
        (particle_count, view_count, source_count, line_count),
    )
    uncollided = total * rng.uniform(0.25, 1.0, total.shape)
    features = _valid_transport_features(rng.uniform(0.0, 2.0, total.shape + (4,)))
    live_times = np.array([30.0, 20.0, 10.0], dtype=np.float64)
    mean_numpy = model.predict_mean_numpy(
        total,
        uncollided,
        features,
        live_times,
    )
    mean_torch = (
        model.predict_mean_torch(
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(mean_numpy, mean_torch, rtol=2e-11, atol=2e-9)
    observations = np.rint(mean_numpy[:2]).astype(np.float64)
    log_numpy = model.cross_log_likelihood_numpy(
        observations,
        total,
        uncollided,
        features,
        live_times,
    )
    log_torch = (
        model.cross_log_likelihood_torch(
            torch.as_tensor(observations, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert log_numpy.shape == (2, particle_count)
    assert np.allclose(log_numpy, log_torch, rtol=1e-10, atol=1e-7)


@pytest.mark.parametrize(
    "branch",
    (
        "fixed_renewal",
        "rate_nodes",
        "station_shared_count",
        "view_independent_count",
        "component_tree",
    ),
)
def test_numpy_subset_cache_matches_direct_arbitrary_view_likelihood(
    branch: str,
) -> None:
    """Every likelihood branch must match direct non-prefix subset calls."""
    model = _subset_likelihood_branch_model(branch)
    observed, total, uncollided, features, live = _subset_likelihood_inputs(model)
    prepared = model.prepare_subset_cross_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
        action_chunk_size=1,
        sample_chunk_size=1,
        state_chunk_size=2,
        view_chunk_size=2,
    )
    subsets = np.asarray(
        (
            ((4, 1, 3), (2, 0, 4)),
            ((3, 0, 1), (1, 4, 2)),
        ),
        dtype=np.int64,
    )

    cached = prepared.evaluate(subsets)

    assert prepared.action_count == 2
    assert prepared.sample_count == 2
    assert prepared.state_count == 3
    assert prepared.view_count == prepared.pair_count == 5
    assert cached.shape == (2, 2, 2, 3)
    for action_index in range(2):
        for candidate_index in range(2):
            selected = subsets[action_index, candidate_index]
            direct = model.cross_log_likelihood_numpy(
                np.take(observed[action_index], selected, axis=1),
                np.take(total[action_index], selected, axis=1),
                np.take(uncollided[action_index], selected, axis=1),
                np.take(features[action_index], selected, axis=1),
                live[selected],
            )
            assert np.allclose(
                cached[action_index, candidate_index],
                direct,
                rtol=2.0e-12,
                atol=5.0e-9,
            )
    reversed_subsets = subsets[..., ::-1]
    assert np.allclose(
        cached,
        prepared.evaluate(reversed_subsets),
        rtol=2.0e-12,
        atol=5.0e-9,
    )
    full = model.cross_log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
    )
    assert np.allclose(prepared.full(), full, rtol=2.0e-12, atol=5.0e-9)


def test_shared_gamma_subset_cache_ignores_every_unselected_view() -> None:
    """Unselected views must not alter shared-Gamma sufficient statistics."""
    model = _subset_likelihood_branch_model("station_shared_count")
    observed, total, uncollided, features, live = _subset_likelihood_inputs(model)
    selected = np.asarray(((4, 1, 3),), dtype=np.int64)
    baseline = model.prepare_subset_cross_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
    ).evaluate(selected)
    changed_observed = observed.copy()
    changed_total = total.copy()
    changed_uncollided = uncollided.copy()
    changed_features = features.copy()
    changed_observed[:, :, (0, 2), 0] += 10_000.0
    changed_total[:, :, (0, 2)] *= 9.0
    changed_uncollided[:, :, (0, 2)] *= 0.1
    changed_features[:, :, (0, 2), ..., :4] += 7.0
    changed = model.prepare_subset_cross_likelihood_numpy(
        changed_observed,
        changed_total,
        changed_uncollided,
        changed_features,
        live,
    ).evaluate(selected)

    assert np.array_equal(baseline, changed)


def test_subset_cache_rejects_invalid_or_duplicate_view_indices() -> None:
    """Subset caches must reject noninteger, duplicate, and invalid views."""
    model = _subset_likelihood_branch_model("fixed_renewal")
    observed, total, uncollided, features, live = _subset_likelihood_inputs(model)
    prepared = model.prepare_subset_cross_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
    )

    with pytest.raises(ValueError, match="integer"):
        prepared.evaluate(np.asarray(((0.0, 1.0),)))
    with pytest.raises(ValueError, match="duplicate"):
        prepared.evaluate(np.asarray(((0, 0),), dtype=np.int64))
    with pytest.raises(ValueError, match="range"):
        prepared.evaluate(np.asarray(((0, 5),), dtype=np.int64))
    with pytest.raises(ValueError, match="shaped"):
        prepared.evaluate(np.asarray((0, 1), dtype=np.int64))


@pytest.mark.parametrize(
    "branch",
    (
        "fixed_renewal",
        "rate_nodes",
        "station_shared_count",
        "view_independent_count",
        "component_tree",
    ),
)
def test_torch_subset_cache_matches_numpy_and_stays_device_resident(
    branch: str,
) -> None:
    """The standard batched Torch cache must equal NumPy on every branch."""
    torch = pytest.importorskip("torch")
    model = _subset_likelihood_branch_model(branch)
    observed, total, uncollided, features, live = _subset_likelihood_inputs(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prepared_numpy = model.prepare_subset_cross_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
    )
    prepared_torch = model.prepare_subset_cross_likelihood_torch(
        torch.as_tensor(observed, device=device, dtype=torch.float64),
        torch.as_tensor(total, device=device, dtype=torch.float64),
        torch.as_tensor(uncollided, device=device, dtype=torch.float64),
        torch.as_tensor(features, device=device, dtype=torch.float64),
        torch.as_tensor(live, device=device, dtype=torch.float64),
        action_chunk_size=1,
        sample_chunk_size=1,
        state_chunk_size=2,
        view_chunk_size=2,
    )
    subsets = np.asarray(
        (
            ((4, 1, 3), (2, 0, 4)),
            ((3, 0, 1), (1, 4, 2)),
        ),
        dtype=np.int64,
    )

    expected = prepared_numpy.evaluate(subsets)
    actual = prepared_torch.evaluate(
        torch.as_tensor(subsets, device=device, dtype=torch.int64)
    )

    assert tuple(actual.shape) == (2, 2, 2, 3)
    assert actual.device == device
    assert actual.dtype == torch.float64
    assert np.allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=1.0e-10,
        atol=1.0e-7,
    )


@pytest.mark.parametrize("scope", ["station_shared", "view_independent"])
def test_shared_gamma_full_likelihood_matches_torch(
    scope: str,
) -> None:
    """The complete-station shared-Gamma likelihood must match Torch."""
    torch = pytest.importorskip("torch")
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        count_discrepancy_concentration=75.0,
        count_discrepancy_scope=scope,
        mark_concentration_source=3000.0,
        mark_concentration_multi_isotope=10000.0,
    )
    rng = np.random.default_rng(2718)
    line_count = len(model.line_identity)
    total = rng.uniform(0.0, 30.0, (3, 2, 2, line_count))
    uncollided = total * rng.uniform(0.25, 1.0, total.shape)
    features = _valid_transport_features(rng.uniform(0.0, 2.0, total.shape + (4,)))
    live = np.asarray([30.0, 15.0], dtype=np.float64)
    observed = np.rint(
        model.predict_mean_numpy(total[:1], uncollided[:1], features[:1], live)[0]
    )

    numpy_full = model.log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live,
    )
    torch_full = (
        model.log_likelihood_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(numpy_full, torch_full, rtol=1e-10, atol=1e-7)


def test_complete_station_likelihood_uses_all_views_and_matches_torch() -> None:
    """The complete-station target must use all views and match Torch."""
    torch = pytest.importorskip("torch")
    model = _model()
    rng = np.random.default_rng(20260730)
    particle_count = 5
    view_count = 4
    source_count = 3
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.0,
        20.0,
        (particle_count, view_count, source_count, line_count),
    )
    uncollided = total * rng.uniform(0.1, 1.0, total.shape)
    features = _valid_transport_features(rng.uniform(0.0, 2.0, total.shape + (4,)))
    live_times = np.asarray([30.0, 20.0, 10.0, 5.0], dtype=np.float64)
    observed = np.rint(
        model.predict_mean_numpy(
            total[:1],
            uncollided[:1],
            features[:1],
            live_times,
        )[0]
    ).astype(np.float64)

    full_numpy = model.log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
    )
    full_torch = (
        model.log_likelihood_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert full_numpy.shape == (particle_count,)
    assert np.allclose(full_numpy, full_torch, rtol=1e-10, atol=1e-7)


def test_zero_total_marks_are_exactly_neutral_numpy_torch_and_cross() -> None:
    """A zero-count spectrum must contribute only its renewal total term."""
    torch = pytest.importorskip("torch")
    hierarchical = GeometryConditionedSpectralModel.nonproduction_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=0.0,
        rate_scale_nodes_j=(0.9, 1.0, 1.1),
        rate_scale_weights_j=(0.25, 0.5, 0.25),
        mark_concentration_source=100.0,
    )
    plain = GeometryConditionedSpectralModel.nonproduction_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=0.0,
        rate_scale_nodes_j=(0.9, 1.0, 1.1),
        rate_scale_weights_j=(0.25, 0.5, 0.25),
        mark_concentration_source=None,
    )
    line_count = len(hierarchical.line_identity)
    total = np.zeros((2, 3, 2, 1, line_count), dtype=np.float64)
    total[..., 3] = np.asarray(
        (5.0, 25.0, 100.0),
        dtype=np.float64,
    )[np.newaxis, :, np.newaxis, np.newaxis]
    uncollided = total.copy()
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    features[..., 3] = 2.0
    observed = np.zeros((2, 4, 2, 851), dtype=np.float64)
    live = np.asarray((0.01, 30.0), dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        numpy_cross = hierarchical.cross_log_likelihood_numpy(
            observed,
            total,
            uncollided,
            features,
            live,
            action_chunk_size=2,
            sample_chunk_size=4,
            state_chunk_size=3,
        )
        numpy_plain = plain.cross_log_likelihood_numpy(
            observed,
            total,
            uncollided,
            features,
            live,
            action_chunk_size=2,
            sample_chunk_size=4,
            state_chunk_size=3,
        )
    numpy_single = hierarchical.log_likelihood_numpy(
        observed[0, 0],
        total[0],
        uncollided[0],
        features[0],
        live,
    )
    torch_cross = (
        hierarchical.cross_log_likelihood_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live, dtype=torch.float64),
            action_chunk_size=2,
            sample_chunk_size=4,
            state_chunk_size=3,
        )
        .detach()
        .cpu()
        .numpy()
    )
    torch_single = (
        hierarchical.log_likelihood_torch(
            torch.as_tensor(observed[0, 0], dtype=torch.float64),
            torch.as_tensor(total[0], dtype=torch.float64),
            torch.as_tensor(uncollided[0], dtype=torch.float64),
            torch.as_tensor(features[0], dtype=torch.float64),
            torch.as_tensor(live, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.isfinite(numpy_cross))
    assert np.array_equal(numpy_cross, numpy_plain)
    assert np.allclose(numpy_cross, torch_cross, rtol=0.0, atol=1.0e-10)
    assert np.allclose(numpy_single, numpy_cross[0, 0], rtol=0.0, atol=1.0e-10)
    assert np.allclose(numpy_single, torch_single, rtol=0.0, atol=1.0e-10)


def test_background_and_source_share_one_pre_dead_time_total() -> None:
    """Background is added once before one dead-time transform of total rate."""
    tau = 2.0e-3
    background_rate = 7.0
    model = _model(
        dead_time_tau_s=tau,
        background_rate_cps=background_rate,
    )
    line_count = len(model.line_identity)
    live_times = np.asarray([3.0, 5.0], dtype=np.float64)
    source_rates = np.asarray([20.0, 40.0], dtype=np.float64)
    total = np.zeros((1, 2, 1, line_count), dtype=np.float64)
    total[0, :, 0, 0] = source_rates
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))

    predicted = model.predict_mean_numpy(
        total,
        total,
        features,
        live_times,
    )
    predicted_total = np.sum(predicted[0], axis=-1)
    source_mean, background_mean = model._pre_dead_time_mean_numpy(
        total,
        total,
        features,
        live_times,
        return_components=True,
    )
    np.testing.assert_allclose(
        np.sum(background_mean[0], axis=-1),
        background_rate * live_times,
    )
    pre_dead_time_counts = np.sum(source_mean[0] + background_mean[0], axis=-1)
    expected_total = pre_dead_time_counts / (
        1.0 + pre_dead_time_counts / live_times * tau
    )

    np.testing.assert_allclose(
        predicted_total,
        expected_total,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    manifest = model.manifest_payload()
    assert manifest["dead_time_application_count"] == 1
    assert manifest["background_semantics"] == (
        "independent_pre_dead_time_pulse_rate_added_once"
    )


def test_cross_likelihood_chunking_matches_unchunked_numpy_and_torch() -> None:
    """Action, predictive-sample, and state chunks must preserve exact scores."""
    torch = pytest.importorskip("torch")
    nodes, weights = rate_scale_mixture_for_half_width(0.10)
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=4.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=300.0,
    )
    rng = np.random.default_rng(9284)
    action_count = 2
    state_count = 11
    sample_count = 5
    view_count = 2
    source_count = 2
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.0,
        10.0,
        (
            action_count,
            state_count,
            view_count,
            source_count,
            line_count,
        ),
    )
    uncollided = total * rng.uniform(0.2, 1.0, total.shape)
    features = _valid_transport_features(rng.uniform(0.0, 1.0, total.shape + (4,)))
    live_times = np.array([2.0, 3.0], dtype=np.float64)
    observations = model.sample_predictive_numpy(
        total[:, :sample_count],
        uncollided[:, :sample_count],
        features[:, :sample_count],
        live_times,
        sample_count=1,
        rng=rng,
    )[:, :, 0].astype(np.float64)
    oracle_numpy = model._cross_log_likelihood_numpy_unchunked(
        observations,
        total,
        uncollided,
        features,
        live_times,
    )
    chunked_numpy = model.cross_log_likelihood_numpy(
        observations,
        total,
        uncollided,
        features,
        live_times,
        action_chunk_size=1,
        sample_chunk_size=2,
        state_chunk_size=3,
    )
    np.testing.assert_allclose(
        chunked_numpy,
        oracle_numpy,
        rtol=2.0e-12,
        atol=2.0e-9,
    )
    oracle_torch = model._cross_log_likelihood_torch_unchunked(
        torch.as_tensor(observations, dtype=torch.float64),
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(features, dtype=torch.float64),
        torch.as_tensor(live_times, dtype=torch.float64),
    )
    chunked_torch = model.cross_log_likelihood_torch(
        torch.as_tensor(observations, dtype=torch.float64),
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(features, dtype=torch.float64),
        torch.as_tensor(live_times, dtype=torch.float64),
        action_chunk_size=1,
        sample_chunk_size=2,
        state_chunk_size=3,
    )
    torch_total = torch.as_tensor(total, dtype=torch.float64)
    prepared = model.prepare_cross_observation_torch(
        torch.as_tensor(observations, dtype=torch.float64),
        reference=torch_total,
    )
    prepared_torch = model.cross_log_likelihood_torch(
        prepared.observed_asvb,
        torch_total,
        torch.as_tensor(uncollided, dtype=torch.float64),
        torch.as_tensor(features, dtype=torch.float64),
        torch.as_tensor(live_times, dtype=torch.float64),
        action_chunk_size=1,
        sample_chunk_size=2,
        state_chunk_size=3,
        prepared_observation=prepared,
    )
    np.testing.assert_allclose(
        chunked_torch.detach().cpu().numpy(),
        oracle_torch.detach().cpu().numpy(),
        rtol=2.0e-12,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        chunked_torch.detach().cpu().numpy(),
        chunked_numpy,
        rtol=1.0e-10,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        prepared_torch.detach().cpu().numpy(),
        chunked_torch.detach().cpu().numpy(),
    )


def test_cuda_cross_likelihood_autotune_matches_explicit_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empirical CUDA state tuning must preserve every exact score."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    model = _model()
    monkeypatch.setattr(
        model,
        "estimate_cross_likelihood_working_set_bytes",
        lambda **_: 0,
    )
    state_count = 256 + 512 + 1024
    line_count = len(model.line_identity)
    total = torch.full(
        (1, state_count, 1, 1, line_count),
        0.25,
        device="cuda",
        dtype=torch.float64,
    )
    uncollided = 0.75 * total
    features = torch.zeros(
        tuple(total.shape) + (5 + DETECTOR_IMPACT_PHASE_COUNT,),
        device="cuda",
        dtype=torch.float64,
    )
    features[..., 4] = 1.25
    features[..., 5:] = 1.0 / DETECTOR_IMPACT_PHASE_COUNT
    observed = torch.zeros(
        (1, 1, 1, int(model.energy_axis_keV.size)),
        device="cuda",
        dtype=torch.float64,
    )
    live_times = torch.ones(1, device="cuda", dtype=torch.float64)
    tuned = model.cross_log_likelihood_torch(
        observed,
        total,
        uncollided,
        features,
        live_times,
    )
    diagnostics = dict(model.last_torch_cross_chunk_diagnostics)
    explicit = model.cross_log_likelihood_torch(
        observed,
        total,
        uncollided,
        features,
        live_times,
        state_chunk_size=256,
    )

    np.testing.assert_array_equal(
        tuned.detach().cpu().numpy(),
        explicit.detach().cpu().numpy(),
    )
    assert diagnostics["mode"] == "empirical_cuda_autotune"
    assert [trial["state_chunk_size"] for trial in diagnostics["trials"]] == [
        256,
        512,
        1024,
    ]
    assert diagnostics["selected_state_chunk_size"] in {256, 512, 1024}


def test_cross_likelihood_working_set_estimate_tracks_chunk_sizes() -> None:
    """The memory estimate must include and respond to all three chunk axes."""
    model = _model()
    unchunked = model.estimate_cross_likelihood_working_set_bytes(
        num_actions=8,
        num_samples=50,
        num_particles=512,
        num_isotopes=15,
        num_views=8,
        action_chunk_size=8,
        sample_chunk_size=50,
        state_chunk_size=512,
    )
    bounded = model.estimate_cross_likelihood_working_set_bytes(
        num_actions=8,
        num_samples=50,
        num_particles=512,
        num_isotopes=15,
        num_views=8,
        action_chunk_size=1,
        sample_chunk_size=10,
        state_chunk_size=8,
    )
    assert bounded < unchunked
    assert bounded > 0


def test_subset_working_set_estimate_tracks_poses_and_candidates() -> None:
    """The arbitrary-subset estimate must cover both resident and search axes."""
    model = _model()
    base = model.estimate_subset_cross_likelihood_working_set_bytes(
        num_actions=1,
        num_samples=50,
        num_particles=512,
        num_source_slots=15,
        num_views=64,
        num_candidates=64,
        subset_size=8,
        action_chunk_size=1,
        sample_chunk_size=10,
        state_chunk_size=128,
        view_chunk_size=8,
    )
    more_candidates = model.estimate_subset_cross_likelihood_working_set_bytes(
        num_actions=1,
        num_samples=50,
        num_particles=512,
        num_source_slots=15,
        num_views=64,
        num_candidates=448,
        subset_size=8,
        action_chunk_size=1,
        sample_chunk_size=10,
        state_chunk_size=128,
        view_chunk_size=8,
    )
    more_source_slots = model.estimate_subset_cross_likelihood_working_set_bytes(
        num_actions=1,
        num_samples=50,
        num_particles=512,
        num_source_slots=30,
        num_views=64,
        num_candidates=64,
        subset_size=8,
        action_chunk_size=1,
        sample_chunk_size=10,
        state_chunk_size=128,
        view_chunk_size=8,
    )
    more_poses = model.estimate_subset_cross_likelihood_working_set_bytes(
        num_actions=4,
        num_samples=50,
        num_particles=512,
        num_source_slots=15,
        num_views=64,
        num_candidates=448,
        subset_size=8,
        action_chunk_size=1,
        sample_chunk_size=10,
        state_chunk_size=128,
        view_chunk_size=8,
    )

    assert 0 < base < more_candidates < more_poses
    assert base < more_source_slots


def test_birth_proposal_score_is_finite_target_only_and_matches_torch() -> None:
    """Proposal matched-filter scores must not invoke the target likelihood."""
    torch = pytest.importorskip("torch")
    model = _model()
    line_count = len(model.line_identity)
    mask = np.asarray(
        [item["isotope"] == "Cs-137" for item in model.line_identity],
        dtype=np.bool_,
    )
    candidate_count = 6
    total = np.zeros((candidate_count, 2, 1, line_count), dtype=np.float64)
    total[:, :, 0, mask] = np.linspace(10.0, 60.0, candidate_count)[:, None, None]
    uncollided = 0.8 * total
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    live_times = np.array([30.0, 30.0], dtype=np.float64)
    observed = np.rint(
        model.predict_mean_numpy(
            total[3:4],
            uncollided[3:4],
            features[3:4],
            live_times,
        )[0]
    )
    numpy_scores = model.birth_proposal_log_scores_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
        target_line_mask_l=mask,
    )
    background = model.predict_mean_numpy(
        np.zeros((1, 2, 1, line_count), dtype=np.float64),
        np.zeros((1, 2, 1, line_count), dtype=np.float64),
        _valid_transport_features(np.zeros((1, 2, 1, line_count, 4), dtype=np.float64)),
        live_times,
    )[0]
    reference = background + 0.15 * observed
    numpy_reference_scores = model.birth_proposal_log_scores_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
        target_line_mask_l=mask,
        reference_mean_vb=reference,
    )
    torch_scores = (
        model.birth_proposal_log_scores_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
            target_line_mask_l=torch.as_tensor(mask),
        )
        .detach()
        .cpu()
        .numpy()
    )
    torch_reference_scores = (
        model.birth_proposal_log_scores_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
            target_line_mask_l=torch.as_tensor(mask),
            reference_mean_vb=torch.as_tensor(
                reference,
                dtype=torch.float64,
            ),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert numpy_scores.shape == (candidate_count,)
    assert np.all(np.isfinite(numpy_scores))
    assert np.allclose(numpy_scores, torch_scores, rtol=1e-10, atol=1e-8)
    assert np.allclose(
        numpy_reference_scores,
        torch_reference_scores,
        rtol=1e-10,
        atol=1e-8,
    )
    assert not np.allclose(numpy_reference_scores, numpy_scores)
    assert int(np.argmax(numpy_scores)) in (2, 3, 4)


def test_birth_proposal_score_is_independent_of_candidate_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Larger runtime batches must preserve exact proposal-only scores."""
    model = _model()
    line_count = len(model.line_identity)
    mask = np.asarray(
        [item["isotope"] == "Cs-137" for item in model.line_identity],
        dtype=np.bool_,
    )
    total = np.zeros((5, 2, 1, line_count), dtype=np.float64)
    total[:, :, 0, mask] = np.linspace(5.0, 45.0, 5)[:, None, None]
    uncollided = 0.75 * total
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    live_times = np.asarray([30.0, 30.0], dtype=np.float64)
    observed = np.rint(
        model.predict_mean_numpy(
            total[2:3],
            uncollided[2:3],
            features[2:3],
            live_times,
        )[0]
    )

    monkeypatch.setattr(
        transport_spectral,
        "BIRTH_PROPOSAL_WORKING_SET_BYTES",
        1,
    )
    scalar_chunks = model.birth_proposal_log_scores_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
        target_line_mask_l=mask,
    )
    monkeypatch.setattr(
        transport_spectral,
        "BIRTH_PROPOSAL_WORKING_SET_BYTES",
        1 << 40,
    )
    one_batch = model.birth_proposal_log_scores_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
        target_line_mask_l=mask,
    )

    np.testing.assert_allclose(
        one_batch,
        scalar_chunks,
        rtol=0.0,
        atol=2.0e-11,
    )


def test_torch_production_paths_reject_float32() -> None:
    """PF/DSS spectral calculations must fail closed outside float64."""
    torch = pytest.importorskip("torch")
    model = _model()
    line_count = len(model.line_identity)
    total = torch.zeros((1, 1, 1, line_count), dtype=torch.float32)
    features = torch.zeros(
        total.shape + (5 + DETECTOR_IMPACT_PHASE_COUNT,),
        dtype=torch.float32,
    )
    with pytest.raises(TypeError, match="float64"):
        model.predict_mean_torch(
            total,
            total,
            features,
            torch.ones(1, dtype=torch.float32),
        )
    with pytest.raises(TypeError, match="float64"):
        nonparalyzable_count_log_probability_torch(
            torch.zeros(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            dead_time_tau_s=1e-4,
        )


def test_renewal_likelihood_has_exact_poisson_zero_dead_time_limit() -> None:
    """At tau=0 the CPU and Torch laws must be the exact Poisson law."""
    torch = pytest.importorskip("torch")
    counts = np.arange(0, 31, dtype=np.float64)
    rates = np.full(counts.shape, 7.25, dtype=np.float64)
    live_times = np.full(counts.shape, 2.5, dtype=np.float64)
    expected = stats.poisson.logpmf(counts, rates * live_times)
    actual = nonparalyzable_count_log_probability_numpy(
        counts,
        rates,
        live_times,
        dead_time_tau_s=0.0,
    )
    actual_torch = (
        nonparalyzable_count_log_probability_torch(
            torch.as_tensor(counts, dtype=torch.float64),
            torch.as_tensor(rates, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
            dead_time_tau_s=0.0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.allclose(actual_torch, expected, rtol=1e-13, atol=1e-13)


def test_renewal_probability_normalizes_and_matches_torch() -> None:
    """The finite-support renewal law must normalize on CPU and Torch."""
    torch = pytest.importorskip("torch")
    rate = 20.0
    live_time = 1.0
    tau = 0.02
    counts = np.arange(0, int(np.floor(live_time / tau)) + 2)
    rates = np.full(counts.shape, rate, dtype=np.float64)
    times = np.full(counts.shape, live_time, dtype=np.float64)
    log_probability = nonparalyzable_count_log_probability_numpy(
        counts,
        rates,
        times,
        dead_time_tau_s=tau,
    )
    torch_probability = (
        nonparalyzable_count_log_probability_torch(
            torch.as_tensor(counts, dtype=torch.float64),
            torch.as_tensor(rates, dtype=torch.float64),
            torch.as_tensor(times, dtype=torch.float64),
            dead_time_tau_s=tau,
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.isclose(np.sum(np.exp(log_probability)), 1.0, atol=2e-13)
    assert np.allclose(log_probability, torch_probability, atol=2e-11)


def _high_precision_renewal_log_probability(
    count: int,
    rate_cps: float,
    live_time_s: float,
    dead_time_tau_s: float,
) -> float:
    """Return an independent positive-term renewal oracle using mpmath."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):
        m = mp.mpf(int(count))
        rate = mp.mpf(str(rate_cps))
        live_time = mp.mpf(str(live_time_s))
        tau = mp.mpf(str(dead_time_tau_s))
        first = rate * max(live_time - (m - 1) * tau, mp.mpf(0))
        second = rate * max(live_time - m * tau, mp.mpf(0))
        width = first - second
        log_terms = []

        def _log_gamma_density(argument: object) -> object:
            """Return the shape-m unit-scale gamma log density."""
            if argument == 0:
                return mp.mpf(0) if m == 1 else mp.ninf
            return (m - 1) * mp.log(argument) - argument - mp.loggamma(m)

        if width > 0:
            mode = min(max(m - 1, second), first)
            log_scale = _log_gamma_density(mode)
            scaled_interval = width * mp.quad(
                lambda unit: mp.exp(
                    _log_gamma_density(second + width * unit) - log_scale
                ),
                [0, 1],
            )
            log_terms.append(log_scale + mp.log(scaled_interval))
        if second > 0:
            log_terms.append(m * mp.log(second) - second - mp.loggamma(m + 1))
        if not log_terms:
            return -np.inf
        maximum = max(log_terms)
        return float(
            maximum + mp.log(sum(mp.exp(value - maximum) for value in log_terms))
        )


def test_renewal_extreme_tails_match_positive_high_precision_oracle() -> None:
    """Physically possible high/low count tails must remain finite and exact."""
    cases = (
        (9_000, 1_500.0, 2.0, 5.813e-9),
        (9_001, 1_500.0, 2.0, 5.813e-9),
        (1_000, 5_000.0, 1.0, 5.813e-9),
        (500, 5.0, 1.0, 1.0e-3),
        (501, 5.0, 1.0, 1.0e-3),
    )
    first_group_actual = nonparalyzable_count_log_probability_numpy(
        np.asarray([case[0] for case in cases[:3]], dtype=np.float64),
        np.asarray([case[1] for case in cases[:3]], dtype=np.float64),
        np.asarray([case[2] for case in cases[:3]], dtype=np.float64),
        dead_time_tau_s=cases[0][3],
    )
    first_group_expected = np.asarray(
        [_high_precision_renewal_log_probability(*case) for case in cases[:3]],
        dtype=np.float64,
    )
    second_group_actual = nonparalyzable_count_log_probability_numpy(
        np.asarray([case[0] for case in cases[3:]], dtype=np.float64),
        np.asarray([case[1] for case in cases[3:]], dtype=np.float64),
        np.asarray([case[2] for case in cases[3:]], dtype=np.float64),
        dead_time_tau_s=cases[3][3],
    )
    second_group_expected = np.asarray(
        [_high_precision_renewal_log_probability(*case) for case in cases[3:]],
        dtype=np.float64,
    )
    saturation_case = (100_000, 1.0, 1.0, 1.0e-5)
    saturation_actual = nonparalyzable_count_log_probability_numpy(
        np.asarray([saturation_case[0]], dtype=np.float64),
        np.asarray([saturation_case[1]], dtype=np.float64),
        np.asarray([saturation_case[2]], dtype=np.float64),
        dead_time_tau_s=saturation_case[3],
    )[0]
    saturation_expected = _high_precision_renewal_log_probability(*saturation_case)
    assert np.all(np.isfinite(first_group_actual))
    assert np.all(np.isfinite(second_group_actual))
    assert np.allclose(
        first_group_actual,
        first_group_expected,
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.allclose(
        second_group_actual,
        second_group_expected,
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.isclose(
        saturation_actual,
        saturation_expected,
        rtol=0.0,
        atol=1.0e-8,
    )


def test_renewal_extreme_tail_numpy_torch_cpu_cuda_equivalence() -> None:
    """Tail recovery must agree on NumPy, Torch CPU, and available CUDA."""
    torch = pytest.importorskip("torch")
    counts = np.asarray([500.0, 501.0, 9_000.0, 9_001.0])
    rates = np.asarray([5.0, 5.0, 1_500.0, 1_500.0])
    live_times = np.asarray([1.0, 1.0, 2.0, 2.0])
    tau = 5.813e-9
    numpy_log = nonparalyzable_count_log_probability_numpy(
        counts,
        rates,
        live_times,
        dead_time_tau_s=tau,
    )

    def _torch_result(
        device: str,
        input_counts: np.ndarray,
        input_rates: np.ndarray,
        input_live_times: np.ndarray,
        input_tau: float,
    ) -> np.ndarray:
        """Evaluate the renewal tail on one Torch device."""
        return (
            nonparalyzable_count_log_probability_torch(
                torch.as_tensor(
                    input_counts,
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    input_rates,
                    dtype=torch.float64,
                    device=device,
                ),
                torch.as_tensor(
                    input_live_times,
                    dtype=torch.float64,
                    device=device,
                ),
                dead_time_tau_s=input_tau,
            )
            .detach()
            .cpu()
            .numpy()
        )

    assert np.allclose(
        _torch_result("cpu", counts, rates, live_times, tau),
        numpy_log,
        rtol=0.0,
        atol=2.0e-10,
    )
    saturation_counts = np.asarray([100_000.0])
    saturation_rates = np.asarray([1.0])
    saturation_live_times = np.asarray([1.0])
    saturation_tau = 1.0e-5
    saturation_numpy = nonparalyzable_count_log_probability_numpy(
        saturation_counts,
        saturation_rates,
        saturation_live_times,
        dead_time_tau_s=saturation_tau,
    )
    assert np.allclose(
        _torch_result(
            "cpu",
            saturation_counts,
            saturation_rates,
            saturation_live_times,
            saturation_tau,
        ),
        saturation_numpy,
        rtol=0.0,
        atol=1.0e-8,
    )
    if torch.cuda.is_available():
        assert np.allclose(
            _torch_result("cuda", counts, rates, live_times, tau),
            numpy_log,
            rtol=0.0,
            atol=2.0e-10,
        )
        assert np.allclose(
            _torch_result(
                "cuda",
                saturation_counts,
                saturation_rates,
                saturation_live_times,
                saturation_tau,
            ),
            saturation_numpy,
            rtol=0.0,
            atol=1.0e-8,
        )


def test_renewal_runtime_scale_numpy_torch_cpu_cuda_equivalence() -> None:
    """Production-scale central counts must not depend on long tail series."""
    torch = pytest.importorskip("torch")
    counts = np.asarray(
        [29_826_600.0, 44_610_000.0, 59_310_000.0],
        dtype=np.float64,
    )
    rates = np.asarray(
        [1_000_000.0, 1_500_000.0, 2_000_000.0],
        dtype=np.float64,
    )
    live_times = np.full(counts.shape, 30.0, dtype=np.float64)
    tau = 5.813e-9
    expected = nonparalyzable_count_log_probability_numpy(
        counts,
        rates,
        live_times,
        dead_time_tau_s=tau,
    )

    for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
        actual = (
            nonparalyzable_count_log_probability_torch(
                torch.as_tensor(counts, dtype=torch.float64, device=device),
                torch.as_tensor(rates, dtype=torch.float64, device=device),
                torch.as_tensor(
                    live_times,
                    dtype=torch.float64,
                    device=device,
                ),
                dead_time_tau_s=tau,
            )
            .detach()
            .cpu()
            .numpy()
        )
        assert np.all(np.isfinite(actual))
        assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-8)


def test_renewal_positive_decomposition_handles_both_count_law_tails() -> None:
    """Stable positive terms must handle both count-law tails consistently."""
    torch = pytest.importorskip("torch")
    counts = np.asarray([100_000.0, 1_000.0], dtype=np.float64)
    rates = np.asarray([1.0, 10_000.0], dtype=np.float64)
    live_times = np.ones(2, dtype=np.float64)
    taus = (1.0e-5, 5.813e-9)
    for index, tau in enumerate(taus):
        numpy_value = nonparalyzable_count_log_probability_numpy(
            counts[index : index + 1],
            rates[index : index + 1],
            live_times[index : index + 1],
            dead_time_tau_s=tau,
        )
        torch_value = (
            nonparalyzable_count_log_probability_torch(
                torch.as_tensor(
                    counts[index : index + 1],
                    dtype=torch.float64,
                ),
                torch.as_tensor(
                    rates[index : index + 1],
                    dtype=torch.float64,
                ),
                torch.as_tensor(
                    live_times[index : index + 1],
                    dtype=torch.float64,
                ),
                dead_time_tau_s=tau,
            )
            .detach()
            .cpu()
            .numpy()
        )
        assert np.all(np.isfinite(numpy_value))
        np.testing.assert_allclose(
            torch_value,
            numpy_value,
            rtol=0.0,
            atol=1.0e-8,
        )


def test_vectorized_renewal_sampler_matches_exact_mean_and_fano() -> None:
    """Inverse-CDF samples must reproduce exact renewal mean and variance."""
    rate = 45.0
    live_time = 1.0
    tau = 0.008
    support = np.arange(0, int(np.floor(live_time / tau)) + 2)
    log_probability = nonparalyzable_count_log_probability_numpy(
        support,
        np.full(support.shape, rate, dtype=np.float64),
        np.full(support.shape, live_time, dtype=np.float64),
        dead_time_tau_s=tau,
    )
    probability = np.exp(log_probability)
    exact_mean = float(np.sum(support * probability))
    exact_variance = float(np.sum(np.square(support - exact_mean) * probability))
    samples = sample_nonparalyzable_counts_numpy(
        np.array([rate], dtype=np.float64),
        np.array([live_time], dtype=np.float64),
        dead_time_tau_s=tau,
        sample_count=40_000,
        rng=np.random.default_rng(20260727),
    )[0]
    sampled_mean = float(np.mean(samples))
    sampled_variance = float(np.var(samples))
    assert abs(sampled_mean - exact_mean) < 0.04
    assert abs(sampled_variance / sampled_mean - exact_variance / exact_mean) < 0.015


def test_renewal_sampler_handles_zero_rate_and_near_saturation() -> None:
    """Inverse-CDF bracketing must cover zero and almost saturated regimes."""
    samples = sample_nonparalyzable_counts_numpy(
        np.array([0.0, 1.0e6], dtype=np.float64),
        np.array([1.0, 0.01], dtype=np.float64),
        dead_time_tau_s=1.0e-4,
        sample_count=1_000,
        rng=np.random.default_rng(12),
    )
    assert np.all(samples[0] == 0)
    assert np.all(samples[1] <= 101)
    assert float(np.mean(samples[1])) > 95.0
    poisson_samples = sample_nonparalyzable_counts_numpy(
        np.array([1.0e4], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        dead_time_tau_s=0.0,
        sample_count=100,
        rng=np.random.default_rng(13),
    )
    assert poisson_samples.shape == (1, 100)


def test_predictive_sampler_preserves_renewal_total_and_mark_shape() -> None:
    """Predictive spectra must contain exact integer conditional marks."""
    model = _model(dead_time_tau_s=1.0e-4, background_rate_cps=0.0)
    line_count = len(model.line_identity)
    total = np.zeros((2, 1, 1, line_count), dtype=np.float64)
    total[:, 0, 0, 0] = np.array([500.0, 800.0])
    uncollided = total.copy()
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    samples = model.sample_predictive_numpy(
        total,
        uncollided,
        features,
        np.array([1.0], dtype=np.float64),
        sample_count=2_000,
        rng=np.random.default_rng(987),
    )
    assert samples.shape == (2, 2_000, 1, 851)
    assert np.all(samples >= 0.0)
    assert np.all(samples == np.floor(samples))
    total_samples = np.sum(samples, axis=-1)
    assert np.all(total_samples <= np.array([500.0, 800.0])[:, None, None] * 2)


def test_untrained_model_fails_runtime_and_production_gates() -> None:
    """A model without the physics-only transport contract fails closed."""
    model = _model()
    assert model.runtime_ready is False
    assert model.production_ready is False
    with pytest.raises(RuntimeError, match="physics-only"):
        model.require_runtime_ready()
    with pytest.raises(RuntimeError, match="all-64 validation"):
        model.require_production_ready()
    assert model.manifest_payload()["runtime_ready"] is False
    assert model.manifest_payload()["production_ready"] is False


def test_physics_only_candidate_is_runtime_ready_but_not_approved() -> None:
    """Physics readiness and application approval remain separate gates."""
    candidate = _runtime_ready_candidate()

    assert candidate.runtime_ready is True
    assert candidate.production_ready is False
    candidate.require_runtime_ready()
    with pytest.raises(RuntimeError, match="all-64"):
        candidate.require_production_ready()

    payload = candidate.manifest_payload()
    assert payload["runtime_ready"] is True
    assert payload["production_ready"] is False
    reconstructed = GeometryConditionedSpectralModel.from_manifest_payload(
        payload,
        detector_green_operator=candidate.detector_green_operator,
    )
    assert reconstructed.runtime_ready is True
    assert reconstructed.production_ready is False
    assert reconstructed.contract_hash_sha256 == candidate.contract_hash_sha256


def test_physics_only_candidate_needs_no_scene_fitted_terms() -> None:
    """Runtime readiness must use no empirical scene-fit correction."""
    model = _runtime_ready_candidate()

    assert model.exact_physical_statistics_ready is True
    assert model.discrepancy_training_ready is False
    assert model.discrepancy_training_manifest is None
    assert model.low_rank_spectral_mean_correction is None
    assert model.runtime_ready is True
    assert model.production_ready is False
    reconstructed = GeometryConditionedSpectralModel.from_manifest_payload(
        model.manifest_payload(),
        detector_green_operator=model.detector_green_operator,
    )
    assert reconstructed.exact_physical_statistics_ready is True
    assert reconstructed.runtime_ready is True


def test_physics_only_prediction_cannot_enter_retired_scatter_order_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-v6 mean and uncertainty must use only detector-cone scatter."""
    model = _runtime_ready_candidate()
    line_count = len(model.line_identity)
    total = np.zeros((1, 1, 1, line_count), dtype=np.float64)
    uncollided = np.zeros_like(total)
    total[..., 0] = 2.0
    uncollided[..., 0] = 1.0
    base_features = np.broadcast_to(
        np.asarray((0.2, 0.3, 0.1, 2.0), dtype=np.float64),
        total.shape + (4,),
    )
    features = _valid_transport_features(base_features)

    def _retired_path(*_args: object, **_kwargs: object) -> np.ndarray:
        """Fail if a production prediction reaches optical-depth orders."""
        raise AssertionError("retired scatter-order path was reached")

    monkeypatch.setattr(
        type(model),
        "_interaction_order_weights_numpy",
        _retired_path,
    )

    mean = model.predict_mean_numpy(
        total,
        uncollided,
        features,
        np.asarray((20.0,), dtype=np.float64),
    )
    concentration = model._component_count_concentration_numpy(
        total,
        uncollided,
        features,
    )

    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(concentration))


def test_detector_cone_scatter_distance_extrapolation_fails_closed() -> None:
    """An active scatter source outside the authenticated grid must abort."""
    model = _runtime_ready_candidate()
    line_count = len(model.line_identity)
    total = np.zeros((1, 1, 1, line_count), dtype=np.float64)
    uncollided = np.zeros_like(total)
    total[..., 0] = 2.0
    uncollided[..., 0] = 1.0
    base_features = np.broadcast_to(
        np.asarray((0.2, 0.3, 0.1, 201.0), dtype=np.float64),
        total.shape + (4,),
    )
    features = _valid_transport_features(base_features)

    with pytest.raises(ValueError, match="Scatter-to-detector distance"):
        model.predict_mean_numpy(
            total,
            uncollided,
            features,
            np.asarray((20.0,), dtype=np.float64),
        )


def test_retired_component_candidate_is_rejected_before_runtime() -> None:
    """A stale learned asset cannot cross the current training boundary."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "geant4"
        / "models"
        / "geometry_conditioned_full_spectrum_ral_eu154_component.json"
    )
    with pytest.raises(ValueError, match="schema-v7"):
        GeometryConditionedSpectralModel.from_manifest_payload(
            json.loads(path.read_text(encoding="utf-8")),
            detector_green_operator=(
                approved_full_spectrum_model().detector_green_operator
            ),
        )


def test_acceptance_contract_file_is_predeclared_and_hash_stable() -> None:
    """The canonical JSON must exactly match the in-code validation contract."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "validation"
        / "full_spectrum_acceptance.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == full_spectrum_acceptance_contract_payload()
    assert len(FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256) == 64


def test_acceptance_contract_owns_fresh_environment_and_compute_inputs() -> None:
    """Acceptance must share dwell and explicitly freeze every backend input."""
    payload = full_spectrum_acceptance_contract_payload()

    assert "schema_version" not in payload
    assert payload["dwell_time_s"] == STANDARD_ACQUISITION_LIVE_TIME_S
    assert payload["dwell_time_s"] == (
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.live_time_s
    )
    assert payload["surface_boundary_probe"] == {
        "schema_version": 3,
        "dwell_time_s": 1.0e-2,
        "surface_emission_epsilon_m": 1.0e-6,
        "native_position_variants": [
            "exact_surface_anchor",
            "air_plus_epsilon",
            "solid_minus_epsilon",
        ],
        "require_nonempty_transport_process_counts": True,
    }
    assert DESIGNATED_VALIDATION_SCENE_SEEDS == (
        3646699724,
        4620708915,
        5193545889,
        7235536511,
        7325752837,
    )
    assert payload["native_process_counter_policy"] == {
        "background_only": "exact_empty_counter_map",
        "source_present": "nonempty_positive_counter_map",
    }
    assert payload["detector_cps_green_reference_efficiency_policy"] == (
        "recomputed_from_authenticated_catalog_and_operator_strict_tolerance_v1"
    )
    assert payload["detector_response_event_policy"] == {
        "primary_emission_model": "independent_gamma_lines",
        "source_bias_cone_policy": "detector_covering",
        "catalog_line_semantics": ("positive_intensity_lines_normalized_per_isotope"),
        "prompt_decay_cascade_transport": False,
        "true_coincidence_summing": "disabled",
        "coincidence_window_s": 1.0e-6,
        "sampling_mode": DETECTOR_GREEN_SAMPLING_MODE,
        "coincidence_semantics": DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
        "counter_semantics": (
            "incident_ge_registered_ge_pulses_and_merged_entry_excess_"
            "ge_multi_entry_pulses_v1"
        ),
    }
    assert set(DESIGNATED_VALIDATION_SCENE_SEEDS).isdisjoint(
        DESIGNATED_TRAINING_SCENE_SEEDS
    )
    assert set(DESIGNATED_VALIDATION_SCENE_SEEDS).isdisjoint({2026072791, 2026072792})
    assert payload["continuous_surface_perturbation"] == {
        "selection": ("first_valid_fixed_order_geometry_only_separable_tangent_v1"),
        "tangent_magnitudes_m": list(ACCEPTANCE_PERTURBATION_TANGENT_MAGNITUDES_M),
        "tangent_directions_uv": [
            list(direction)
            for direction in ACCEPTANCE_PERTURBATION_TANGENT_DIRECTIONS_UV
        ],
        "minimum_surface_displacement_m": (
            ACCEPTANCE_PERTURBATION_MINIMUM_DISPLACEMENT_M
        ),
        "minimum_absolute_log_inverse_square_rate_ratio": (
            ACCEPTANCE_PERTURBATION_MINIMUM_LOG_RATE_SEPARATION
        ),
        "minimum_detector_bearing_angle_rad": (
            ACCEPTANCE_PERTURBATION_MINIMUM_BEARING_ANGLE_RAD
        ),
        "separability_logic": "inverse_square_rate_or_detector_bearing",
        "uses_observation_counts": False,
        "uses_detector_response": False,
        "uses_candidate_model_likelihood": False,
    }
    assert payload["environment"] == {
        "room_size_xyz_m": list(ACCEPTANCE_ROOM_SIZE_XYZ),
        "detector_pose_xyz_m": list(ACCEPTANCE_DETECTOR_POSE_XYZ),
        "target_blocked_fraction": ACCEPTANCE_OBSTACLE_BLOCKED_FRACTION,
        "passage_width_m": ACCEPTANCE_PASSAGE_WIDTH_M,
        "surface_chart_max_edge_m": ACCEPTANCE_SURFACE_CHART_MAX_EDGE_M,
        "obstacle_material": MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.obstacle_material,
        "room_boundary_thickness_m": (
            MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.room_boundary_thickness_m
        ),
    }
    assert payload["geometry_compute"] == {
        "use_gpu": ACCEPTANCE_GEOMETRY_USE_GPU,
        "device": ACCEPTANCE_GEOMETRY_DEVICE,
        "dtype": ACCEPTANCE_GEOMETRY_DTYPE,
    }


def test_validation_manifest_and_line_identity_are_immutable_snapshots() -> None:
    """External mutation must not authorize an already-created model."""
    validation = {
        "schema_version": 1,
        "all_passed": False,
    }
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=5.813e-9,
        background_rate_cps=5.0,
        validation_manifest=validation,
    )
    original_hash = model.contract_hash_sha256
    validation["all_passed"] = True
    assert model.production_ready is False
    assert model.contract_hash_sha256 == original_hash
    returned_line = dict(model.line_identity[0])
    returned_line["energy_keV"] = -1.0
    assert float(model.line_identity[0]["energy_keV"]) > 0.0


def test_hierarchical_likelihood_matches_torch_and_background_is_exact() -> None:
    """The calibrated mixture must match Torch and leave K=0 marks exact."""
    torch = pytest.importorskip("torch")
    nodes, weights = rate_scale_mixture_for_half_width(0.20)
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=7.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=300.0,
    )
    plain = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=7.0,
    )
    rng = np.random.default_rng(1729)
    line_count = len(model.line_identity)
    total = rng.uniform(0.0, 25.0, (3, 2, 2, line_count))
    uncollided = 0.7 * total
    features = _valid_transport_features(rng.uniform(0.0, 1.0, total.shape + (4,)))
    live_times = np.array([2.0, 3.0], dtype=np.float64)
    observed = model.sample_predictive_numpy(
        total[:1],
        uncollided[:1],
        features[:1],
        live_times,
        sample_count=2,
        rng=rng,
    )[0]
    numpy_log = model.cross_log_likelihood_numpy(
        observed,
        total,
        uncollided,
        features,
        live_times,
    )
    torch_log = (
        model.cross_log_likelihood_torch(
            torch.as_tensor(observed, dtype=torch.float64),
            torch.as_tensor(total, dtype=torch.float64),
            torch.as_tensor(uncollided, dtype=torch.float64),
            torch.as_tensor(features, dtype=torch.float64),
            torch.as_tensor(live_times, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.allclose(numpy_log, torch_log, rtol=1e-10, atol=1e-7)

    zero = np.zeros((2, 2, 1, line_count), dtype=np.float64)
    zero_features = _valid_transport_features(
        np.zeros(zero.shape + (4,), dtype=np.float64)
    )
    background_observed = plain.sample_predictive_numpy(
        zero[:1],
        zero[:1],
        zero_features[:1],
        live_times,
        sample_count=3,
        rng=rng,
    )[0]
    hierarchical_background = model.cross_log_likelihood_numpy(
        background_observed,
        zero,
        zero,
        zero_features,
        live_times,
    )
    exact_background = plain.cross_log_likelihood_numpy(
        background_observed,
        zero,
        zero,
        zero_features,
        live_times,
    )
    assert np.allclose(
        hierarchical_background,
        exact_background,
        rtol=0.0,
        atol=2e-12,
    )


def test_predictive_sampler_uses_one_rate_scale_for_all_station_views() -> None:
    """Station-shared scale draws must induce cross-view total covariance."""
    nodes = (0.5, 1.0, 1.5)
    weights = (0.25, 0.50, 0.25)
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137",),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=1_000.0,
    )
    line_count = len(model.line_identity)
    total = np.zeros((1, 2, 1, line_count), dtype=np.float64)
    total[:, :, :, 0] = 100.0
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    samples = model.sample_predictive_numpy(
        total,
        total,
        features,
        np.ones(2, dtype=np.float64),
        sample_count=1_500,
        rng=np.random.default_rng(82),
    )
    totals = np.sum(samples[0], axis=-1)
    correlation = float(np.corrcoef(totals[:, 0], totals[:, 1])[0, 1])
    assert samples.dtype == np.int64
    assert correlation > 0.75


def test_innovation_integrates_configured_rate_and_mark_discrepancy() -> None:
    """Innovation gates must use the same calibrated noise as likelihood."""
    nodes, weights = rate_scale_mixture_for_half_width(0.20)
    calibrated = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137",),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        rate_scale_nodes_j=nodes,
        rate_scale_weights_j=weights,
        mark_concentration_source=300.0,
    )
    exact = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137",),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
    )
    line_count = len(calibrated.line_identity)
    total = np.zeros((1, 1, 1, line_count), dtype=np.float64)
    total[..., 0] = 100_000.0
    features = _valid_transport_features(np.zeros(total.shape + (4,), dtype=np.float64))
    expected = calibrated.predict_mean_numpy(
        total,
        total,
        features,
        np.ones(1, dtype=np.float64),
    )[0]
    observed = np.rint(expected).astype(np.float64)
    peak = int(np.argmax(observed[0]))
    observed[0, peak] += 10_000.0
    arguments = (
        observed,
        total,
        total,
        features,
        np.ones(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
    )
    calibrated_result = calibrated.posterior_predictive_innovation_numpy(
        *arguments,
        confidence=0.99,
    )
    exact_result = exact.posterior_predictive_innovation_numpy(
        *arguments,
        confidence=0.99,
    )

    assert (
        calibrated_result["renewal_total_max_abs_z"]
        < (exact_result["renewal_total_max_abs_z"])
    )
    assert (
        calibrated_result["conditional_mark_pearson"]
        < (exact_result["conditional_mark_pearson"])
    )


def test_physics_only_innovation_uses_component_discrepancy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physics-only component uncertainty must govern innovation diagnostics."""
    model = _runtime_ready_candidate()
    assert model.exact_physical_statistics_ready is True
    sentinel = {
        "renewal_total_max_abs_z": 1.25,
        "renewal_total_within_confidence": True,
        "conditional_mark_pearson": 2.5,
        "conditional_mark_degrees_of_freedom": 3,
        "conditional_mark_tail_probability": 0.5,
        "conditional_mark_upper_tail_probability": 0.25,
        "confidence": 0.99,
    }
    calls = 0

    def _discrepancy_path(self: object, *args: object, **kwargs: object) -> object:
        """Return a sentinel while recording the selected numerical path."""
        nonlocal calls
        del self, args, kwargs
        calls += 1
        return sentinel

    monkeypatch.setattr(
        GeometryConditionedSpectralModel,
        "_discrepancy_innovation_numpy",
        _discrepancy_path,
    )
    line_count = len(model.line_identity)
    total = np.ones((1, 1, 1, line_count), dtype=np.float64)
    features = _valid_transport_features(
        np.zeros(total.shape + (4,), dtype=np.float64)
    )
    observed = np.zeros(
        (1, np.asarray(model.energy_axis_keV).size),
        dtype=np.float64,
    )

    result = model.posterior_predictive_innovation_numpy(
        observed,
        total,
        total,
        features,
        np.ones(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        confidence=0.99,
    )

    assert calls == 1
    assert result == sentinel


def test_predictive_action_seeds_are_invariant_to_action_batch_width() -> None:
    """Canonical action substreams must survive arbitrary outer batching."""
    model = GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137", "Eu-154"),
        dead_time_tau_s=2.0e-5,
        background_rate_cps=6.0,
        mark_concentration_source=300.0,
    )
    rng = np.random.default_rng(951)
    action_count = 5
    state_count = 3
    view_count = 2
    source_count = 2
    line_count = len(model.line_identity)
    total = rng.uniform(
        0.0,
        20.0,
        (
            action_count,
            state_count,
            view_count,
            source_count,
            line_count,
        ),
    )
    uncollided = 0.6 * total
    features = _valid_transport_features(rng.uniform(0.0, 1.0, total.shape + (4,)))
    live_times = np.array([2.0, 3.0], dtype=np.float64)
    action_seeds = np.asarray([19, 23, 29, 31, 37], dtype=np.int64)
    all_actions = model.sample_predictive_numpy(
        total,
        uncollided,
        features,
        live_times,
        sample_count=4,
        rng=np.random.default_rng(1),
        action_seeds_a=action_seeds,
    )
    split_actions = np.concatenate(
        [
            model.sample_predictive_numpy(
                total[start:stop],
                uncollided[start:stop],
                features[start:stop],
                live_times,
                sample_count=4,
                rng=np.random.default_rng(999 + start),
                action_seeds_a=action_seeds[start:stop],
            )
            for start, stop in ((0, 2), (2, 3), (3, 5))
        ],
        axis=0,
    )
    assert np.array_equal(all_actions, split_actions)


def test_schema_seven_manifest_excludes_scene_fitted_and_legacy_terms() -> None:
    """Schema seven binds catalog lines and component-aware mark uncertainty."""
    candidate = _runtime_ready_candidate()
    manifest = candidate.manifest_payload()

    assert manifest["schema_version"] == 7
    assert manifest["runtime_ready"] is True
    assert manifest["production_ready"] is False
    assert "discrepancy_training" not in manifest
    assert "low_rank_spectral_mean_correction" not in manifest
    assert manifest["mark_concentration_source"] is None
    assert "count_discrepancy_concentration" not in manifest
    assert "maximum_scatter_order" not in manifest
    assert manifest["scatter_shape"] == (
        "detector_cone_joint_energy_impact_single_compton_v1"
    )
    assert manifest["detector_green_operator_contract_sha256"] == (
        candidate.detector_green_operator.contract_hash_sha256
    )
    assert all(
        float(row["branching_weight"]) > 0.0 and float(row["energy_keV"]) > 0.0
        for row in manifest["line_identity"]
    )


def test_runtime_factory_reconstructs_and_authenticates_schema_seven() -> None:
    """Live construction must authenticate model, catalog, and Green identity."""
    runtime = approved_runtime_config()
    model = geometry_conditioned_model_from_runtime_config(runtime)

    assert model.runtime_ready is True
    assert model.production_ready is True
    corrupted = json.loads(json.dumps(runtime))
    corrupted["full_spectrum_generative_model"]["background_rate_cps"] = 6.0
    with pytest.raises(ValueError, match="reconstruct"):
        geometry_conditioned_model_from_runtime_config(corrupted)
    corrupted = json.loads(json.dumps(runtime))
    corrupted["full_spectrum_contract_hash_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        geometry_conditioned_model_from_runtime_config(corrupted)
    corrupted = json.loads(json.dumps(runtime))
    corrupted["full_spectrum_generative_model"][
        "detector_green_operator_contract_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="reconstruct"):
        geometry_conditioned_model_from_runtime_config(corrupted)


def test_nonalgorithm_acceptance_metadata_does_not_expire_model_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renaming an acceptance run must not invalidate an approved algorithm."""
    model = approved_full_spectrum_model()
    contract_hash = model.contract_hash_sha256

    monkeypatch.setattr(
        transport_spectral,
        "FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256",
        "0" * 64,
    )

    assert model.contract_hash_sha256 == contract_hash
    assert model.production_ready is True
    model.require_production_ready()


def test_catalog_independent_approval_preserves_target_profile_contract() -> None:
    """Approval transfer changes provenance but not catalog-derived physics."""
    source = approved_full_spectrum_model()
    target = GeometryConditionedSpectralModel.physics_only_native(
        ("Cs-137",),
        dead_time_tau_s=source.dead_time_tau_s,
        background_rate_cps=source.background_rate_cps,
        detector_green_operator=source.detector_green_operator,
    )
    target_contract = target.contract_hash_sha256

    approved = with_catalog_independent_production_approval(
        target,
        approved_source=source,
    )

    assert approved.contract_hash_sha256 == target_contract
    assert approved.production_ready is True
    assert approved.validation_manifest is not None
    assert approved.validation_manifest["schema_version"] == 7
    assert approved.validation_manifest["application_validation_isotopes"] == (
        "Co-60",
        "Cs-137",
        "Eu-154",
    )


def test_catalog_independent_approval_rejects_algorithm_drift() -> None:
    """A background or other algorithm-contract change cannot reuse evidence."""
    source = approved_full_spectrum_model()
    changed = GeometryConditionedSpectralModel.physics_only_native(
        ("Cs-137",),
        dead_time_tau_s=source.dead_time_tau_s,
        background_rate_cps=source.background_rate_cps + 1.0,
        detector_green_operator=source.detector_green_operator,
    )

    with pytest.raises(RuntimeError, match="algorithm differs"):
        with_catalog_independent_production_approval(
            changed,
            approved_source=source,
        )


def test_catalog_independent_approval_digest_tampering_fails_closed() -> None:
    """Transferred evidence cannot authorize a different core digest."""
    source = approved_full_spectrum_model()
    target = GeometryConditionedSpectralModel.physics_only_native(
        ("Cs-137",),
        dead_time_tau_s=source.dead_time_tau_s,
        background_rate_cps=source.background_rate_cps,
        detector_green_operator=source.detector_green_operator,
    )
    approved = with_catalog_independent_production_approval(
        target,
        approved_source=source,
    )
    validation = approved.manifest_payload()["validation"]
    validation["approved_catalog_independent_contract_sha256"] = "0" * 64
    tampered = GeometryConditionedSpectralModel.physics_only_native(
        ("Cs-137",),
        dead_time_tau_s=source.dead_time_tau_s,
        background_rate_cps=source.background_rate_cps,
        detector_green_operator=source.detector_green_operator,
        validation_manifest=validation,
    )

    assert tampered.production_ready is False
    with pytest.raises(RuntimeError, match="catalog-independent"):
        tampered.require_production_ready()


@pytest.mark.parametrize(
    "field_name",
    (
        "energy_min_keV",
        "energy_max_keV",
        "bin_width_keV",
        "energy_bin_count",
        "background_cps",
        "dead_time_tau_s",
    ),
)
def test_runtime_factory_rejects_numeric_string_scalars(
    field_name: str,
) -> None:
    """Runtime identity scalars retain their exact JSON numeric types."""
    runtime = approved_runtime_config()
    runtime[field_name] = str(runtime[field_name])

    with pytest.raises(ValueError, match=field_name):
        geometry_conditioned_model_from_runtime_config(runtime)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    (
        (("dead_time_tau_s",), "5.813e-09"),
        (("background_rate_cps",), "5.0"),
        (("rate_scale_mixture", "nodes", 0), "1.0"),
        (("rate_scale_mixture", "weights", 0), "1.0"),
        (("line_identity", 0, "isotope"), 137),
        (("line_identity", 0, "branching_weight"), "1.0"),
    ),
)
def test_manifest_factory_rejects_scalar_type_coercion(
    field_path: tuple[object, ...],
    replacement: object,
) -> None:
    """File-backed model fields must not gain meaning through coercion."""
    approved = approved_full_spectrum_model()
    payload = json.loads(json.dumps(approved.manifest_payload()))
    target: object = payload
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        GeometryConditionedSpectralModel.from_manifest_payload(
            payload,
            detector_green_operator=approved.detector_green_operator,
        )


@pytest.mark.parametrize("field_name", ("value", "threshold"))
def test_validation_metrics_reject_numeric_strings(field_name: str) -> None:
    """Validation evidence retains exact JSON-number semantics."""
    approved = approved_full_spectrum_model()
    payload = json.loads(json.dumps(approved.manifest_payload()))
    validation = payload["validation"]
    metric_id = next(iter(validation["metrics"]))
    validation["metrics"][metric_id][field_name] = str(
        validation["metrics"][metric_id][field_name]
    )

    with pytest.raises((TypeError, ValueError)):
        GeometryConditionedSpectralModel.from_manifest_payload(
            payload,
            detector_green_operator=approved.detector_green_operator,
        )


def test_schema_four_rejects_scene_fitted_transport_response() -> None:
    """A retired fitted response cannot cross the schema-four boundary."""
    approved = approved_full_spectrum_model()
    payload = json.loads(json.dumps(approved.manifest_payload()))
    payload["additive_noncollided_transport_response"]["model"] = (
        "retired_scene_fitted_response"
    )

    with pytest.raises(ValueError, match="scene-fitted"):
        GeometryConditionedSpectralModel.from_manifest_payload(
            payload,
            detector_green_operator=approved.detector_green_operator,
        )


def test_runtime_factory_authenticates_file_backed_schema_four(
    tmp_path: Path,
) -> None:
    """A file-backed schema-four model must match byte and model hashes."""
    runtime = approved_runtime_config()
    manifest = runtime.pop("full_spectrum_generative_model")
    raw_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    asset_path = tmp_path / "approved.json"
    asset_path.write_bytes(raw_bytes)
    runtime["full_spectrum_generative_model_path"] = asset_path.name
    runtime["full_spectrum_generative_model_file_sha256"] = hashlib.sha256(
        raw_bytes
    ).hexdigest()

    reconstructed = geometry_conditioned_model_from_runtime_config(
        runtime,
        run_root=tmp_path,
    )
    assert reconstructed.production_ready is True
    runtime["full_spectrum_generative_model_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file SHA-256"):
        geometry_conditioned_model_from_runtime_config(
            runtime,
            run_root=tmp_path,
        )


def test_runtime_factory_rejects_duplicate_model_asset_keys(
    tmp_path: Path,
) -> None:
    """A pinned model asset must reject duplicate JSON keys."""
    runtime = approved_runtime_config()
    manifest = runtime.pop("full_spectrum_generative_model")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    raw_bytes = ('{"schema_version":4,' + canonical.removeprefix("{")).encode()
    asset_path = tmp_path / "duplicate.json"
    asset_path.write_bytes(raw_bytes)
    runtime["full_spectrum_generative_model_path"] = asset_path.name
    runtime["full_spectrum_generative_model_file_sha256"] = hashlib.sha256(
        raw_bytes
    ).hexdigest()

    with pytest.raises(ValueError, match="canonical JSON"):
        geometry_conditioned_model_from_runtime_config(
            runtime,
            run_root=tmp_path,
        )
