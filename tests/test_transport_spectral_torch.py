"""Tests for exact bulk-device full-spectrum Torch sampling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import spectrum.transport_spectral as transport_spectral
from tests.green_test_support import write_synthetic_detector_green_artifact

torch = pytest.importorskip("torch")

from spectrum.predictive_torch import (  # noqa: E402
    ComponentTreeMarkParameters,
    nonparalyzable_count_cdf_torch,
    sample_action_seeded_torch,
    sample_mean_one_gamma_torch,
    sample_multinomial_counts_torch,
    sample_nonparalyzable_counts_torch,
    sample_predictive_action_torch,
)
from spectrum.transport_spectral import (  # noqa: E402
    DETECTOR_IMPACT_PHASE_COUNT,
    GeometryConditionedSpectralModel,
    PhysicalComponentDiscrepancy,
    nonparalyzable_count_cdf_numpy,
    nonparalyzable_count_log_probability_numpy,
)


@pytest.fixture(scope="module")
def _detector_green_manifest(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Publish one immutable synthetic Green artifact for this test module."""
    return write_synthetic_detector_green_artifact(
        tmp_path_factory.mktemp("detector-green-torch") / "operator"
    )


@pytest.fixture(autouse=True)
def _use_explicit_detector_green(
    _detector_green_manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route sampler models through an authenticated synthetic operator."""
    monkeypatch.setattr(
        transport_spectral,
        "DEFAULT_DETECTOR_GREEN_OPERATOR_MANIFEST",
        _detector_green_manifest,
    )


def _generator(seed: int, *, device: str = "cpu") -> "torch.Generator":
    """Return one explicitly seeded Torch generator."""
    return torch.Generator(device=device).manual_seed(seed)


def _hierarchical_parameters(
    leading_shape: tuple[int, ...],
    view_count: int,
    *,
    device: str = "cpu",
) -> ComponentTreeMarkParameters:
    """Return a small aligned component-aware tree for sampler tests."""
    leaves = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    tree_concentration = torch.full(
        leading_shape + (view_count, 2),
        80.0,
        dtype=torch.float64,
        device=device,
    )
    leaf_concentration = torch.full(
        leading_shape + (view_count, 3),
        90.0,
        dtype=torch.float64,
        device=device,
    )
    return ComponentTreeMarkParameters(
        leaf_group_mask_hb=leaves,
        tree_left_mask_tb=torch.stack((leaves[0], leaves[1])),
        tree_right_mask_tb=torch.stack((leaves[1] + leaves[2], leaves[2])),
        tree_depth_t=torch.tensor([0, 1], device=device),
        tree_left_child_t=torch.tensor([-1, -2], device=device),
        tree_right_child_t=torch.tensor([1, -3], device=device),
        tree_concentration_xvt=tree_concentration,
        leaf_concentration_xvh=leaf_concentration,
    )


def _integration_model(branch: str) -> GeometryConditionedSpectralModel:
    """Return a compact model selecting one NumPy predictive branch."""
    keywords: dict[str, object] = {}
    if branch == "legacy_fraction":
        keywords["mark_concentration_source"] = 40.0
    elif branch in ("legacy_view_count", "legacy_station_count"):
        keywords["count_discrepancy_concentration"] = 30.0
        keywords["count_discrepancy_scope"] = (
            "view_independent" if branch == "legacy_view_count" else "station_shared"
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
        raise ValueError(f"Unknown integration branch: {branch!r}.")
    return GeometryConditionedSpectralModel.nonproduction_native(
        ("Cs-137",),
        dead_time_tau_s=1.0e-5,
        background_rate_cps=1.0,
        **keywords,
    )


def _integration_inputs(
    model: GeometryConditionedSpectralModel,
    *,
    action_count: int = 2,
    device: str = "cpu",
) -> tuple[object, object, object, object]:
    """Return action/state/view transport tensors for class integration tests."""
    line_count = len(model.line_identity)
    total = torch.linspace(
        16.0,
        28.0,
        steps=action_count * 2 * 2 * line_count,
        device=device,
        dtype=torch.float64,
    ).reshape(action_count, 2, 2, 1, line_count)
    uncollided = total * 0.7
    features = torch.zeros(
        tuple(total.shape) + (len(model.transport_feature_order),),
        device=device,
        dtype=torch.float64,
    )
    features[..., 3] = 1.25
    features[..., 4:] = 1.0 / DETECTOR_IMPACT_PHASE_COUNT
    live_times = torch.tensor(
        [0.25, 0.4],
        device=device,
        dtype=torch.float64,
    )
    return total, uncollided, features, live_times


def test_renewal_cdf_torch_matches_numpy() -> None:
    """Torch renewal CDF values must match the established NumPy equation."""
    thresholds = np.arange(-1, 90, dtype=np.int64).reshape(7, 13)
    rates = np.linspace(0.0, 2.0e5, thresholds.size).reshape(thresholds.shape)
    live = np.linspace(0.01, 3.0, thresholds.size).reshape(thresholds.shape)
    expected = nonparalyzable_count_cdf_numpy(
        thresholds,
        rates,
        live,
        dead_time_tau_s=1.0e-4,
    )
    actual = nonparalyzable_count_cdf_torch(
        torch.as_tensor(thresholds),
        torch.as_tensor(rates, dtype=torch.float64),
        torch.as_tensor(live, dtype=torch.float64),
        dead_time_tau_s=1.0e-4,
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-15)


def test_renewal_sampler_matches_exact_moments_and_support() -> None:
    """Torch inverse-CDF renewal draws must match exact mean and Fano factor."""
    rate = 45.0
    live_time = 1.0
    tau = 0.008
    support = np.arange(0, int(np.floor(live_time / tau)) + 2)
    probability = np.exp(
        nonparalyzable_count_log_probability_numpy(
            support,
            np.full(support.shape, rate, dtype=np.float64),
            np.full(support.shape, live_time, dtype=np.float64),
            dead_time_tau_s=tau,
        )
    )
    exact_mean = float(np.sum(support * probability))
    exact_variance = float(np.sum(np.square(support - exact_mean) * probability))
    samples = sample_nonparalyzable_counts_torch(
        torch.tensor([0.0, rate, 1.0e6], dtype=torch.float64),
        torch.tensor([1.0, live_time, 0.01], dtype=torch.float64),
        dead_time_tau_s=tau,
        sample_count=40_000,
        generator=_generator(20260727),
    )
    central = samples[1].to(torch.float64)
    sampled_mean = float(torch.mean(central))
    sampled_variance = float(torch.var(central, correction=0))

    assert samples.dtype == torch.int64
    assert torch.all(samples[0] == 0)
    assert torch.all(samples[2] <= 2)
    assert abs(sampled_mean - exact_mean) < 0.05
    assert abs(sampled_variance / sampled_mean - exact_variance / exact_mean) < 0.02


def test_mean_one_gamma_sampler_matches_moments() -> None:
    """The canonical Torch Gamma kernel must preserve means and variances."""
    concentrations = torch.tensor(
        [0.2, 1.0, 5.0, 100.0],
        dtype=torch.float64,
    )[:, None].expand(4, 120_000)
    samples = sample_mean_one_gamma_torch(
        concentrations,
        generator=_generator(811),
    )
    means = torch.mean(samples, dim=1).numpy()
    variances = torch.var(samples, dim=1, correction=0).numpy()
    expected_variances = 1.0 / concentrations[:, 0].numpy()

    np.testing.assert_allclose(means, np.ones(4), rtol=0.012, atol=0.006)
    np.testing.assert_allclose(
        variances,
        expected_variances,
        rtol=0.035,
        atol=0.004,
    )


def test_gamma_sampler_rejects_missing_canonical_torch_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Torch compatibility gap must fail instead of selecting another RNG."""
    monkeypatch.setattr(torch, "_standard_gamma", None)

    with pytest.raises(RuntimeError, match="requires torch._standard_gamma"):
        sample_mean_one_gamma_torch(
            torch.ones(2, dtype=torch.float64),
            generator=_generator(811),
        )


def test_balanced_multinomial_tree_preserves_totals_and_moments() -> None:
    """Conditional-binomial tree draws must be exact variable-total rows."""
    row_count = 80_000
    probabilities = torch.tensor(
        [[0.2, 0.3, 0.5], [0.0, 0.8, 0.2]],
        dtype=torch.float64,
    ).repeat(row_count // 2, 1)
    totals = torch.tensor([20, 7], dtype=torch.int64).repeat(row_count // 2)
    first = sample_multinomial_counts_torch(
        totals,
        probabilities,
        generator=_generator(17),
    )
    replay = sample_multinomial_counts_torch(
        totals,
        probabilities,
        generator=_generator(17),
    )

    assert torch.equal(first, replay)
    assert torch.equal(torch.sum(first, dim=-1), totals)
    assert first.dtype == torch.int64
    first_family = first[0::2].to(torch.float64).mean(dim=0).numpy()
    second_family = first[1::2].to(torch.float64).mean(dim=0).numpy()
    np.testing.assert_allclose(first_family, [4.0, 6.0, 10.0], atol=0.035)
    np.testing.assert_allclose(second_family, [0.0, 5.6, 1.4], atol=0.025)


def test_gamma_poisson_predictive_sampler_preserves_count_moments() -> None:
    """View-independent Gamma-Poisson totals must retain mean and variance."""
    source = torch.tensor(
        [[[45.0, 30.0, 15.0], [20.0, 15.0, 5.0]]],
        dtype=torch.float64,
    )
    background = torch.zeros_like(source)
    live = torch.tensor([1.0, 1.0], dtype=torch.float64)
    concentration = torch.tensor([[12.0, 30.0]], dtype=torch.float64)
    sample_count = 80_000
    samples = sample_predictive_action_torch(
        source,
        background,
        live,
        sample_count=sample_count,
        generator=_generator(93),
        rate_scale_nodes_j=torch.ones(1, dtype=torch.float64),
        rate_scale_weights_j=torch.ones(1, dtype=torch.float64),
        dead_time_tau_s=0.0,
        mark_model="fixed_multinomial",
        count_scope="view_independent_gamma_poisson",
        count_concentration_xv=concentration,
    )
    totals = torch.sum(samples[0], dim=-1).to(torch.float64)
    means = torch.mean(totals, dim=0).numpy()
    variances = torch.var(totals, dim=0, correction=0).numpy()
    expected_mean = np.asarray([90.0, 40.0])
    expected_variance = expected_mean + np.square(expected_mean) / np.asarray(
        [12.0, 30.0]
    )

    np.testing.assert_allclose(means, expected_mean, rtol=0.005, atol=0.15)
    np.testing.assert_allclose(
        variances,
        expected_variance,
        rtol=0.025,
        atol=0.5,
    )
    assert samples.dtype == torch.int64
    assert torch.all(samples >= 0)


def test_hierarchical_marks_preserve_partition_means_and_row_totals() -> None:
    """Hierarchical mark latents must preserve mean peak and group fractions."""
    source = torch.tensor(
        [[[20.0, 10.0, 15.0, 5.0, 30.0, 20.0]]],
        dtype=torch.float64,
    )
    background = torch.zeros_like(source)
    sample_count = 70_000
    hierarchy = _hierarchical_parameters((1,), 1)
    samples = sample_predictive_action_torch(
        source,
        background,
        torch.ones(1, dtype=torch.float64),
        sample_count=sample_count,
        generator=_generator(187),
        rate_scale_nodes_j=torch.ones(1, dtype=torch.float64),
        rate_scale_weights_j=torch.ones(1, dtype=torch.float64),
        dead_time_tau_s=0.0,
        mark_model="component_dirichlet_tree_hierarchical",
        count_scope="view_independent_gamma_poisson",
        count_concentration_xv=torch.full((1, 1), 40.0, dtype=torch.float64),
        hierarchical_marks=hierarchy,
    )[0, :, 0]
    aggregate = torch.sum(samples, dim=0).to(torch.float64)
    total = torch.sum(aggregate)
    peak_fraction = float(torch.sum(aggregate[:2]) / total)
    first_group_fraction = float(torch.sum(aggregate[2:4]) / total)
    expected = source[0, 0] / torch.sum(source[0, 0])

    assert samples.dtype == torch.int64
    assert torch.all(torch.sum(samples, dim=-1) >= 0)
    assert peak_fraction == pytest.approx(float(torch.sum(expected[:2])), abs=0.004)
    assert first_group_fraction == pytest.approx(
        float(torch.sum(expected[2:4])),
        abs=0.004,
    )


def test_fraction_dirichlet_marks_are_exact_and_replayable() -> None:
    """Fraction-Dirichlet marks must conserve totals and replay by seed."""
    source = torch.tensor(
        [[[25.0, 15.0, 10.0, 5.0]]],
        dtype=torch.float64,
    )

    def _run() -> object:
        """Draw the same fraction-Dirichlet stream."""
        return sample_predictive_action_torch(
            source,
            torch.ones_like(source),
            torch.ones(1, dtype=torch.float64),
            sample_count=512,
            generator=_generator(409),
            rate_scale_nodes_j=torch.ones(1, dtype=torch.float64),
            rate_scale_weights_j=torch.ones(1, dtype=torch.float64),
            dead_time_tau_s=0.0,
            count_scope="view_independent_gamma_poisson",
            count_concentration_xv=torch.full(
                (1, 1),
                50.0,
                dtype=torch.float64,
            ),
            mark_model="fraction_dirichlet_multinomial",
            mark_concentration_xv=torch.full(
                (1, 1),
                40.0,
                dtype=torch.float64,
            ),
        )

    first = _run()
    replay = _run()
    assert torch.equal(first, replay)
    assert first.dtype == torch.int64
    assert torch.all(torch.sum(first, dim=-1) >= 0)
    assert torch.var(first[0, :, 0, 0].to(torch.float64)) > 0.0


def test_retired_component_scale_mark_model_is_rejected() -> None:
    """The retired two-point component-scale path must be unreachable."""
    source = torch.ones((1, 1, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="Unsupported predictive mark_model"):
        sample_predictive_action_torch(
            source,
            torch.ones_like(source),
            torch.ones(1, dtype=torch.float64),
            sample_count=2,
            generator=_generator(419),
            rate_scale_nodes_j=torch.ones(1, dtype=torch.float64),
            rate_scale_weights_j=torch.ones(1, dtype=torch.float64),
            dead_time_tau_s=0.0,
            mark_model="station_shared_two_point_component_scale",
        )


def test_station_shared_gamma_poisson_scale_correlates_view_totals() -> None:
    """A station-shared Gamma scale must couple predictive view totals."""
    source = torch.full((1, 2, 3), 20.0, dtype=torch.float64)
    samples = sample_predictive_action_torch(
        source,
        torch.zeros_like(source),
        torch.ones(2, dtype=torch.float64),
        sample_count=20_000,
        generator=_generator(431),
        rate_scale_nodes_j=torch.ones(1, dtype=torch.float64),
        rate_scale_weights_j=torch.ones(1, dtype=torch.float64),
        dead_time_tau_s=0.0,
        count_scope="station_shared_gamma_poisson",
        count_concentration_xv=torch.full((1,), 8.0, dtype=torch.float64),
        mark_model="fixed_multinomial",
    )
    totals = torch.sum(samples[0], dim=-1).to(torch.float64)
    correlation = torch.corrcoef(totals.T)[0, 1]
    assert correlation > 0.5


@pytest.mark.parametrize(
    "branch",
    (
        "fixed_renewal",
        "legacy_fraction",
        "legacy_view_count",
        "legacy_station_count",
        "component_tree",
    ),
)
def test_model_sampler_maps_every_numpy_predictive_branch(branch: str) -> None:
    """The model wrapper must reproduce every NumPy branch on Torch tensors."""
    model = _integration_model(branch)
    total, uncollided, features, live_times = _integration_inputs(model)
    seeds = np.asarray([1201, 1213], dtype=np.int64)

    first = model.sample_predictive_torch(
        total,
        uncollided,
        features,
        live_times,
        sample_count=3,
        action_seeds_a=seeds,
    )
    replay = model.sample_predictive_torch(
        total,
        uncollided,
        features,
        live_times,
        sample_count=3,
        action_seeds_a=seeds,
    )

    assert first.device == total.device
    assert first.dtype == torch.int64
    assert tuple(first.shape) == (
        2,
        2,
        3,
        2,
        int(model.energy_axis_keV.size),
    )
    assert torch.equal(first, replay)
    assert torch.all(first >= 0)


@pytest.mark.parametrize("branch", ("fixed_renewal", "component_tree"))
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ),
)
def test_exact_slot_overlay_matches_materialized_state(
    branch: str,
    device: str,
) -> None:
    """Bounded slot replacement must equal an explicitly materialized oracle."""
    model = _integration_model(branch)
    action_count = 2
    accepted_state_count = 5
    proposal_state_count = 3
    view_count = 2
    source_slot_count = 3
    line_count = len(model.line_identity)
    total = torch.linspace(
        12.0,
        31.0,
        steps=(
            action_count
            * accepted_state_count
            * view_count
            * source_slot_count
            * line_count
        ),
        device=device,
        dtype=torch.float64,
    ).reshape(
        action_count,
        accepted_state_count,
        view_count,
        source_slot_count,
        line_count,
    )
    uncollided = 0.7 * total
    features = torch.zeros(
        tuple(total.shape) + (len(model.transport_feature_order),),
        device=device,
        dtype=torch.float64,
    )
    features[..., 4] = 1.0
    features[..., 5:] = 1.0 / DETECTOR_IMPACT_PHASE_COUNT
    replacement_total = torch.linspace(
        4.0,
        9.0,
        steps=(
            action_count
            * proposal_state_count
            * view_count
            * line_count
        ),
        device=device,
        dtype=torch.float64,
    ).reshape(
        action_count,
        proposal_state_count,
        view_count,
        1,
        line_count,
    )
    replacement_uncollided = 0.6 * replacement_total
    replacement_features = torch.zeros(
        tuple(replacement_total.shape) + (len(model.transport_feature_order),),
        device=device,
        dtype=torch.float64,
    )
    replacement_features[..., 4] = 1.5
    replacement_features[..., 5:] = 1.0 / DETECTOR_IMPACT_PHASE_COUNT
    particle_indices = torch.tensor(
        [4, 1, 3],
        device=device,
        dtype=torch.long,
    )
    observed = torch.zeros(
        (action_count, 1, view_count, int(model.energy_axis_keV.size)),
        device=device,
        dtype=torch.float64,
    )
    live_times = torch.tensor(
        [0.25, 0.4],
        device=device,
        dtype=torch.float64,
    )
    total_before = total.clone()
    uncollided_before = uncollided.clone()
    features_before = features.clone()

    materialized_total = torch.index_select(total, -4, particle_indices)
    materialized_uncollided = torch.index_select(
        uncollided,
        -4,
        particle_indices,
    )
    materialized_features = torch.index_select(features, -5, particle_indices)
    materialized_total[..., 1:2, :] = replacement_total
    materialized_uncollided[..., 1:2, :] = replacement_uncollided
    materialized_features[..., 1:2, :, :] = replacement_features
    expected = model.cross_log_likelihood_torch(
        observed,
        materialized_total,
        materialized_uncollided,
        materialized_features,
        live_times,
        state_chunk_size=2,
    )
    actual = model.cross_log_likelihood_replace_slots_torch(
        observed,
        total,
        uncollided,
        features,
        replacement_total,
        replacement_uncollided,
        replacement_features,
        live_times,
        particle_indices_n=particle_indices,
        slot_start=1,
        slot_stop=2,
        state_chunk_size=2,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(total, total_before)
    assert torch.equal(uncollided, uncollided_before)
    assert torch.equal(features, features_before)
    assert model.last_torch_slot_overlay_diagnostics == {
        "mode": "bounded_exact_slot_overlay",
        "chunk_selection_mode": "explicit_or_non_cuda",
        "proposal_state_count": 3,
        "accepted_state_count": 5,
        "state_chunk_size": 2,
        "slab_count": 2,
        "replacement_slot_count": 1,
        "source_slot_count": 3,
        "scratch_peak_bytes": (
            action_count
            * 2
            * view_count
            * source_slot_count
            * line_count
            * 8
            * (2 + len(model.transport_feature_order))
        ),
        "full_history_clone_count": 0,
    }


def test_exact_slot_overlay_rejects_misaligned_replacement() -> None:
    """Slot replacement must fail before evaluating a malformed state."""
    model = _integration_model("fixed_renewal")
    total, uncollided, features, live_times = _integration_inputs(model)
    observed = torch.zeros(
        (2, 1, 2, int(model.energy_axis_keV.size)),
        dtype=torch.float64,
    )
    replacement = total[:, :1, :, :, :]
    replacement_features = features[:, :1, :, :, :, :-1]

    with pytest.raises(ValueError, match="Replacement slot tensors"):
        model.cross_log_likelihood_replace_slots_torch(
            observed,
            total,
            uncollided,
            features,
            replacement,
            replacement,
            replacement_features,
            live_times,
            particle_indices_n=torch.tensor([0]),
            slot_start=0,
            slot_stop=1,
        )

    valid_replacement = total[:, :1, :, :1, :]
    valid_replacement_features = features[:, :1, :, :1, :, :]
    with pytest.raises(TypeError, match="particle indices must be a Torch tensor"):
        model.cross_log_likelihood_replace_slots_torch(
            observed,
            total,
            uncollided,
            features,
            valid_replacement,
            valid_replacement,
            valid_replacement_features,
            live_times,
            particle_indices_n=[0],
            slot_start=0,
            slot_stop=1,
        )

    with pytest.raises(ValueError, match="share device and dtype"):
        model.cross_log_likelihood_replace_slots_torch(
            observed,
            total,
            uncollided,
            features,
            valid_replacement.to(dtype=torch.float32),
            valid_replacement.to(dtype=torch.float32),
            valid_replacement_features.to(dtype=torch.float32),
            live_times,
            particle_indices_n=torch.tensor([0], dtype=torch.long),
            slot_start=0,
            slot_stop=1,
        )


def test_model_action_streams_are_chunk_order_invariant_and_aligned() -> None:
    """Class-level action streams must be invariant and require aligned seeds."""
    model = _integration_model("fixed_renewal")
    total, uncollided, features, live_times = _integration_inputs(
        model,
        action_count=3,
    )
    seeds = (1301, 1303, 1307)

    def _sample(indices: Sequence[int], selected_seeds: Sequence[int]) -> object:
        """Sample one explicit action subset through the model API."""
        return model.sample_predictive_torch(
            total[list(indices)],
            uncollided[list(indices)],
            features[list(indices)],
            live_times,
            sample_count=2,
            action_seeds_a=selected_seeds,
        )

    complete = _sample((0, 1, 2), seeds)
    split = torch.cat(
        (
            _sample((0,), seeds[:1]),
            _sample((1, 2), seeds[1:]),
        ),
        dim=0,
    )
    permutation = (2, 0, 1)
    permuted = _sample(
        permutation,
        tuple(seeds[index] for index in permutation),
    )

    assert torch.equal(complete, split)
    assert torch.equal(permuted, complete[list(permutation)])
    with pytest.raises(ValueError, match="one seed for the leading action axis"):
        _sample((0, 1), seeds[:1])


def test_model_sampler_requires_explicit_rng_control() -> None:
    """The model sampler must reject an uncontrolled global Torch RNG stream."""
    model = _integration_model("fixed_renewal")
    total, uncollided, features, live_times = _integration_inputs(model)
    with pytest.raises(ValueError, match="generator or action seeds"):
        model.sample_predictive_torch(
            total,
            uncollided,
            features,
            live_times,
            sample_count=1,
        )
    first = model.sample_predictive_torch(
        total[0],
        uncollided[0],
        features[0],
        live_times,
        sample_count=2,
        generator=_generator(1381),
    )
    replay = model.sample_predictive_torch(
        total[0],
        uncollided[0],
        features[0],
        live_times,
        sample_count=2,
        generator=_generator(1381),
    )
    assert tuple(first.shape) == (
        2,
        2,
        2,
        int(model.energy_axis_keV.size),
    )
    assert torch.equal(first, replay)


def test_model_action_scheduler_accepts_batches_larger_than_32() -> None:
    """The seed scheduler bound must follow the actual prepared action batch."""
    model = _integration_model("fixed_renewal")
    action_count = 33
    total, uncollided, features, live_times = _integration_inputs(
        model,
        action_count=action_count,
    )
    samples = model.sample_predictive_torch(
        total,
        uncollided,
        features,
        live_times,
        sample_count=1,
        action_seeds_a=tuple(range(2001, 2001 + action_count)),
    )
    assert int(samples.shape[0]) == action_count
    assert samples.dtype == torch.int64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_hierarchical_sampler_stays_on_cuda() -> None:
    """Production hierarchical class integration must stay CUDA resident."""
    model = _integration_model("component_tree")
    total, uncollided, features, live_times = _integration_inputs(
        model,
        device="cuda",
    )
    first = model.sample_predictive_torch(
        total,
        uncollided,
        features,
        live_times,
        sample_count=2,
        action_seeds_a=(1409, 1423),
    )
    replay = model.sample_predictive_torch(
        total,
        uncollided,
        features,
        live_times,
        sample_count=2,
        action_seeds_a=(1409, 1423),
    )

    assert first.is_cuda
    assert first.dtype == torch.int64
    assert torch.equal(first, replay)


def test_action_seed_scheduler_is_split_and_order_invariant() -> None:
    """Canonical action streams must survive splitting and permutation."""
    action_count = 5
    source = torch.arange(
        1,
        action_count * 2 * 6 + 1,
        dtype=torch.float64,
    ).reshape(action_count, 2, 6)
    background = torch.full_like(source, 0.5)
    seeds = (19, 23, 29, 31, 37)

    def _sample(indices: Sequence[int], selected_seeds: Sequence[int]) -> object:
        """Sample an explicit subset with its canonical seeds."""
        selected_source = source[list(indices)]
        selected_background = background[list(indices)]

        def _one(local_index: int, generator: object) -> object:
            """Sample one action on the shared CPU device."""
            return sample_predictive_action_torch(
                selected_source[local_index],
                selected_background[local_index],
                torch.ones(2, dtype=torch.float64),
                sample_count=9,
                generator=generator,
                rate_scale_nodes_j=torch.tensor(
                    [0.8, 1.2],
                    dtype=torch.float64,
                ),
                rate_scale_weights_j=torch.tensor(
                    [0.4, 0.6],
                    dtype=torch.float64,
                ),
                dead_time_tau_s=1.0e-4,
                mark_model="fixed_multinomial",
                count_scope="renewal",
            )

        return sample_action_seeded_torch(
            selected_seeds,
            reference=selected_source,
            sampler=_one,
            maximum_action_count=8,
        )

    complete = _sample(tuple(range(action_count)), seeds)
    split = torch.cat(
        (
            _sample((0, 1), seeds[:2]),
            _sample((2,), seeds[2:3]),
            _sample((3, 4), seeds[3:]),
        ),
        dim=0,
    )
    permutation = (3, 0, 4, 1, 2)
    permuted = _sample(
        permutation,
        tuple(seeds[index] for index in permutation),
    )

    assert torch.equal(complete, split)
    assert torch.equal(permuted, complete[list(permutation)])
    with pytest.raises(ValueError, match="declared bound"):
        sample_action_seeded_torch(
            seeds,
            reference=source,
            sampler=lambda index, generator: source[index],
            maximum_action_count=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hierarchical_action_sampler_stays_on_cuda_and_replays() -> None:
    """Production-shaped hierarchical samples must remain CUDA int64 tensors."""
    device = "cuda"
    source = torch.tensor(
        [
            [[20.0, 10.0, 15.0, 5.0, 30.0, 20.0]],
            [[10.0, 20.0, 5.0, 15.0, 20.0, 30.0]],
        ],
        dtype=torch.float64,
        device=device,
    )
    hierarchy = _hierarchical_parameters((), 1, device=device)
    seeds = (101, 103)

    def _run() -> object:
        """Run the same two canonical CUDA action streams."""

        def _one(action_index: int, generator: object) -> object:
            """Sample one hierarchical CUDA action."""
            return sample_predictive_action_torch(
                source[action_index],
                torch.zeros_like(source[action_index]),
                torch.ones(1, dtype=torch.float64, device=device),
                sample_count=32,
                generator=generator,
                rate_scale_nodes_j=torch.ones(
                    1,
                    dtype=torch.float64,
                    device=device,
                ),
                rate_scale_weights_j=torch.ones(
                    1,
                    dtype=torch.float64,
                    device=device,
                ),
                dead_time_tau_s=5.813e-9,
                mark_model="component_dirichlet_tree_hierarchical",
                count_scope="view_independent_gamma_poisson",
                count_concentration_xv=torch.full(
                    (1,),
                    100.0,
                    dtype=torch.float64,
                    device=device,
                ),
                hierarchical_marks=hierarchy,
            )

        return sample_action_seeded_torch(
            seeds,
            reference=source,
            sampler=_one,
            maximum_action_count=4,
        )

    first = _run()
    replay = _run()

    assert first.is_cuda
    assert first.dtype == torch.int64
    assert torch.equal(first, replay)
    assert torch.all(first >= 0)
