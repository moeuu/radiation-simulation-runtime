"""Exact bulk-device Torch samplers for full-spectrum prediction.

The helpers in this module contain no observation-model calibration or
transport approximation.  They implement the same renewal, Gamma-Poisson,
and conditional-mark distributions used by the shared full-spectrum model,
while keeping all bulk arrays on their Torch device.  Independent action
streams are scheduled with one explicitly seeded generator per action so
caller batching and action order cannot change an action's random stream.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real


_MAX_EXACT_FLOAT64_INTEGER = (1 << 53) - 1
_UINT64_MASK = (1 << 64) - 1


def _torch_module() -> object:
    """Return Torch or raise a focused optional-dependency error."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Torch predictive sampling requires torch.") from error
    return torch


def _require_float64_tensor(value: object, *, name: str) -> object:
    """Return one Torch float64 tensor without changing its device."""
    torch = _torch_module()
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} must be a torch.float64 tensor.")
    return tensor


def _validate_generator_device(generator: object, reference: object) -> None:
    """Require a Torch generator on the reference tensor's device type."""
    torch = _torch_module()
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator.")
    generator_device = torch.device(generator.device)
    reference_device = torch.as_tensor(reference).device
    if generator_device.type != reference_device.type:
        raise ValueError(
            "The Torch generator and predictive tensors must use the same device type."
        )
    if (
        generator_device.type == "cuda"
        and generator_device.index is not None
        and reference_device.index is not None
        and generator_device.index != reference_device.index
    ):
        raise ValueError(
            "The Torch generator and predictive tensors must use the same CUDA device."
        )


def _positive_integer(value: object, *, name: str) -> int:
    """Return a strictly positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _positive_finite(value: object, *, name: str) -> float:
    """Return a strictly positive finite scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _nonnegative_finite(value: object, *, name: str) -> float:
    """Return a finite nonnegative scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return resolved


def nonparalyzable_count_cdf_torch(
    count_threshold: object,
    incident_rates_cps: object,
    live_times_s: object,
    *,
    dead_time_tau_s: float,
) -> object:
    """Return ``P(M <= m)`` for a nonparalyzable detector starting live."""
    torch = _torch_module()
    rates = _require_float64_tensor(
        incident_rates_cps,
        name="incident_rates_cps",
    )
    thresholds = torch.as_tensor(
        count_threshold,
        device=rates.device,
        dtype=torch.int64,
    )
    live_times = torch.as_tensor(
        live_times_s,
        device=rates.device,
        dtype=torch.float64,
    )
    thresholds, rates, live_times = torch.broadcast_tensors(
        thresholds,
        rates,
        live_times,
    )
    tau = _nonnegative_finite(dead_time_tau_s, name="dead_time_tau_s")
    invalid = torch.stack(
        (
            torch.any(~torch.isfinite(rates)),
            torch.any(rates < 0.0),
            torch.any(~torch.isfinite(live_times)),
            torch.any(live_times <= 0.0),
        )
    ).any()
    if bool(invalid.item()):
        raise ValueError("Renewal CDF rates and live times are invalid.")
    remaining = live_times - thresholds.to(torch.float64) * tau
    argument = rates * torch.clamp(remaining, min=0.0)
    cdf = torch.special.gammaincc(
        thresholds.to(torch.float64) + 1.0,
        argument,
    )
    cdf = torch.where(thresholds < 0, torch.zeros_like(cdf), cdf)
    cdf = torch.where(remaining <= 0.0, torch.ones_like(cdf), cdf)
    return torch.clamp(cdf, min=0.0, max=1.0)


def sample_nonparalyzable_counts_torch(
    incident_rates_cps: object,
    live_times_s: object,
    *,
    dead_time_tau_s: float,
    sample_count: int,
    generator: object,
) -> object:
    """Draw exact renewal totals by device-side inverse-CDF bisection."""
    torch = _torch_module()
    rates = _require_float64_tensor(
        incident_rates_cps,
        name="incident_rates_cps",
    )
    _validate_generator_device(generator, rates)
    live_times = torch.as_tensor(
        live_times_s,
        device=rates.device,
        dtype=torch.float64,
    )
    rates, live_times = torch.broadcast_tensors(rates, live_times)
    resolved_sample_count = _positive_integer(
        sample_count,
        name="sample_count",
    )
    tau = _nonnegative_finite(dead_time_tau_s, name="dead_time_tau_s")
    invalid = torch.stack(
        (
            torch.any(~torch.isfinite(rates)),
            torch.any(rates < 0.0),
            torch.any(~torch.isfinite(live_times)),
            torch.any(live_times <= 0.0),
        )
    ).any()
    if bool(invalid.item()):
        raise ValueError("Renewal sampling rates and live times are invalid.")
    sample_shape = tuple(rates.shape) + (resolved_sample_count,)
    expanded_rates = rates.unsqueeze(-1).expand(sample_shape)
    expanded_times = live_times.unsqueeze(-1).expand(sample_shape)
    if tau == 0.0:
        samples = torch.poisson(
            expanded_rates * expanded_times,
            generator=generator,
        )
        overflow = torch.any(samples > _MAX_EXACT_FLOAT64_INTEGER)
        if bool(overflow.item()):
            raise OverflowError("Poisson renewal draw exceeds exact int64 support.")
        return samples.to(torch.int64)

    support_float = torch.floor(expanded_times / tau) + 1.0
    support_invalid = torch.any(support_float > float(_MAX_EXACT_FLOAT64_INTEGER))
    if bool(support_invalid.item()):
        raise OverflowError(
            "Renewal count support exceeds exact float64 integer support."
        )
    support_maximum = support_float.to(torch.int64)
    uniform = torch.rand(
        sample_shape,
        device=rates.device,
        dtype=torch.float64,
        generator=generator,
    )
    poisson_mean = expanded_rates * expanded_times
    initial_high = torch.ceil(
        poisson_mean + 10.0 * torch.sqrt(poisson_mean + 1.0) + 10.0
    )
    high = torch.minimum(
        torch.clamp(initial_high, min=0.0).to(torch.int64),
        support_maximum,
    )
    high_cdf = nonparalyzable_count_cdf_torch(
        high,
        expanded_rates,
        expanded_times,
        dead_time_tau_s=tau,
    )
    unresolved = high_cdf < uniform
    for _ in range(64):
        if not bool(torch.any(unresolved).item()):
            break
        expanded_high = torch.minimum(
            torch.clamp(2 * high + 1, min=1),
            support_maximum,
        )
        stalled = unresolved & (expanded_high == high)
        if bool(torch.any(stalled).item()):
            raise RuntimeError(
                "Renewal inverse-CDF upper support failed to bracket a draw."
            )
        high = torch.where(unresolved, expanded_high, high)
        high_cdf = nonparalyzable_count_cdf_torch(
            high,
            expanded_rates,
            expanded_times,
            dead_time_tau_s=tau,
        )
        unresolved = high_cdf < uniform
    else:  # pragma: no cover - int64 support is bracketed within 64 doublings
        raise RuntimeError("Renewal inverse-CDF bracketing did not converge.")

    low = torch.full(sample_shape, -1, device=rates.device, dtype=torch.int64)
    for _ in range(64):
        active = high - low > 1
        if not bool(torch.any(active).item()):
            return high
        midpoint = low + (high - low) // 2
        cdf = nonparalyzable_count_cdf_torch(
            midpoint,
            expanded_rates,
            expanded_times,
            dead_time_tau_s=tau,
        )
        move_high = active & (cdf >= uniform)
        move_low = active & ~move_high
        high = torch.where(move_high, midpoint, high)
        low = torch.where(move_low, midpoint, low)
    raise RuntimeError("Renewal inverse-CDF integer bisection did not converge.")


def sample_standard_gamma_torch(
    concentration: object,
    *,
    generator: object,
) -> object:
    """Draw unit-scale Gamma variates through the required Torch kernel."""
    torch = _torch_module()
    alpha = _require_float64_tensor(concentration, name="concentration")
    _validate_generator_device(generator, alpha)
    invalid = torch.any(~torch.isfinite(alpha)) | torch.any(alpha <= 0.0)
    if bool(invalid.item()):
        raise ValueError("Gamma concentrations must be finite and positive.")
    if alpha.numel() == 0:
        return torch.empty_like(alpha)
    standard_gamma = getattr(torch, "_standard_gamma", None)
    if not callable(standard_gamma):
        raise RuntimeError(
            "Production predictive sampling requires torch._standard_gamma."
        )
    try:
        return standard_gamma(alpha, generator=generator)
    except TypeError as exc:
        raise RuntimeError(
            "Production predictive sampling requires generator-aware "
            "torch._standard_gamma."
        ) from exc


def sample_mean_one_gamma_torch(
    concentration: object,
    *,
    generator: object,
) -> object:
    """Draw mean-one Gamma scales for positive concentration parameters."""
    alpha = _require_float64_tensor(concentration, name="concentration")
    return (
        sample_standard_gamma_torch(
            alpha,
            generator=generator,
        )
        / alpha
    )


def sample_beta_torch(
    alpha: object,
    beta: object,
    *,
    generator: object,
) -> object:
    """Draw Beta variates as ratios of independent device Gamma draws."""
    torch = _torch_module()
    alpha_tensor = _require_float64_tensor(alpha, name="alpha")
    beta_tensor = torch.as_tensor(
        beta,
        device=alpha_tensor.device,
        dtype=torch.float64,
    )
    alpha_tensor, beta_tensor = torch.broadcast_tensors(
        alpha_tensor,
        beta_tensor,
    )
    first = sample_standard_gamma_torch(
        alpha_tensor,
        generator=generator,
    )
    second = sample_standard_gamma_torch(
        beta_tensor,
        generator=generator,
    )
    denominator = first + second
    return first / torch.clamp(
        denominator,
        min=torch.finfo(torch.float64).tiny,
    )


def sample_multinomial_counts_torch(
    totals: object,
    probabilities: object,
    *,
    generator: object,
) -> object:
    """Draw variable-total multinomials by a balanced conditional-binomial tree.

    A multinomial over ``B`` bins factorizes exactly into conditional binomials
    on a balanced binary partition of the bins.  The implementation therefore
    supports a different total for every row with only ``ceil(log2(B))``
    sequential distribution calls and no event-level count expansion.
    """
    torch = _torch_module()
    probability = _require_float64_tensor(
        probabilities,
        name="probabilities",
    )
    _validate_generator_device(generator, probability)
    count = torch.as_tensor(totals, device=probability.device)
    if count.dtype != torch.int64:
        raise TypeError("totals must be a torch.int64 tensor.")
    if probability.ndim < 1 or tuple(count.shape) != tuple(probability.shape[:-1]):
        raise ValueError(
            "Multinomial totals must match every probability axis except bins."
        )
    bin_count = int(probability.shape[-1])
    if bin_count <= 0:
        raise ValueError("Multinomial probabilities require at least one bin.")
    mass = torch.sum(probability, dim=-1)
    invalid = torch.stack(
        (
            torch.any(count < 0),
            torch.any(count > _MAX_EXACT_FLOAT64_INTEGER),
            torch.any(~torch.isfinite(probability)),
            torch.any(probability < 0.0),
            torch.any((count > 0) & (mass <= 0.0)),
        )
    ).any()
    if bool(invalid.item()):
        raise ValueError(
            "Multinomial totals or probabilities are outside exact support."
        )

    padded_bin_count = 1 << (bin_count - 1).bit_length()
    if padded_bin_count != bin_count:
        padding = torch.zeros(
            tuple(probability.shape[:-1]) + (padded_bin_count - bin_count,),
            device=probability.device,
            dtype=torch.float64,
        )
        probability = torch.cat((probability, padding), dim=-1)
    probability_levels = [probability]
    while int(probability_levels[-1].shape[-1]) > 1:
        current = probability_levels[-1]
        probability_levels.append(
            current.reshape(tuple(current.shape[:-1]) + (-1, 2)).sum(dim=-1)
        )

    level_counts = count.to(torch.float64).unsqueeze(-1)
    for depth in range(len(probability_levels) - 1, 0, -1):
        children = probability_levels[depth - 1]
        left_mass = children[..., 0::2]
        right_mass = children[..., 1::2]
        parent_mass = left_mass + right_mass
        left_probability = torch.where(
            parent_mass > 0.0,
            left_mass
            / torch.clamp(
                parent_mass,
                min=torch.finfo(torch.float64).tiny,
            ),
            torch.zeros_like(parent_mass),
        )
        left_count = torch.binomial(
            level_counts,
            left_probability,
            generator=generator,
        )
        level_counts = torch.stack(
            (left_count, level_counts - left_count),
            dim=-1,
        ).flatten(start_dim=-2)
    result = level_counts[..., :bin_count].to(torch.int64)
    if bool(torch.any(torch.sum(result, dim=-1) != count).item()):
        raise RuntimeError("Balanced multinomial sampling lost row totals.")
    return result


@dataclass(frozen=True)
class ComponentTreeMarkParameters:
    """Define the component-aware production mark-partition tree."""

    leaf_group_mask_hb: object
    tree_left_mask_tb: object
    tree_right_mask_tb: object
    tree_depth_t: object
    tree_left_child_t: object
    tree_right_child_t: object
    tree_concentration_xvt: object
    leaf_concentration_xvh: object


def _safe_dirichlet_probabilities(
    probabilities: object,
    *,
    concentration: float,
    generator: object,
) -> object:
    """Draw Dirichlet probabilities while preserving zero-support entries."""
    torch = _torch_module()
    probability = _require_float64_tensor(
        probabilities,
        name="probabilities",
    )
    resolved_concentration = _positive_finite(
        concentration,
        name="concentration",
    )
    alpha = probability * resolved_concentration
    positive = alpha > 0.0
    safe_alpha = torch.where(positive, alpha, torch.ones_like(alpha))
    gamma_draw = sample_standard_gamma_torch(
        safe_alpha,
        generator=generator,
    )
    gamma_draw = torch.where(positive, gamma_draw, torch.zeros_like(gamma_draw))
    gamma_sum = torch.sum(gamma_draw, dim=-1, keepdim=True)
    return torch.where(
        gamma_sum > 0.0,
        gamma_draw / torch.clamp(gamma_sum, min=torch.finfo(torch.float64).tiny),
        probability,
    )


def _sample_component_tree_marks_torch(
    totals: object,
    probabilities: object,
    parameters: ComponentTreeMarkParameters,
    *,
    generator: object,
) -> object:
    """Draw component-aware Dirichlet-tree marks in batched device form."""
    torch = _torch_module()
    probability = _require_float64_tensor(
        probabilities,
        name="probabilities",
    )
    total = torch.as_tensor(totals, device=probability.device, dtype=torch.int64)
    leaf_masks = torch.as_tensor(
        parameters.leaf_group_mask_hb,
        device=probability.device,
        dtype=torch.float64,
    )
    left_masks = torch.as_tensor(
        parameters.tree_left_mask_tb,
        device=probability.device,
        dtype=torch.float64,
    )
    right_masks = torch.as_tensor(
        parameters.tree_right_mask_tb,
        device=probability.device,
        dtype=torch.float64,
    )
    depths = torch.as_tensor(
        parameters.tree_depth_t,
        device=probability.device,
        dtype=torch.long,
    )
    left_children = torch.as_tensor(
        parameters.tree_left_child_t,
        device=probability.device,
        dtype=torch.long,
    )
    right_children = torch.as_tensor(
        parameters.tree_right_child_t,
        device=probability.device,
        dtype=torch.long,
    )
    if (
        leaf_masks.ndim != 2
        or left_masks.ndim != 2
        or tuple(right_masks.shape) != tuple(left_masks.shape)
        or int(leaf_masks.shape[1]) != int(probability.shape[-1])
        or int(left_masks.shape[1]) != int(probability.shape[-1])
        or tuple(depths.shape) != (int(left_masks.shape[0]),)
        or tuple(left_children.shape) != tuple(depths.shape)
        or tuple(right_children.shape) != tuple(depths.shape)
    ):
        raise ValueError("Component-tree mark topology is not bin-aligned.")
    leaf_membership = torch.sum(leaf_masks, dim=0)
    node_partition = torch.sum(left_masks + right_masks, dim=1)
    mask_invalid = torch.stack(
        (
            torch.any(~torch.isfinite(leaf_masks)),
            torch.any(leaf_masks < 0.0),
            torch.any(torch.abs(leaf_membership - 1.0) > 1.0e-12),
            torch.any(~torch.isfinite(left_masks)),
            torch.any(~torch.isfinite(right_masks)),
            torch.any(left_masks < 0.0),
            torch.any(right_masks < 0.0),
            torch.any(node_partition <= 0.0),
            depths[0] != 0,
        )
    ).any()
    if bool(mask_invalid.item()):
        raise ValueError("Component-tree mark partition masks are invalid.")

    raw_tree_concentration = torch.as_tensor(
        parameters.tree_concentration_xvt,
        device=probability.device,
        dtype=torch.float64,
    )
    raw_leaf_concentration = torch.as_tensor(
        parameters.leaf_concentration_xvh,
        device=probability.device,
        dtype=torch.float64,
    )
    try:
        tree_concentration = torch.broadcast_to(
            raw_tree_concentration.unsqueeze(-3),
            probability.shape[:-1] + (int(left_masks.shape[0]),),
        )
        leaf_concentration = torch.broadcast_to(
            raw_leaf_concentration.unsqueeze(-3),
            probability.shape[:-1] + (int(leaf_masks.shape[0]),),
        )
    except RuntimeError as error:
        raise ValueError(
            "Component-tree concentrations are not state/view aligned."
        ) from error
    concentration_invalid = torch.stack(
        (
            torch.any(~torch.isfinite(tree_concentration)),
            torch.any(tree_concentration <= 0.0),
            torch.any(~torch.isfinite(leaf_concentration)),
            torch.any(leaf_concentration <= 0.0),
        )
    ).any()
    if bool(concentration_invalid.item()):
        raise ValueError("Component-tree concentrations must be positive.")

    left_mass = torch.einsum("...b,tb->...t", probability, left_masks)
    right_mass = torch.einsum("...b,tb->...t", probability, right_masks)
    parent_mass = left_mass + right_mass
    branch_probability = torch.where(
        parent_mass > 0.0,
        left_mass
        / torch.clamp(parent_mass, min=torch.finfo(torch.float64).tiny),
        torch.zeros_like(parent_mass),
    )
    interior = (branch_probability > 0.0) & (branch_probability < 1.0)
    safe_alpha = torch.where(
        interior,
        tree_concentration * branch_probability,
        torch.ones_like(branch_probability),
    )
    safe_beta = torch.where(
        interior,
        tree_concentration * (1.0 - branch_probability),
        torch.ones_like(branch_probability),
    )
    random_branch_probability = sample_beta_torch(
        safe_alpha,
        safe_beta,
        generator=generator,
    )
    sampled_branch_probability = torch.where(
        interior,
        random_branch_probability,
        branch_probability,
    )
    node_mass = torch.zeros_like(sampled_branch_probability)
    node_mass[..., 0] = 1.0
    leaf_mass = torch.zeros(
        probability.shape[:-1] + (int(leaf_masks.shape[0]),),
        device=probability.device,
        dtype=torch.float64,
    )
    maximum_depth = int(torch.max(depths).item())
    for depth in range(maximum_depth + 1):
        node_ids = torch.nonzero(depths == depth, as_tuple=False).flatten()
        if int(node_ids.numel()) == 0:
            continue
        parent = node_mass[..., node_ids]
        left_value = parent * sampled_branch_probability[..., node_ids]
        right_value = parent - left_value
        left_target = left_children[node_ids]
        right_target = right_children[node_ids]
        left_nodes = left_target >= 0
        right_nodes = right_target >= 0
        if bool(torch.any(left_nodes).item()):
            node_mass[..., left_target[left_nodes]] = left_value[..., left_nodes]
        if bool(torch.any(right_nodes).item()):
            node_mass[..., right_target[right_nodes]] = right_value[..., right_nodes]
        if bool(torch.any(~left_nodes).item()):
            leaf_mass[..., -left_target[~left_nodes] - 1] = left_value[..., ~left_nodes]
        if bool(torch.any(~right_nodes).item()):
            leaf_mass[..., -right_target[~right_nodes] - 1] = right_value[
                ..., ~right_nodes
            ]

    base_leaf_mass = torch.einsum("...b,hb->...h", probability, leaf_masks)
    mapped_base_leaf_mass = torch.einsum(
        "...h,hb->...b",
        base_leaf_mass,
        leaf_masks,
    )
    within_probability = torch.where(
        mapped_base_leaf_mass > 0.0,
        probability
        / torch.clamp(
            mapped_base_leaf_mass,
            min=torch.finfo(torch.float64).tiny,
        ),
        torch.zeros_like(probability),
    )
    mapped_leaf_concentration = torch.einsum(
        "...h,hb->...b",
        leaf_concentration,
        leaf_masks,
    )
    within_alpha = within_probability * mapped_leaf_concentration
    positive_within = within_alpha > 0.0
    within_gamma = sample_standard_gamma_torch(
        torch.where(
            positive_within,
            within_alpha,
            torch.ones_like(within_alpha),
        ),
        generator=generator,
    )
    within_gamma = torch.where(
        positive_within,
        within_gamma,
        torch.zeros_like(within_gamma),
    )
    within_group_sums = torch.einsum(
        "...b,hb->...h",
        within_gamma,
        leaf_masks,
    )
    within_sum_by_bin = torch.einsum(
        "...h,hb->...b",
        within_group_sums,
        leaf_masks,
    )
    random_within_probabilities = torch.where(
        within_sum_by_bin > 0.0,
        within_gamma
        / torch.clamp(
            within_sum_by_bin,
            min=torch.finfo(torch.float64).tiny,
        ),
        within_probability,
    )
    random_leaf_mass_by_bin = torch.einsum(
        "...h,hb->...b",
        leaf_mass,
        leaf_masks,
    )
    random_probability = random_leaf_mass_by_bin * random_within_probabilities
    normalization = torch.sum(random_probability, dim=-1, keepdim=True)
    random_probability = torch.where(
        normalization > 0.0,
        random_probability
        / torch.clamp(normalization, min=torch.finfo(torch.float64).tiny),
        probability,
    )
    return sample_multinomial_counts_torch(
        total,
        random_probability,
        generator=generator,
    )


def sample_predictive_action_torch(
    source_mean_xvb: object,
    background_mean_xvb: object,
    live_times_s_v: object,
    *,
    sample_count: int,
    generator: object,
    rate_scale_nodes_j: object,
    rate_scale_weights_j: object,
    dead_time_tau_s: float,
    mark_model: str,
    count_scope: str = "renewal",
    count_concentration_xv: object | None = None,
    mark_concentration_xv: object | None = None,
    hierarchical_marks: ComponentTreeMarkParameters | None = None,
) -> object:
    """Draw one action's future spectra from prepared pre-dead-time means.

    ``source_mean_xvb`` and ``background_mean_xvb`` end in view/bin axes.
    Any leading axes are retained and one predictive-sample axis is inserted
    immediately before the view axis.  Rate-scale draws are shared across all
    views of one station, while view-independent count discrepancy is sampled
    separately per view.  ``mark_model`` is mandatory so unsupported retired
    marking branches cannot silently fall back to another distribution.
    """
    torch = _torch_module()
    source_mean = _require_float64_tensor(
        source_mean_xvb,
        name="source_mean_xvb",
    )
    _validate_generator_device(generator, source_mean)
    background_mean = torch.as_tensor(
        background_mean_xvb,
        device=source_mean.device,
        dtype=torch.float64,
    )
    live_times = torch.as_tensor(
        live_times_s_v,
        device=source_mean.device,
        dtype=torch.float64,
    )
    resolved_sample_count = _positive_integer(
        sample_count,
        name="sample_count",
    )
    tau = _nonnegative_finite(dead_time_tau_s, name="dead_time_tau_s")
    if mark_model not in (
        "fixed_multinomial",
        "fraction_dirichlet_multinomial",
        "component_dirichlet_tree_hierarchical",
    ):
        raise ValueError(f"Unsupported predictive mark_model: {mark_model!r}.")
    if (mark_model == "component_dirichlet_tree_hierarchical") != (
        hierarchical_marks is not None
    ):
        raise ValueError(
            "Hierarchical mark parameters must exactly match the hierarchical "
            "mark model."
        )
    if (mark_model == "fraction_dirichlet_multinomial") != (
        mark_concentration_xv is not None
    ):
        raise ValueError(
            "Mark concentrations must exactly match the fraction-Dirichlet mark model."
        )
    if (
        source_mean.ndim < 2
        or tuple(background_mean.shape) != tuple(source_mean.shape)
        or tuple(live_times.shape) != (int(source_mean.shape[-2]),)
    ):
        raise ValueError("Prepared predictive means are not view/bin aligned.")
    invalid = torch.stack(
        (
            torch.any(~torch.isfinite(source_mean)),
            torch.any(source_mean < 0.0),
            torch.any(~torch.isfinite(background_mean)),
            torch.any(background_mean < 0.0),
            torch.any(~torch.isfinite(live_times)),
            torch.any(live_times <= 0.0),
        )
    ).any()
    if bool(invalid.item()):
        raise ValueError("Prepared predictive means contain invalid values.")

    nodes = torch.as_tensor(
        rate_scale_nodes_j,
        device=source_mean.device,
        dtype=torch.float64,
    ).reshape(-1)
    weights = torch.as_tensor(
        rate_scale_weights_j,
        device=source_mean.device,
        dtype=torch.float64,
    ).reshape(-1)
    mixture_invalid = torch.stack(
        (
            torch.any(~torch.isfinite(nodes)),
            torch.any(nodes < 0.0),
            torch.any(~torch.isfinite(weights)),
            torch.any(weights < 0.0),
            torch.sum(weights) <= 0.0,
        )
    ).any()
    if (
        nodes.numel() == 0
        or tuple(weights.shape) != tuple(nodes.shape)
        or bool(mixture_invalid.item())
    ):
        raise ValueError("Rate-scale mixture nodes and weights are invalid.")
    normalized_weights = weights / torch.sum(weights)
    cumulative_weights = torch.cumsum(normalized_weights, dim=0)
    cumulative_weights[-1] = 1.0
    leading_shape = tuple(source_mean.shape[:-2])
    scale_shape = leading_shape + (resolved_sample_count,)
    uniform_scale = torch.rand(
        scale_shape,
        device=source_mean.device,
        dtype=torch.float64,
        generator=generator,
    )
    node_indices = torch.searchsorted(
        cumulative_weights,
        uniform_scale.contiguous(),
        right=False,
    ).clamp(max=int(nodes.numel()) - 1)
    sampled_scale = nodes[node_indices]
    node_source = source_mean.unsqueeze(-3) * sampled_scale.unsqueeze(-1).unsqueeze(-1)
    pre_mean = background_mean.unsqueeze(-3) + node_source
    pre_total = torch.sum(pre_mean, dim=-1)
    live = live_times.reshape((1,) * (pre_total.ndim - 1) + (-1,))
    mark_pre_mean = pre_mean

    if count_scope == "renewal":
        if count_concentration_xv is not None:
            raise ValueError("Renewal counts do not accept Gamma concentration.")
        totals = sample_nonparalyzable_counts_torch(
            pre_total / live,
            torch.broadcast_to(live, pre_total.shape),
            dead_time_tau_s=tau,
            sample_count=1,
            generator=generator,
        ).squeeze(-1)
    elif count_scope in (
        "view_independent_gamma_poisson",
        "station_shared_gamma_poisson",
    ):
        if count_concentration_xv is None:
            raise ValueError("Gamma-Poisson counts require concentration.")
        recorded_total_mean = pre_total / (1.0 + pre_total / live * tau)
        raw_concentration = torch.as_tensor(
            count_concentration_xv,
            device=source_mean.device,
            dtype=torch.float64,
        )
        try:
            if count_scope == "view_independent_gamma_poisson":
                base_concentration = torch.broadcast_to(
                    raw_concentration,
                    leading_shape + (int(source_mean.shape[-2]),),
                )
                concentration = torch.broadcast_to(
                    base_concentration.unsqueeze(-2),
                    pre_total.shape,
                )
                count_scale = sample_mean_one_gamma_torch(
                    concentration,
                    generator=generator,
                )
            else:
                base_concentration = torch.broadcast_to(
                    raw_concentration,
                    leading_shape,
                )
                concentration = torch.broadcast_to(
                    base_concentration.unsqueeze(-1),
                    leading_shape + (resolved_sample_count,),
                )
                count_scale = sample_mean_one_gamma_torch(
                    concentration,
                    generator=generator,
                ).unsqueeze(-1)
        except RuntimeError as error:
            raise ValueError(
                "Gamma-Poisson concentrations are not state/view aligned."
            ) from error
        totals_float = torch.poisson(
            recorded_total_mean * count_scale,
            generator=generator,
        )
        count_invalid = torch.any(~torch.isfinite(totals_float)) | torch.any(
            totals_float > _MAX_EXACT_FLOAT64_INTEGER
        )
        if bool(count_invalid.item()):
            raise OverflowError("Gamma-Poisson draw exceeds exact int64 support.")
        totals = totals_float.to(torch.int64)
    else:
        raise ValueError(f"Unsupported predictive count_scope: {count_scope!r}.")

    if mark_model == "component_dirichlet_tree_hierarchical":
        source_active = torch.sum(node_source, dim=-1) > 0.0
        if bool(torch.any(~source_active).item()):
            background_only_totals = sample_nonparalyzable_counts_torch(
                pre_total / live,
                torch.broadcast_to(live, pre_total.shape),
                dead_time_tau_s=tau,
                sample_count=1,
                generator=generator,
            ).squeeze(-1)
            totals = torch.where(
                source_active,
                totals,
                background_only_totals,
            )

    mark_total = torch.sum(mark_pre_mean, dim=-1)
    probabilities = torch.where(
        mark_total.unsqueeze(-1) > 0.0,
        mark_pre_mean
        / torch.clamp(
            mark_total.unsqueeze(-1),
            min=torch.finfo(torch.float64).tiny,
        ),
        torch.zeros_like(mark_pre_mean),
    )
    zero_rate = pre_total <= 0.0
    zero_count_sentinel = torch.zeros_like(probabilities)
    zero_count_sentinel[..., 0] = 1.0
    probabilities = torch.where(
        zero_rate.unsqueeze(-1),
        zero_count_sentinel,
        probabilities,
    )
    if mark_model == "fraction_dirichlet_multinomial":
        raw_mark_concentration = torch.as_tensor(
            mark_concentration_xv,
            device=source_mean.device,
            dtype=torch.float64,
        )
        try:
            base_mark_concentration = torch.broadcast_to(
                raw_mark_concentration,
                leading_shape + (int(source_mean.shape[-2]),),
            )
            base_mark_concentration = torch.broadcast_to(
                base_mark_concentration.unsqueeze(-2),
                pre_total.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "Mark concentrations are not state/view aligned."
            ) from error
        source_total = torch.sum(node_source, dim=-1)
        source_fraction = torch.where(
            pre_total > 0.0,
            source_total
            / torch.clamp(
                pre_total,
                min=torch.finfo(torch.float64).tiny,
            ),
            torch.zeros_like(source_total),
        )
        concentration = base_mark_concentration / torch.clamp(
            torch.square(source_fraction),
            min=1.0e-12,
        )
        alpha = probabilities * concentration.unsqueeze(-1)
        positive_alpha = alpha > 0.0
        gamma_draws = sample_standard_gamma_torch(
            torch.where(
                positive_alpha,
                alpha,
                torch.ones_like(alpha),
            ),
            generator=generator,
        )
        gamma_draws = torch.where(
            positive_alpha,
            gamma_draws,
            torch.zeros_like(gamma_draws),
        )
        gamma_sum = torch.sum(gamma_draws, dim=-1, keepdim=True)
        random_probabilities = torch.where(
            gamma_sum > 0.0,
            gamma_draws
            / torch.clamp(
                gamma_sum,
                min=torch.finfo(torch.float64).tiny,
            ),
            probabilities,
        )
        probabilities = torch.where(
            source_fraction.unsqueeze(-1) > 0.0,
            random_probabilities,
            probabilities,
        )
    if mark_model in (
        "fixed_multinomial",
        "fraction_dirichlet_multinomial",
    ):
        samples = sample_multinomial_counts_torch(
            totals,
            probabilities,
            generator=generator,
        )
    else:
        if hierarchical_marks is None:  # pragma: no cover - checked above
            raise RuntimeError("Hierarchical mark parameters disappeared.")
        samples = _sample_component_tree_marks_torch(
            totals,
            probabilities,
            hierarchical_marks,
            generator=generator,
        )
    if samples.dtype != torch.int64 or tuple(samples.shape) != tuple(
        probabilities.shape
    ):
        raise RuntimeError("Torch predictive samples are not int64 and aligned.")
    return samples


def sample_action_seeded_torch(
    action_seeds_a: Sequence[Integral],
    *,
    reference: object,
    sampler: Callable[[int, object], object],
    maximum_action_count: int,
) -> object:
    """Schedule bounded, independently seeded action samplers on one device.

    The loop schedules one bulk-device sampler per action; it never iterates
    over particles, samples, views, or spectrum bins.  Reconstructing each
    generator from the canonical action seed makes output invariant to outer
    action chunking and ordering.
    """
    torch = _torch_module()
    tensor = torch.as_tensor(reference)
    limit = _positive_integer(
        maximum_action_count,
        name="maximum_action_count",
    )
    try:
        raw_seeds = tuple(action_seeds_a)
        seeds = tuple(int(seed) for seed in raw_seeds)
    except (TypeError, ValueError) as error:
        raise TypeError("action_seeds_a must be an integer sequence.") from error
    if not seeds:
        raise ValueError("action_seeds_a must not be empty.")
    if len(seeds) > limit:
        raise ValueError("Action-seeded scheduling exceeds its declared bound.")
    if any(
        isinstance(raw_seed, bool) or not isinstance(raw_seed, Integral)
        for raw_seed in raw_seeds
    ):
        raise TypeError("action_seeds_a must contain integers only.")
    outputs = []
    for action_index, raw_seed in enumerate(seeds):
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(int(raw_seed) & _UINT64_MASK)
        output = sampler(action_index, generator)
        if not isinstance(output, torch.Tensor):
            raise TypeError("An action sampler must return a Torch tensor.")
        if output.device != tensor.device:
            raise ValueError("An action sampler returned a tensor on another device.")
        outputs.append(output)
    first_shape = tuple(outputs[0].shape)
    if any(tuple(output.shape) != first_shape for output in outputs):
        raise ValueError("Action samplers must return one common output shape.")
    return torch.stack(outputs, dim=0)


def sample_geometry_conditioned_predictive_torch(
    model: object,
    total_line_contributions_xvsl: object,
    uncollided_line_contributions_xvsl: object,
    transport_features_xvslf: object,
    live_times_s_v: object,
    *,
    sample_count: int,
    generator: object | None = None,
    action_seeds_a: object | None = None,
) -> object:
    """Prepare and sample every geometry-conditioned predictive branch.

    This adapter deliberately owns the Torch sampling orchestration so the
    shared model class needs only a thin delegate.  It consumes the model's
    existing Torch mean and concentration APIs, preserving the exact branch
    selection and latent-variable meanings of ``sample_predictive_numpy``.
    """
    torch = _torch_module()
    total = torch.as_tensor(total_line_contributions_xvsl)
    if action_seeds_a is None and generator is None:
        raise ValueError(
            "Torch predictive sampling requires a generator or action seeds."
        )
    if action_seeds_a is not None and total.ndim < 4:
        raise ValueError("Torch action seeds require a leading action transport axis.")

    component = model.physical_component_discrepancy
    component_tree_marks = bool(
        component is not None
        and component.mark_latent_model == "component_dirichlet_tree_hierarchical"
    )
    direct_mean = None
    scatter_mean = None
    if component_tree_marks:
        direct_mean, scatter_mean, background_mean = model._pre_dead_time_mean_torch(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_physical_components=True,
        )
        source_mean = direct_mean + scatter_mean
    else:
        source_mean, background_mean = model._pre_dead_time_mean_torch(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
            live_times_s_v,
            return_components=True,
        )

    rate_nodes, rate_weights, peak_mask, continuum_groups = (
        model._torch_likelihood_constants(source_mean)
    )
    if component is not None:
        count_scope = "view_independent_gamma_poisson"
        count_concentration = model._component_count_concentration_torch(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
            transport_features_xvslf,
        )
        count_concentration_has_action_axis = action_seeds_a is not None
    elif model.count_discrepancy_concentration is not None:
        count_scope = f"{model.count_discrepancy_scope}_gamma_poisson"
        count_concentration = torch.as_tensor(
            float(model.count_discrepancy_concentration),
            device=source_mean.device,
            dtype=torch.float64,
        )
        count_concentration_has_action_axis = False
    else:
        count_scope = "renewal"
        count_concentration = None
        count_concentration_has_action_axis = False

    mark_concentration = None
    hierarchy_concentration = None
    hierarchy = None
    if component_tree_marks:
        if component is None:  # pragma: no cover - implied above
            raise RuntimeError("Hierarchical component settings disappeared.")
        mark_model = "component_dirichlet_tree_hierarchical"
        component_means = torch.stack(
            (direct_mean, scatter_mean, background_mean),
            dim=-2,
        ).unsqueeze(-4)
        raw_tree_concentration, raw_leaf_concentration = (
            model._component_tree_mark_concentrations_torch(
                total_line_contributions_xvsl,
                uncollided_line_contributions_xvsl,
                component_means,
            )
        )
        tree_concentration = raw_tree_concentration.squeeze(-3)
        leaf_concentration = raw_leaf_concentration.squeeze(-3)
        hierarchy_concentration = (tree_concentration, leaf_concentration)
        (
            leaf_masks,
            left_masks,
            right_masks,
            _domains,
            depths,
            left_children,
            right_children,
        ) = model._torch_mark_tree_constants(source_mean)
        hierarchy = ComponentTreeMarkParameters(
            leaf_group_mask_hb=leaf_masks,
            tree_left_mask_tb=left_masks,
            tree_right_mask_tb=right_masks,
            tree_depth_t=depths,
            tree_left_child_t=left_children,
            tree_right_child_t=right_children,
            tree_concentration_xvt=tree_concentration,
            leaf_concentration_xvh=leaf_concentration,
        )
    else:
        mark_model = "fraction_dirichlet_multinomial"
        mark_concentration = model._base_mark_concentration_torch(
            total_line_contributions_xvsl,
            uncollided_line_contributions_xvsl,
        )

    def _sample_prepared_action(
        action_index: int | None,
        action_generator: object,
    ) -> object:
        """Sample one prepared action or the complete unseeded batch."""
        selected_source = (
            source_mean if action_index is None else source_mean[action_index]
        )
        selected_background = (
            background_mean if action_index is None else background_mean[action_index]
        )
        selected_count_concentration = count_concentration
        if (
            action_index is not None
            and count_concentration_has_action_axis
            and count_concentration is not None
        ):
            selected_count_concentration = count_concentration[action_index]
        selected_mark_concentration = mark_concentration
        if action_index is not None and mark_concentration is not None:
            selected_mark_concentration = mark_concentration[action_index]
        selected_hierarchy = hierarchy
        if hierarchy is not None:
            if hierarchy_concentration is None:
                raise RuntimeError("Component-tree concentrations disappeared.")
            selected_tree_concentration = hierarchy_concentration[0]
            selected_leaf_concentration = hierarchy_concentration[1]
            if action_index is not None:
                selected_tree_concentration = selected_tree_concentration[action_index]
                selected_leaf_concentration = selected_leaf_concentration[action_index]
            selected_hierarchy = ComponentTreeMarkParameters(
                leaf_group_mask_hb=hierarchy.leaf_group_mask_hb,
                tree_left_mask_tb=hierarchy.tree_left_mask_tb,
                tree_right_mask_tb=hierarchy.tree_right_mask_tb,
                tree_depth_t=hierarchy.tree_depth_t,
                tree_left_child_t=hierarchy.tree_left_child_t,
                tree_right_child_t=hierarchy.tree_right_child_t,
                tree_concentration_xvt=selected_tree_concentration,
                leaf_concentration_xvh=selected_leaf_concentration,
            )
        return sample_predictive_action_torch(
            selected_source,
            selected_background,
            live_times_s_v,
            sample_count=sample_count,
            generator=action_generator,
            rate_scale_nodes_j=rate_nodes,
            rate_scale_weights_j=rate_weights,
            dead_time_tau_s=float(model.dead_time_tau_s),
            mark_model=mark_model,
            count_scope=count_scope,
            count_concentration_xv=selected_count_concentration,
            mark_concentration_xv=selected_mark_concentration,
            hierarchical_marks=selected_hierarchy,
        )

    if action_seeds_a is None:
        return _sample_prepared_action(None, generator)
    try:
        resolved_action_seeds = tuple(action_seeds_a)
    except TypeError as error:
        raise TypeError("action_seeds_a must be an integer sequence.") from error
    if len(resolved_action_seeds) != int(source_mean.shape[0]):
        raise ValueError(
            "action_seeds_a must provide one seed for the leading action axis."
        )

    def _sample_seeded_action(
        action_index: int,
        action_generator: object,
    ) -> object:
        """Sample one canonical action stream from prepared tensors."""
        return _sample_prepared_action(action_index, action_generator)

    return sample_action_seeded_torch(
        resolved_action_seeds,
        reference=source_mean,
        sampler=_sample_seeded_action,
        maximum_action_count=int(source_mean.shape[0]),
    )
