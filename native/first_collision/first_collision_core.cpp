#include "first_collision_core.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace rotating_shield::calibration::first_collision {
namespace {

void RequireFiniteNonnegative(
    const double value,
    const std::string& field
) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            field + " must be finite and nonnegative."
        );
    }
}

void RequireUnitInterval(
    const double value,
    const std::string& field
) {
    if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
        throw std::invalid_argument(
            field + " must lie in the half-open interval [0, 1)."
        );
    }
}

std::size_t SelectInteractionChannel(
    const MaterialSegment& segment,
    const double total_cross_section_per_m,
    const double process_unit_interval
) {
    RequireUnitInterval(
        process_unit_interval,
        "process_unit_interval"
    );
    const double target = process_unit_interval
        * total_cross_section_per_m;
    double cumulative = 0.0;
    std::size_t last_positive = segment.channels.size();
    for (
        std::size_t index = 0;
        index < segment.channels.size();
        ++index
    ) {
        const double cross_section = (
            segment.channels[index].macroscopic_cross_section_per_m
        );
        if (cross_section <= 0.0) {
            continue;
        }
        last_positive = index;
        cumulative += cross_section;
        if (target < cumulative) {
            return index;
        }
    }
    if (last_positive < segment.channels.size()) {
        return last_positive;
    }
    throw std::invalid_argument(
        "A collidable segment has no positive interaction channel."
    );
}

}  // namespace

void RequireCalibrationOnly(const bool calibration_only) {
    if (!calibration_only) {
        throw std::runtime_error(
            "First-collision variance reduction is calibration-only and is "
            "not integrated with the standard Geant4 runtime."
        );
    }
}

double SegmentMacroscopicCrossSectionPerM(
    const MaterialSegment& segment
) {
    RequireFiniteNonnegative(segment.length_m, "segment.length_m");
    double total = 0.0;
    std::set<std::uint64_t> process_tokens;
    for (const auto& channel : segment.channels) {
        if (!process_tokens.insert(channel.process_token).second) {
            throw std::invalid_argument(
                "Interaction process tokens must be unique within a segment."
            );
        }
        RequireFiniteNonnegative(
            channel.macroscopic_cross_section_per_m,
            "channel.macroscopic_cross_section_per_m"
        );
        if (
            channel.macroscopic_cross_section_per_m
                > std::numeric_limits<double>::max() - total
        ) {
            throw std::overflow_error(
                "Segment macroscopic cross section overflowed."
            );
        }
        total += channel.macroscopic_cross_section_per_m;
    }
    if (!std::isfinite(total)) {
        throw std::overflow_error(
            "Segment macroscopic cross section is not finite."
        );
    }
    return total;
}

double SegmentOpticalDepth(const MaterialSegment& segment) {
    const double cross_section = SegmentMacroscopicCrossSectionPerM(
        segment
    );
    const double optical_depth = cross_section * segment.length_m;
    if (!std::isfinite(optical_depth)) {
        throw std::overflow_error("Segment optical depth is not finite.");
    }
    return optical_depth;
}

double NoCollisionProbability(const double optical_depth) {
    RequireFiniteNonnegative(optical_depth, "optical_depth");
    return std::exp(-optical_depth);
}

double CollisionProbability(const double optical_depth) {
    RequireFiniteNonnegative(optical_depth, "optical_depth");
    return -std::expm1(-optical_depth);
}

FirstCollisionLocation SampleConditionalFirstCollision(
    const MaterialSegment& segment,
    const std::size_t segment_index,
    const SegmentRandomDraw& draw
) {
    RequireUnitInterval(
        draw.collision_unit_interval,
        "collision_unit_interval"
    );
    const double total_cross_section = (
        SegmentMacroscopicCrossSectionPerM(segment)
    );
    const double optical_depth = total_cross_section * segment.length_m;
    const double collision_probability = CollisionProbability(
        optical_depth
    );
    if (!(total_cross_section > 0.0 && collision_probability > 0.0)) {
        throw std::invalid_argument(
            "Conditional first-collision sampling requires positive optical "
            "depth."
        );
    }
    double sampled_optical_depth = -std::log1p(
        -draw.collision_unit_interval * collision_probability
    );
    sampled_optical_depth = std::clamp(
        sampled_optical_depth,
        0.0,
        optical_depth
    );
    const double distance_m = std::clamp(
        sampled_optical_depth / total_cross_section,
        0.0,
        segment.length_m
    );
    const std::size_t process_index = SelectInteractionChannel(
        segment,
        total_cross_section,
        draw.process_unit_interval
    );
    const auto& channel = segment.channels[process_index];
    const double survival_to_collision = std::exp(
        -sampled_optical_depth
    );
    const double local_analog_density = (
        total_cross_section * survival_to_collision
    );

    FirstCollisionLocation location;
    location.segment_index = segment_index;
    location.segment_token = segment.segment_token;
    location.distance_from_segment_start_m = distance_m;
    location.optical_depth_from_segment_start = sampled_optical_depth;
    location.process_token = channel.process_token;
    location.process_probability = (
        channel.macroscopic_cross_section_per_m / total_cross_section
    );
    location.conditional_density_per_m = (
        local_analog_density / collision_probability
    );
    location.local_analog_density_per_m = local_analog_density;
    return location;
}

ForcedCollisionResult SplitHeterogeneousPath(
    const std::vector<MaterialSegment>& segments,
    const std::vector<SegmentRandomDraw>& draws,
    const std::vector<SegmentLineageIds>& lineage_ids,
    const std::uint64_t primary_history_id,
    const std::uint64_t initial_lineage_id,
    const double initial_weight,
    const bool calibration_only
) {
    RequireCalibrationOnly(calibration_only);
    if (segments.empty()) {
        throw std::invalid_argument(
            "A heterogeneous path must contain at least one segment."
        );
    }
    if (draws.size() != segments.size()) {
        throw std::invalid_argument(
            "Each path segment requires one random draw pair."
        );
    }
    if (lineage_ids.size() != segments.size()) {
        throw std::invalid_argument(
            "Each path segment requires collision and survivor lineage IDs."
        );
    }
    if (!std::isfinite(initial_weight) || initial_weight <= 0.0) {
        throw std::invalid_argument(
            "initial_weight must be finite and positive."
        );
    }

    std::set<std::uint64_t> used_lineage_ids = {initial_lineage_id};
    double total_optical_depth = 0.0;
    for (std::size_t index = 0; index < segments.size(); ++index) {
        RequireUnitInterval(
            draws[index].collision_unit_interval,
            "collision_unit_interval"
        );
        RequireUnitInterval(
            draws[index].process_unit_interval,
            "process_unit_interval"
        );
        const double optical_depth = SegmentOpticalDepth(segments[index]);
        if (
            optical_depth
                > std::numeric_limits<double>::max()
                    - total_optical_depth
        ) {
            throw std::overflow_error(
                "Path optical depth overflowed."
            );
        }
        total_optical_depth += optical_depth;
        const auto& ids = lineage_ids[index];
        if (
            ids.collision_lineage_id == ids.survivor_lineage_id
            || !used_lineage_ids.insert(
                ids.collision_lineage_id
            ).second
            || !used_lineage_ids.insert(
                ids.survivor_lineage_id
            ).second
        ) {
            throw std::invalid_argument(
                "Branch lineage IDs must be globally unique."
            );
        }
    }

    ForcedCollisionResult result;
    result.primary_history_id = primary_history_id;
    result.initial_weight = initial_weight;
    result.total_optical_depth = total_optical_depth;
    result.no_collision_probability = NoCollisionProbability(
        total_optical_depth
    );
    result.collision_probability = CollisionProbability(
        total_optical_depth
    );

    double survivor_weight = initial_weight;
    double survivor_probability = 1.0;
    std::uint64_t survivor_lineage_id = initial_lineage_id;
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const auto& segment = segments[index];
        const auto& ids = lineage_ids[index];
        const double segment_optical_depth = SegmentOpticalDepth(segment);
        const double local_survival_probability = NoCollisionProbability(
            segment_optical_depth
        );
        const double local_collision_probability = CollisionProbability(
            segment_optical_depth
        );

        if (local_collision_probability > 0.0) {
            auto location = SampleConditionalFirstCollision(
                segment,
                index,
                draws[index]
            );
            BranchAccounting collision;
            collision.kind = BranchKind::kCollision;
            collision.primary_history_id = primary_history_id;
            collision.parent_lineage_id = survivor_lineage_id;
            collision.child_lineage_id = ids.collision_lineage_id;
            collision.segment_index = index;
            collision.segment_token = segment.segment_token;
            collision.parent_weight = survivor_weight;
            collision.local_branch_probability = (
                local_collision_probability
            );
            collision.cumulative_probability = (
                survivor_probability * local_collision_probability
            );
            collision.branch_weight = (
                survivor_weight * local_collision_probability
            );
            collision.has_collision_location = true;
            collision.collision = location;
            collision.absolute_analog_density_per_m = (
                survivor_probability
                * location.local_analog_density_per_m
            );
            result.collision_branches.push_back(collision);
        }

        BranchAccounting survivor;
        survivor.kind = index + 1 == segments.size()
            ? BranchKind::kUncollided
            : BranchKind::kIntermediateSurvivor;
        survivor.primary_history_id = primary_history_id;
        survivor.parent_lineage_id = survivor_lineage_id;
        survivor.child_lineage_id = ids.survivor_lineage_id;
        survivor.segment_index = index;
        survivor.segment_token = segment.segment_token;
        survivor.parent_weight = survivor_weight;
        survivor.local_branch_probability = local_survival_probability;
        survivor.cumulative_probability = (
            survivor_probability * local_survival_probability
        );
        survivor.branch_weight = (
            survivor_weight * local_survival_probability
        );
        result.survivor_branches.push_back(survivor);

        survivor_weight = survivor.branch_weight;
        survivor_probability = survivor.cumulative_probability;
        survivor_lineage_id = ids.survivor_lineage_id;
    }

    double leaf_weight_sum = survivor_weight;
    for (const auto& collision : result.collision_branches) {
        leaf_weight_sum += collision.branch_weight;
    }
    result.final_leaf_weight_sum = leaf_weight_sum;
    const double conservation_tolerance = (
        64.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, initial_weight)
        * static_cast<double>(segments.size() + 1)
    );
    if (std::abs(leaf_weight_sum - initial_weight) > conservation_tolerance) {
        throw std::runtime_error(
            "Forced first-collision branch weights do not conserve the "
            "primary history weight."
        );
    }
    return result;
}

}  // namespace rotating_shield::calibration::first_collision
