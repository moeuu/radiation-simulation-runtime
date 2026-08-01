#ifndef ROTATING_SHIELD_FIRST_COLLISION_CORE_HH
#define ROTATING_SHIELD_FIRST_COLLISION_CORE_HH

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rotating_shield::calibration::first_collision {

// This core is intentionally not wired to the standard Geant4 runtime. It is
// an exact mathematical building block for calibration-only variance
// reduction after a navigator has supplied homogeneous material segments.
inline constexpr bool kStandardRuntimeIntegrationEnabled = false;

struct InteractionChannel {
    std::uint64_t process_token = 0;
    double macroscopic_cross_section_per_m = 0.0;
};

struct MaterialSegment {
    std::uint64_t segment_token = 0;
    double length_m = 0.0;
    std::vector<InteractionChannel> channels;
};

struct SegmentRandomDraw {
    double collision_unit_interval = 0.0;
    double process_unit_interval = 0.0;
};

struct SegmentLineageIds {
    std::uint64_t collision_lineage_id = 0;
    std::uint64_t survivor_lineage_id = 0;
};

enum class BranchKind {
    kCollision,
    kIntermediateSurvivor,
    kUncollided,
};

struct FirstCollisionLocation {
    std::size_t segment_index = 0;
    std::uint64_t segment_token = 0;
    double distance_from_segment_start_m = 0.0;
    double optical_depth_from_segment_start = 0.0;
    std::uint64_t process_token = 0;
    double process_probability = 0.0;
    double conditional_density_per_m = 0.0;
    double local_analog_density_per_m = 0.0;
};

struct BranchAccounting {
    BranchKind kind = BranchKind::kUncollided;
    std::uint64_t primary_history_id = 0;
    std::uint64_t parent_lineage_id = 0;
    std::uint64_t child_lineage_id = 0;
    std::size_t segment_index = 0;
    std::uint64_t segment_token = 0;
    double parent_weight = 0.0;
    double local_branch_probability = 0.0;
    double cumulative_probability = 0.0;
    double branch_weight = 0.0;
    bool has_collision_location = false;
    FirstCollisionLocation collision;
    double absolute_analog_density_per_m = 0.0;
};

struct ForcedCollisionResult {
    std::uint64_t primary_history_id = 0;
    double initial_weight = 0.0;
    double total_optical_depth = 0.0;
    double no_collision_probability = 0.0;
    double collision_probability = 0.0;
    std::vector<BranchAccounting> collision_branches;
    std::vector<BranchAccounting> survivor_branches;
    double final_leaf_weight_sum = 0.0;
};

// Fail closed unless a caller explicitly declares the calibration-only
// context. Standard runtime integration requires a separate reviewed wrapper.
void RequireCalibrationOnly(bool calibration_only);

double SegmentMacroscopicCrossSectionPerM(const MaterialSegment& segment);

double SegmentOpticalDepth(const MaterialSegment& segment);

double NoCollisionProbability(double optical_depth);

double CollisionProbability(double optical_depth);

FirstCollisionLocation SampleConditionalFirstCollision(
    const MaterialSegment& segment,
    std::size_t segment_index,
    const SegmentRandomDraw& draw
);

ForcedCollisionResult SplitHeterogeneousPath(
    const std::vector<MaterialSegment>& segments,
    const std::vector<SegmentRandomDraw>& draws,
    const std::vector<SegmentLineageIds>& lineage_ids,
    std::uint64_t primary_history_id,
    std::uint64_t initial_lineage_id,
    double initial_weight,
    bool calibration_only
);

}  // namespace rotating_shield::calibration::first_collision

#endif
