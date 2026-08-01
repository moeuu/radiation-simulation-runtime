#include "first_collision_core.hh"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace first_collision =
    rotating_shield::calibration::first_collision;

namespace {

const char* BranchKindToken(const first_collision::BranchKind kind) {
    switch (kind) {
        case first_collision::BranchKind::kCollision:
            return "collision";
        case first_collision::BranchKind::kIntermediateSurvivor:
            return "survivor";
        case first_collision::BranchKind::kUncollided:
            return "uncollided";
    }
    throw std::runtime_error("Unknown first-collision branch kind.");
}

void WriteBranch(const first_collision::BranchAccounting& branch) {
    std::cout
        << "BRANCH"
        << " kind=" << BranchKindToken(branch.kind)
        << " primary=" << branch.primary_history_id
        << " parent_lineage=" << branch.parent_lineage_id
        << " child_lineage=" << branch.child_lineage_id
        << " segment_index=" << branch.segment_index
        << " segment_token=" << branch.segment_token
        << " parent_weight=" << branch.parent_weight
        << " local_probability=" << branch.local_branch_probability
        << " cumulative_probability=" << branch.cumulative_probability
        << " branch_weight=" << branch.branch_weight
        << " has_collision="
        << (branch.has_collision_location ? 1 : 0);
    if (branch.has_collision_location) {
        std::cout
            << " distance_m="
            << branch.collision.distance_from_segment_start_m
            << " collision_tau="
            << branch.collision.optical_depth_from_segment_start
            << " process_token=" << branch.collision.process_token
            << " process_probability="
            << branch.collision.process_probability
            << " conditional_density_per_m="
            << branch.collision.conditional_density_per_m
            << " local_analog_density_per_m="
            << branch.collision.local_analog_density_per_m
            << " absolute_analog_density_per_m="
            << branch.absolute_analog_density_per_m;
    }
    std::cout << "\n";
}

}  // namespace

int main() {
    try {
        std::string header;
        int calibration_only = 0;
        double initial_weight = 0.0;
        std::uint64_t primary_history_id = 0;
        std::uint64_t initial_lineage_id = 0;
        std::size_t segment_count = 0;
        if (
            !(std::cin
                >> header
                >> calibration_only
                >> initial_weight
                >> primary_history_id
                >> initial_lineage_id
                >> segment_count)
            || header != "PATH"
        ) {
            throw std::runtime_error(
                "Expected PATH calibration weight primary lineage count."
            );
        }

        std::vector<first_collision::MaterialSegment> segments;
        std::vector<first_collision::SegmentRandomDraw> draws;
        std::vector<first_collision::SegmentLineageIds> lineages;
        segments.reserve(segment_count);
        draws.reserve(segment_count);
        lineages.reserve(segment_count);
        for (std::size_t index = 0; index < segment_count; ++index) {
            std::string segment_header;
            first_collision::MaterialSegment segment;
            std::size_t channel_count = 0;
            if (
                !(std::cin
                    >> segment_header
                    >> segment.segment_token
                    >> segment.length_m
                    >> channel_count)
                || segment_header != "SEGMENT"
            ) {
                throw std::runtime_error(
                    "Expected SEGMENT token length channel_count."
                );
            }
            segment.channels.reserve(channel_count);
            for (
                std::size_t channel_index = 0;
                channel_index < channel_count;
                ++channel_index
            ) {
                first_collision::InteractionChannel channel;
                if (
                    !(std::cin
                        >> channel.process_token
                        >> channel.macroscopic_cross_section_per_m)
                ) {
                    throw std::runtime_error(
                        "Failed to read an interaction channel."
                    );
                }
                segment.channels.push_back(channel);
            }
            first_collision::SegmentRandomDraw draw;
            first_collision::SegmentLineageIds lineage;
            if (
                !(std::cin
                    >> draw.collision_unit_interval
                    >> draw.process_unit_interval
                    >> lineage.collision_lineage_id
                    >> lineage.survivor_lineage_id)
            ) {
                throw std::runtime_error(
                    "Failed to read segment draws and lineage IDs."
                );
            }
            segments.push_back(std::move(segment));
            draws.push_back(draw);
            lineages.push_back(lineage);
        }

        const auto result = first_collision::SplitHeterogeneousPath(
            segments,
            draws,
            lineages,
            primary_history_id,
            initial_lineage_id,
            initial_weight,
            calibration_only == 1
        );
        std::cout << std::setprecision(17);
        std::cout
            << "RESULT"
            << " standard_runtime_enabled="
            << (
                first_collision::kStandardRuntimeIntegrationEnabled
                    ? 1
                    : 0
            )
            << " primary=" << result.primary_history_id
            << " initial_weight=" << result.initial_weight
            << " total_tau=" << result.total_optical_depth
            << " no_collision_probability="
            << result.no_collision_probability
            << " collision_probability="
            << result.collision_probability
            << " leaf_weight_sum=" << result.final_leaf_weight_sum
            << "\n";
        for (const auto& branch : result.collision_branches) {
            WriteBranch(branch);
        }
        for (const auto& branch : result.survivor_branches) {
            WriteBranch(branch);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "FIRST_COLLISION_ERROR " << exc.what() << "\n";
        return 2;
    }
}
