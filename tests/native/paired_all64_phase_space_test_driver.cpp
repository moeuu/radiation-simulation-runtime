#include "paired_all64_phase_space.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phase = rotating_shield::paired_all64;

namespace {

void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireNear(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string& message
) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void RequireFailure(Callable&& callable, const std::string& message) {
    bool failed = false;
    try {
        callable();
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    Require(failed, message);
}

phase::Crossing PrimaryCrossing(
    const std::uint64_t branch_id,
    const std::array<double, 3>& position,
    const std::array<double, 3>& direction,
    const std::uint32_t source_index = 0U,
    const std::uint32_t line_index = 0U
) {
    phase::Crossing crossing;
    crossing.branch_id = branch_id;
    crossing.source_index = source_index;
    crossing.line_index = line_index;
    crossing.pdg_code = 22;
    crossing.particle_name = "gamma";
    crossing.position_m = position;
    crossing.direction = direction;
    crossing.polarization = {0.0, 0.0, 0.0};
    crossing.kinetic_energy_mev = 0.662;
    crossing.mass_mev = 0.0;
    crossing.charge_eplus = 0.0;
    crossing.global_time_s = 1.0e-9;
    crossing.proper_time_s = 0.0;
    return crossing;
}

std::vector<phase::HistoryEstimatorIdentity> ExactIdentities() {
    return {
        {0U, 0U, 0U, 0U, 2U, 2.0},
        {1U, 0U, 0U, 0U, 2U, 2.0},
        {2U, 0U, 0U, 1U, 2U, 2.0},
        {3U, 0U, 0U, 1U, 2U, 2.0},
        {4U, 1U, 2U, 0U, 1U, 0.5},
        {5U, 1U, 2U, 0U, 1U, 0.5},
        {6U, 1U, 2U, 0U, 1U, 0.5},
        {7U, 1U, 2U, 0U, 1U, 0.5},
    };
}

std::vector<double> PairScores(const std::size_t pair) {
    const double p = static_cast<double>(pair);
    return {
        p + 1.0, 0.0,
        p + 3.0, 2.0,
        10.0 + 2.0 * p, 1.0,
        14.0 + 2.0 * p, 5.0,
        0.0, 0.0,
        2.0, 0.0,
        4.0, 2.0,
        6.0, 2.0,
    };
}

}  // namespace

int main() {
    try {
        RequireFailure(
            [] {
                phase::RequireDedicatedProfile(
                    phase::kDedicatedProfile,
                    true
                );
            },
            "standard runtime accepted paired profile"
        );
        const auto profile = phase::RequireDedicatedProfile(
            phase::kDedicatedProfile,
            false
        );
        const phase::Boundary boundary{{0.0, 0.0, 0.0}, 1.0};
        phase::CaptureAccumulator worker_a(boundary);
        phase::CaptureAccumulator worker_b(boundary);
        worker_a.RegisterHistory(0U, 0U, 0U, 0U, 2U, 2.0);
        worker_b.RegisterHistory(1U, 0U, 0U, 0U, 2U, 2.0);
        worker_a.RegisterHistory(2U, 0U, 0U, 1U, 2U, 2.0);
        worker_b.RegisterHistory(3U, 0U, 0U, 1U, 2U, 2.0);
        worker_a.RegisterHistory(4U, 3U, 5U, 0U, 1U, 0.5);
        worker_b.RegisterHistory(5U, 3U, 5U, 0U, 1U, 0.5);

        auto first = PrimaryCrossing(
            1U,
            {1.0, 0.0, 0.0},
            {-1.0, 0.0, 0.0}
        );
        Require(
            worker_a.RecordFirstInwardCrossing(0U, first)
                == phase::CaptureResult::kCaptured,
            "first branch crossing was not captured"
        );
        Require(
            worker_a.RecordFirstInwardCrossing(0U, first)
                == phase::CaptureResult::kAlreadyCaptured,
            "branch reentry was captured twice"
        );
        auto secondary = PrimaryCrossing(
            2U,
            {0.0, 1.0, 0.0},
            {0.0, -1.0, 0.0}
        );
        secondary.parent_branch_id = 1U;
        secondary.generation = 1U;
        secondary.gamma_interaction_count = 1U;
        secondary.interaction_flags =
            phase::InteractionFlags::kSecondaryLineage
            | phase::InteractionFlags::kInteracted;
        Require(
            worker_a.RecordFirstInwardCrossing(0U, secondary)
                == phase::CaptureResult::kCaptured,
            "secondary gamma branch was not captured"
        );
        auto other = PrimaryCrossing(
            7U,
            {-1.0, 0.0, 0.0},
            {1.0, 0.0, 0.0}
        );
        other.pdg_code = 11;
        other.particle_name = "e-";
        other.mass_mev = 0.51099895;
        other.charge_eplus = -1.0;
        worker_b.RecordFirstInwardCrossing(1U, other);

        auto outward = PrimaryCrossing(
            9U,
            {1.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            3U,
            5U
        );
        RequireFailure(
            [&] {
                worker_a.RecordFirstInwardCrossing(4U, outward);
            },
            "outward capture crossing was accepted"
        );
        auto weighted = PrimaryCrossing(
            11U,
            {1.0, 0.0, 0.0},
            {-1.0, 0.0, 0.0},
            3U,
            5U
        );
        weighted.weight = 0.5;
        RequireFailure(
            [&] {
                worker_a.RecordFirstInwardCrossing(4U, weighted);
            },
            "weighted capture crossing was accepted"
        );
        auto mislabeled = PrimaryCrossing(
            12U,
            {1.0, 0.0, 0.0},
            {-1.0, 0.0, 0.0}
        );
        RequireFailure(
            [&] {
                worker_a.RecordFirstInwardCrossing(4U, mislabeled);
            },
            "mislabeled source-line crossing was accepted"
        );

        const auto merged = phase::MergeWorkerBanks(
            profile,
            {worker_a.Finalize(), worker_b.Finalize()}
        );
        Require(merged.histories.size() == 6U, "history merge lost zeros");
        for (std::size_t index = 0U; index < merged.histories.size(); ++index) {
            Require(
                merged.histories.at(index).original_history_id == index,
                "history merge is not canonical"
            );
        }
        Require(
            merged.histories[0].crossings.size() == 2U
                && merged.histories[1].crossings.size() == 1U
                && merged.histories[2].crossings.empty()
                && merged.histories[5].crossings.empty(),
            "event-grouped crossing counts are wrong"
        );

        const auto payload = phase::SerializeBank(profile, merged);
        const auto roundtrip = phase::DeserializeBank(profile, payload);
        Require(
            roundtrip.histories.size() == merged.histories.size(),
            "bank roundtrip changed history count"
        );
        Require(
            roundtrip.histories[4].source_index == 3U
                && roundtrip.histories[4].line_index == 5U
                && roundtrip.histories[4].angle_stratum_index == 0U
                && roundtrip.histories[4].angle_stratum_count == 1U
                && roundtrip.histories[4].estimator_coefficient == 0.5,
            "zero-crossing history lost estimator identity"
        );
        Require(
            roundtrip.histories[1].crossings[0].particle_name == "e-"
                && roundtrip.histories[1].crossings[0].pdg_code == 11
                && roundtrip.histories[1].crossings[0].charge_eplus == -1.0,
            "non-gamma particle state was not preserved"
        );
        auto corrupt = payload;
        corrupt.at(24) ^= 0x01U;
        RequireFailure(
            [&] {
                phase::DeserializeBank(profile, corrupt);
            },
            "corrupted bank payload passed checksum"
        );

        const std::string identity(64U, 'c');
        const auto replay_seed = phase::DeriveReplaySeed(7U, identity, 12U);
        const phase::ReplaySchedule schedule(
            profile,
            merged,
            12U,
            replay_seed
        );
        Require(schedule.EventCount() == 6U, "replay dropped zero event");
        Require(
            schedule.Event(0).primaries.size() == 2U
                && schedule.Event(4).primaries.empty(),
            "replay did not group crossings by original event"
        );
        Require(
            schedule.Event(4).source_index == 3U
                && schedule.Event(4).line_index == 5U
                && schedule.Event(4).angle_stratum_index == 0U
                && schedule.Event(4).angle_stratum_count == 1U
                && schedule.Event(4).estimator_coefficient == 0.5,
            "replay event lost authenticated estimator identity"
        );
        Require(
            phase::ReplaySchedule::FullWorldReplayRequired(),
            "replay does not require full world"
        );
        Require(
            !phase::ReplaySchedule::KillOutwardCrossings(),
            "replay would kill outward/reentry branches"
        );

        constexpr std::size_t feature_count = 2U;
        phase::PairedScoreAccumulator scores(
            profile,
            ExactIdentities(),
            feature_count
        );
        for (std::size_t pair = 0U; pair < phase::kShieldPairCount; ++pair) {
            scores.SubmitPairScores(
                static_cast<std::uint32_t>(pair),
                PairScores(pair)
            );
        }
        Require(scores.Complete(), "all-64 score matrix is incomplete");
        const auto covariance = scores.FinalizeExact(
            "incident_gamma_bin_count_per_primary_history"
        );
        Require(covariance.group_count == 3U, "heterogeneous groups pooled");
        Require(
            covariance.groups.at(0).history_count == 2U
                && covariance.groups.at(1).history_count == 2U
                && covariance.groups.at(2).history_count == 4U,
            "exact covariance group quotas are wrong"
        );
        RequireNear(
            covariance.estimate_by_pair_feature.at(0U),
            62.0,
            1.0e-12,
            "fixed-quota estimate first feature is wrong"
        );
        RequireNear(
            covariance.estimate_by_pair_feature.at(63U * 2U),
            818.0,
            1.0e-12,
            "fixed-quota estimate last pair is wrong"
        );
        RequireNear(
            covariance.centered_factor_by_history.at(0U),
            -2.0 * std::sqrt(2.0),
            1.0e-12,
            "within-stratum centered factor is wrong"
        );
        RequireNear(
            covariance.centered_factor_by_history.at(
                4U * phase::kShieldPairCount * feature_count
            ),
            -std::sqrt(3.0),
            1.0e-12,
            "zero-score history was omitted from exact covariance"
        );
        RequireNear(
            covariance.total_cross_pair_covariance.front(),
            1000.0 / 3.0,
            1.0e-10,
            "exact stratified covariance first entry is wrong"
        );
        RequireNear(
            covariance.total_cross_pair_covariance.back(),
            1000.0 / 3.0,
            1.0e-10,
            "exact stratified covariance last entry is wrong"
        );

        const auto approximate =
            scores.FinalizeApproximateBlockDiagnostic(4U);
        Require(
            approximate.semantics
                == phase::kApproximateBlockDiagnosticSemantics,
            "pooled diagnostic is not labeled approximate"
        );
        Require(
            approximate.covariance_factor_by_block
                != covariance.centered_factor_by_history,
            "approximate pooled diagnostic masquerades as exact factors"
        );

        const auto covariance_payload =
            phase::SerializeCovarianceArtifact(covariance);
        Require(!covariance_payload.empty(), "covariance was not serialized");
        const auto covariance_roundtrip =
            phase::DeserializeCovarianceArtifact(covariance_payload);
        Require(
            covariance_roundtrip.artifact_sha256
                == covariance.artifact_sha256
                && covariance_roundtrip.group_assignment_sha256
                    == covariance.group_assignment_sha256
                && covariance_roundtrip.groups.size()
                    == covariance.groups.size(),
            "covariance roundtrip changed authenticated content"
        );
        auto corrupt_covariance = covariance_payload;
        corrupt_covariance.at(32) ^= 0x01U;
        RequireFailure(
            [&] {
                phase::DeserializeCovarianceArtifact(corrupt_covariance);
            },
            "corrupted covariance payload passed checksum"
        );

        auto incomplete_identities = ExactIdentities();
        incomplete_identities.erase(incomplete_identities.begin() + 2);
        RequireFailure(
            [&] {
                phase::PairedScoreAccumulator invalid(
                    profile,
                    incomplete_identities,
                    feature_count
                );
            },
            "incomplete fixed-quota stratum was accepted"
        );

        std::cout << "replay_seed=" << replay_seed << "\n";
        std::cout << "bank_sha256="
                  << phase::BankPayloadSha256(profile, merged) << "\n";
        std::cout << "bank_size=" << payload.size() << "\n";
        std::cout << "group_assignment_sha256="
                  << covariance.group_assignment_sha256 << "\n";
        std::cout << "artifact_sha256="
                  << covariance.artifact_sha256 << "\n";
        std::cout << "group_count=" << covariance.group_count << "\n";
        std::cout << "estimate_first="
                  << covariance.estimate_by_pair_feature.front() << "\n";
        std::cout << "estimate_pair63_feature0="
                  << covariance.estimate_by_pair_feature.at(126U) << "\n";
        std::cout << "factor_first="
                  << covariance.centered_factor_by_history.front() << "\n";
        std::cout << "zero_history_factor="
                  << covariance.centered_factor_by_history.at(
                      4U * phase::kShieldPairCount * feature_count
                  ) << "\n";
        std::cout << "covariance_first="
                  << covariance.total_cross_pair_covariance.front() << "\n";
        std::cout << "covariance_last="
                  << covariance.total_cross_pair_covariance.back() << "\n";
        std::cout << "approximate_semantics="
                  << approximate.semantics << "\n";
        std::cout << "covariance_payload_size="
                  << covariance_payload.size() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
