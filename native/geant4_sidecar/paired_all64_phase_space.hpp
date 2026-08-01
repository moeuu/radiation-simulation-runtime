#pragma once

// Dedicated calibration-only core for paired all-64 phase-space replay.
//
// This file intentionally has no Geant4 dependency.  The native sidecar can
// integrate it at three narrow boundaries:
//
// 1. register every original primary history and capture the first inward
//    detector-boundary crossing of each transported-particle branch;
// 2. serialize the event-grouped bank and replay all crossings belonging to an
//    original history as primaries in one G4Event;
// 3. submit one complete history-by-feature score matrix per shield pair and
//    save the resulting exact within-stratum original-history covariance.
//
// The core is not an observation-runtime shortcut.  Its profile gate rejects
// standard runtime selection, weighted histories, incomplete particle restart
// states, outward capture crossings, and incomplete all-64 score sets.

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rotating_shield::paired_all64 {

inline constexpr std::uint32_t kBankSchemaVersion = 3;
inline constexpr std::size_t kShieldPairCount = 64;
inline constexpr const char* kDedicatedProfile =
    "geant4_phase_space_paired_all64_v3";
inline constexpr const char* kBankFormat =
    "event_grouped_detector_boundary_v3";
inline constexpr const char* kCovarianceSemantics =
    "stratified_fixed_quota_original_history_covariance_v1";
inline constexpr const char* kApproximateBlockDiagnosticSemantics =
    "approximate_pooled_hash_block_diagnostic_v1";

enum class InteractionFlags : std::uint32_t {
    kNone = 0,
    kInteracted = 1U << 0U,
    kSecondaryLineage = 1U << 1U,
};

InteractionFlags operator|(InteractionFlags left, InteractionFlags right);
bool HasInteractionFlag(InteractionFlags value, InteractionFlags flag);

struct DedicatedProfile {
    std::string name;
};

// Return the only profile token accepted by this core.
//
// `standard_runtime` must be false.  Keeping this explicit at the API boundary
// prevents a phase-space bank with paired counterfactual semantics from being
// mistaken for 64 independent production observations.
DedicatedProfile RequireDedicatedProfile(
    const std::string& profile,
    bool standard_runtime
);

struct Boundary {
    std::array<double, 3> center_m{};
    double radius_m = 0.0;
};

struct Crossing {
    std::uint64_t branch_id = 0;
    std::uint64_t parent_branch_id = 0;
    std::uint32_t source_index = 0;
    std::uint32_t line_index = 0;
    std::int32_t pdg_code = 0;
    std::string particle_name;
    std::uint32_t generation = 0;
    std::uint32_t gamma_interaction_count = 0;
    InteractionFlags interaction_flags = InteractionFlags::kNone;
    std::array<double, 3> position_m{};
    std::array<double, 3> direction{};
    std::array<double, 3> polarization{};
    double kinetic_energy_mev = 0.0;
    double mass_mev = 0.0;
    double charge_eplus = 0.0;
    double global_time_s = 0.0;
    double proper_time_s = 0.0;
    double weight = 1.0;
};

struct History {
    std::uint64_t original_history_id = 0;
    std::uint32_t source_index = 0;
    std::uint32_t line_index = 0;
    std::uint32_t angle_stratum_index = 0;
    std::uint32_t angle_stratum_count = 1;
    // External coefficient of this original-history score in the fixed-quota
    // estimator. It is authenticated in the bank but is never assigned to a
    // Geant4 track; all capture and replay tracks remain unit weight.
    double estimator_coefficient = 1.0;
    std::vector<Crossing> crossings;
};

struct HistoryEstimatorIdentity {
    std::uint64_t original_history_id = 0;
    std::uint32_t source_index = 0;
    std::uint32_t line_index = 0;
    std::uint32_t angle_stratum_index = 0;
    std::uint32_t angle_stratum_count = 1;
    double estimator_coefficient = 1.0;
};

struct Bank {
    Boundary boundary;
    std::vector<History> histories;
};

enum class CaptureResult {
    kCaptured,
    kAlreadyCaptured,
};

// Thread-local capture accumulator.
//
// Construct one accumulator per Geant4 worker.  RegisterHistory must be called
// even when a history produces no crossing so that zero-score histories remain
// in the paired covariance.  RecordFirstInwardCrossing stores at most one
// crossing for each `(original_history_id, branch_id)` and returns
// kAlreadyCaptured on reentry.  The Geant4 stepping integration must kill a
// branch immediately after kCaptured; sibling branches continue normally.
class CaptureAccumulator {
public:
    explicit CaptureAccumulator(Boundary boundary);

    void RegisterHistory(
        std::uint64_t original_history_id,
        std::uint32_t source_index,
        std::uint32_t line_index,
        std::uint32_t angle_stratum_index,
        std::uint32_t angle_stratum_count,
        double estimator_coefficient
    );

    CaptureResult RecordFirstInwardCrossing(
        std::uint64_t original_history_id,
        const Crossing& crossing
    );

    const Boundary& GetBoundary() const noexcept;
    std::size_t HistoryCount() const noexcept;
    std::size_t CrossingCount() const noexcept;

    Bank Finalize() const;

private:
    Boundary boundary_;
    std::vector<History> histories_;
    std::unordered_map<std::uint64_t, std::size_t> history_index_;
    std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>>
        captured_branches_;
};

// Merge worker-local banks into one canonical bank.
//
// Histories must be disjoint across workers.  The result is sorted by original
// history ID, and crossings within each history are sorted by branch ID.
Bank MergeWorkerBanks(
    const DedicatedProfile& profile,
    const std::vector<Bank>& worker_banks
);

// Serialize one canonical little-endian bank.
std::vector<std::uint8_t> SerializeBank(
    const DedicatedProfile& profile,
    const Bank& bank
);

// Parse and fully validate one canonical bank.  Unknown schema versions,
// trailing bytes, incomplete particle states, non-unit weights, noncanonical
// ordering, and corrupted payloads fail closed.
Bank DeserializeBank(
    const DedicatedProfile& profile,
    const std::vector<std::uint8_t>& payload
);

void WriteBank(
    const DedicatedProfile& profile,
    const Bank& bank,
    std::ostream& output
);

Bank ReadBank(
    const DedicatedProfile& profile,
    std::istream& input
);

// Return the lowercase SHA-256 of the exact serialized bank payload.
std::string BankPayloadSha256(
    const DedicatedProfile& profile,
    const Bank& bank
);

// Return a stable pair-specific seed independent of replay iteration order.
std::uint64_t DeriveReplaySeed(
    std::uint64_t root_seed,
    const std::string& bank_payload_sha256,
    std::uint32_t shield_pair_id
);

// Return a stable original-history seed for a replay event.
std::uint64_t DeriveHistoryReplaySeed(
    std::uint64_t replay_seed,
    std::uint64_t original_history_id
);

struct ReplayEvent {
    std::uint64_t original_history_id = 0;
    std::uint32_t source_index = 0;
    std::uint32_t line_index = 0;
    std::uint32_t angle_stratum_index = 0;
    std::uint32_t angle_stratum_count = 1;
    double estimator_coefficient = 1.0;
    std::uint64_t random_seed = 0;
    std::vector<Crossing> primaries;
};

// Canonical replay schedule.
//
// It includes zero-primary events so every original history contributes an
// explicit zero score.  All crossings from one history are returned in one
// event.  The integration must instantiate the full original world plus the
// selected shields and detector, and must not kill outward crossings or
// reentries during replay.
class ReplaySchedule {
public:
    ReplaySchedule(
        const DedicatedProfile& profile,
        const Bank& bank,
        std::uint32_t shield_pair_id,
        std::uint64_t replay_seed
    );

    std::size_t EventCount() const noexcept;
    const ReplayEvent& Event(std::size_t index) const;
    std::uint32_t ShieldPairId() const noexcept;

    static constexpr bool FullWorldReplayRequired() noexcept {
        return true;
    }

    static constexpr bool KillOutwardCrossings() noexcept {
        return false;
    }

private:
    std::uint32_t shield_pair_id_ = 0;
    std::vector<ReplayEvent> events_;
};

struct StratumCovarianceDescriptor {
    std::uint32_t source_index = 0;
    std::uint32_t line_index = 0;
    std::uint32_t angle_stratum_index = 0;
    std::uint32_t angle_stratum_count = 1;
    std::size_t history_count = 0;
    double estimator_coefficient = 1.0;
};

// Exact fixed-quota covariance artifact.
//
// For group g with n_g histories, external estimator coefficient a_g, and
// raw replay score y_h, each factor row is
//
//   a_g * sqrt(n_g / (n_g - 1)) * (y_h - mean_g).
//
// Consequently F^T F is the unbiased covariance of
// sum_g a_g * sum_{h in g} y_h. Every bank history has one row, including
// histories whose replay score is identically zero. Groups are never pooled
// across source, line, or angle stratum.
struct CrossPairStratifiedCovariance {
    std::size_t history_count = 0;
    std::size_t group_count = 0;
    std::size_t feature_count = 0;
    std::string score_semantics;
    std::string group_assignment_sha256;
    std::vector<std::uint64_t> original_history_ids;
    std::vector<std::uint32_t> history_group_indices;
    std::vector<StratumCovarianceDescriptor> groups;
    std::vector<double> estimate_by_pair_feature;
    std::vector<double> first_sum_by_group_pair_feature;
    std::vector<double> centered_factor_by_history;
    std::array<double, kShieldPairCount * kShieldPairCount>
        total_cross_pair_covariance{};
    std::string artifact_sha256;
};

// Optional pooled block diagnostic retained only for coarse convergence
// visualization. It is approximate, may pool heterogeneous groups, and is
// never serialized or accepted as the exact covariance artifact.
struct ApproximateCrossPairBlockDiagnostic {
    std::size_t history_count = 0;
    std::size_t block_count = 0;
    std::size_t histories_per_block = 0;
    std::size_t feature_count = 0;
    std::string semantics = kApproximateBlockDiagnosticSemantics;
    std::vector<double> pooled_mean_by_pair_feature;
    std::vector<double> covariance_factor_by_block;
};

// Batched all-64 score collector.
//
// SubmitPairScores accepts a complete row-major
// `(history_count, feature_count)` matrix.  It deliberately has no scalar score
// insertion API so the native integration cannot accidentally leave a partial
// pair result that is indistinguishable from a real zero.
class PairedScoreAccumulator {
public:
    PairedScoreAccumulator(
        const DedicatedProfile& profile,
        std::vector<HistoryEstimatorIdentity> history_identities,
        std::size_t feature_count
    );

    void SubmitPairScores(
        std::uint32_t shield_pair_id,
        const std::vector<double>& history_feature_scores
    );

    bool Complete() const noexcept;

    CrossPairStratifiedCovariance FinalizeExact(
        const std::string& score_semantics
    ) const;

    ApproximateCrossPairBlockDiagnostic
    FinalizeApproximateBlockDiagnostic(std::size_t block_count) const;

private:
    std::vector<HistoryEstimatorIdentity> history_identities_;
    std::size_t feature_count_ = 0;
    std::vector<double> scores_;
    std::array<bool, kShieldPairCount> submitted_{};
};

// Serialize the numeric covariance artifact in canonical little-endian order.
// The returned SHA-256 is stored in `artifact.artifact_sha256`.
std::vector<std::uint8_t> SerializeCovarianceArtifact(
    const CrossPairStratifiedCovariance& artifact
);

// Parse, authenticate, and validate one covariance artifact.
CrossPairStratifiedCovariance DeserializeCovarianceArtifact(
    const std::vector<std::uint8_t>& payload
);

}  // namespace rotating_shield::paired_all64
