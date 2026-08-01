#pragma once

// Geant4 integration for the dedicated paired all-64 calibration profile.
//
// This API is intentionally separate from the standard observation sidecar.
// It provides:
//
// * a material-free parallel-world sphere that captures the first inward
//   transported-particle crossing of every branch;
// * event-grouped replay primaries for the complete mass world;
// * worker-local original-history score matrices that feed the dependency-free
//   paired covariance core.
//
// The caller still owns the experiment-specific mass-world construction,
// shield-pair poses, source schedule, detector scoring, and track-information
// class.  The narrow hooks needed for those objects are documented in
// docs/paired_all64_phase_space_native_integration.md.

#include "paired_all64_phase_space.hpp"

#include <G4VUserEventInformation.hh>
#include <G4VUserParallelWorld.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4VUserPrimaryParticleInformation.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class G4Event;
class G4Track;

namespace rotating_shield::paired_all64::geant4 {

inline constexpr const char* kParallelWorldName =
    "PairedAll64CaptureParallelWorld";
inline constexpr const char* kCaptureSphereLogicalName =
    "PairedAll64CaptureSphereLV";
inline constexpr const char* kCaptureSpherePhysicalName =
    "PairedAll64CaptureSpherePV";

struct AxisAlignedBox {
    std::array<double, 3> minimum_m{};
    std::array<double, 3> maximum_m{};
};

struct CaptureTransportContract {
    bool standard_runtime = true;
    bool detector_and_shields_omitted = false;
    bool full_secondary_transport = false;
    bool unit_primary_sampling = false;
    bool unit_track_weights = false;
    bool detector_response_disabled = false;
    bool background_disabled = false;
    bool dead_time_disabled = false;
};

struct CaptureGeometryContract {
    Boundary boundary;
    AxisAlignedBox world_bounds_m;
    std::vector<std::array<double, 3>> source_positions_m;
    std::vector<AxisAlignedBox> enclosed_detector_and_shield_bounds_m;
};

// Fail before Geant4 initialization unless the dedicated capture run preserves
// the exact unit-weight, full-environment transport contract.
void ValidateCapturePreflight(
    const DedicatedProfile& profile,
    const CaptureTransportContract& transport,
    const CaptureGeometryContract& geometry
);

struct HistoryIdentity {
    std::uint64_t original_history_id = 0;
    std::uint32_t source_index = 0;
    std::uint32_t line_index = 0;
    std::uint32_t angle_stratum_index = 0;
    std::uint32_t angle_stratum_count = 1;
    // Authenticated fixed-quota estimator coefficient. This value is event
    // metadata only and must never be copied to a Geant4 track weight.
    double estimator_coefficient = 1.0;
};

struct TrackLineage {
    std::uint64_t branch_id = 0;
    std::uint64_t parent_branch_id = 0;
    std::uint32_t source_index = 0;
    std::uint32_t line_index = 0;
    std::uint32_t generation = 0;
    std::uint32_t gamma_interaction_count = 0;
    bool secondary_lineage = false;
};

// The sidecar supplies immutable event and track identities.  Returning a
// default or guessed identity is forbidden; implementations must throw when
// the source schedule or track information is missing.
class CaptureIdentityProvider {
public:
    virtual ~CaptureIdentityProvider() = default;

    virtual HistoryIdentity IdentityForEvent(const G4Event& event) const = 0;
    virtual TrackLineage LineageForTrack(const G4Track& track) const = 0;
};

class CaptureWorkerState;

// Shared registry for worker-local CaptureAccumulator instances.
class CaptureCollector {
public:
    explicit CaptureCollector(Boundary boundary);

    std::shared_ptr<CaptureWorkerState> AcquireLocal();
    Bank FinalizeMerged(const DedicatedProfile& profile) const;
    std::size_t WorkerCount() const;

private:
    Boundary boundary_;
    mutable std::mutex registry_mutex_;
    std::vector<std::shared_ptr<CaptureWorkerState>> worker_states_;
};

// Material-free capture sphere.  The sensitive detector is attached to the
// outside ghost-world logical volume so the entry boundary is recorded before
// any inside step or interaction can occur.
class CaptureParallelWorld final : public G4VUserParallelWorld {
public:
    CaptureParallelWorld(
        Boundary boundary,
        std::shared_ptr<CaptureCollector> collector,
        std::shared_ptr<const CaptureIdentityProvider> identity_provider
    );

    void Construct() override;
    void ConstructSD() override;

private:
    Boundary boundary_;
    std::shared_ptr<CaptureCollector> collector_;
    std::shared_ptr<const CaptureIdentityProvider> identity_provider_;
};

struct ReplayTransportContract {
    bool standard_runtime = true;
    bool full_original_world = false;
    bool detector_present = false;
    bool selected_shields_present = false;
    bool full_secondary_transport = false;
    bool unit_track_weights = false;
    bool capture_boundary_is_absorbing = true;
    bool kill_outward_crossings = true;
};

// Fail before replay unless the original environment remains present and the
// capture surface has no absorbing or one-way semantics.
void ValidateReplayPreflight(
    const DedicatedProfile& profile,
    const ReplayTransportContract& transport
);

// User information attached to every replay G4PrimaryParticle.  The sidecar's
// tracking action reads this object to initialize its normal track-information
// object without replacing the standard detector scorer.
class ReplayPrimaryInformation final
    : public G4VUserPrimaryParticleInformation {
public:
    ReplayPrimaryInformation(
        std::uint64_t original_history_id,
        Crossing crossing
    );

    std::uint64_t OriginalHistoryId() const noexcept;
    const Crossing& State() const noexcept;
    void Print() const override;

private:
    std::uint64_t original_history_id_ = 0;
    Crossing crossing_;
};

// Event information is available even for zero-primary replay histories.
class ReplayEventInformation final : public G4VUserEventInformation {
public:
    ReplayEventInformation(
        std::uint64_t original_history_id,
        std::uint32_t source_index,
        std::uint32_t line_index,
        std::uint32_t angle_stratum_index,
        std::uint32_t angle_stratum_count,
        double estimator_coefficient,
        std::uint32_t shield_pair_id,
        std::uint64_t random_seed
    );

    std::uint64_t OriginalHistoryId() const noexcept;
    std::uint32_t SourceIndex() const noexcept;
    std::uint32_t LineIndex() const noexcept;
    std::uint32_t AngleStratumIndex() const noexcept;
    std::uint32_t AngleStratumCount() const noexcept;
    double EstimatorCoefficient() const noexcept;
    std::uint32_t ShieldPairId() const noexcept;
    std::uint64_t RandomSeed() const noexcept;
    void Print() const override;

private:
    std::uint64_t original_history_id_ = 0;
    std::uint32_t source_index_ = 0;
    std::uint32_t line_index_ = 0;
    std::uint32_t angle_stratum_index_ = 0;
    std::uint32_t angle_stratum_count_ = 1;
    double estimator_coefficient_ = 1.0;
    std::uint32_t shield_pair_id_ = 0;
    std::uint64_t random_seed_ = 0;
};

// Construct one Geant4 event per original history.  Every captured crossing in
// that history is injected as a unit-weight primary of its original Geant4
// species; histories with no crossing deliberately receive no primary vertex.
class ReplayPrimaryGeneratorAction final
    : public G4VUserPrimaryGeneratorAction {
public:
    explicit ReplayPrimaryGeneratorAction(
        std::shared_ptr<const ReplaySchedule> schedule
    );

    void GeneratePrimaries(G4Event* event) override;

private:
    std::shared_ptr<const ReplaySchedule> schedule_;
};

const ReplayEventInformation* ReplayEventIdentity(const G4Event& event);
const ReplayPrimaryInformation* ReplayPrimaryIdentity(const G4Track& track);

class ReplayScoreWorkerState;

// One complete feature vector is submitted at EndOfEvent for every original
// history, including an explicit all-zero vector for a zero-primary event.
// There is intentionally no scalar insertion API.
class ReplayPairScoreMatrixCollector {
public:
    ReplayPairScoreMatrixCollector(
        const ReplaySchedule& schedule,
        std::size_t feature_count
    );

    std::shared_ptr<ReplayScoreWorkerState> AcquireLocal();

    void SubmitHistoryScores(
        const std::shared_ptr<ReplayScoreWorkerState>& worker,
        std::uint64_t original_history_id,
        const std::vector<double>& feature_scores
    );

    std::vector<double> FinalizePairScores() const;
    const std::vector<std::uint64_t>& OriginalHistoryIds() const noexcept;
    std::size_t FeatureCount() const noexcept;

private:
    std::vector<std::uint64_t> history_ids_;
    std::unordered_map<std::uint64_t, std::size_t> history_index_;
    std::size_t feature_count_ = 0;
    mutable std::mutex registry_mutex_;
    std::vector<std::shared_ptr<ReplayScoreWorkerState>> worker_states_;
};

// Connect 64 completed original-history matrices to PairedScoreAccumulator.
class All64ReplayScoreCoordinator {
public:
    All64ReplayScoreCoordinator(
        const DedicatedProfile& profile,
        const Bank& bank,
        std::size_t feature_count
    );

    void SubmitCompletedPair(
        std::uint32_t shield_pair_id,
        const ReplayPairScoreMatrixCollector& collector
    );

    bool Complete() const noexcept;

    CrossPairStratifiedCovariance FinalizeExact(
        const std::string& score_semantics
    ) const;

    ApproximateCrossPairBlockDiagnostic
    FinalizeApproximateBlockDiagnostic(std::size_t block_count) const;

private:
    std::vector<std::uint64_t> history_ids_;
    std::vector<HistoryEstimatorIdentity> history_identities_;
    std::size_t feature_count_ = 0;
    PairedScoreAccumulator accumulator_;
};

}  // namespace rotating_shield::paired_all64::geant4
