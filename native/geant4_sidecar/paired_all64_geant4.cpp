#include "paired_all64_geant4.hpp"

#include <G4DynamicParticle.hh>
#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4Gamma.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4ParallelWorldProcess.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4PrimaryParticle.hh>
#include <G4PrimaryVertex.hh>
#include <G4SDManager.hh>
#include <G4Sphere.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4SystemOfUnits.hh>
#include <G4Threading.hh>
#include <G4Track.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSensitiveDetector.hh>
#include <G4ios.hh>
#include <Randomize.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rotating_shield::paired_all64::geant4 {

class CaptureWorkerState {
public:
    explicit CaptureWorkerState(const Boundary& boundary)
        : accumulator(boundary) {}

    CaptureAccumulator accumulator;
};

class ReplayScoreWorkerState {
public:
    std::unordered_map<std::uint64_t, std::vector<double>> score_by_history;
};

namespace {

void RequireProfile(const DedicatedProfile& profile) {
    if (profile.name != kDedicatedProfile) {
        throw std::invalid_argument(
            "Geant4 paired replay requires its dedicated calibration profile."
        );
    }
}

bool IsFiniteVector(const std::array<double, 3>& value) {
    return std::all_of(
        value.begin(),
        value.end(),
        [](const double component) { return std::isfinite(component); }
    );
}

double SquaredDistance(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    double total = 0.0;
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const double difference = left[axis] - right[axis];
        total += difference * difference;
    }
    return total;
}

void ValidateBox(const AxisAlignedBox& box, const std::string& label) {
    if (!IsFiniteVector(box.minimum_m) || !IsFiniteVector(box.maximum_m)) {
        throw std::invalid_argument(label + " AABB must be finite.");
    }
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        if (!(box.maximum_m[axis] > box.minimum_m[axis])) {
            throw std::invalid_argument(
                label + " AABB must have positive extent."
            );
        }
    }
}

std::vector<std::array<double, 3>> Corners(const AxisAlignedBox& box) {
    std::vector<std::array<double, 3>> corners;
    corners.reserve(8U);
    for (std::size_t selector = 0; selector < 8U; ++selector) {
        corners.push_back({
            (selector & 1U) == 0U ? box.minimum_m[0] : box.maximum_m[0],
            (selector & 2U) == 0U ? box.minimum_m[1] : box.maximum_m[1],
            (selector & 4U) == 0U ? box.minimum_m[2] : box.maximum_m[2],
        });
    }
    return corners;
}

long ReplaySeedAsLong(const std::uint64_t seed) {
    static_assert(
        std::numeric_limits<unsigned long>::digits >= 64,
        "Exact replay seeding requires a 64-bit unsigned long."
    );
    const auto positive_mask = static_cast<std::uint64_t>(
        std::numeric_limits<long>::max()
    );
    const std::uint64_t folded = seed & positive_mask;
    return static_cast<long>(folded == 0U ? 1U : folded);
}

std::vector<std::uint64_t> HistoryIds(const Bank& bank) {
    std::vector<std::uint64_t> result;
    result.reserve(bank.histories.size());
    for (const auto& history : bank.histories) {
        result.push_back(history.original_history_id);
    }
    return result;
}

std::vector<HistoryEstimatorIdentity> HistoryEstimatorIdentities(
    const Bank& bank
) {
    std::vector<HistoryEstimatorIdentity> result;
    result.reserve(bank.histories.size());
    for (const auto& history : bank.histories) {
        result.push_back(HistoryEstimatorIdentity{
            history.original_history_id,
            history.source_index,
            history.line_index,
            history.angle_stratum_index,
            history.angle_stratum_count,
            history.estimator_coefficient,
        });
    }
    return result;
}

class CaptureSensitiveDetector final : public G4VSensitiveDetector {
public:
    CaptureSensitiveDetector(
        std::shared_ptr<CaptureCollector> collector,
        std::shared_ptr<const CaptureIdentityProvider> identity_provider
    ) : G4VSensitiveDetector("PairedAll64CaptureBoundarySD"),
        collector_(std::move(collector)),
        identity_provider_(std::move(identity_provider)),
        worker_(collector_ == nullptr ? nullptr : collector_->AcquireLocal()) {
        if (
            collector_ == nullptr
            || identity_provider_ == nullptr
            || worker_ == nullptr
        ) {
            throw std::invalid_argument(
                "Capture sensitive detector requires collector and identity "
                "provider."
            );
        }
    }

    void Initialize(G4HCofThisEvent*) override {
        auto* event_manager = G4EventManager::GetEventManager();
        const auto* event = event_manager == nullptr
            ? nullptr
            : event_manager->GetConstCurrentEvent();
        if (event == nullptr) {
            throw std::runtime_error(
                "Capture initialization cannot resolve the current event."
            );
        }
        current_history_ = identity_provider_->IdentityForEvent(*event);
        worker_->accumulator.RegisterHistory(
            current_history_.original_history_id,
            current_history_.source_index,
            current_history_.line_index,
            current_history_.angle_stratum_index,
            current_history_.angle_stratum_count,
            current_history_.estimator_coefficient
        );
        history_registered_ = true;
    }

    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        if (step == nullptr) {
            throw std::runtime_error("Capture received a null Geant4 step.");
        }
        const auto* pre = step->GetPreStepPoint();
        const auto* post = step->GetPostStepPoint();
        if (pre == nullptr || post == nullptr) {
            throw std::runtime_error(
                "Capture received a step without both ghost step points."
            );
        }
        const auto* post_volume = post->GetPhysicalVolume();
        if (
            post->GetStepStatus() != fGeomBoundary
            || post_volume == nullptr
            || post_volume->GetName() != kCaptureSpherePhysicalName
        ) {
            return false;
        }
        const auto* pre_volume = pre->GetPhysicalVolume();
        if (
            pre_volume != nullptr
            && pre_volume->GetName() == kCaptureSpherePhysicalName
        ) {
            return false;
        }
        if (!history_registered_) {
            throw std::runtime_error(
                "Capture boundary was reached before history registration."
            );
        }
        auto* track = step->GetTrack();
        if (track == nullptr || track->GetDefinition() == nullptr) {
            throw std::runtime_error(
                "Capture boundary crossing is missing its physical track."
            );
        }
        const TrackLineage lineage =
            identity_provider_->LineageForTrack(*track);
        Crossing crossing;
        crossing.branch_id = lineage.branch_id;
        crossing.parent_branch_id = lineage.parent_branch_id;
        crossing.source_index = lineage.source_index;
        crossing.line_index = lineage.line_index;
        crossing.pdg_code =
            track->GetDefinition()->GetPDGEncoding();
        crossing.particle_name =
            track->GetDefinition()->GetParticleName();
        crossing.generation = lineage.generation;
        crossing.gamma_interaction_count =
            lineage.gamma_interaction_count;
        crossing.interaction_flags = InteractionFlags::kNone;
        if (lineage.gamma_interaction_count > 0U) {
            crossing.interaction_flags =
                crossing.interaction_flags
                | InteractionFlags::kInteracted;
        }
        if (lineage.secondary_lineage) {
            crossing.interaction_flags =
                crossing.interaction_flags
                | InteractionFlags::kSecondaryLineage;
        }
        crossing.position_m = {
            post->GetPosition().x() / m,
            post->GetPosition().y() / m,
            post->GetPosition().z() / m,
        };
        crossing.direction = {
            post->GetMomentumDirection().x(),
            post->GetMomentumDirection().y(),
            post->GetMomentumDirection().z(),
        };
        crossing.polarization = {
            post->GetPolarization().x(),
            post->GetPolarization().y(),
            post->GetPolarization().z(),
        };
        crossing.kinetic_energy_mev = post->GetKineticEnergy() / MeV;
        const auto* dynamic_particle = track->GetDynamicParticle();
        crossing.mass_mev = (
            dynamic_particle == nullptr
                ? track->GetDefinition()->GetPDGMass()
                : dynamic_particle->GetMass()
        ) / MeV;
        crossing.charge_eplus = (
            dynamic_particle == nullptr
                ? track->GetDefinition()->GetPDGCharge()
                : dynamic_particle->GetCharge()
        ) / eplus;
        crossing.global_time_s = post->GetGlobalTime() / s;
        crossing.proper_time_s = track->GetProperTime() / s;
        crossing.weight = post->GetWeight();

        const CaptureResult result =
            worker_->accumulator.RecordFirstInwardCrossing(
                current_history_.original_history_id,
                crossing
            );
        if (result == CaptureResult::kAlreadyCaptured) {
            throw std::runtime_error(
                "A captured branch reached the boundary again instead of "
                "being killed."
            );
        }
        track->SetTrackStatus(fStopAndKill);
        return true;
    }

private:
    std::shared_ptr<CaptureCollector> collector_;
    std::shared_ptr<const CaptureIdentityProvider> identity_provider_;
    std::shared_ptr<CaptureWorkerState> worker_;
    HistoryIdentity current_history_;
    bool history_registered_ = false;
};

}  // namespace

void ValidateCapturePreflight(
    const DedicatedProfile& profile,
    const CaptureTransportContract& transport,
    const CaptureGeometryContract& geometry
) {
    RequireProfile(profile);
    if (transport.standard_runtime) {
        throw std::invalid_argument(
            "Paired phase-space capture is calibration-only."
        );
    }
    if (
        !transport.detector_and_shields_omitted
        || !transport.full_secondary_transport
        || !transport.unit_primary_sampling
        || !transport.unit_track_weights
        || !transport.detector_response_disabled
        || !transport.background_disabled
        || !transport.dead_time_disabled
    ) {
        throw std::invalid_argument(
            "Capture requires omitted detector/shields, full secondary "
            "transport, unit histories, and disabled response/background/"
            "dead-time sampling."
        );
    }
    if (
        !IsFiniteVector(geometry.boundary.center_m)
        || !std::isfinite(geometry.boundary.radius_m)
        || geometry.boundary.radius_m <= 0.0
    ) {
        throw std::invalid_argument(
            "Capture boundary must be finite and positive."
        );
    }
    ValidateBox(geometry.world_bounds_m, "World");
    const double radius = geometry.boundary.radius_m;
    const double tolerance = std::max(1.0e-9, radius * 1.0e-10);
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        if (
            geometry.boundary.center_m[axis] - radius
                <= geometry.world_bounds_m.minimum_m[axis] + tolerance
            || geometry.boundary.center_m[axis] + radius
                >= geometry.world_bounds_m.maximum_m[axis] - tolerance
        ) {
            throw std::invalid_argument(
                "Capture sphere must lie strictly inside the mass world."
            );
        }
    }
    if (geometry.source_positions_m.empty()) {
        throw std::invalid_argument(
            "Capture preflight requires every source position."
        );
    }
    for (const auto& source : geometry.source_positions_m) {
        if (!IsFiniteVector(source)) {
            throw std::invalid_argument(
                "Capture source positions must be finite."
            );
        }
        if (
            SquaredDistance(source, geometry.boundary.center_m)
            <= (radius + tolerance) * (radius + tolerance)
        ) {
            throw std::invalid_argument(
                "Every capture source must lie strictly outside the sphere."
            );
        }
    }
    if (geometry.enclosed_detector_and_shield_bounds_m.empty()) {
        throw std::invalid_argument(
            "Capture preflight requires detector/shield enclosure bounds."
        );
    }
    for (
        std::size_t box_index = 0;
        box_index < geometry.enclosed_detector_and_shield_bounds_m.size();
        ++box_index
    ) {
        const auto& box =
            geometry.enclosed_detector_and_shield_bounds_m[box_index];
        ValidateBox(box, "Detector/shield");
        for (const auto& corner : Corners(box)) {
            if (
                SquaredDistance(corner, geometry.boundary.center_m)
                >= (radius - tolerance) * (radius - tolerance)
            ) {
                throw std::invalid_argument(
                    "Capture sphere must strictly enclose every detector/"
                    "shield AABB."
                );
            }
        }
    }
}

CaptureCollector::CaptureCollector(const Boundary boundary)
    : boundary_(boundary) {
    if (
        !IsFiniteVector(boundary_.center_m)
        || !std::isfinite(boundary_.radius_m)
        || boundary_.radius_m <= 0.0
    ) {
        throw std::invalid_argument(
            "CaptureCollector requires a finite positive boundary."
        );
    }
}

std::shared_ptr<CaptureWorkerState> CaptureCollector::AcquireLocal() {
    static thread_local std::unordered_map<
        const CaptureCollector*,
        std::weak_ptr<CaptureWorkerState>
    > local_states;
    const auto existing = local_states.find(this);
    if (existing != local_states.end()) {
        if (auto state = existing->second.lock()) {
            return state;
        }
    }
    auto state = std::make_shared<CaptureWorkerState>(boundary_);
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        worker_states_.push_back(state);
    }
    local_states[this] = state;
    return state;
}

Bank CaptureCollector::FinalizeMerged(
    const DedicatedProfile& profile
) const {
    std::vector<Bank> worker_banks;
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        worker_banks.reserve(worker_states_.size());
        for (const auto& state : worker_states_) {
            if (state == nullptr) {
                throw std::runtime_error(
                    "Capture worker registry contains a null state."
                );
            }
            worker_banks.push_back(state->accumulator.Finalize());
        }
    }
    return MergeWorkerBanks(profile, worker_banks);
}

std::size_t CaptureCollector::WorkerCount() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return worker_states_.size();
}

CaptureParallelWorld::CaptureParallelWorld(
    const Boundary boundary,
    std::shared_ptr<CaptureCollector> collector,
    std::shared_ptr<const CaptureIdentityProvider> identity_provider
) : G4VUserParallelWorld(kParallelWorldName),
    boundary_(boundary),
    collector_(std::move(collector)),
    identity_provider_(std::move(identity_provider)) {
    if (collector_ == nullptr || identity_provider_ == nullptr) {
        throw std::invalid_argument(
            "CaptureParallelWorld requires collector and identity provider."
        );
    }
}

void CaptureParallelWorld::Construct() {
    auto* ghost_world = GetWorld();
    if (
        ghost_world == nullptr
        || ghost_world->GetLogicalVolume() == nullptr
    ) {
        throw std::runtime_error(
            "Geant4 did not provide the capture parallel world."
        );
    }
    auto* sphere_solid = new G4Sphere(
        kCaptureSphereLogicalName,
        0.0,
        boundary_.radius_m * m,
        0.0,
        CLHEP::twopi,
        0.0,
        CLHEP::pi
    );
    auto* sphere_logical = new G4LogicalVolume(
        sphere_solid,
        nullptr,
        kCaptureSphereLogicalName
    );
    new G4PVPlacement(
        nullptr,
        G4ThreeVector(
            boundary_.center_m[0] * m,
            boundary_.center_m[1] * m,
            boundary_.center_m[2] * m
        ),
        sphere_logical,
        kCaptureSpherePhysicalName,
        ghost_world->GetLogicalVolume(),
        false,
        0,
        false
    );
}

void CaptureParallelWorld::ConstructSD() {
    auto* ghost_world = GetWorld();
    if (
        ghost_world == nullptr
        || ghost_world->GetLogicalVolume() == nullptr
    ) {
        throw std::runtime_error(
            "Capture SD construction cannot resolve the ghost world."
        );
    }
    auto* detector = new CaptureSensitiveDetector(
        collector_,
        identity_provider_
    );
    G4SDManager::GetSDMpointer()->AddNewDetector(detector);
    SetSensitiveDetector(ghost_world->GetLogicalVolume(), detector);
}

void ValidateReplayPreflight(
    const DedicatedProfile& profile,
    const ReplayTransportContract& transport
) {
    RequireProfile(profile);
    if (transport.standard_runtime) {
        throw std::invalid_argument(
            "Paired phase-space replay is calibration-only."
        );
    }
    if (
        !transport.full_original_world
        || !transport.detector_present
        || !transport.selected_shields_present
        || !transport.full_secondary_transport
        || !transport.unit_track_weights
        || transport.capture_boundary_is_absorbing
        || transport.kill_outward_crossings
    ) {
        throw std::invalid_argument(
            "Replay requires the complete original world, detector, selected "
            "shields, full secondary transport, unit tracks, and unrestricted "
            "outward/reentry transport."
        );
    }
    if (
        !ReplaySchedule::FullWorldReplayRequired()
        || ReplaySchedule::KillOutwardCrossings()
    ) {
        throw std::logic_error(
            "ReplaySchedule static transport contract is inconsistent."
        );
    }
}

ReplayPrimaryInformation::ReplayPrimaryInformation(
    const std::uint64_t original_history_id,
    Crossing crossing
) : original_history_id_(original_history_id),
    crossing_(std::move(crossing)) {}

std::uint64_t ReplayPrimaryInformation::OriginalHistoryId() const noexcept {
    return original_history_id_;
}

const Crossing& ReplayPrimaryInformation::State() const noexcept {
    return crossing_;
}

void ReplayPrimaryInformation::Print() const {
    G4cout
        << "PairedAll64 replay primary history="
        << original_history_id_
        << " branch="
        << crossing_.branch_id
        << G4endl;
}

ReplayEventInformation::ReplayEventInformation(
    const std::uint64_t original_history_id,
    const std::uint32_t source_index,
    const std::uint32_t line_index,
    const std::uint32_t angle_stratum_index,
    const std::uint32_t angle_stratum_count,
    const double estimator_coefficient,
    const std::uint32_t shield_pair_id,
    const std::uint64_t random_seed
) : original_history_id_(original_history_id),
    source_index_(source_index),
    line_index_(line_index),
    angle_stratum_index_(angle_stratum_index),
    angle_stratum_count_(angle_stratum_count),
    estimator_coefficient_(estimator_coefficient),
    shield_pair_id_(shield_pair_id),
    random_seed_(random_seed) {
    if (
        angle_stratum_count_ == 0U
        || angle_stratum_index_ >= angle_stratum_count_
        || !std::isfinite(estimator_coefficient_)
        || estimator_coefficient_ <= 0.0
    ) {
        throw std::invalid_argument(
            "Replay event estimator identity is invalid."
        );
    }
}

std::uint64_t ReplayEventInformation::OriginalHistoryId() const noexcept {
    return original_history_id_;
}

std::uint32_t ReplayEventInformation::SourceIndex() const noexcept {
    return source_index_;
}

std::uint32_t ReplayEventInformation::LineIndex() const noexcept {
    return line_index_;
}

std::uint32_t ReplayEventInformation::AngleStratumIndex() const noexcept {
    return angle_stratum_index_;
}

std::uint32_t ReplayEventInformation::AngleStratumCount() const noexcept {
    return angle_stratum_count_;
}

double ReplayEventInformation::EstimatorCoefficient() const noexcept {
    return estimator_coefficient_;
}

std::uint32_t ReplayEventInformation::ShieldPairId() const noexcept {
    return shield_pair_id_;
}

std::uint64_t ReplayEventInformation::RandomSeed() const noexcept {
    return random_seed_;
}

void ReplayEventInformation::Print() const {
    G4cout
        << "PairedAll64 replay event history="
        << original_history_id_
        << " source="
        << source_index_
        << " line="
        << line_index_
        << " angle_stratum="
        << angle_stratum_index_
        << "/"
        << angle_stratum_count_
        << " external_coefficient="
        << estimator_coefficient_
        << " pair="
        << shield_pair_id_
        << " seed="
        << random_seed_
        << G4endl;
}

ReplayPrimaryGeneratorAction::ReplayPrimaryGeneratorAction(
    std::shared_ptr<const ReplaySchedule> schedule
) : schedule_(std::move(schedule)) {
    if (schedule_ == nullptr) {
        throw std::invalid_argument(
            "Replay primary generator requires a schedule."
        );
    }
}

void ReplayPrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
    if (event == nullptr) {
        throw std::invalid_argument(
            "Replay primary generator received a null event."
        );
    }
    const int event_id = event->GetEventID();
    if (
        event_id < 0
        || static_cast<std::size_t>(event_id) >= schedule_->EventCount()
    ) {
        throw std::out_of_range(
            "Replay Geant4 event ID is outside the schedule."
        );
    }
    if (event->GetUserInformation() != nullptr) {
        throw std::runtime_error(
            "Replay event already contains user information."
        );
    }
    const auto& replay_event = schedule_->Event(
        static_cast<std::size_t>(event_id)
    );
    G4Random::setTheSeed(ReplaySeedAsLong(replay_event.random_seed));
    event->SetUserInformation(new ReplayEventInformation(
        replay_event.original_history_id,
        replay_event.source_index,
        replay_event.line_index,
        replay_event.angle_stratum_index,
        replay_event.angle_stratum_count,
        replay_event.estimator_coefficient,
        schedule_->ShieldPairId(),
        replay_event.random_seed
    ));
    for (const auto& crossing : replay_event.primaries) {
        auto* vertex = new G4PrimaryVertex(
            G4ThreeVector(
                crossing.position_m[0] * m,
                crossing.position_m[1] * m,
                crossing.position_m[2] * m
            ),
            crossing.global_time_s * s
        );
        auto* definition = G4ParticleTable::GetParticleTable()->FindParticle(
            crossing.particle_name
        );
        if (
            definition == nullptr
            || definition->GetPDGEncoding() != crossing.pdg_code
        ) {
            delete vertex;
            throw std::runtime_error(
                "Replay cannot resolve the exact captured Geant4 particle "
                "definition."
            );
        }
        auto* primary = new G4PrimaryParticle(definition);
        primary->SetKineticEnergy(crossing.kinetic_energy_mev * MeV);
        primary->SetMass(crossing.mass_mev * MeV);
        primary->SetCharge(crossing.charge_eplus * eplus);
        primary->SetProperTime(crossing.proper_time_s * s);
        primary->SetMomentumDirection(G4ThreeVector(
            crossing.direction[0],
            crossing.direction[1],
            crossing.direction[2]
        ));
        primary->SetPolarization(G4ThreeVector(
            crossing.polarization[0],
            crossing.polarization[1],
            crossing.polarization[2]
        ));
        primary->SetWeight(1.0);
        primary->SetUserInformation(new ReplayPrimaryInformation(
            replay_event.original_history_id,
            crossing
        ));
        vertex->SetPrimary(primary);
        event->AddPrimaryVertex(vertex);
    }
}

const ReplayEventInformation* ReplayEventIdentity(const G4Event& event) {
    return dynamic_cast<const ReplayEventInformation*>(
        event.GetUserInformation()
    );
}

const ReplayPrimaryInformation* ReplayPrimaryIdentity(const G4Track& track) {
    const auto* dynamic_particle = track.GetDynamicParticle();
    const auto* primary = dynamic_particle == nullptr
        ? nullptr
        : dynamic_particle->GetPrimaryParticle();
    return primary == nullptr
        ? nullptr
        : dynamic_cast<const ReplayPrimaryInformation*>(
            primary->GetUserInformation()
        );
}

ReplayPairScoreMatrixCollector::ReplayPairScoreMatrixCollector(
    const ReplaySchedule& schedule,
    const std::size_t feature_count
) : feature_count_(feature_count) {
    if (feature_count_ == 0U) {
        throw std::invalid_argument(
            "Replay score feature count must be positive."
        );
    }
    history_ids_.reserve(schedule.EventCount());
    history_index_.reserve(schedule.EventCount());
    for (std::size_t index = 0; index < schedule.EventCount(); ++index) {
        const std::uint64_t history_id =
            schedule.Event(index).original_history_id;
        if (!history_index_.emplace(history_id, index).second) {
            throw std::invalid_argument(
                "Replay schedule contains a duplicate history ID."
            );
        }
        history_ids_.push_back(history_id);
    }
    if (history_ids_.empty()) {
        throw std::invalid_argument(
            "Replay score collector requires a nonempty schedule."
        );
    }
}

std::shared_ptr<ReplayScoreWorkerState>
ReplayPairScoreMatrixCollector::AcquireLocal() {
    static thread_local std::unordered_map<
        const ReplayPairScoreMatrixCollector*,
        std::weak_ptr<ReplayScoreWorkerState>
    > local_states;
    const auto existing = local_states.find(this);
    if (existing != local_states.end()) {
        if (auto state = existing->second.lock()) {
            return state;
        }
    }
    auto state = std::make_shared<ReplayScoreWorkerState>();
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        worker_states_.push_back(state);
    }
    local_states[this] = state;
    return state;
}

void ReplayPairScoreMatrixCollector::SubmitHistoryScores(
    const std::shared_ptr<ReplayScoreWorkerState>& worker,
    const std::uint64_t original_history_id,
    const std::vector<double>& feature_scores
) {
    if (worker == nullptr) {
        throw std::invalid_argument(
            "Replay score submission requires a worker state."
        );
    }
    if (history_index_.count(original_history_id) == 0U) {
        throw std::invalid_argument(
            "Replay score references a history outside the schedule."
        );
    }
    if (feature_scores.size() != feature_count_) {
        throw std::invalid_argument(
            "Replay score feature vector has the wrong size."
        );
    }
    for (const double value : feature_scores) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(
                "Replay history scores must be finite and nonnegative."
            );
        }
    }
    if (
        !worker->score_by_history.emplace(
            original_history_id,
            feature_scores
        ).second
    ) {
        throw std::invalid_argument(
            "Replay history scores were submitted more than once."
        );
    }
}

std::vector<double>
ReplayPairScoreMatrixCollector::FinalizePairScores() const {
    std::vector<double> matrix(
        history_ids_.size() * feature_count_,
        0.0
    );
    std::vector<bool> submitted(history_ids_.size(), false);
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (const auto& worker : worker_states_) {
            if (worker == nullptr) {
                throw std::runtime_error(
                    "Replay score worker registry contains a null state."
                );
            }
            for (const auto& [history_id, scores] : worker->score_by_history) {
                const auto index_iterator = history_index_.find(history_id);
                if (index_iterator == history_index_.end()) {
                    throw std::logic_error(
                        "Replay worker contains an unknown history."
                    );
                }
                const std::size_t row = index_iterator->second;
                if (submitted[row]) {
                    throw std::runtime_error(
                        "Replay history was completed on multiple workers."
                    );
                }
                submitted[row] = true;
                std::copy(
                    scores.begin(),
                    scores.end(),
                    matrix.begin()
                        + static_cast<std::ptrdiff_t>(row * feature_count_)
                );
            }
        }
    }
    if (
        !std::all_of(
            submitted.begin(),
            submitted.end(),
            [](const bool value) { return value; }
        )
    ) {
        throw std::runtime_error(
            "Replay pair is incomplete: every original history, including "
            "zero-primary events, must submit a score vector."
        );
    }
    return matrix;
}

const std::vector<std::uint64_t>&
ReplayPairScoreMatrixCollector::OriginalHistoryIds() const noexcept {
    return history_ids_;
}

std::size_t ReplayPairScoreMatrixCollector::FeatureCount() const noexcept {
    return feature_count_;
}

All64ReplayScoreCoordinator::All64ReplayScoreCoordinator(
    const DedicatedProfile& profile,
    const Bank& bank,
    const std::size_t feature_count
) : history_ids_(HistoryIds(bank)),
    history_identities_(HistoryEstimatorIdentities(bank)),
    feature_count_(feature_count),
    accumulator_(profile, history_identities_, feature_count_) {
    RequireProfile(profile);
    if (history_ids_.empty() || feature_count_ == 0U) {
        throw std::invalid_argument(
            "All-64 replay scores require histories and features."
        );
    }
}

void All64ReplayScoreCoordinator::SubmitCompletedPair(
    const std::uint32_t shield_pair_id,
    const ReplayPairScoreMatrixCollector& collector
) {
    if (
        collector.OriginalHistoryIds() != history_ids_
        || collector.FeatureCount() != feature_count_
    ) {
        throw std::invalid_argument(
            "Replay pair score contract differs from the capture bank."
        );
    }
    accumulator_.SubmitPairScores(
        shield_pair_id,
        collector.FinalizePairScores()
    );
}

bool All64ReplayScoreCoordinator::Complete() const noexcept {
    return accumulator_.Complete();
}

CrossPairStratifiedCovariance
All64ReplayScoreCoordinator::FinalizeExact(
    const std::string& score_semantics
) const {
    return accumulator_.FinalizeExact(score_semantics);
}

ApproximateCrossPairBlockDiagnostic
All64ReplayScoreCoordinator::FinalizeApproximateBlockDiagnostic(
    const std::size_t block_count
) const {
    return accumulator_.FinalizeApproximateBlockDiagnostic(block_count);
}

}  // namespace rotating_shield::paired_all64::geant4
