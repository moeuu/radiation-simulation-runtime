#include "paired_all64_geant4.hpp"

#include <G4Box.hh>
#include <G4EmStandardPhysics_option4.hh>
#include <G4Electron.hh>
#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4Gamma.hh>
#include <G4LogicalVolume.hh>
#include <G4MTRunManager.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4ParallelWorldPhysics.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleGun.hh>
#include <G4PhysListFactory.hh>
#include <G4Positron.hh>
#include <G4RunManager.hh>
#include <G4RunManagerFactory.hh>
#include <G4SystemOfUnits.hh>
#include <G4Track.hh>
#include <G4VModularPhysicsList.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace phase = rotating_shield::paired_all64;
namespace integration = rotating_shield::paired_all64::geant4;

namespace {

enum class CaptureMode {
    kNormal,
    kWeightedGamma,
    kElectron,
    kPositron,
};

std::unique_ptr<G4RunManager> retained_run_manager;

void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class TestIdentityProvider final
    : public integration::CaptureIdentityProvider {
public:
    integration::HistoryIdentity IdentityForEvent(
        const G4Event& event
    ) const override {
        return {
            static_cast<std::uint64_t>(event.GetEventID() + 100),
            0U,
            0U,
            0U,
            1U,
            0.25,
        };
    }

    integration::TrackLineage LineageForTrack(
        const G4Track& track
    ) const override {
        auto* manager = G4EventManager::GetEventManager();
        const auto* event = manager == nullptr
            ? nullptr
            : manager->GetConstCurrentEvent();
        if (event == nullptr) {
            throw std::runtime_error("Test lineage cannot resolve event.");
        }
        const std::uint64_t history_id =
            static_cast<std::uint64_t>(event->GetEventID() + 100);
        return {
            (history_id << 20U)
                | static_cast<std::uint64_t>(track.GetTrackID()),
            0U,
            0U,
            0U,
            0U,
            0U,
            false,
        };
    }
};

class TestDetectorConstruction final
    : public G4VUserDetectorConstruction {
public:
    TestDetectorConstruction(
        const phase::Boundary boundary,
        std::shared_ptr<integration::CaptureCollector> collector,
        std::shared_ptr<const integration::CaptureIdentityProvider> provider
    ) {
        RegisterParallelWorld(new integration::CaptureParallelWorld(
            boundary,
            std::move(collector),
            std::move(provider)
        ));
    }

    G4VPhysicalVolume* Construct() override {
        auto* vacuum = G4NistManager::Instance()->FindOrBuildMaterial(
            "G4_Galactic"
        );
        auto* solid = new G4Box("TestWorldSolid", 5.0 * m, 5.0 * m, 5.0 * m);
        auto* logical = new G4LogicalVolume(
            solid,
            vacuum,
            "TestWorldLogical"
        );
        return new G4PVPlacement(
            nullptr,
            {},
            logical,
            "TestWorldPhysical",
            nullptr,
            false,
            0,
            false
        );
    }
};

class TestCaptureGenerator final
    : public G4VUserPrimaryGeneratorAction {
public:
    explicit TestCaptureGenerator(const CaptureMode mode)
        : mode_(mode), gun_(1) {
        G4ParticleDefinition* definition = nullptr;
        if (mode_ == CaptureMode::kElectron) {
            definition = G4Electron::Definition();
        } else if (mode_ == CaptureMode::kPositron) {
            definition = G4Positron::Definition();
        } else {
            definition = G4Gamma::Definition();
        }
        gun_.SetParticleDefinition(definition);
        gun_.SetParticlePosition({2.0 * m, 0.0, 0.0});
        gun_.SetParticleEnergy(1.0 * MeV);
    }

    void GeneratePrimaries(G4Event* event) override {
        if (event == nullptr) {
            throw std::runtime_error("Test generator received null event.");
        }
        gun_.SetParticleMomentumDirection(
            event->GetEventID() == 0
                ? G4ThreeVector(-1.0, 0.0, 0.0)
                : G4ThreeVector(1.0, 0.0, 0.0)
        );
        gun_.GeneratePrimaryVertex(event);
        if (mode_ == CaptureMode::kWeightedGamma) {
            auto* vertex = event->GetPrimaryVertex();
            auto* primary = vertex == nullptr ? nullptr : vertex->GetPrimary();
            if (primary == nullptr) {
                throw std::runtime_error(
                    "Weighted test failed to construct a primary."
                );
            }
            primary->SetWeight(0.5);
        }
    }

private:
    CaptureMode mode_ = CaptureMode::kNormal;
    G4ParticleGun gun_;
};

class TestActionInitialization final
    : public G4VUserActionInitialization {
public:
    explicit TestActionInitialization(const CaptureMode mode) : mode_(mode) {}

    void Build() const override {
        SetUserAction(new TestCaptureGenerator(mode_));
    }

private:
    CaptureMode mode_ = CaptureMode::kNormal;
};

phase::Bank RunCaptureIntegration(
    const phase::DedicatedProfile& profile,
    const phase::Boundary& boundary,
    const CaptureMode mode,
    const int thread_count = 1
) {
    auto collector =
        std::make_shared<integration::CaptureCollector>(boundary);
    auto provider = std::make_shared<TestIdentityProvider>();
    retained_run_manager.reset(G4RunManagerFactory::CreateRunManager(
        thread_count > 1
            ? G4RunManagerType::MTOnly
            : G4RunManagerType::SerialOnly
    ));
    auto* run_manager = retained_run_manager.get();
    if (thread_count > 1) {
        auto* mt_manager = dynamic_cast<G4MTRunManager*>(run_manager);
        if (mt_manager == nullptr) {
            throw std::runtime_error(
                "Geant4 build cannot create the requested MT run manager."
            );
        }
        mt_manager->SetNumberOfThreads(thread_count);
    }
    run_manager->SetUserInitialization(new TestDetectorConstruction(
        boundary,
        collector,
        provider
    ));
    G4PhysListFactory factory;
    auto* physics = factory.GetReferencePhysList("FTFP_BERT");
    physics->ReplacePhysics(new G4EmStandardPhysics_option4());
    physics->RegisterPhysics(new G4ParallelWorldPhysics(
        integration::kParallelWorldName,
        false
    ));
    run_manager->SetUserInitialization(physics);
    run_manager->SetUserInitialization(new TestActionInitialization(mode));
    run_manager->Initialize();
    run_manager->BeamOn(mode == CaptureMode::kNormal ? 2 : 1);
    const phase::Bank bank = collector->FinalizeMerged(profile);
    return bank;
}

void TestPreflight(const phase::DedicatedProfile& profile) {
    integration::CaptureTransportContract capture_transport;
    capture_transport.standard_runtime = false;
    capture_transport.detector_and_shields_omitted = true;
    capture_transport.full_secondary_transport = true;
    capture_transport.unit_primary_sampling = true;
    capture_transport.unit_track_weights = true;
    capture_transport.detector_response_disabled = true;
    capture_transport.background_disabled = true;
    capture_transport.dead_time_disabled = true;
    integration::CaptureGeometryContract capture_geometry;
    capture_geometry.boundary = {{0.0, 0.0, 0.0}, 1.0};
    capture_geometry.world_bounds_m = {
        {-5.0, -5.0, -5.0},
        {5.0, 5.0, 5.0},
    };
    capture_geometry.source_positions_m = {{2.0, 0.0, 0.0}};
    capture_geometry.enclosed_detector_and_shield_bounds_m = {{
        {-0.2, -0.2, -0.2},
        {0.2, 0.2, 0.2},
    }};
    integration::ValidateCapturePreflight(
        profile,
        capture_transport,
        capture_geometry
    );

    integration::ReplayTransportContract replay_transport;
    replay_transport.standard_runtime = false;
    replay_transport.full_original_world = true;
    replay_transport.detector_present = true;
    replay_transport.selected_shields_present = true;
    replay_transport.full_secondary_transport = true;
    replay_transport.unit_track_weights = true;
    replay_transport.capture_boundary_is_absorbing = false;
    replay_transport.kill_outward_crossings = false;
    integration::ValidateReplayPreflight(profile, replay_transport);

    bool rejected_standard = false;
    try {
        capture_transport.standard_runtime = true;
        integration::ValidateCapturePreflight(
            profile,
            capture_transport,
            capture_geometry
        );
    } catch (const std::invalid_argument&) {
        rejected_standard = true;
    }
    Require(rejected_standard, "Standard runtime did not fail closed.");

    bool rejected_inside_source = false;
    try {
        capture_transport.standard_runtime = false;
        capture_geometry.source_positions_m = {{0.5, 0.0, 0.0}};
        integration::ValidateCapturePreflight(
            profile,
            capture_transport,
            capture_geometry
        );
    } catch (const std::invalid_argument&) {
        rejected_inside_source = true;
    }
    Require(
        rejected_inside_source,
        "A source inside the capture sphere did not fail preflight."
    );
}

void TestReplayAndScores(
    const phase::DedicatedProfile& profile,
    const phase::Bank& bank
) {
    const std::string bank_sha = phase::BankPayloadSha256(profile, bank);
    const std::uint64_t pair_seed =
        phase::DeriveReplaySeed(12345U, bank_sha, 0U);
    auto schedule = std::make_shared<phase::ReplaySchedule>(
        profile,
        bank,
        0U,
        pair_seed
    );
    integration::ReplayPrimaryGeneratorAction generator(schedule);
    G4Event first_event(0);
    generator.GeneratePrimaries(&first_event);
    const auto* first_identity =
        integration::ReplayEventIdentity(first_event);
    Require(first_identity != nullptr, "Replay event identity is missing.");
    Require(
        first_identity->OriginalHistoryId()
            == bank.histories.at(0).original_history_id,
        "Replay event history identity changed."
    );
    Require(
        first_identity->SourceIndex() == 0U
            && first_identity->LineIndex() == 0U
            && first_identity->AngleStratumIndex() == 0U
            && first_identity->AngleStratumCount() == 1U
            && first_identity->EstimatorCoefficient() == 0.25,
        "Replay event lost its external estimator identity."
    );
    Require(
        first_event.GetNumberOfPrimaryVertex() == 1,
        "Captured crossing did not become one replay vertex."
    );
    const auto* primary = first_event.GetPrimaryVertex()->GetPrimary();
    const auto* primary_identity =
        dynamic_cast<const integration::ReplayPrimaryInformation*>(
            primary->GetUserInformation()
        );
    Require(
        primary_identity != nullptr
            && primary_identity->OriginalHistoryId()
                == bank.histories.at(0).original_history_id,
        "Replay primary lineage is missing."
    );
    Require(primary->GetWeight() == 1.0, "Replay primary is weighted.");

    G4Event zero_event(1);
    generator.GeneratePrimaries(&zero_event);
    Require(
        zero_event.GetNumberOfPrimaryVertex() == 0,
        "Zero-crossing history gained a synthetic primary."
    );
    Require(
        integration::ReplayEventIdentity(zero_event) != nullptr,
        "Zero-primary event lost its original-history identity."
    );

    integration::All64ReplayScoreCoordinator coordinator(
        profile,
        bank,
        2U
    );
    for (
        std::uint32_t pair = 0U;
        pair < phase::kShieldPairCount;
        ++pair
    ) {
        const std::uint64_t seed =
            phase::DeriveReplaySeed(12345U, bank_sha, pair);
        phase::ReplaySchedule pair_schedule(
            profile,
            bank,
            pair,
            seed
        );
        integration::ReplayPairScoreMatrixCollector collector(
            pair_schedule,
            2U
        );
        auto worker = collector.AcquireLocal();
        collector.SubmitHistoryScores(
            worker,
            bank.histories.at(0).original_history_id,
            {
                static_cast<double>(pair + 1U),
                2.0,
            }
        );
        collector.SubmitHistoryScores(
            worker,
            bank.histories.at(1).original_history_id,
            {0.0, 0.0}
        );
        coordinator.SubmitCompletedPair(pair, collector);
    }
    Require(coordinator.Complete(), "All-64 score set is incomplete.");
    const auto covariance = coordinator.FinalizeExact(
        phase::kCovarianceSemantics
    );
    Require(
        covariance.history_count == 2U
            && covariance.group_count == 1U
            && covariance.feature_count == 2U
            && !covariance.artifact_sha256.empty(),
        "All-64 covariance artifact is incomplete."
    );
    const auto approximate =
        coordinator.FinalizeApproximateBlockDiagnostic(2U);
    Require(
        approximate.semantics
            == phase::kApproximateBlockDiagnosticSemantics,
        "Optional pooled diagnostic is not marked approximate."
    );
}

void VerifyCapturedParticleRoundTrip(
    const phase::DedicatedProfile& profile,
    const phase::Bank& bank,
    const std::string& expected_name,
    const int expected_pdg,
    const double expected_charge
) {
    const auto payload = phase::SerializeBank(profile, bank);
    const auto restored = phase::DeserializeBank(profile, payload);
    const std::string digest = phase::BankPayloadSha256(profile, restored);
    auto schedule = std::make_shared<phase::ReplaySchedule>(
        profile,
        restored,
        0U,
        phase::DeriveReplaySeed(811U, digest, 0U)
    );
    integration::ReplayPrimaryGeneratorAction generator(schedule);
    G4Event replay_event(0);
    generator.GeneratePrimaries(&replay_event);
    Require(
        replay_event.GetNumberOfPrimaryVertex() == 1,
        "Transported particle did not round-trip to one replay primary."
    );
    const auto* replay_primary =
        replay_event.GetPrimaryVertex()->GetPrimary();
    Require(
        replay_primary != nullptr
            && replay_primary->GetParticleDefinition() != nullptr
            && replay_primary->GetParticleDefinition()->GetParticleName()
                == expected_name
            && replay_primary->GetPDGcode() == expected_pdg
            && std::abs(replay_primary->GetCharge() / eplus - expected_charge)
                < 1.0e-12,
        "Replay primary species, PDG, or charge changed during round-trip."
    );
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const std::string mode_name = argc > 1 ? argv[1] : "normal";
        const bool multi_threaded = mode_name == "mt";
        const CaptureMode mode = mode_name == "weighted"
            ? CaptureMode::kWeightedGamma
            : (
                mode_name == "non_gamma" || mode_name == "electron"
                    ? CaptureMode::kElectron
                    : (
                        mode_name == "positron"
                            ? CaptureMode::kPositron
                            : CaptureMode::kNormal
                    )
            );
        if (
            mode_name != "normal"
            && mode_name != "mt"
            && mode_name != "weighted"
            && mode_name != "non_gamma"
            && mode_name != "electron"
            && mode_name != "positron"
        ) {
            throw std::invalid_argument("Unknown capture test mode.");
        }
        const auto profile = phase::RequireDedicatedProfile(
            phase::kDedicatedProfile,
            false
        );
        const phase::Boundary boundary = {{0.0, 0.0, 0.0}, 1.0};
        if (mode == CaptureMode::kWeightedGamma) {
            (void)RunCaptureIntegration(profile, boundary, mode);
            throw std::runtime_error(
                "Invalid capture input unexpectedly produced a bank."
            );
        }
        if (
            mode == CaptureMode::kElectron
            || mode == CaptureMode::kPositron
        ) {
            const phase::Bank particle_bank = RunCaptureIntegration(
                profile,
                boundary,
                mode
            );
            Require(
                particle_bank.histories.size() == 1U
                    && particle_bank.histories.at(0).crossings.size() == 1U,
                "Non-gamma transport branch was not captured."
            );
            const auto& particle =
                particle_bank.histories.at(0).crossings.at(0);
            const bool electron = mode == CaptureMode::kElectron;
            const int expected_pdg = electron ? 11 : -11;
            const std::string expected_name = electron ? "e-" : "e+";
            const double expected_charge = electron ? -1.0 : 1.0;
            Require(
                particle.pdg_code == expected_pdg
                    && particle.particle_name == expected_name
                    && std::abs(particle.mass_mev - 0.51099895) < 1.0e-6
                    && std::abs(
                        particle.charge_eplus - expected_charge
                    ) < 1.0e-12,
                "Captured charged-particle restart state is incomplete."
            );
            VerifyCapturedParticleRoundTrip(
                profile,
                particle_bank,
                expected_name,
                expected_pdg,
                expected_charge
            );
            retained_run_manager.reset();
            std::cout
                << (
                    electron
                        ? "paired_all64_geant4_electron_ok\n"
                        : "paired_all64_geant4_positron_ok\n"
                );
            return 0;
        }
        TestPreflight(profile);
        const phase::Bank bank = RunCaptureIntegration(
            profile,
            boundary,
            mode,
            multi_threaded ? 2 : 1
        );
        Require(bank.histories.size() == 2U, "Capture lost a history.");
        Require(
            bank.histories.at(0).original_history_id == 100U
                && bank.histories.at(0).crossings.size() == 1U
                && bank.histories.at(0).angle_stratum_index == 0U
                && bank.histories.at(0).angle_stratum_count == 1U
                && bank.histories.at(0).estimator_coefficient == 0.25,
            "Inward history was not captured exactly once."
        );
        Require(
            bank.histories.at(1).original_history_id == 101U
                && bank.histories.at(1).crossings.empty(),
            "Zero-crossing history was not preserved."
        );
        const auto& crossing = bank.histories.at(0).crossings.at(0);
        const double radius = std::sqrt(
            crossing.position_m[0] * crossing.position_m[0]
            + crossing.position_m[1] * crossing.position_m[1]
            + crossing.position_m[2] * crossing.position_m[2]
        );
        Require(
            std::abs(radius - 1.0) < 1.0e-7,
            "Parallel-world crossing is not on the capture sphere."
        );
        Require(
            std::abs(crossing.position_m[0] - 1.0) < 1.0e-7
                && std::abs(crossing.position_m[1]) < 1.0e-12
                && std::abs(crossing.position_m[2]) < 1.0e-12
                && std::abs(crossing.direction[0] + 1.0) < 1.0e-12
                && std::abs(crossing.kinetic_energy_mev - 1.0) < 1.0e-12
                && crossing.gamma_interaction_count == 0U
                && !phase::HasInteractionFlag(
                    crossing.interaction_flags,
                    phase::InteractionFlags::kInteracted
                ),
            "Capture state is not the interaction-free inward post-step "
            "boundary state."
        );
        const double expected_time_s =
            1.0 / 299792458.0;
        Require(
            std::abs(crossing.global_time_s - expected_time_s) < 1.0e-14,
            "Capture time is not the exact one-metre boundary flight time."
        );
        TestReplayAndScores(profile, bank);
        retained_run_manager.reset();
        std::cout
            << (
                multi_threaded
                    ? "paired_all64_geant4_mt_ok\n"
                    : "paired_all64_geant4_ok\n"
            );
        return 0;
    } catch (const std::exception& error) {
        retained_run_manager.reset();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
