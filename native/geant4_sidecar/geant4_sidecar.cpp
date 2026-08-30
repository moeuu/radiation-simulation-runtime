#include <G4Box.hh>
#include <G4BOptrForceCollision.hh>
#include <G4BOptrForceCollisionTrackData.hh>
#include <G4BiasingProcessInterface.hh>
#include <G4EmCalculator.hh>
#include <G4EmParameters.hh>
#include <G4EmStandardPhysics_option4.hh>
#include <G4Event.hh>
#include <G4Gamma.hh>
#include <G4GenericBiasingPhysics.hh>
#include <G4GeometryManager.hh>
#include <G4HadronicParameters.hh>
#include <G4IonTable.hh>
#include <G4LogicalVolume.hh>
#include <G4LossTableManager.hh>
#include <G4Material.hh>
#ifdef G4MULTITHREADED
#include <G4MTRunManager.hh>
#endif
#include <G4NistManager.hh>
#include <G4NuclearLevelData.hh>
#include <G4NuclideTable.hh>
#include <G4PVPlacement.hh>
#include <G4ParticleGun.hh>
#include <G4PhysListFactory.hh>
#include <G4PhysicsModelCatalog.hh>
#include <G4PrimaryParticle.hh>
#include <G4PrimaryVertex.hh>
#include <G4ProcessType.hh>
#include <G4ProcessManager.hh>
#include <G4RotationMatrix.hh>
#include <G4RadioactiveDecayPhysics.hh>
#include <G4RadioactiveDecay.hh>
#include <G4GenericIon.hh>
#include <G4GammaGeneralProcess.hh>
#include <G4PhysicsListHelper.hh>
#include <G4Triton.hh>
#include <G4VPhysicsConstructor.hh>
#include <G4RunManager.hh>
#include <G4RunManagerFactory.hh>
#include <G4SDManager.hh>
#include <G4Sphere.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4TessellatedSolid.hh>
#include <G4ThreeVector.hh>
#include <G4TriangularFacet.hh>
#include <G4Track.hh>
#include <G4TrackingManager.hh>
#include <G4Types.hh>
#include <G4UserEventAction.hh>
#include <G4UserStackingAction.hh>
#include <G4UserSteppingAction.hh>
#include <G4UserTrackingAction.hh>
#include <G4VModularPhysicsList.hh>
#include <G4VAtomDeexcitation.hh>
#include <G4VBiasingOperation.hh>
#include <G4VBiasingOperator.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VParticleChange.hh>
#include <G4VProcess.hh>
#include <G4VSensitiveDetector.hh>
#include <G4VUserTrackInformation.hh>
#include <G4UAtomicDeexcitation.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4VisAttributes.hh>
#include <G4Version.hh>
#include <G4ios.hh>
#include <Randomize.hh>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kDefaultCrystalRadiusM = 0.038;
constexpr double kDefaultCrystalLengthM = 0.076;
constexpr double kDefaultHousingThicknessM = 0.0015;
constexpr double kProductionCutRangeMm = 0.7;
constexpr const char* kReferencePhysicsListName = "FTFP_BERT";
constexpr const char* kElectromagneticPhysicsConstructorName =
    "G4EmStandardPhysics_option4";
constexpr const char* kGeant4VersionTag = "geant4-11-03-patch-02";
constexpr const char* kGeant4PhysicsContractId =
    "geant4_11_3_2_ftfp_bert_em_option4_cut_0p7mm_v1";
constexpr const char* kGeant4PhysicsContractSha256 =
    "7b21112ff768478e38c798affb42ccacd7212f308397d44c1de045cc9f158550";
constexpr const char* kMaterialResolutionContractId =
    "exported_density_mass_composition_except_g4_air_v1";
constexpr double kDefaultDetectorCoincidenceWindowS = 1.0e-6;
constexpr double kDefaultShieldContactRadiusM = kDefaultCrystalRadiusM + kDefaultHousingThicknessM;
constexpr double kDefaultShieldTransmissionScale = 0.6989700043360189;
constexpr double kDefaultFeShieldTvlThicknessM = 0.05;
constexpr double kDefaultPbShieldTvlThicknessM = 0.022;
constexpr double kDefaultFeShieldThicknessM =
    kDefaultFeShieldTvlThicknessM * kDefaultShieldTransmissionScale;
constexpr double kDefaultPbShieldThicknessM =
    kDefaultPbShieldTvlThicknessM * kDefaultShieldTransmissionScale;
constexpr double kDefaultFeShieldInnerRadiusM = kDefaultShieldContactRadiusM;
constexpr double kDefaultPbShieldInnerRadiusM =
    kDefaultFeShieldInnerRadiusM + kDefaultFeShieldThicknessM;
constexpr double kWorldDaughterMarginM = 0.5;
constexpr const char* kShieldPoseContractId =
    "spherical_octant_positive_xyz_incoming_index_v1";
constexpr const char* kShieldPoseContractSha256 =
    "0732f12d1f2aa83607560643484652da6ba942d02a0fdfdf4b0fda4e6d3116fd";
constexpr const char* kNativeActionIdentityContractId =
    "geant4_native_action_identity_v1";

struct MaterialSpec {
    std::string name;
    double density_g_cm3 = -1.0;
    std::string preset_name;
    std::map<std::string, double> composition_by_mass;
};

struct VolumeSpec {
    std::string path;
    std::string shape;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double sx = -1.0;
    double sy = -1.0;
    double sz = -1.0;
    double radius_m = -1.0;
    MaterialSpec material;
    std::vector<std::array<double, 9>> triangles;
    std::string transport_group;
    std::string transport_mode = "geant4";
};

struct LineSpec {
    double energy_keV;
    double intensity;
};

struct SourceSpec {
    std::string isotope;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double intensity_cps_1m = std::numeric_limits<double>::quiet_NaN();
    double activity_bq = std::numeric_limits<double>::quiet_NaN();
    double anchor_x = 0.0;
    double anchor_y = 0.0;
    double anchor_z = 0.0;
    long surface_chart_id = -1;
    double surface_u = -1.0;
    double surface_v = -1.0;
    double surface_normal_x = 0.0;
    double surface_normal_y = 0.0;
    double surface_normal_z = 0.0;
    double surface_emission_epsilon_m = 0.0;
    std::string surface_emission_policy_sha256;
};

struct NuclideSpec {
    std::string isotope;
    int atomic_number = 0;
    int mass_number = 0;
    double geant4_excitation_keV = 0.0;
    double half_life_s = 0.0;
    std::string prompt_cascade_model;
    std::vector<LineSpec> gamma_lines;
    std::vector<LineSpec> transport_gamma_lines;
};

struct DetectorSpec {
    double crystal_radius_m = kDefaultCrystalRadiusM;
    double crystal_length_m = kDefaultCrystalLengthM;
    double housing_thickness_m = kDefaultHousingThicknessM;
    double coincidence_window_s = kDefaultDetectorCoincidenceWindowS;
    std::string crystal_shape = "sphere";
    std::string crystal_material = "cebr3";
    std::string housing_material = "aluminum";
};

struct ShieldSpec {
    std::string kind;
    std::string path;
    std::string shape = "spherical_octant_shell";
    double inner_radius_m = kDefaultFeShieldInnerRadiusM;
    double outer_radius_m = kDefaultFeShieldInnerRadiusM + kDefaultFeShieldThicknessM;
    double thickness_m = kDefaultFeShieldThicknessM;
    double sx = 0.25;
    double sy = 0.08;
    double sz = 0.25;
    MaterialSpec material;
};

ShieldSpec DefaultFeShieldSpec() {
    ShieldSpec shield;
    shield.kind = "fe";
    shield.shape = "spherical_octant_shell";
    shield.inner_radius_m = kDefaultFeShieldInnerRadiusM;
    shield.thickness_m = kDefaultFeShieldThicknessM;
    shield.outer_radius_m = shield.inner_radius_m + shield.thickness_m;
    shield.material.name = "fe";
    shield.material.preset_name = "iron";
    return shield;
}

ShieldSpec DefaultPbShieldSpec() {
    ShieldSpec shield;
    shield.kind = "pb";
    shield.shape = "spherical_octant_shell";
    shield.inner_radius_m = kDefaultPbShieldInnerRadiusM;
    shield.thickness_m = kDefaultPbShieldThicknessM;
    shield.outer_radius_m = shield.inner_radius_m + shield.thickness_m;
    shield.material.name = "pb";
    shield.material.preset_name = "lead";
    return shield;
}

std::optional<ShieldSpec> ValidateParsedShield(
    ShieldSpec shield,
    const std::string& use_angle_attenuation
) {
    if (shield.kind != "fe" && shield.kind != "pb") {
        throw std::runtime_error("SHIELD kind must be exactly fe or pb.");
    }
    if (shield.path.empty()) {
        throw std::runtime_error("SHIELD path must be nonempty.");
    }
    if (shield.shape != "spherical_octant_shell") {
        throw std::runtime_error(
            "SHIELD shape must be exactly spherical_octant_shell."
        );
    }
    if (use_angle_attenuation != "0") {
        throw std::runtime_error(
            "SHIELD use_angle_attenuation must be exactly 0."
        );
    }
    if (
        !std::isfinite(shield.inner_radius_m)
        || shield.inner_radius_m < 0.0
        || !std::isfinite(shield.outer_radius_m)
        || !std::isfinite(shield.thickness_m)
        || shield.thickness_m < 0.0
    ) {
        throw std::runtime_error(
            "SHIELD radii and thickness must be finite and nonnegative."
        );
    }
    const double expected_outer_radius_m =
        shield.inner_radius_m + shield.thickness_m;
    if (shield.outer_radius_m != expected_outer_radius_m) {
        throw std::runtime_error(
            "SHIELD outer_radius_m must equal inner_radius_m + thickness_m "
            "exactly."
        );
    }
    if (shield.thickness_m == 0.0) {
        return std::nullopt;
    }
    if (shield.material.name.empty() && shield.material.preset_name.empty()) {
        throw std::runtime_error(
            "Positive-thickness SHIELD must declare a material."
        );
    }
    return shield;
}

struct PoseSpec {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
};

class RuntimeDetectorState {
public:
    void Update(const PoseSpec& pose) {
        if (
            !std::isfinite(pose.x)
            || !std::isfinite(pose.y)
            || !std::isfinite(pose.z)
        ) {
            throw std::runtime_error(
                "Runtime detector center requires finite coordinates."
            );
        }
        generation_.fetch_add(1, std::memory_order_acq_rel);
        x_.store(pose.x * m, std::memory_order_relaxed);
        y_.store(pose.y * m, std::memory_order_relaxed);
        z_.store(pose.z * m, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
    }

    G4ThreeVector Center() const {
        while (true) {
            const auto before = generation_.load(std::memory_order_acquire);
            if ((before & 1U) != 0U) {
                continue;
            }
            const G4ThreeVector center(
                x_.load(std::memory_order_relaxed),
                y_.load(std::memory_order_relaxed),
                z_.load(std::memory_order_relaxed)
            );
            const auto after = generation_.load(std::memory_order_acquire);
            if (before == after) {
                return center;
            }
        }
    }

private:
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<double> x_{0.0};
    std::atomic<double> y_{0.0};
    std::atomic<double> z_{0.0};
};

struct SceneSpec {
    std::string scene_hash;
    std::string surface_source_contract_sha256;
    std::string nuclide_catalog_sha256;
    std::string usd_path;
    double room_x = 10.0;
    double room_y = 20.0;
    double room_z = 10.0;
    DetectorSpec detector;
    std::optional<ShieldSpec> fe_shield;
    std::optional<ShieldSpec> pb_shield;
    std::vector<SourceSpec> sources;
    std::map<std::string, NuclideSpec> nuclides;
    std::vector<VolumeSpec> volumes;
};

struct RequestSpec {
    int step_id = -1;
    double dwell_time_s = std::numeric_limits<double>::quiet_NaN();
    long seed = -1;
    PoseSpec detector_pose;
    PoseSpec fe_pose;
    PoseSpec pb_pose;
    bool has_step = false;
    bool has_detector_pose = false;
    bool has_fe_pose = false;
    bool has_pb_pose = false;
    std::string shield_pose_contract_id;
    std::string shield_pose_contract_sha256;
    std::string native_action_contract_id;
    std::string native_action_sha256;
    int fe_orientation_index = -1;
    int pb_orientation_index = -1;
};

struct SimulationResult {
    std::vector<double> spectrum_counts;
    std::vector<double> spectrum_count_variance;
    std::map<std::string, std::string> metadata;
};

struct TransportOptions {
    double background_cps = 0.0;
    std::string source_rate_model = "detector_cps_1m";
    std::string source_bias_mode = "detector_cone";
    std::string source_bias_cone_policy = "detector_covering";
    double source_bias_isotropic_fraction = 1.0;
    std::string detector_scoring_mode = "full_transport";
    std::string secondary_transport_mode = "full_transport";
    double primary_sampling_fraction = 1.0;
    long long target_sampled_primaries = 0;
    long long mean_calibration_histories_per_source_line = 0;
    int mean_calibration_angle_strata_mu = 1;
    int mean_calibration_angle_strata_phi = 1;
    bool mean_calibration_forced_collision = false;
    bool validation_entry_class_spectra = false;
    bool sample_detector_response = false;
    std::string detector_green_operator_path;
    std::string detector_green_operator_binary_sha256;
    std::string detector_green_operator_contract_sha256;
    std::string primary_emission_model = "independent_gamma_lines";
    bool decay_comparison_diagnostic = false;
    double decay_comparison_energy_max_keV = 3400.0;
    bool decay_comparison_energy_max_overridden = false;
};

enum class DetectorEntryClass {
    kUncollidedPrimary = 0,
    kInteractedPrimary = 1,
    kSecondary = 2,
};

std::size_t DetectorEntryClassIndex(const DetectorEntryClass entry_class) {
    const int index = static_cast<int>(entry_class);
    if (index < 0 || index > 2) {
        throw std::runtime_error("Detector entry class is outside its support.");
    }
    return static_cast<std::size_t>(index);
}

std::string DetectorEntryClassToken(const DetectorEntryClass entry_class) {
    if (entry_class == DetectorEntryClass::kUncollidedPrimary) {
        return "uncollided_primary";
    }
    if (entry_class == DetectorEntryClass::kInteractedPrimary) {
        return "interacted_primary";
    }
    if (entry_class == DetectorEntryClass::kSecondary) {
        return "secondary";
    }
    throw std::runtime_error("Detector entry class is outside its support.");
}

struct EnergyDeposit {
    double energy_keV = 0.0;
    double weight = 1.0;
    DetectorEntryClass entry_class = DetectorEntryClass::kUncollidedPrimary;
    std::string isotope;
    std::string source_token;
    std::string line_token;
    std::size_t primary_batch_index =
        std::numeric_limits<std::size_t>::max();
    long long primary_history_index = -1;
    long long bias_branch_lineage_id = -1;
    long long step_deposit_count = 0;
    double global_time_s = 0.0;
    double impact_parameter_fraction =
        std::numeric_limits<double>::quiet_NaN();
};

struct WeightedEventDeposit {
    double edep_mev = 0.0;
    double weight = 1.0;
    DetectorEntryClass entry_class = DetectorEntryClass::kUncollidedPrimary;
    std::size_t primary_batch_index = std::numeric_limits<std::size_t>::max();
    long long primary_history_index = -1;
    long long bias_branch_lineage_id = -1;
    double global_time_s = 0.0;
    long long step_deposit_count = 0;
    double primary_event_time_s = 0.0;
    double impact_parameter_fraction =
        std::numeric_limits<double>::quiet_NaN();
};

DetectorEntryClass MergeDetectorEntryClass(
    const DetectorEntryClass current,
    const DetectorEntryClass incoming
) {
    return static_cast<int>(incoming) > static_cast<int>(current) ? incoming : current;
}

std::string NormalizeToken(const std::string& token) {
    std::string result = token;
    std::size_t pos = 0;
    while ((pos = result.find("%20", pos)) != std::string::npos) {
        result.replace(pos, 3, " ");
        pos += 1;
    }
    return result;
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool IsLowercaseSha256(const std::string& value) {
    if (value.size() != 64U) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](const char character) {
        return (
            (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f')
        );
    });
}

std::string NormalizeModeToken(std::string value) {
    value = ToLower(std::move(value));
    for (char& ch : value) {
        if (ch == '-') {
            ch = '_';
        }
    }
    return value;
}

std::string JoinSet(const std::set<std::string>& values, const std::string& separator) {
    std::ostringstream stream;
    bool first = true;
    for (const auto& value : values) {
        if (!first) {
            stream << separator;
        }
        stream << value;
        first = false;
    }
    return stream.str();
}

std::string SerializeDouble(const double value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return stream.str();
}

std::string SerializeDoubleVector(const std::vector<double>& values) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << values[index];
    }
    return stream.str();
}

std::string SanitizeMetadataToken(std::string value) {
    for (char& ch : value) {
        if (std::isspace(static_cast<unsigned char>(ch)) || ch == ',' || ch == '=') {
            ch = '_';
        }
    }
    return value.empty() ? "unknown" : value;
}

std::string SerializeCounterMap(const std::map<std::string, long long>& values) {
    if (values.empty()) {
        return "-";
    }
    std::ostringstream stream;
    bool first = true;
    for (const auto& item : values) {
        if (!first) {
            stream << ",";
        }
        first = false;
        stream << SanitizeMetadataToken(item.first) << ":" << item.second;
    }
    return stream.str();
}

std::string SerializeSparseBinCounts(const std::map<int, long long>& values) {
    if (values.empty()) {
        return "-";
    }
    std::ostringstream stream;
    bool first = true;
    for (const auto& item : values) {
        if (item.first < 0 || item.second < 0) {
            throw std::runtime_error(
                "Sparse spectrum counts require nonnegative bins and counts."
            );
        }
        if (!first) {
            stream << ",";
        }
        first = false;
        stream << item.first << ":" << item.second;
    }
    return stream.str();
}

std::string SerializeSparseDoubleMap(const std::map<int, double>& values) {
    if (values.empty()) {
        return "-";
    }
    std::ostringstream stream;
    stream << std::setprecision(17);
    bool first = true;
    for (const auto& item : values) {
        if (
            item.first < 0
            || !std::isfinite(item.second)
            || item.second < 0.0
        ) {
            throw std::runtime_error(
                "Sparse moments require nonnegative indices and values."
            );
        }
        if (item.second == 0.0) {
            continue;
        }
        if (!first) {
            stream << ",";
        }
        first = false;
        stream << item.first << ":" << item.second;
    }
    return first ? "-" : stream.str();
}

std::string SerializeSparseSecondMoments(
    const std::map<std::pair<int, int>, double>& values
) {
    if (values.empty()) {
        return "-";
    }
    std::ostringstream stream;
    stream << std::setprecision(17);
    bool first = true;
    for (const auto& item : values) {
        if (
            item.first.first < 0
            || item.first.second < item.first.first
            || !std::isfinite(item.second)
            || item.second < 0.0
        ) {
            throw std::runtime_error(
                "Sparse second moments require ordered nonnegative indices "
                "and values."
            );
        }
        if (item.second == 0.0) {
            continue;
        }
        if (!first) {
            stream << ",";
        }
        first = false;
        stream << item.first.first << ":" << item.first.second << ":"
               << item.second;
    }
    return first ? "-" : stream.str();
}

std::string SerializeTopCounterMap(
    const std::map<std::string, long long>& values,
    const std::size_t limit
) {
    std::vector<std::pair<std::string, long long>> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second > rhs.second;
        }
        return lhs.first < rhs.first;
    });
    if (sorted.size() > limit) {
        sorted.resize(limit);
    }
    std::map<std::string, long long> top(sorted.begin(), sorted.end());
    return SerializeCounterMap(top);
}

std::map<std::string, std::string> ParseFields(const std::vector<std::string>& tokens, std::size_t start_index = 1) {
    std::map<std::string, std::string> fields;
    for (std::size_t index = start_index; index < tokens.size(); ++index) {
        const auto separator = tokens[index].find('=');
        if (separator == std::string::npos) {
            continue;
        }
        fields[tokens[index].substr(0, separator)] = NormalizeToken(tokens[index].substr(separator + 1));
    }
    return fields;
}

void RequireExactFields(
    const std::map<std::string, std::string>& fields,
    const std::set<std::string>& expected,
    const std::string& record_type
) {
    std::set<std::string> actual;
    for (const auto& entry : fields) {
        actual.insert(entry.first);
    }
    if (actual != expected) {
        throw std::runtime_error(
            "Request " + record_type + " fields are missing or unknown."
        );
    }
}

std::vector<std::string> Split(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

double ParseDouble(const std::map<std::string, std::string>& fields, const std::string& key, double fallback = 0.0) {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second == "-") {
        return fallback;
    }
    return std::stod(it->second);
}

long ParseLong(const std::map<std::string, std::string>& fields, const std::string& key, long fallback = 0) {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second == "-") {
        return fallback;
    }
    return std::stol(it->second);
}

std::string ParseString(const std::map<std::string, std::string>& fields, const std::string& key, const std::string& fallback = "") {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second == "-") {
        return fallback;
    }
    return it->second;
}

std::vector<LineSpec> GammaLinesForIsotope(
    const SceneSpec& scene,
    const std::string& isotope
) {
    const auto evaluated = scene.nuclides.find(isotope);
    if (evaluated == scene.nuclides.end()) {
        throw std::runtime_error(
            "Source isotope is absent from the authenticated nuclide "
            "catalog: " + isotope
        );
    }
    if (evaluated->second.transport_gamma_lines.empty()) {
        throw std::runtime_error(
            "Evaluated nuclide has no positive transport gamma lines: "
            + isotope
        );
    }
    return evaluated->second.transport_gamma_lines;
}

double SigmaEnergyKeV(const double energy_keV) {
    return std::max(0.5 * std::sqrt(std::max(0.0, energy_keV)) - 1.5, 0.5);
}

template <typename Value>
Value ReadDetectorGreenValue(std::ifstream& input) {
    Value value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(Value));
    if (!input) {
        throw std::runtime_error("Detector Green operator binary is truncated.");
    }
    return value;
}

class DetectorGreenOperator {
public:
    explicit DetectorGreenOperator(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "Failed to open detector Green operator: " + path
            );
        }
        std::array<char, 8> magic{};
        input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        const std::array<char, 8> expected_magic = {
            'R', 'S', 'G', 'K', 'V', '3', '\0', '\0'
        };
        const auto schema_version = ReadDetectorGreenValue<std::uint32_t>(input);
        energy_node_count_ = ReadDetectorGreenValue<std::uint32_t>(input);
        impact_bin_count_ = ReadDetectorGreenValue<std::uint32_t>(input);
        output_bin_count_ = ReadDetectorGreenValue<std::uint32_t>(input);
        output_energy_min_keV_ = ReadDetectorGreenValue<double>(input);
        output_bin_width_keV_ = ReadDetectorGreenValue<double>(input);
        const double domain_min_keV = ReadDetectorGreenValue<double>(input);
        const double domain_max_keV = ReadDetectorGreenValue<double>(input);
        if (
            magic != expected_magic
            || schema_version != 3U
            || energy_node_count_ < 2U
            || impact_bin_count_ < 1U
            || output_bin_count_ != 851U
            || output_energy_min_keV_ != 0.0
            || output_bin_width_keV_ != 2.0
            || domain_min_keV != 0.0
            || domain_max_keV != 1700.0
        ) {
            throw std::runtime_error(
                "Detector Green operator header is incompatible."
            );
        }
        energy_nodes_keV_.reserve(energy_node_count_);
        for (std::uint32_t index = 0; index < energy_node_count_; ++index) {
            energy_nodes_keV_.push_back(
                ReadDetectorGreenValue<double>(input)
            );
        }
        impact_edges_fraction_.reserve(impact_bin_count_ + 1U);
        for (std::uint32_t index = 0; index <= impact_bin_count_; ++index) {
            impact_edges_fraction_.push_back(
                ReadDetectorGreenValue<double>(input)
            );
        }
        if (
            energy_nodes_keV_.front() != 0.0
            || energy_nodes_keV_.back() != 1700.0
            || !std::is_sorted(
                energy_nodes_keV_.begin(),
                energy_nodes_keV_.end()
            )
            || std::adjacent_find(
                energy_nodes_keV_.begin(),
                energy_nodes_keV_.end()
            ) != energy_nodes_keV_.end()
            || impact_edges_fraction_.front() != 0.0
            || impact_edges_fraction_.back() != 1.0
            || !std::is_sorted(
                impact_edges_fraction_.begin(),
                impact_edges_fraction_.end()
            )
            || std::adjacent_find(
                impact_edges_fraction_.begin(),
                impact_edges_fraction_.end()
            ) != impact_edges_fraction_.end()
        ) {
            throw std::runtime_error(
                "Detector Green energy or impact axis is invalid."
            );
        }
        const std::size_t column_count = (
            static_cast<std::size_t>(energy_node_count_)
            * static_cast<std::size_t>(impact_bin_count_)
        );
        const std::size_t probability_count = (
            column_count * static_cast<std::size_t>(output_bin_count_)
        );
        cdfs_.assign(probability_count, 0.0);
        for (std::size_t column = 0; column < column_count; ++column) {
            double cumulative = 0.0;
            for (
                std::uint32_t output_index = 0;
                output_index < output_bin_count_;
                ++output_index
            ) {
                const float raw = ReadDetectorGreenValue<float>(input);
                if (!std::isfinite(raw) || raw < 0.0F) {
                    throw std::runtime_error(
                        "Detector Green probability is invalid."
                    );
                }
                cumulative += static_cast<double>(raw);
                cdfs_[
                    column * static_cast<std::size_t>(output_bin_count_)
                    + static_cast<std::size_t>(output_index)
                ] = cumulative;
            }
            if (
                !std::isfinite(cumulative)
                || std::abs(cumulative - 1.0) > 1.0e-5
            ) {
                throw std::runtime_error(
                    "Detector Green response column does not preserve one "
                    "registered pulse."
                );
            }
            for (
                std::uint32_t output_index = 0;
                output_index < output_bin_count_;
                ++output_index
            ) {
                cdfs_[
                    column * static_cast<std::size_t>(output_bin_count_)
                    + static_cast<std::size_t>(output_index)
                ] /= cumulative;
            }
            cdfs_[
                (column + 1U)
                    * static_cast<std::size_t>(output_bin_count_)
                - 1U
            ] = 1.0;
        }
        for (std::size_t index = 0; index < column_count; ++index) {
            const double histories = ReadDetectorGreenValue<double>(input);
            if (!std::isfinite(histories) || histories < 2.0) {
                throw std::runtime_error(
                    "Detector Green effective history count is invalid."
                );
            }
        }
        detection_probabilities_.assign(column_count, 0.0);
        for (std::size_t index = 0; index < column_count; ++index) {
            const double detection = ReadDetectorGreenValue<double>(input);
            if (
                !std::isfinite(detection)
                || detection < 0.0
                || detection > 1.0
            ) {
                throw std::runtime_error(
                    "Detector Green pulse-detection probability is invalid."
                );
            }
            detection_probabilities_[index] = detection;
        }
        if (input.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error(
                "Detector Green operator contains trailing bytes."
            );
        }
    }

    int SampleBin(
        const double incident_energy_keV,
        const double impact_parameter_fraction,
        std::mt19937_64& rng
    ) const {
        if (
            !std::isfinite(incident_energy_keV)
            || incident_energy_keV < 0.0
            || incident_energy_keV > energy_nodes_keV_.back()
            || !std::isfinite(impact_parameter_fraction)
            || impact_parameter_fraction < -1.0e-9
            || impact_parameter_fraction > 1.0 + 1.0e-9
        ) {
            throw std::runtime_error(
                "Detector Green input state is outside its closed domain."
            );
        }
        const double impact = std::clamp(
            impact_parameter_fraction,
            0.0,
            1.0
        );
        auto impact_iterator = std::upper_bound(
            impact_edges_fraction_.begin(),
            impact_edges_fraction_.end(),
            impact
        );
        std::size_t impact_index = impact_iterator
            == impact_edges_fraction_.begin()
            ? 0U
            : static_cast<std::size_t>(
                std::distance(
                    impact_edges_fraction_.begin(),
                    impact_iterator
                ) - 1
            );
        impact_index = std::min(
            impact_index,
            static_cast<std::size_t>(impact_bin_count_ - 1U)
        );
        auto upper_iterator = std::lower_bound(
            energy_nodes_keV_.begin(),
            energy_nodes_keV_.end(),
            incident_energy_keV
        );
        std::size_t upper_index = static_cast<std::size_t>(
            std::distance(energy_nodes_keV_.begin(), upper_iterator)
        );
        upper_index = std::min(
            upper_index,
            static_cast<std::size_t>(energy_node_count_ - 1U)
        );
        std::size_t lower_index = upper_index;
        double upper_weight = 0.0;
        if (energy_nodes_keV_[upper_index] != incident_energy_keV) {
            if (upper_index == 0U) {
                throw std::runtime_error(
                    "Detector Green energy bracketing failed."
                );
            }
            lower_index = upper_index - 1U;
            upper_weight = (
                incident_energy_keV - energy_nodes_keV_[lower_index]
            ) / (
                energy_nodes_keV_[upper_index]
                - energy_nodes_keV_[lower_index]
            );
        }
        const double node_draw = std::generate_canonical<double, 53>(rng);
        const std::size_t node_index = node_draw < upper_weight
            ? upper_index
            : lower_index;
        const std::size_t column = (
            node_index * static_cast<std::size_t>(impact_bin_count_)
            + impact_index
        );
        const double detection_draw = std::generate_canonical<double, 53>(rng);
        if (detection_draw >= detection_probabilities_[column]) {
            return -1;
        }
        const auto begin = cdfs_.begin()
            + static_cast<std::ptrdiff_t>(
                column * static_cast<std::size_t>(output_bin_count_)
            );
        const auto end = begin
            + static_cast<std::ptrdiff_t>(output_bin_count_);
        const double response_draw = std::generate_canonical<double, 53>(rng);
        const auto sampled = std::lower_bound(begin, end, response_draw);
        const int sampled_bin = static_cast<int>(
            std::distance(begin, sampled == end ? std::prev(end) : sampled)
        );
        const double source_energy_keV = energy_nodes_keV_[node_index];
        if (source_energy_keV <= 0.0) {
            return 0;
        }
        const double source_bin_anchor = (
            std::floor(
                (source_energy_keV - output_energy_min_keV_)
                / output_bin_width_keV_
            ) + 0.5
        );
        const double target_bin_anchor = (
            std::floor(
                (incident_energy_keV - output_energy_min_keV_)
                / output_bin_width_keV_
            ) + 0.5
        );
        const double aligned_coordinate = (
            (static_cast<double>(sampled_bin) + 0.5)
            * target_bin_anchor / source_bin_anchor
            - 0.5
        );
        const int raw_aligned_lower = static_cast<int>(
            std::floor(aligned_coordinate)
        );
        double aligned_fraction = aligned_coordinate
            - static_cast<double>(raw_aligned_lower);
        const int aligned_lower = std::clamp(
            raw_aligned_lower,
            0,
            static_cast<int>(output_bin_count_ - 1U)
        );
        const int aligned_upper = std::clamp(
            raw_aligned_lower + 1,
            0,
            static_cast<int>(output_bin_count_ - 1U)
        );
        if (aligned_lower == aligned_upper) {
            aligned_fraction = 0.0;
        }
        return std::generate_canonical<double, 53>(rng) < aligned_fraction
            ? aligned_upper
            : aligned_lower;
    }

    double ReferencePulseDetectionProbability(
        const double incident_energy_keV,
        const double detector_target_radius_m
    ) const {
        if (
            !std::isfinite(incident_energy_keV)
            || incident_energy_keV <= 0.0
            || incident_energy_keV > energy_nodes_keV_.back()
            || !std::isfinite(detector_target_radius_m)
            || detector_target_radius_m <= 0.0
            || detector_target_radius_m >= 1.0
        ) {
            throw std::runtime_error(
                "Detector Green reference-efficiency input is invalid."
            );
        }
        auto upper_iterator = std::lower_bound(
            energy_nodes_keV_.begin(),
            energy_nodes_keV_.end(),
            incident_energy_keV
        );
        std::size_t upper_index = static_cast<std::size_t>(
            std::distance(energy_nodes_keV_.begin(), upper_iterator)
        );
        upper_index = std::min(
            upper_index,
            static_cast<std::size_t>(energy_node_count_ - 1U)
        );
        std::size_t lower_index = upper_index;
        double upper_weight = 0.0;
        if (energy_nodes_keV_[upper_index] != incident_energy_keV) {
            if (upper_index == 0U) {
                throw std::runtime_error(
                    "Detector Green reference-energy bracketing failed."
                );
            }
            lower_index = upper_index - 1U;
            upper_weight = (
                incident_energy_keV - energy_nodes_keV_[lower_index]
            ) / (
                energy_nodes_keV_[upper_index]
                - energy_nodes_keV_[lower_index]
            );
        }
        const double radius_ratio = detector_target_radius_m;
        const double normalization = std::max(
            1.0 - std::sqrt(
                std::max(0.0, 1.0 - radius_ratio * radius_ratio)
            ),
            std::numeric_limits<double>::min()
        );
        double efficiency = 0.0;
        for (
            std::size_t impact_index = 0;
            impact_index < static_cast<std::size_t>(impact_bin_count_);
            ++impact_index
        ) {
            const double lower_impact = impact_edges_fraction_[impact_index];
            const double upper_impact = impact_edges_fraction_[impact_index + 1U];
            const double phase_weight = (
                std::sqrt(
                    std::max(
                        0.0,
                        1.0 - std::pow(radius_ratio * lower_impact, 2.0)
                    )
                )
                - std::sqrt(
                    std::max(
                        0.0,
                        1.0 - std::pow(radius_ratio * upper_impact, 2.0)
                    )
                )
            ) / normalization;
            const std::size_t lower_column = (
                lower_index * static_cast<std::size_t>(impact_bin_count_)
                + impact_index
            );
            const std::size_t upper_column = (
                upper_index * static_cast<std::size_t>(impact_bin_count_)
                + impact_index
            );
            const double phase_efficiency = (
                (1.0 - upper_weight)
                    * detection_probabilities_[lower_column]
                + upper_weight * detection_probabilities_[upper_column]
            );
            efficiency += phase_weight * phase_efficiency;
        }
        if (
            !std::isfinite(efficiency)
            || efficiency <= 0.0
            || efficiency > 1.0 + 1.0e-12
        ) {
            throw std::runtime_error(
                "Detector Green reference pulse efficiency is invalid."
            );
        }
        return std::min(efficiency, 1.0);
    }

private:
    std::uint32_t energy_node_count_ = 0U;
    std::uint32_t impact_bin_count_ = 0U;
    std::uint32_t output_bin_count_ = 0U;
    double output_energy_min_keV_ = 0.0;
    double output_bin_width_keV_ = 0.0;
    std::vector<double> energy_nodes_keV_;
    std::vector<double> impact_edges_fraction_;
    std::vector<double> cdfs_;
    std::vector<double> detection_probabilities_;
};

std::vector<long long> SampleUniformHistogramSubset(
    const std::vector<long long>& histogram,
    const long long subset_size,
    std::mt19937_64& rng
) {
    const long long total = std::accumulate(
        histogram.begin(),
        histogram.end(),
        0LL
    );
    if (subset_size < 0 || subset_size > total) {
        throw std::runtime_error("Histogram subset size is invalid");
    }
    std::vector<long long> result(histogram.size(), 0LL);
    if (subset_size == 0) {
        return result;
    }
    if (subset_size == total) {
        return histogram;
    }
    const bool draw_rejected = subset_size > total - subset_size;
    const long long draw_count = draw_rejected
        ? total - subset_size
        : subset_size;
    if (draw_rejected) {
        result = histogram;
    }
    std::vector<long long> fenwick(histogram.size() + 1, 0LL);
    for (std::size_t index = 0; index < histogram.size(); ++index) {
        for (
            std::size_t cursor = index + 1;
            cursor < fenwick.size();
            cursor += cursor & (~cursor + 1)
        ) {
            fenwick[cursor] += histogram[index];
        }
    }
    long long remaining = total;
    for (long long draw_index = 0; draw_index < draw_count; ++draw_index) {
        std::uniform_int_distribution<long long> order_distribution(
            0LL,
            remaining - 1LL
        );
        const long long order = order_distribution(rng);
        std::size_t index = 0;
        long long prefix = 0;
        std::size_t step = 1;
        while ((step << 1U) < fenwick.size()) {
            step <<= 1U;
        }
        for (; step > 0; step >>= 1U) {
            const std::size_t next = index + step;
            if (
                next < fenwick.size()
                && prefix + fenwick[next] <= order
            ) {
                index = next;
                prefix += fenwick[next];
            }
        }
        if (index >= histogram.size()) {
            throw std::runtime_error(
                "Histogram subset order statistic is outside its bins"
            );
        }
        if (draw_rejected) {
            --result[index];
        } else {
            ++result[index];
        }
        for (
            std::size_t cursor = index + 1;
            cursor < fenwick.size();
            cursor += cursor & (~cursor + 1)
        ) {
            --fenwick[cursor];
        }
        --remaining;
    }
    return result;
}

long long SampleNonparalyzableAcceptedCount(
    const long long incident_count,
    const double live_time_s,
    const double dead_time_tau_s,
    std::mt19937_64& rng
) {
    if (
        incident_count < 0
        || !(live_time_s > 0.0)
        || dead_time_tau_s < 0.0
    ) {
        throw std::runtime_error(
            "Nonparalyzable event-time parameters are invalid"
        );
    }
    if (incident_count == 0 || dead_time_tau_s == 0.0) {
        return incident_count;
    }
    long long accepted = 0;
    long long remaining = incident_count;
    double arrival_time_s = 0.0;
    double next_live_time_s = -std::numeric_limits<double>::infinity();
    while (remaining > 0) {
        const double uniform = std::max(
            std::generate_canonical<double, 53>(rng),
            std::numeric_limits<double>::min()
        );
        const double remaining_fraction = -std::expm1(
            std::log(uniform) / static_cast<double>(remaining)
        );
        arrival_time_s += (
            live_time_s - arrival_time_s
        ) * remaining_fraction;
        if (arrival_time_s >= next_live_time_s) {
            ++accepted;
            next_live_time_s = arrival_time_s + dead_time_tau_s;
        }
        --remaining;
    }
    return accepted;
}

double BackgroundShape(const double energy_keV) {
    const double low_energy_scatter = 0.62 * std::exp(-std::max(0.0, energy_keV) / 260.0);
    const double long_tail = 0.30 * std::exp(-std::max(0.0, energy_keV) / 1050.0);
    const double potassium_line = 0.08 * std::exp(
        -0.5 * std::pow((energy_keV - 1460.0) / 38.0, 2.0)
    );
    return std::max(0.0, low_energy_scatter + long_tail + potassium_line);
}

void AddBackgroundSpectrum(
    std::vector<double>& spectrum,
    std::vector<double>* spectrum_variance,
    const double bin_width_keV,
    const double dwell_time_s,
    const TransportOptions& options,
    std::mt19937_64& rng
) {
    if (options.background_cps <= 0.0 || spectrum.empty()) {
        return;
    }
    std::vector<double> shape(spectrum.size(), 0.0);
    double normalization = 0.0;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        const double energy_keV = (static_cast<double>(index) + 0.5) * bin_width_keV;
        shape[index] = BackgroundShape(energy_keV);
        normalization += shape[index];
    }
    if (normalization <= 0.0) {
        return;
    }
    const double expected_total = options.background_cps * std::max(0.0, dwell_time_s);
    for (std::size_t index = 0; index < spectrum.size(); ++index) {
        const double expected = expected_total * shape[index] / normalization;
        std::poisson_distribution<long> distribution(std::max(0.0, expected));
        const double sampled = static_cast<double>(distribution(rng));
        spectrum[index] += sampled;
        if (spectrum_variance != nullptr && index < spectrum_variance->size()) {
            // Every sampled background pulse is an independent unit-weight
            // entry, so its realized sumw2 contribution is the sampled count.
            (*spectrum_variance)[index] += sampled;
        }
    }
}

double InverseSquareScale(
    const double source_x,
    const double source_y,
    const double source_z,
    const double detector_x,
    const double detector_y,
    const double detector_z
) {
    const double dx = detector_x - source_x;
    const double dy = detector_y - source_y;
    const double dz = detector_z - source_z;
    const double distance_sq = dx * dx + dy * dy + dz * dz;
    if (distance_sq <= 1.0e-12) {
        return 0.0;
    }
    return 1.0 / distance_sq;
}

double SphereSolidAngleFraction(const double distance_m, const double radius_m) {
    const double radius = std::max(0.0, radius_m);
    if (radius <= 0.0) {
        return 0.0;
    }
    if (distance_m <= radius) {
        return 0.5;
    }
    const double ratio = std::clamp(radius / std::max(distance_m, 1.0e-12), 0.0, 1.0);
    return 0.5 * (1.0 - std::sqrt(std::max(0.0, 1.0 - ratio * ratio)));
}

double DetectorCpsGeometryScale(
    const double source_x,
    const double source_y,
    const double source_z,
    const double detector_x,
    const double detector_y,
    const double detector_z,
    const DetectorSpec& detector
) {
    const double dx = detector_x - source_x;
    const double dy = detector_y - source_y;
    const double dz = detector_z - source_z;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (distance <= 1.0e-12) {
        return 0.0;
    }
    const double radius = std::max(0.0, detector.crystal_radius_m);
    if (radius <= 0.0) {
        return InverseSquareScale(
            source_x,
            source_y,
            source_z,
            detector_x,
            detector_y,
            detector_z
        );
    }
    const double reference_distance = std::max(1.0, radius);
    const double reference_fraction = std::max(
        SphereSolidAngleFraction(reference_distance, radius),
        1.0e-12
    );
    return SphereSolidAngleFraction(std::max(distance, radius), radius)
        / reference_fraction;
}

double DetectorReferenceAcceptance(const DetectorSpec& detector) {
    constexpr double kReferenceDistanceM = 1.0;
    const double radius_m = std::max(1.0e-9, detector.crystal_radius_m);
    return std::clamp(
        SphereSolidAngleFraction(kReferenceDistanceM, radius_m),
        1.0e-12,
        1.0
    );
}

std::string NormalizeSourceBiasMode(const std::string& mode) {
    auto normalized = NormalizeModeToken(mode);
    if (normalized.empty() || normalized == "none" || normalized == "isotropic") {
        return "analog";
    }
    if (normalized == "mixture_cone" || normalized == "cone_isotropic") {
        return "mixture_cone_isotropic";
    }
    if (normalized == "detector" || normalized == "detector_directed" || normalized == "detector_cone") {
        return "detector_cone";
    }
    return normalized;
}

bool UsesSourceBias(const TransportOptions& options) {
    return NormalizeSourceBiasMode(options.source_bias_mode) == "mixture_cone_isotropic";
}

std::string NormalizeSourceRateModel(const std::string& mode) {
    auto normalized = NormalizeModeToken(mode);
    if (
        normalized.empty()
        || normalized == "detector"
        || normalized == "detector_cps"
        || normalized == "detector_cps_1m"
        || normalized == "detector_count_rate"
    ) {
        return "detector_cps_1m";
    }
    if (
        normalized == "activity_bq"
        || normalized == "parent_activity_bq"
        || normalized == "parent_decay_activity_bq"
    ) {
        return "parent_decay_activity_bq";
    }
    if (
        normalized == "isotropic"
        || normalized == "isotropic_emission"
        || normalized == "isotropic_emission_equivalent"
        || normalized == "emission_equivalent"
    ) {
        return "isotropic_emission_equivalent";
    }
    return normalized;
}

std::string NormalizePrimaryEmissionModel(const std::string& mode) {
    const auto normalized = NormalizeModeToken(mode);
    if (
        normalized.empty()
        || normalized == "gamma_lines"
        || normalized == "independent_gamma_lines"
    ) {
        return "independent_gamma_lines";
    }
    if (
        normalized == "radioactive_decay"
        || normalized == "geant4_radioactive_decay"
    ) {
        return "geant4_radioactive_decay";
    }
    return normalized;
}

std::string NormalizeDetectorScoringMode(const std::string& mode) {
    const auto normalized = NormalizeModeToken(mode);
    if (
        normalized.empty()
        || normalized == "full"
        || normalized == "full_transport"
        || normalized == "energy_deposit"
    ) {
        return "full_transport";
    }
    if (
        normalized == "fast"
        || normalized == "incident_energy"
        || normalized == "incident_gamma_energy"
        || normalized == "perfect_absorption"
    ) {
        return "incident_gamma_energy";
    }
    return normalized;
}

bool UsesFastDetectorScoring(const TransportOptions& options) {
    return NormalizeDetectorScoringMode(options.detector_scoring_mode) == "incident_gamma_energy";
}

std::string NormalizeSecondaryTransportMode(const std::string& mode) {
    const auto normalized = NormalizeModeToken(mode);
    if (
        normalized.empty()
        || normalized == "full"
        || normalized == "full_transport"
        || normalized == "all_particles"
    ) {
        return "full_transport";
    }
    if (
        normalized == "gamma_only"
        || normalized == "photon_only"
        || normalized == "kill_charged"
        || normalized == "kill_charged_secondaries"
    ) {
        return "gamma_only";
    }
    return normalized;
}

bool UsesGammaOnlySecondaryTransport(const TransportOptions& options) {
    return NormalizeSecondaryTransportMode(options.secondary_transport_mode) == "gamma_only";
}

double DetectorTargetRadiusM(const DetectorSpec& detector) {
    return std::max(
        1.0e-9,
        detector.crystal_radius_m + std::max(0.0, detector.housing_thickness_m)
    );
}

double EffectiveConeHalfAngleRad(
    const SourceSpec& source,
    const SceneSpec& scene,
    const RequestSpec& request,
    const TransportOptions& options
) {
    const double dx = request.detector_pose.x - source.x;
    const double dy = request.detector_pose.y - source.y;
    const double dz = request.detector_pose.z - source.z;
    const double distance_m = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double target_radius_m = DetectorTargetRadiusM(scene.detector);
    double covering_angle = CLHEP::pi;
    if (distance_m > target_radius_m) {
        covering_angle = std::asin(std::clamp(target_radius_m / distance_m, 0.0, 1.0));
    }
    if (options.source_bias_cone_policy != "detector_covering") {
        throw std::runtime_error("Unsupported source-bias cone policy");
    }
    return std::clamp(covering_angle, 1.0e-9, CLHEP::pi);
}

double ConeSolidAngleSr(const double half_angle_rad) {
    const double theta = std::clamp(half_angle_rad, 0.0, CLHEP::pi);
    return std::max(2.0 * CLHEP::pi * (1.0 - std::cos(theta)), 1.0e-18);
}

bool UseTheoryTvlProfile(const std::string& physics_profile) {
    std::string profile = physics_profile;
    std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return profile.find("theory_tvl") != std::string::npos || profile.find("ideal_tvl") != std::string::npos;
}

double MuFromTvlMm(const double tvl_mm) {
    return std::log(10.0) / (std::max(tvl_mm, 1.0e-12) / 10.0);
}

double TvlMmForShield(const std::string& isotope, const std::string& shield_kind) {
    const bool is_fe = shield_kind == "fe";
    if (isotope == "Cs-137") {
        return is_fe ? 50.0 : 22.0;
    }
    if (isotope == "Co-60") {
        return is_fe ? 67.0 : 40.0;
    }
    if (isotope == "Eu-154") {
        return is_fe ? 57.7 : 28.1;
    }
    return is_fe ? 50.0 : 22.0;
}

std::array<double, 3> ShieldNormalFromPose(const PoseSpec& pose) {
    const double norm = std::sqrt(
        pose.qw * pose.qw + pose.qx * pose.qx + pose.qy * pose.qy + pose.qz * pose.qz
    );
    if (norm <= 1.0e-12) {
        const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
        return {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    }
    const double w = pose.qw / norm;
    const double x = pose.qx / norm;
    const double y = pose.qy / norm;
    const double z = pose.qz / norm;
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - z * w);
    const double r02 = 2.0 * (x * z + y * w);
    const double r10 = 2.0 * (x * y + z * w);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - x * w);
    const double r20 = 2.0 * (x * z - y * w);
    const double r21 = 2.0 * (y * z + x * w);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);
    const double local = 1.0 / std::sqrt(3.0);
    std::array<double, 3> normal = {
        r00 * local + r01 * local + r02 * local,
        r10 * local + r11 * local + r12 * local,
        r20 * local + r21 * local + r22 * local,
    };
    const double mag = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (mag <= 1.0e-12) {
        return {local, local, local};
    }
    return {normal[0] / mag, normal[1] / mag, normal[2] / mag};
}

std::array<double, 3> PhysicalShieldNormalFromOrientationIndex(
    const int orientation_index
) {
    if (orientation_index < 0 || orientation_index >= 8) {
        throw std::runtime_error("Shield orientation index must be in [0, 7].");
    }
    const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    const int incoming_x = orientation_index < 4 ? 1 : -1;
    const int incoming_y = (orientation_index % 4) < 2 ? 1 : -1;
    const int incoming_z = (orientation_index % 2) == 0 ? 1 : -1;
    return {
        -static_cast<double>(incoming_x) * inv_sqrt3,
        -static_cast<double>(incoming_y) * inv_sqrt3,
        -static_cast<double>(incoming_z) * inv_sqrt3,
    };
}

void ValidateShieldPoseContract(const RequestSpec& request) {
    if (request.shield_pose_contract_id != kShieldPoseContractId) {
        throw std::runtime_error(
            "Request shield_pose_contract_id is missing or incompatible."
        );
    }
    if (request.shield_pose_contract_sha256 != kShieldPoseContractSha256) {
        throw std::runtime_error(
            "Request shield_pose_contract_sha256 is missing or incompatible."
        );
    }
    const std::array<std::pair<std::string, std::pair<int, PoseSpec>>, 2> entries = {{
        {"Fe", {request.fe_orientation_index, request.fe_pose}},
        {"Pb", {request.pb_orientation_index, request.pb_pose}},
    }};
    for (const auto& entry : entries) {
        const auto expected = PhysicalShieldNormalFromOrientationIndex(
            entry.second.first
        );
        const auto actual = ShieldNormalFromPose(entry.second.second);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (std::abs(expected[axis] - actual[axis]) > 1.0e-8) {
                throw std::runtime_error(
                    entry.first
                    + " shield quaternion violates the shared local "
                    "(+X,+Y,+Z) octant-pose contract."
                );
            }
        }
    }
}

void ValidateNativeActionIdentityContract(const RequestSpec& request) {
    if (
        !request.has_step
        || !request.has_detector_pose
        || !request.has_fe_pose
        || !request.has_pb_pose
    ) {
        throw std::runtime_error(
            "Native action request requires one STEP and all three POSE records."
        );
    }
    if (
        request.step_id < 0
        || request.seed < 0
        || !std::isfinite(request.dwell_time_s)
        || request.dwell_time_s <= 0.0
    ) {
        throw std::runtime_error(
            "Native action step, seed, and dwell values are invalid."
        );
    }
    if (request.native_action_contract_id != kNativeActionIdentityContractId) {
        throw std::runtime_error(
            "Request native_action_contract_id is missing or incompatible."
        );
    }
    if (
        request.native_action_sha256.size() != 64
        || !std::all_of(
            request.native_action_sha256.begin(),
            request.native_action_sha256.end(),
            [](const unsigned char ch) {
                return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
            }
        )
    ) {
        throw std::runtime_error(
            "Request native_action_sha256 must be one lowercase SHA-256 digest."
        );
    }
    const std::array<PoseSpec, 3> poses = {
        request.detector_pose,
        request.fe_pose,
        request.pb_pose,
    };
    for (const auto& pose : poses) {
        const std::array<double, 7> values = {
            pose.x, pose.y, pose.z, pose.qw, pose.qx, pose.qy, pose.qz,
        };
        if (!std::all_of(values.begin(), values.end(), [](const double value) {
            return std::isfinite(value);
        })) {
            throw std::runtime_error("Native action poses must be finite.");
        }
        const double quaternion_norm = std::sqrt(
            pose.qw * pose.qw
            + pose.qx * pose.qx
            + pose.qy * pose.qy
            + pose.qz * pose.qz
        );
        if (!(quaternion_norm > 1.0e-12)) {
            throw std::runtime_error(
                "Native action pose quaternions must have positive norm."
            );
        }
    }
}

std::array<std::array<double, 3>, 3> ShieldAxesFromPose(const PoseSpec& pose) {
    const double norm = std::sqrt(
        pose.qw * pose.qw + pose.qx * pose.qx + pose.qy * pose.qy + pose.qz * pose.qz
    );
    if (norm <= 1.0e-12) {
        return {{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        }};
    }
    const double w = pose.qw / norm;
    const double x = pose.qx / norm;
    const double y = pose.qy / norm;
    const double z = pose.qz / norm;
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - z * w);
    const double r02 = 2.0 * (x * z + y * w);
    const double r10 = 2.0 * (x * y + z * w);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - x * w);
    const double r20 = 2.0 * (x * z - y * w);
    const double r21 = 2.0 * (y * z + x * w);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);
    return {{
        {r00, r10, r20},
        {r01, r11, r21},
        {r02, r12, r22}
    }};
}

void AddShieldAxisMetadata(
    SimulationResult& result,
    const std::string& prefix,
    const PoseSpec& pose
) {
    const auto axes = ShieldAxesFromPose(pose);
    const std::array<std::string, 3> names = {"x", "y", "z"};
    const std::array<std::string, 3> components = {"x", "y", "z"};
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        for (std::size_t component = 0; component < axes[axis].size(); ++component) {
            result.metadata[
                prefix + "_shield_axis_" + names[axis] + "_" + components[component]
            ] = std::to_string(axes[axis][component]);
        }
    }
}

void AddNativeActionPoseMetadata(
    SimulationResult& result,
    const std::string& prefix,
    const PoseSpec& pose
) {
    result.metadata["native_action_" + prefix + "_pose_x"] = SerializeDouble(pose.x);
    result.metadata["native_action_" + prefix + "_pose_y"] = SerializeDouble(pose.y);
    result.metadata["native_action_" + prefix + "_pose_z"] = SerializeDouble(pose.z);
    result.metadata["native_action_" + prefix + "_quat_w"] = SerializeDouble(pose.qw);
    result.metadata["native_action_" + prefix + "_quat_x"] = SerializeDouble(pose.qx);
    result.metadata["native_action_" + prefix + "_quat_y"] = SerializeDouble(pose.qy);
    result.metadata["native_action_" + prefix + "_quat_z"] = SerializeDouble(pose.qz);
}

int SignWithTolerance(const double value, const double tolerance = 1.0e-6) {
    if (std::abs(value) < tolerance) {
        return 0;
    }
    return value > 0.0 ? 1 : -1;
}

bool ShieldBlocksSourceDirection(
    const SourceSpec& source,
    const PoseSpec& detector_pose,
    const PoseSpec& shield_pose
) {
    const std::array<double, 3> direction = {
        source.x - detector_pose.x,
        source.y - detector_pose.y,
        source.z - detector_pose.z,
    };
    const auto normal = ShieldNormalFromPose(shield_pose);
    for (std::size_t idx = 0; idx < 3; ++idx) {
        if (SignWithTolerance(direction[idx]) != SignWithTolerance(normal[idx])) {
            return false;
        }
    }
    return true;
}

double TheoryTvlTransmission(
    const SourceSpec& source,
    const SceneSpec& scene,
    const RequestSpec& request
) {
    double exponent = 0.0;
    if (
        scene.fe_shield.has_value()
        && ShieldBlocksSourceDirection(source, request.detector_pose, request.fe_pose)
    ) {
        const double thickness_cm = scene.fe_shield->thickness_m * 100.0;
        exponent += MuFromTvlMm(TvlMmForShield(source.isotope, "fe")) * thickness_cm;
    }
    if (
        scene.pb_shield.has_value()
        && ShieldBlocksSourceDirection(source, request.detector_pose, request.pb_pose)
    ) {
        const double thickness_cm = scene.pb_shield->thickness_m * 100.0;
        exponent += MuFromTvlMm(TvlMmForShield(source.isotope, "pb")) * thickness_cm;
    }
    return std::exp(-exponent);
}

std::map<std::string, std::map<std::string, double>> PresetCompositionByMass() {
    return {
        {"air", {{"N", 0.755}, {"O", 0.232}, {"Ar", 0.013}}},
        {"water", {{"H", 0.1119}, {"O", 0.8881}}},
        {"concrete", {{"O", 0.525}, {"Si", 0.325}, {"Ca", 0.090}, {"Al", 0.060}}},
        {"aluminum", {{"Al", 1.0}}},
        {"iron", {{"Fe", 1.0}}},
        {"lead", {{"Pb", 1.0}}},
        {"steel", {{"Fe", 0.98}, {"C", 0.02}}},
        {"stainless_steel", {{"Fe", 0.70}, {"Cr", 0.19}, {"Ni", 0.10}, {"C", 0.01}}},
        {"cebr3", {{"Ce", 0.455}, {"Br", 0.545}}},
    };
}

double PresetDensity(const std::string& name) {
    if (name == "air") {
        return 0.001225;
    }
    if (name == "water") {
        return 1.0;
    }
    if (name == "concrete") {
        return 2.3;
    }
    if (name == "aluminum") {
        return 2.7;
    }
    if (name == "iron" || name == "fe") {
        return 7.87;
    }
    if (name == "lead" || name == "pb") {
        return 11.34;
    }
    if (name == "steel") {
        return 7.85;
    }
    if (name == "stainless_steel") {
        return 8.0;
    }
    if (name == "cebr3") {
        return 5.1;
    }
    return -1.0;
}

std::string NormalizeMaterialName(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "fe") {
        return "iron";
    }
    if (value == "pb") {
        return "lead";
    }
    if (value == "alu" || value == "aluminium") {
        return "aluminum";
    }
    return value;
}

std::string EnergyMetadataToken(const double energy_keV) {
    std::ostringstream stream;
    stream << "e" << std::fixed << std::setprecision(1) << energy_keV;
    std::string token = stream.str();
    for (char& ch : token) {
        if (ch == '.') {
            ch = 'p';
        }
    }
    return token;
}

std::string MaterialSpecName(const MaterialSpec& material) {
    const std::string raw = material.name.empty() ? material.preset_name : material.name;
    return NormalizeMaterialName(raw.empty() ? "air" : raw);
}

std::string CompositionCacheKey(const std::map<std::string, double>& composition_by_mass) {
    std::ostringstream stream;
    bool first = true;
    for (const auto& item : composition_by_mass) {
        if (!first) {
            stream << ";";
        }
        first = false;
        stream << item.first << ":" << std::setprecision(12) << item.second;
    }
    return stream.str();
}

G4Material* ResolveAttenuationMaterial(
    const MaterialSpec& material,
    const std::string& fallback_name
) {
    const std::string normalized_name = NormalizeMaterialName(
        MaterialSpecName(material).empty() ? fallback_name : MaterialSpecName(material)
    );
    auto* nist = G4NistManager::Instance();
    if (normalized_name == "air") {
        return nist->FindOrBuildMaterial("G4_AIR");
    }

    std::map<std::string, double> composition = material.composition_by_mass;
    if (composition.empty()) {
        const auto presets = PresetCompositionByMass();
        const auto preset_it = presets.find(normalized_name);
        if (preset_it != presets.end()) {
            composition = preset_it->second;
        }
    }
    if (composition.empty()) {
        throw std::runtime_error(
            "Material '" + normalized_name
            + "' has no explicit or authenticated preset composition."
        );
    }

    const double density_g_cm3 = material.density_g_cm3 > 0.0
        ? material.density_g_cm3
        : PresetDensity(normalized_name);
    if (!std::isfinite(density_g_cm3) || density_g_cm3 <= 0.0) {
        throw std::runtime_error(
            "Material '" + normalized_name
            + "' has no finite positive density."
        );
    }
    double mass_fraction_sum = 0.0;
    for (const auto& item : composition) {
        if (
            item.first.empty()
            || !std::isfinite(item.second)
            || item.second <= 0.0
        ) {
            throw std::runtime_error(
                "Material '" + normalized_name
                + "' has an invalid mass-composition entry."
            );
        }
        mass_fraction_sum += item.second;
    }
    if (std::abs(mass_fraction_sum - 1.0) > 1.0e-9) {
        throw std::runtime_error(
            "Material '" + normalized_name
            + "' mass fractions do not sum to one."
        );
    }
    const std::string key = normalized_name
        + "|rho=" + std::to_string(density_g_cm3)
        + "|comp=" + CompositionCacheKey(composition);
    static std::mutex cache_mutex;
    static std::map<std::string, G4Material*> cache;
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto cached = cache.find(key);
    if (cached != cache.end()) {
        return cached->second;
    }
    auto* resolved = new G4Material(
        "Attenuation_" + SanitizeMetadataToken(normalized_name)
            + "_" + std::to_string(cache.size()),
        density_g_cm3 * g / cm3,
        static_cast<G4int>(composition.size())
    );
    for (const auto& item : composition) {
        resolved->AddElement(nist->FindOrBuildElement(item.first), item.second);
    }
    cache[key] = resolved;
    return resolved;
}

double Geant4GammaMuCmInv(G4Material* material, const double energy_keV) {
    if (material == nullptr || energy_keV <= 0.0) {
        return 0.0;
    }
    G4EmCalculator calculator;
    const double length = calculator.ComputeGammaAttenuationLength(
        energy_keV * keV,
        material
    );
    if (!std::isfinite(length) || length <= 0.0) {
        return 0.0;
    }
    const double length_cm = length / cm;
    if (!std::isfinite(length_cm) || length_cm <= 0.0) {
        return 0.0;
    }
    return 1.0 / length_cm;
}

void AddMaterialMuMetadata(
    SimulationResult& result,
    const SceneSpec& scene,
    const std::string& material_token,
    G4Material* material,
    const std::set<std::string>& isotopes
) {
    const std::string clean_material = SanitizeMetadataToken(material_token);
    for (const auto& isotope : isotopes) {
        for (const auto& line : GammaLinesForIsotope(scene, isotope)) {
            const std::string energy_token = EnergyMetadataToken(line.energy_keV);
            const double mu_cm_inv = Geant4GammaMuCmInv(material, line.energy_keV);
            const std::string base = "geant4_mu_cm_inv_"
                + clean_material + "_" + SanitizeMetadataToken(isotope)
                + "_" + energy_token;
            result.metadata[base] = std::to_string(mu_cm_inv);
        }
    }
}

void AddGeant4MaterialMuMetadata(SimulationResult& result, const SceneSpec& scene) {
    std::set<std::string> isotopes;
    for (const auto& source : scene.sources) {
        if (!source.isotope.empty()) {
            isotopes.insert(source.isotope);
        }
    }
    if (isotopes.empty()) {
        return;
    }
    if (scene.fe_shield.has_value()) {
        AddMaterialMuMetadata(
            result,
            scene,
            "shield_fe",
            ResolveAttenuationMaterial(scene.fe_shield->material, "iron"),
            isotopes
        );
    }
    if (scene.pb_shield.has_value()) {
        AddMaterialMuMetadata(
            result,
            scene,
            "shield_pb",
            ResolveAttenuationMaterial(scene.pb_shield->material, "lead"),
            isotopes
        );
    }
    std::set<std::string> emitted_materials;
    for (const auto& volume : scene.volumes) {
        if (volume.transport_mode != "geant4") {
            continue;
        }
        const std::string material_name = MaterialSpecName(volume.material);
        if (material_name.empty() || emitted_materials.count(material_name) > 0) {
            continue;
        }
        emitted_materials.insert(material_name);
        AddMaterialMuMetadata(
            result,
            scene,
            "obstacle_" + material_name,
            ResolveAttenuationMaterial(volume.material, material_name),
            isotopes
        );
    }
}

class TransportTrackInformation : public G4VUserTrackInformation {
public:
    TransportTrackInformation(
        const std::size_t primary_batch_index,
        const std::size_t source_index,
        const long long primary_history_index,
        const int angle_stratum_index,
        const int angle_stratum_count,
        const bool gamma_interacted,
        const bool secondary_lineage,
        const long long bias_branch_lineage_id,
        const bool force_collision_clone,
        const double primary_event_time_s
    ) : G4VUserTrackInformation("RotatingShieldTransportTrackInformation"),
        primary_batch_index_(primary_batch_index),
        source_index_(source_index),
        primary_history_index_(primary_history_index),
        angle_stratum_index_(angle_stratum_index),
        angle_stratum_count_(angle_stratum_count),
        gamma_interacted_(gamma_interacted),
        secondary_lineage_(secondary_lineage),
        bias_branch_lineage_id_(bias_branch_lineage_id),
        force_collision_clone_(force_collision_clone),
        primary_event_time_s_(primary_event_time_s) {
        if (bias_branch_lineage_id_ < 0) {
            throw std::runtime_error(
                "Transport track branch lineage must be nonnegative."
            );
        }
    }

    std::size_t PrimaryBatchIndex() const {
        return primary_batch_index_;
    }

    bool GammaInteracted() const {
        return gamma_interacted_;
    }

    bool SecondaryLineage() const {
        return secondary_lineage_;
    }

    long long BiasBranchLineageId() const {
        return bias_branch_lineage_id_;
    }

    bool ForceCollisionClone() const {
        return force_collision_clone_;
    }

    long long ActiveForceCollisionSplitId() const {
        return active_force_collision_split_id_;
    }

    std::size_t SourceIndex() const {
        return source_index_;
    }

    long long PrimaryHistoryIndex() const {
        return primary_history_index_;
    }

    int AngleStratumIndex() const {
        return angle_stratum_index_;
    }

    int AngleStratumCount() const {
        return angle_stratum_count_;
    }

    double PrimaryEventTimeS() const {
        return primary_event_time_s_;
    }

    void MarkGammaInteracted() {
        gamma_interacted_ = true;
    }

    void BeginForceCollisionSplit(const long long split_id) {
        if (split_id < 0 || active_force_collision_split_id_ >= 0) {
            throw std::runtime_error(
                "Force-collision split lineage is invalid or already active."
            );
        }
        active_force_collision_split_id_ = split_id;
    }

    void EndForceCollisionSplit() {
        if (active_force_collision_split_id_ < 0) {
            throw std::runtime_error(
                "Force-collision split lineage is not active."
            );
        }
        active_force_collision_split_id_ = -1;
    }

private:
    std::size_t primary_batch_index_ = std::numeric_limits<std::size_t>::max();
    std::size_t source_index_ = std::numeric_limits<std::size_t>::max();
    long long primary_history_index_ = -1;
    int angle_stratum_index_ = -1;
    int angle_stratum_count_ = 0;
    bool gamma_interacted_ = false;
    bool secondary_lineage_ = false;
    long long bias_branch_lineage_id_ = -1;
    bool force_collision_clone_ = false;
    long long active_force_collision_split_id_ = -1;
    double primary_event_time_s_ = 0.0;
};

const TransportTrackInformation* TrackInformation(const G4Track* track) {
    if (track == nullptr) {
        return nullptr;
    }
    return dynamic_cast<const TransportTrackInformation*>(
        track->GetUserInformation()
    );
}

TransportTrackInformation* MutableTrackInformation(G4Track* track) {
    if (track == nullptr) {
        return nullptr;
    }
    return dynamic_cast<TransportTrackInformation*>(
        track->GetUserInformation()
    );
}

int ForceCollisionModelId() {
    static const int model_id = G4PhysicsModelCatalog::GetModelID(
        "model_GenBiasForceCollision"
    );
    if (model_id < 0) {
        throw std::runtime_error(
            "Geant4 force-collision model ID is unavailable."
        );
    }
    return model_id;
}

const G4BOptrForceCollisionTrackData* ForceCollisionTrackData(
    const G4Track* track
) {
    if (track == nullptr) {
        return nullptr;
    }
    return dynamic_cast<const G4BOptrForceCollisionTrackData*>(
        track->GetAuxiliaryTrackInformation(ForceCollisionModelId())
    );
}

bool HasActiveForceCollisionScheme(const G4Track* track) {
    const auto* data = ForceCollisionTrackData(track);
    return data != nullptr && !data->IsFreeFromBiasing();
}

struct ForceCollisionRunSummary {
    long long split_count = 0;
    double maximum_absolute_weight_error = 0.0;
    double maximum_relative_weight_error = 0.0;
};

class ForceCollisionDiagnostics {
public:
    struct SplitRecord {
        long long collision_branch_lineage_id = -1;
        long long survivor_branch_lineage_id = -1;
        double initial_weight = 0.0;
        double survivor_terminal_weight =
            std::numeric_limits<double>::quiet_NaN();
        double collision_terminal_weight =
            std::numeric_limits<double>::quiet_NaN();
    };

    struct LocalState {
        std::map<long long, SplitRecord> splits;
    };

    using LocalHandle = std::shared_ptr<LocalState>;

    LocalHandle AcquireLocal() {
        static thread_local std::unordered_map<
            const ForceCollisionDiagnostics*,
            std::weak_ptr<LocalState>
        > thread_states;
        const auto existing = thread_states.find(this);
        if (existing != thread_states.end()) {
            if (auto state = existing->second.lock()) {
                return state;
            }
        }
        auto state = std::make_shared<LocalState>();
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            local_states_.push_back(state);
        }
        thread_states[this] = state;
        return state;
    }

    long long BeginSplit(
        LocalState* state,
        const long long parent_branch_lineage_id,
        const double initial_weight
    ) {
        if (
            state == nullptr
            || parent_branch_lineage_id < 0
            || !std::isfinite(initial_weight)
            || initial_weight <= 0.0
        ) {
            throw std::runtime_error(
                "Force-collision split requires valid lineage and weight."
            );
        }
        const long long split_id = next_split_id_.fetch_add(
            1,
            std::memory_order_relaxed
        );
        if (
            split_id < kFirstBiasBranchLineageId
            || !state->splits.emplace(
                split_id,
                SplitRecord{
                    parent_branch_lineage_id,
                    split_id,
                    initial_weight,
                }
            ).second
        ) {
            throw std::runtime_error(
                "Force-collision split identifier is invalid or duplicated."
            );
        }
        return split_id;
    }

    static long long SurvivorBranchLineageId(
        const LocalState* state,
        const long long split_id
    ) {
        if (state == nullptr) {
            throw std::runtime_error(
                "Force-collision branch lookup requires local diagnostics."
            );
        }
        const auto found = state->splits.find(split_id);
        if (found == state->splits.end()) {
            throw std::runtime_error(
                "Force-collision split identifier is unknown."
            );
        }
        return found->second.survivor_branch_lineage_id;
    }

    static void RecordTerminalBranch(
        LocalState* state,
        const long long split_id,
        const long long branch_lineage_id,
        const double terminal_weight
    ) {
        if (
            state == nullptr
            || split_id < 0
            || branch_lineage_id < 0
            || !std::isfinite(terminal_weight)
            || terminal_weight < 0.0
        ) {
            throw std::runtime_error(
                "Force-collision terminal branch is invalid."
            );
        }
        const auto found = state->splits.find(split_id);
        if (found == state->splits.end()) {
            throw std::runtime_error(
                "Force-collision terminal references an unknown split."
            );
        }
        auto& record = found->second;
        double* destination = nullptr;
        if (branch_lineage_id == record.survivor_branch_lineage_id) {
            destination = &record.survivor_terminal_weight;
        } else if (
            branch_lineage_id == record.collision_branch_lineage_id
        ) {
            destination = &record.collision_terminal_weight;
        } else {
            throw std::runtime_error(
                "Force-collision terminal branch does not belong to split."
            );
        }
        if (std::isfinite(*destination)) {
            throw std::runtime_error(
                "Force-collision terminal branch was recorded twice."
            );
        }
        *destination = terminal_weight;
    }

    ForceCollisionRunSummary ValidateAndSummarize() const {
        ForceCollisionRunSummary summary;
        for (const auto& state : RegisteredStates()) {
            for (const auto& item : state->splits) {
                const auto& record = item.second;
                if (
                    !std::isfinite(record.survivor_terminal_weight)
                    || !std::isfinite(record.collision_terminal_weight)
                ) {
                    throw std::runtime_error(
                        "Force-collision split did not produce both terminal "
                        "branches."
                    );
                }
                if (
                    record.survivor_terminal_weight < 0.0
                    || record.survivor_terminal_weight
                        > record.initial_weight
                ) {
                    throw std::runtime_error(
                        "Force-collision survivor weight is outside its "
                        "parent branch mass."
                    );
                }
                // The forced-collision original can terminate at its physical
                // interaction, where Geant4 transfers the occurrence weight
                // to the particle change and leaves no meaningful parent
                // track weight. Its exact branch probability mass is the
                // complement of the force-free-flight survivor mass.
                const double collision_probability_mass = (
                    record.initial_weight
                    - record.survivor_terminal_weight
                );
                const double terminal_sum = (
                    record.survivor_terminal_weight
                    + collision_probability_mass
                );
                const double absolute_error = std::abs(
                    terminal_sum - record.initial_weight
                );
                const double relative_error = absolute_error
                    / std::max(
                        record.initial_weight,
                        std::numeric_limits<double>::min()
                    );
                summary.split_count += 1;
                summary.maximum_absolute_weight_error = std::max(
                    summary.maximum_absolute_weight_error,
                    absolute_error
                );
                summary.maximum_relative_weight_error = std::max(
                    summary.maximum_relative_weight_error,
                    relative_error
                );
                const double tolerance = (
                    1.0e-9 * std::max(1.0, record.initial_weight)
                );
                if (absolute_error > tolerance) {
                    std::ostringstream message;
                    message
                        << std::setprecision(17)
                        << "Force-collision terminal branch weights do not "
                        << "conserve their parent weight: split_id="
                        << item.first
                        << ", initial=" << record.initial_weight
                        << ", survivor="
                        << record.survivor_terminal_weight
                        << ", collision="
                        << record.collision_terminal_weight
                        << ", terminal_sum=" << terminal_sum
                        << ", absolute_error=" << absolute_error
                        << ", tolerance=" << tolerance;
                    throw std::runtime_error(message.str());
                }
            }
        }
        return summary;
    }

    void Clear() {
        for (const auto& state : RegisteredStates()) {
            state->splits.clear();
        }
        next_split_id_.store(
            kFirstBiasBranchLineageId,
            std::memory_order_relaxed
        );
    }

private:
    std::vector<LocalHandle> RegisteredStates() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return local_states_;
    }

    static constexpr long long kFirstBiasBranchLineageId = 1LL << 40;
    mutable std::mutex registry_mutex_;
    std::vector<LocalHandle> local_states_;
    std::atomic<long long> next_split_id_{kFirstBiasBranchLineageId};
};

class CalibrationFirstCollisionOperator : public G4VBiasingOperator {
public:
    CalibrationFirstCollisionOperator(
        const G4String& name,
        ForceCollisionDiagnostics* diagnostics
    ) : G4VBiasingOperator(name),
        delegate_(std::make_unique<G4BOptrForceCollision>(
            "gamma",
            name + "_delegate"
        )),
        diagnostics_state_(
            diagnostics == nullptr ? nullptr : diagnostics->AcquireLocal()
        ),
        diagnostics_(diagnostics) {
        if (diagnostics_ == nullptr) {
            throw std::runtime_error(
                "Calibration force collision requires diagnostics."
            );
        }
    }

    void Configure() final {
        delegate_->Configure();
    }

    void ConfigureForWorker() final {
        delegate_->ConfigureForWorker();
    }

    void StartRun() final {
        delegate_->StartRun();
    }

    void StartTracking(const G4Track* track) final {
        current_track_ = track;
        delegate_->StartTracking(track);
    }

    void EndTracking() final {
        if (pending_clone_track_ != nullptr) {
            throw std::runtime_error(
                "A gamma track ended before its proposed force-collision "
                "clone was registered."
            );
        }
        auto* information = MutableTrackInformation(
            const_cast<G4Track*>(current_track_)
        );
        if (
            information != nullptr
            && information->ActiveForceCollisionSplitId() >= 0
            && (
                information->ForceCollisionClone()
                || information->GammaInteracted()
            )
        ) {
            // Geant4's clone carries the force-free-flight survivor. An
            // original track that has interacted carries the already applied
            // forced-occurrence weight. Either is a weighted endpoint even
            // when no subsequent stepping callback is issued.
            CloseActiveSplit(current_track_);
        }
        if (
            information != nullptr
            && information->ActiveForceCollisionSplitId() >= 0
        ) {
            std::ostringstream message;
            const auto* volume = (
                current_track_ == nullptr
                    ? nullptr
                    : current_track_->GetVolume()
            );
            const auto* force_data = ForceCollisionTrackData(current_track_);
            message
                << "A gamma track ended before its force-collision split "
                << "reached a statistically weighted endpoint: split_id="
                << information->ActiveForceCollisionSplitId()
                << ", track_id="
                << (
                    current_track_ == nullptr
                        ? -1
                        : current_track_->GetTrackID()
                )
                << ", status="
                << (
                    current_track_ == nullptr
                        ? -1
                        : static_cast<int>(
                            current_track_->GetTrackStatus()
                        )
                )
                << ", volume="
                << (volume == nullptr ? "<null>" : volume->GetName())
                << ", weight="
                << (
                    current_track_ == nullptr
                        ? std::numeric_limits<double>::quiet_NaN()
                        : current_track_->GetWeight()
                )
                << ", gamma_interacted="
                << information->GammaInteracted()
                << ", force_clone="
                << information->ForceCollisionClone()
                << ", geant_bias_free="
                << (
                    force_data == nullptr
                        ? -1
                        : static_cast<int>(force_data->IsFreeFromBiasing())
                );
            throw std::runtime_error(message.str());
        }
        delegate_->EndTracking();
        current_track_ = nullptr;
    }

private:
    bool ShouldDelegate(const G4Track* track) const {
        if (
            track == nullptr
            || track->GetDefinition() != G4Gamma::Definition()
        ) {
            return false;
        }
        const auto* information = TrackInformation(track);
        if (information == nullptr) {
            throw std::runtime_error(
                "Calibration force collision encountered gamma transport "
                "without original-history lineage."
            );
        }
        if (information->GammaInteracted()) {
            // This callback can precede the next stepping action after a
            // weighted forced occurrence. The current track weight already
            // contains the occurrence factor, so finish the collision branch
            // before considering any later material volume.
            if (information->ActiveForceCollisionSplitId() >= 0) {
                CloseActiveSplit(track);
            }
            return false;
        }
        const bool active_scheme = HasActiveForceCollisionScheme(track);
        if (active_scheme) {
            if (
                information->ActiveForceCollisionSplitId() < 0
                && pending_clone_track_ == track
                && !information->GammaInteracted()
                && !information->SecondaryLineage()
            ) {
                // Geant4 asks the occurrence/final-state operators in the
                // same GPIL cycle after the cloning proposal has changed its
                // auxiliary state to `toBeCloned`, but before OperationApplied
                // gives this wrapper the clone on which lineage is installed.
                return true;
            }
            if (
                information->ActiveForceCollisionSplitId() < 0
                || information->GammaInteracted()
            ) {
                std::ostringstream message;
                const auto* volume = track->GetVolume();
                message
                    << "Active force-collision state disagrees with track "
                    << "lineage: track_id=" << track->GetTrackID()
                    << ", parent_id=" << track->GetParentID()
                    << ", step=" << track->GetCurrentStepNumber()
                    << ", volume="
                    << (volume == nullptr ? "<null>" : volume->GetName())
                    << ", split_id="
                    << information->ActiveForceCollisionSplitId()
                    << ", gamma_interacted="
                    << information->GammaInteracted()
                    << ", secondary_lineage="
                    << information->SecondaryLineage()
                    << ", force_clone="
                    << information->ForceCollisionClone();
                throw std::runtime_error(message.str());
            }
            return true;
        }
        if (information->ActiveForceCollisionSplitId() >= 0) {
            throw std::runtime_error(
                "Force-collision lineage stayed active after Geant4 reset."
            );
        }
        return (
            !information->GammaInteracted()
            && !information->SecondaryLineage()
        );
    }

    G4VBiasingOperation* ProposeNonPhysicsBiasingOperation(
        const G4Track* track,
        const G4BiasingProcessInterface* calling_process
    ) final {
        if (!ShouldDelegate(track)) {
            return nullptr;
        }
        auto* operation = delegate_->GetProposedNonPhysicsBiasingOperation(
            track,
            calling_process
        );
        if (
            operation != nullptr
            && operation->GetName() == "Cloning"
        ) {
            if (
                pending_clone_track_ != nullptr
                && pending_clone_track_ != track
            ) {
                throw std::runtime_error(
                    "A second force-collision clone was proposed before the "
                    "first clone was registered."
                );
            }
            pending_clone_track_ = track;
        }
        return operation;
    }

    G4VBiasingOperation* ProposeOccurenceBiasingOperation(
        const G4Track* track,
        const G4BiasingProcessInterface* calling_process
    ) final {
        return ShouldDelegate(track)
            ? delegate_->GetProposedOccurenceBiasingOperation(
                track,
                calling_process
            )
            : nullptr;
    }

    G4VBiasingOperation* ProposeFinalStateBiasingOperation(
        const G4Track* track,
        const G4BiasingProcessInterface* calling_process
    ) final {
        return ShouldDelegate(track)
            ? delegate_->GetProposedFinalStateBiasingOperation(
                track,
                calling_process
            )
            : nullptr;
    }

    void OperationApplied(
        const G4BiasingProcessInterface* calling_process,
        const G4BiasingAppliedCase biasing_case,
        G4VBiasingOperation* operation_applied,
        const G4VParticleChange* particle_change
    ) final {
        delegate_->ReportOperationApplied(
            calling_process,
            biasing_case,
            operation_applied,
            particle_change
        );
        if (
            biasing_case == BAC_NonPhysics
            && operation_applied != nullptr
            && operation_applied->GetName() == "Cloning"
        ) {
            RegisterClone(particle_change);
        }
    }

    void OperationApplied(
        const G4BiasingProcessInterface* calling_process,
        const G4BiasingAppliedCase biasing_case,
        G4VBiasingOperation* occurrence_operation,
        const G4double weight_for_occurrence,
        G4VBiasingOperation* final_state_operation,
        const G4VParticleChange* particle_change
    ) final {
        const double pre_occurrence_track_weight = (
            current_track_ == nullptr
                ? std::numeric_limits<double>::quiet_NaN()
                : current_track_->GetWeight()
        );
        delegate_->ReportOperationApplied(
            calling_process,
            biasing_case,
            occurrence_operation,
            weight_for_occurrence,
            final_state_operation,
            particle_change
        );
        auto* information = MutableTrackInformation(
            const_cast<G4Track*>(current_track_)
        );
        if (
            information != nullptr
            && information->ActiveForceCollisionSplitId() >= 0
            && !information->ForceCollisionClone()
        ) {
            const double terminal_weight = (
                pre_occurrence_track_weight * weight_for_occurrence
            );
            if (
                !(std::isfinite(terminal_weight)
                  && terminal_weight >= 0.0)
            ) {
                throw std::runtime_error(
                    "A forced occurrence produced an invalid branch weight."
                );
            }
            // The particle change is reported before Geant4 applies the
            // occurrence weight. Record the exact statistical branch mass
            // here so photoelectric termination and surviving Compton tracks
            // share the same conservation accounting.
            ForceCollisionDiagnostics::RecordTerminalBranch(
                diagnostics_state_.get(),
                information->ActiveForceCollisionSplitId(),
                information->BiasBranchLineageId(),
                terminal_weight
            );
            information->EndForceCollisionSplit();
        }
    }

    void ExitBiasing(
        const G4Track* track,
        const G4BiasingProcessInterface* calling_process
    ) final {
        delegate_->ExitingBiasing(track, calling_process);
        // Exiting one attached material volume is the statistical endpoint
        // of its force-collision split. Close it here before the same track
        // can enter another attached material and receive a new split.
        const auto* information = TrackInformation(track);
        if (
            information != nullptr
            && information->ActiveForceCollisionSplitId() >= 0
            && !information->ForceCollisionClone()
        ) {
            throw std::runtime_error(
                "A force-collision clone exited biasing without a weighted "
                "physical occurrence."
            );
        }
        CloseActiveSplit(track);
    }

    void CloseActiveSplit(const G4Track* track) const {
        auto* mutable_track = const_cast<G4Track*>(track);
        auto* information = MutableTrackInformation(mutable_track);
        if (
            information == nullptr
            || information->ActiveForceCollisionSplitId() < 0
        ) {
            return;
        }
        const double terminal_weight = (
            track == nullptr
                ? std::numeric_limits<double>::quiet_NaN()
                : track->GetWeight()
        );
        if (!(std::isfinite(terminal_weight) && terminal_weight >= 0.0)) {
            throw std::runtime_error(
                "A force-collision branch ended with an invalid weight."
            );
        }
        ForceCollisionDiagnostics::RecordTerminalBranch(
            diagnostics_state_.get(),
            information->ActiveForceCollisionSplitId(),
            information->BiasBranchLineageId(),
            terminal_weight
        );
        information->EndForceCollisionSplit();
    }

    void RegisterClone(const G4VParticleChange* particle_change) {
        auto* parent_information = MutableTrackInformation(
            const_cast<G4Track*>(current_track_)
        );
        if (
            current_track_ == nullptr
            || pending_clone_track_ != current_track_
            || parent_information == nullptr
            || parent_information->GammaInteracted()
            || parent_information->SecondaryLineage()
            || parent_information->ActiveForceCollisionSplitId() >= 0
            || particle_change == nullptr
            || particle_change->GetNumberOfSecondaries() != 1
        ) {
            std::ostringstream message;
            message
                << "Force-collision cloning did not expose one eligible "
                << "lineaged clone: current_track="
                << (current_track_ == nullptr ? 0 : 1)
                << ", pending_matches="
                << (
                    current_track_ != nullptr
                    && pending_clone_track_ == current_track_
                )
                << ", parent_information="
                << (parent_information == nullptr ? 0 : 1)
                << ", gamma_interacted="
                << (
                    parent_information == nullptr
                        ? -1
                        : static_cast<int>(
                            parent_information->GammaInteracted()
                        )
                )
                << ", secondary_lineage="
                << (
                    parent_information == nullptr
                        ? -1
                        : static_cast<int>(
                            parent_information->SecondaryLineage()
                        )
                )
                << ", split_id="
                << (
                    parent_information == nullptr
                        ? -2
                        : parent_information->ActiveForceCollisionSplitId()
                )
                << ", particle_change="
                << (particle_change == nullptr ? 0 : 1)
                << ", secondaries="
                << (
                    particle_change == nullptr
                        ? -1
                        : particle_change->GetNumberOfSecondaries()
                );
            throw std::runtime_error(message.str());
        }
        auto* clone = particle_change->GetSecondary(0);
        if (
            clone == nullptr
            || clone->GetUserInformation() != nullptr
            || !HasActiveForceCollisionScheme(clone)
            || !std::isfinite(clone->GetWeight())
            || clone->GetWeight() <= 0.0
        ) {
            throw std::runtime_error(
                "Force-collision clone state or weight is invalid."
            );
        }
        const long long split_id = diagnostics_->BeginSplit(
            diagnostics_state_.get(),
            parent_information->BiasBranchLineageId(),
            clone->GetWeight()
        );
        const long long clone_branch_lineage = (
            ForceCollisionDiagnostics::SurvivorBranchLineageId(
                diagnostics_state_.get(),
                split_id
            )
        );
        parent_information->BeginForceCollisionSplit(split_id);
        auto* clone_information = new TransportTrackInformation(
            parent_information->PrimaryBatchIndex(),
            parent_information->SourceIndex(),
            parent_information->PrimaryHistoryIndex(),
            parent_information->AngleStratumIndex(),
            parent_information->AngleStratumCount(),
            false,
            false,
            clone_branch_lineage,
            true,
            parent_information->PrimaryEventTimeS()
        );
        clone_information->BeginForceCollisionSplit(split_id);
        clone->SetUserInformation(clone_information);
        pending_clone_track_ = nullptr;
    }

    std::unique_ptr<G4BOptrForceCollision> delegate_;
    ForceCollisionDiagnostics::LocalHandle diagnostics_state_;
    ForceCollisionDiagnostics* diagnostics_ = nullptr;
    const G4Track* current_track_ = nullptr;
    const G4Track* pending_clone_track_ = nullptr;
};

class TransportDiagnostics {
public:
    struct LocalState {
        long long total_track_steps = 0;
        long long detector_hit_steps = 0;
        long long secondary_count = 0;
        long long killed_non_gamma_secondary_count = 0;
        std::unordered_map<const G4VProcess*, long long> process_counts;
        std::unordered_map<const G4VPhysicalVolume*, long long> volume_step_counts;
    };

    using LocalHandle = std::shared_ptr<LocalState>;

    LocalHandle AcquireLocal() {
        static thread_local std::unordered_map<
            const TransportDiagnostics*,
            std::weak_ptr<LocalState>
        > thread_states;
        const auto existing = thread_states.find(this);
        if (existing != thread_states.end()) {
            if (auto state = existing->second.lock()) {
                return state;
            }
        }
        auto state = std::make_shared<LocalState>();
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            local_states_.push_back(state);
        }
        thread_states[this] = state;
        return state;
    }

    void Clear() {
        for (const auto& state : RegisteredStates()) {
            *state = LocalState{};
        }
    }

    static void AddStep(LocalState* state, const G4Step* step) {
        if (state == nullptr || step == nullptr) {
            return;
        }
        state->total_track_steps += 1;
        const auto* pre_point = step->GetPreStepPoint();
        const auto* post_point = step->GetPostStepPoint();
        const auto* volume = pre_point == nullptr
            ? nullptr
            : pre_point->GetPhysicalVolume();
        state->volume_step_counts[volume] += 1;
        const auto* process = post_point == nullptr
            ? nullptr
            : post_point->GetProcessDefinedStep();
        const auto* gamma_general = dynamic_cast<
            const G4GammaGeneralProcess*
        >(process);
        if (
            gamma_general != nullptr
            && gamma_general->GetSelectedProcess() != nullptr
        ) {
            process = gamma_general->GetSelectedProcess();
        }
        state->process_counts[process] += 1;
        const auto* secondaries = step->GetSecondaryInCurrentStep();
        if (secondaries != nullptr) {
            state->secondary_count += static_cast<long long>(
                secondaries->size()
            );
        }
    }

    static void AddDetectorHitStep(LocalState* state) {
        if (state != nullptr) {
            state->detector_hit_steps += 1;
        }
    }

    static void AddKilledNonGammaSecondary(LocalState* state) {
        if (state != nullptr) {
            state->killed_non_gamma_secondary_count += 1;
        }
    }

    long long TotalTrackSteps() const {
        long long total = 0;
        for (const auto& state : RegisteredStates()) {
            total += state->total_track_steps;
        }
        return total;
    }

    long long DetectorHitSteps() const {
        long long total = 0;
        for (const auto& state : RegisteredStates()) {
            total += state->detector_hit_steps;
        }
        return total;
    }

    long long SecondaryCount() const {
        long long total = 0;
        for (const auto& state : RegisteredStates()) {
            total += state->secondary_count;
        }
        return total;
    }

    long long KilledNonGammaSecondaryCount() const {
        long long total = 0;
        for (const auto& state : RegisteredStates()) {
            total += state->killed_non_gamma_secondary_count;
        }
        return total;
    }

    std::map<std::string, long long> ProcessCounts() const {
        std::map<std::string, long long> counts;
        for (const auto& state : RegisteredStates()) {
            for (const auto& item : state->process_counts) {
                const std::string name = item.first == nullptr
                    ? "unknown"
                    : item.first->GetProcessName();
                counts[name] += item.second;
            }
        }
        return counts;
    }

    std::map<std::string, long long> VolumeStepCounts() const {
        std::map<std::string, long long> counts;
        for (const auto& state : RegisteredStates()) {
            for (const auto& item : state->volume_step_counts) {
                const std::string name = item.first == nullptr
                    ? "WorldBoundary"
                    : item.first->GetName();
                counts[name] += item.second;
            }
        }
        return counts;
    }

    long long ProcessCountForAliases(
        const std::set<std::string>& aliases
    ) const {
        long long total = 0;
        for (const auto& item : ProcessCounts()) {
            if (aliases.count(ToLower(item.first)) > 0) {
                total += item.second;
            }
        }
        return total;
    }

private:
    std::vector<LocalHandle> RegisteredStates() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return local_states_;
    }

    mutable std::mutex registry_mutex_;
    std::vector<LocalHandle> local_states_;
};

class EventStore {
public:
    struct TimedStepDeposit {
        double global_time_s = 0.0;
        double edep_mev = 0.0;
        double weight = 0.0;
        DetectorEntryClass entry_class =
            DetectorEntryClass::kUncollidedPrimary;
        std::size_t primary_batch_index =
            std::numeric_limits<std::size_t>::max();
        long long primary_history_index = -1;
        long long step_deposit_count = 1;
        double primary_event_time_s = 0.0;
        double impact_parameter_fraction =
            std::numeric_limits<double>::quiet_NaN();
    };

    struct BranchPulse {
        bool initialized = false;
        double start_time_s = 0.0;
        double edep_mev = 0.0;
        double weight = 0.0;
        DetectorEntryClass entry_class =
            DetectorEntryClass::kUncollidedPrimary;
        std::size_t primary_batch_index =
            std::numeric_limits<std::size_t>::max();
        long long primary_history_index = -1;
        long long step_deposit_count = 0;
        double primary_event_time_s = 0.0;
        double impact_parameter_fraction =
            std::numeric_limits<double>::quiet_NaN();
    };

    struct LocalState {
        bool event_active = false;
        std::map<long long, std::vector<TimedStepDeposit>>
            current_branch_deposits;
        std::vector<WeightedEventDeposit> event_deposits;
    };

    using LocalHandle = std::shared_ptr<LocalState>;

    explicit EventStore(
        const double coincidence_window_s =
            kDefaultDetectorCoincidenceWindowS
    ) : coincidence_window_s_(coincidence_window_s) {
        if (
            !std::isfinite(coincidence_window_s_)
            || coincidence_window_s_ <= 0.0
        ) {
            throw std::runtime_error(
                "Detector coincidence window must be finite and positive."
            );
        }
    }

    double CoincidenceWindowS() const {
        return coincidence_window_s_;
    }

    LocalHandle AcquireLocal() {
        static thread_local std::unordered_map<
            const EventStore*,
            std::weak_ptr<LocalState>
        > thread_states;
        const auto existing = thread_states.find(this);
        if (existing != thread_states.end()) {
            if (auto state = existing->second.lock()) {
                return state;
            }
        }
        auto state = std::make_shared<LocalState>();
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            local_states_.push_back(state);
        }
        thread_states[this] = state;
        return state;
    }

    static void BeginEvent(LocalState* state) {
        if (state == nullptr) {
            return;
        }
        state->event_active = true;
        state->current_branch_deposits.clear();
    }

    static void AddEnergyDeposit(
        LocalState* state,
        const double edep_mev,
        const double global_time_s,
        const double track_weight,
        const DetectorEntryClass entry_class,
        const std::size_t primary_batch_index,
        const long long primary_history_index,
        const long long bias_branch_lineage_id,
        const double primary_event_time_s,
        const double impact_parameter_fraction =
            std::numeric_limits<double>::quiet_NaN()
    ) {
        if (state == nullptr) {
            return;
        }
        if (!(std::isfinite(edep_mev) && edep_mev > 0.0)) {
            return;
        }
        if (
            !std::isfinite(global_time_s)
            || global_time_s < 0.0
            || !std::isfinite(track_weight)
            || track_weight < 0.0
            || primary_batch_index
                == std::numeric_limits<std::size_t>::max()
            || primary_history_index < 0
            || bias_branch_lineage_id < 0
            || (
                std::isfinite(impact_parameter_fraction)
                && (
                    impact_parameter_fraction < -1.0e-9
                    || impact_parameter_fraction > 1.0 + 1.0e-9
                )
            )
        ) {
            throw std::runtime_error(
                "Detected Geant4 energy has invalid weight or lineage."
            );
        }
        if (!state->event_active) {
            BeginEvent(state);
        }
        state->current_branch_deposits[bias_branch_lineage_id].push_back({
            global_time_s,
            edep_mev,
            track_weight,
            entry_class,
            primary_batch_index,
            primary_history_index,
            1,
            primary_event_time_s,
            impact_parameter_fraction,
        });
    }

    static void EndEvent(
        LocalState* state,
        const double coincidence_window_s
    ) {
        if (state == nullptr || !state->event_active) {
            return;
        }
        for (auto& item : state->current_branch_deposits) {
            auto& deposits = item.second;
            std::sort(
                deposits.begin(),
                deposits.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.global_time_s < rhs.global_time_s;
                }
            );
            const bool incident_gamma_entries = std::any_of(
                deposits.begin(),
                deposits.end(),
                [](const auto& deposit) {
                    return std::isfinite(deposit.impact_parameter_fraction);
                }
            );
            if (incident_gamma_entries) {
                const auto& first = deposits.front();
                for (const auto& deposit : deposits) {
                    const double weight_tolerance = (
                        1.0e-15
                        + 1.0e-12
                            * std::max(
                                std::abs(first.weight),
                                std::abs(deposit.weight)
                            )
                    );
                    if (
                        !std::isfinite(deposit.impact_parameter_fraction)
                        || std::abs(first.weight - deposit.weight)
                            > weight_tolerance
                        || first.primary_batch_index
                            != deposit.primary_batch_index
                        || first.primary_history_index
                            != deposit.primary_history_index
                        || std::abs(
                            first.primary_event_time_s
                            - deposit.primary_event_time_s
                        ) > 1.0e-15
                    ) {
                        throw std::runtime_error(
                            "Incident-gamma branch mixed weights, scoring "
                            "modes, or original history lineage."
                        );
                    }
                    state->event_deposits.push_back({
                        deposit.edep_mev,
                        deposit.weight,
                        deposit.entry_class,
                        deposit.primary_batch_index,
                        deposit.primary_history_index,
                        item.first,
                        deposit.global_time_s,
                        deposit.step_deposit_count,
                        deposit.primary_event_time_s,
                        deposit.impact_parameter_fraction,
                    });
                }
                continue;
            }
            BranchPulse pulse;
            const auto flush_pulse = [&]() {
                if (!pulse.initialized || pulse.edep_mev <= 0.0) {
                    return;
                }
                state->event_deposits.push_back({
                    pulse.edep_mev,
                    pulse.weight,
                    pulse.entry_class,
                    pulse.primary_batch_index,
                    pulse.primary_history_index,
                    item.first,
                    pulse.start_time_s,
                    pulse.step_deposit_count,
                    pulse.primary_event_time_s,
                    pulse.impact_parameter_fraction,
                });
            };
            for (const auto& deposit : deposits) {
                if (
                    !pulse.initialized
                    || deposit.global_time_s - pulse.start_time_s
                        > coincidence_window_s
                ) {
                    flush_pulse();
                    pulse = BranchPulse{};
                    pulse.initialized = true;
                    pulse.start_time_s = deposit.global_time_s;
                    pulse.weight = deposit.weight;
                    pulse.entry_class = deposit.entry_class;
                    pulse.primary_batch_index = deposit.primary_batch_index;
                    pulse.primary_history_index = deposit.primary_history_index;
                    pulse.step_deposit_count = 0;
                    pulse.primary_event_time_s = deposit.primary_event_time_s;
                    pulse.impact_parameter_fraction = (
                        deposit.impact_parameter_fraction
                    );
                } else {
                    const double weight_tolerance = (
                        1.0e-15
                        + 1.0e-12
                            * std::max(
                                std::abs(pulse.weight),
                                std::abs(deposit.weight)
                            )
                    );
                    if (
                        std::abs(pulse.weight - deposit.weight)
                            > weight_tolerance
                        || pulse.primary_batch_index
                            != deposit.primary_batch_index
                        || pulse.primary_history_index
                            != deposit.primary_history_index
                        || std::abs(
                            pulse.primary_event_time_s
                            - deposit.primary_event_time_s
                        ) > 1.0e-15
                        || (
                            std::isfinite(pulse.impact_parameter_fraction)
                            != std::isfinite(
                                deposit.impact_parameter_fraction
                            )
                        )
                        || (
                            std::isfinite(pulse.impact_parameter_fraction)
                            && std::abs(
                                pulse.impact_parameter_fraction
                                - deposit.impact_parameter_fraction
                            ) > 1.0e-12
                        )
                    ) {
                        throw std::runtime_error(
                            "One detector pulse mixed weights or original "
                            "history lineage."
                        );
                    }
                    pulse.entry_class = MergeDetectorEntryClass(
                        pulse.entry_class,
                        deposit.entry_class
                    );
                }
                pulse.edep_mev += deposit.edep_mev;
                pulse.step_deposit_count += deposit.step_deposit_count;
            }
            flush_pulse();
        }
        state->current_branch_deposits.clear();
        state->event_active = false;
    }

    std::vector<WeightedEventDeposit> TakeEventDepositsMeV() {
        std::vector<WeightedEventDeposit> deposits;
        for (const auto& state : RegisteredStates()) {
            deposits.reserve(
                deposits.size() + state->event_deposits.size()
            );
            std::move(
                state->event_deposits.begin(),
                state->event_deposits.end(),
                std::back_inserter(deposits)
            );
            state->event_deposits.clear();
        }
        std::sort(
            deposits.begin(),
            deposits.end(),
            [](const auto& lhs, const auto& rhs) {
                return std::tie(
                    lhs.primary_history_index,
                    lhs.bias_branch_lineage_id,
                    lhs.global_time_s
                ) < std::tie(
                    rhs.primary_history_index,
                    rhs.bias_branch_lineage_id,
                    rhs.global_time_s
                );
            }
        );
        return deposits;
    }

    void ClearDeposits() {
        for (const auto& state : RegisteredStates()) {
            *state = LocalState{};
        }
    }

private:
    double coincidence_window_s_ = kDefaultDetectorCoincidenceWindowS;

    std::vector<LocalHandle> RegisteredStates() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return local_states_;
    }

    mutable std::mutex registry_mutex_;
    std::vector<LocalHandle> local_states_;
};

std::size_t PrimaryBatchIndexForTrack(const G4Track* track) {
    const auto* information = TrackInformation(track);
    return information == nullptr
        ? std::numeric_limits<std::size_t>::max()
        : information->PrimaryBatchIndex();
}

long long PrimaryHistoryIndexForTrack(const G4Track* track) {
    const auto* information = TrackInformation(track);
    return information == nullptr
        ? -1
        : information->PrimaryHistoryIndex();
}

long long BiasBranchLineageIdForTrack(const G4Track* track) {
    const auto* information = TrackInformation(track);
    return information == nullptr
        ? -1
        : information->BiasBranchLineageId();
}

double PrimaryEventTimeForTrack(const G4Track* track) {
    const auto* information = TrackInformation(track);
    return information == nullptr ? 0.0 : information->PrimaryEventTimeS();
}

DetectorEntryClass ClassifyDetectorEntryTrack(const G4Track* track) {
    if (track == nullptr || track->GetDefinition() != G4Gamma::Definition()) {
        return DetectorEntryClass::kInteractedPrimary;
    }
    const auto* information = TrackInformation(track);
    if (information != nullptr) {
        if (information->SecondaryLineage()) {
            return DetectorEntryClass::kSecondary;
        }
        if (information->GammaInteracted()) {
            return DetectorEntryClass::kInteractedPrimary;
        }
    } else if (track->GetParentID() > 0) {
        return DetectorEntryClass::kSecondary;
    }
    return DetectorEntryClass::kUncollidedPrimary;
}

class CrystalSensitiveDetector : public G4VSensitiveDetector {
public:
    CrystalSensitiveDetector(
        EventStore* store,
        TransportDiagnostics* diagnostics,
        const std::string& detector_scoring_mode
    ) : G4VSensitiveDetector("CrystalSD"),
        event_state_(store == nullptr ? nullptr : store->AcquireLocal()),
        coincidence_window_s_(
            store == nullptr
                ? kDefaultDetectorCoincidenceWindowS
                : store->CoincidenceWindowS()
        ),
        diagnostics_state_(
            diagnostics == nullptr ? nullptr : diagnostics->AcquireLocal()
        ),
        score_energy_deposits_(NormalizeDetectorScoringMode(detector_scoring_mode) == "full_transport") {}

    void Initialize(G4HCofThisEvent*) override {
        EventStore::BeginEvent(event_state_.get());
    }

    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        if (step == nullptr) {
            return false;
        }
        if (!score_energy_deposits_) {
            return false;
        }
        const auto* track = step->GetTrack();
        const double weight = track == nullptr ? 1.0 : track->GetWeight();
        const double edep_mev = step->GetTotalEnergyDeposit() / MeV;
        const auto* post_step = step->GetPostStepPoint();
        const double global_time_s = post_step == nullptr
            ? (track == nullptr ? 0.0 : track->GetGlobalTime() / s)
            : post_step->GetGlobalTime() / s;
        if (edep_mev > 0.0) {
            TransportDiagnostics::AddDetectorHitStep(
                diagnostics_state_.get()
            );
        }
        EventStore::AddEnergyDeposit(
            event_state_.get(),
            edep_mev,
            global_time_s,
            weight,
            ClassifyDetectorEntryTrack(track),
            PrimaryBatchIndexForTrack(track),
            PrimaryHistoryIndexForTrack(track),
            BiasBranchLineageIdForTrack(track),
            PrimaryEventTimeForTrack(track)
        );
        return true;
    }

    void EndOfEvent(G4HCofThisEvent*) override {
        EventStore::EndEvent(
            event_state_.get(),
            coincidence_window_s_
        );
    }

private:
    EventStore::LocalHandle event_state_;
    double coincidence_window_s_ = kDefaultDetectorCoincidenceWindowS;
    TransportDiagnostics::LocalHandle diagnostics_state_;
    bool score_energy_deposits_ = true;
};

class Geant4SceneConstruction : public G4VUserDetectorConstruction {
public:
    Geant4SceneConstruction(
        const SceneSpec* scene,
        const RequestSpec* request,
        EventStore* event_store,
        TransportDiagnostics* diagnostics,
        ForceCollisionDiagnostics* force_collision_diagnostics,
        std::string detector_scoring_mode,
        const bool place_shields,
        const bool mean_calibration_forced_collision
    ) : scene_(scene),
        request_(request),
        event_store_(event_store),
        diagnostics_(diagnostics),
        force_collision_diagnostics_(force_collision_diagnostics),
        detector_scoring_mode_(std::move(detector_scoring_mode)),
        place_shields_(place_shields),
        mean_calibration_forced_collision_(
            mean_calibration_forced_collision
        ) {}

    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* world_material = ResolveMaterial("air", -1.0, {}, "air");
        world_material_ = world_material;
        force_collision_leaf_logicals_.clear();
        world_half_extents_m_ = ComputeWorldHalfExtentsM();
        ValidateMovablePoses(*request_);
        auto* world_solid = new G4Box(
            "World",
            world_half_extents_m_[0] * m,
            world_half_extents_m_[1] * m,
            world_half_extents_m_[2] * m
        );
        auto* world_logic = new G4LogicalVolume(world_solid, world_material, "WorldLV");
        auto* world_physical = new G4PVPlacement(
            nullptr,
            G4ThreeVector(),
            world_logic,
            "WorldPV",
            nullptr,
            false,
            0,
            true
        );
        for (std::size_t index = 0; index < scene_->volumes.size(); ++index) {
            const auto& volume = scene_->volumes[index];
            auto* material = ResolveMaterial(
                volume.material.name,
                volume.material.density_g_cm3,
                volume.material.composition_by_mass,
                volume.material.preset_name
            );
            auto* solid = BuildSolid(volume, "StaticSolid_" + std::to_string(index));
            auto* logic = new G4LogicalVolume(solid, material, "StaticLV_" + std::to_string(index));
            if (
                mean_calibration_forced_collision_
                && ToLower(volume.transport_mode) == "geant4"
                && material != world_material_
            ) {
                force_collision_leaf_logicals_.push_back(logic);
            }
            logic->SetVisAttributes(G4VisAttributes::GetInvisible());
            auto rotation = QuaternionToPlacementRotation(volume.qw, volume.qx, volume.qy, volume.qz);
            G4ThreeVector placement(volume.tx * m, volume.ty * m, volume.tz * m);
            if (volume.shape == "mesh") {
                rotation = std::make_unique<G4RotationMatrix>();
                placement = G4ThreeVector();
            }
            new G4PVPlacement(
                rotation.release(),
                placement,
                logic,
                volume.path,
                world_logic,
                false,
                static_cast<int>(index),
                true
            );
        }
        if (place_shields_) {
            if (scene_->fe_shield.has_value()) {
                fe_shield_physical_ = BuildShield(
                    *scene_->fe_shield,
                    request_->fe_pose,
                    world_logic,
                    1001
                );
                RegisterForceCollisionShieldLeaf(fe_shield_physical_);
            }
            if (scene_->pb_shield.has_value()) {
                pb_shield_physical_ = BuildShield(
                    *scene_->pb_shield,
                    request_->pb_pose,
                    world_logic,
                    1002
                );
                RegisterForceCollisionShieldLeaf(pb_shield_physical_);
            }
        }
        BuildDetector(world_logic, nist);
        current_detector_pose_ = request_->detector_pose;
        current_fe_pose_ = request_->fe_pose;
        current_pb_pose_ = request_->pb_pose;
        movable_poses_initialized_ = true;
        return world_physical;
    }

    bool UpdateMovablePoses(const RequestSpec& request) {
        ValidateMovablePoses(request);
        const bool detector_changed = (
            !movable_poses_initialized_
            || !PosesEqual(current_detector_pose_, request.detector_pose)
        );
        const bool fe_changed = (
            place_shields_
            && scene_->fe_shield.has_value()
            && (
                !movable_poses_initialized_
                || !PosesEqual(current_fe_pose_, request.fe_pose)
            )
        );
        const bool pb_changed = (
            place_shields_
            && scene_->pb_shield.has_value()
            && (
                !movable_poses_initialized_
                || !PosesEqual(current_pb_pose_, request.pb_pose)
            )
        );
        if (!detector_changed && !fe_changed && !pb_changed) {
            return false;
        }

        G4GeometryManager::GetInstance()->OpenGeometry();
        if (detector_changed) {
            UpdatePhysicalPose(
                detector_housing_physical_,
                request.detector_pose
            );
        }
        if (fe_changed) {
            UpdatePhysicalPose(fe_shield_physical_, request.fe_pose);
        }
        if (pb_changed) {
            UpdatePhysicalPose(pb_shield_physical_, request.pb_pose);
        }
        current_detector_pose_ = request.detector_pose;
        current_fe_pose_ = request.fe_pose;
        current_pb_pose_ = request.pb_pose;
        movable_poses_initialized_ = true;
        if (auto* run_manager = G4RunManager::GetRunManager()) {
            run_manager->GeometryHasBeenModified();
        }
        return true;
    }

    void ConstructSDandField() override {
        auto* sd_manager = G4SDManager::GetSDMpointer();
        auto* crystal_sd = new CrystalSensitiveDetector(
            event_store_,
            diagnostics_,
            detector_scoring_mode_
        );
        sd_manager->AddNewDetector(crystal_sd);
        SetSensitiveDetector("DetectorCrystalLV", crystal_sd);
        AttachCalibrationForceCollisionOperators();
    }

    std::size_t ForceCollisionLeafCount() const {
        return force_collision_leaf_logicals_.size();
    }

private:
    static bool PosesEqual(const PoseSpec& lhs, const PoseSpec& rhs) {
        return (
            lhs.x == rhs.x
            && lhs.y == rhs.y
            && lhs.z == rhs.z
            && lhs.qw == rhs.qw
            && lhs.qx == rhs.qx
            && lhs.qy == rhs.qy
            && lhs.qz == rhs.qz
        );
    }

    static void ValidateFinitePose(
        const PoseSpec& pose,
        const std::string& label
    ) {
        if (
            !std::isfinite(pose.x)
            || !std::isfinite(pose.y)
            || !std::isfinite(pose.z)
            || !std::isfinite(pose.qw)
            || !std::isfinite(pose.qx)
            || !std::isfinite(pose.qy)
            || !std::isfinite(pose.qz)
        ) {
            throw std::runtime_error(
                label + " pose requires finite translation and quaternion."
            );
        }
    }

    void ValidateSphereInsideWorld(
        const PoseSpec& pose,
        const double radius_m,
        const std::string& label
    ) const {
        ValidateFinitePose(pose, label);
        if (!std::isfinite(radius_m) || radius_m < 0.0) {
            throw std::runtime_error(
                label + " world-containment radius is invalid."
            );
        }
        const std::array<double, 3> coordinates = {
            pose.x,
            pose.y,
            pose.z,
        };
        for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
            if (
                std::abs(coordinates[axis]) + radius_m
                    >= world_half_extents_m_[axis]
            ) {
                throw std::runtime_error(
                    label
                    + " pose lies outside the persistent Geant4 world."
                );
            }
        }
    }

    void ValidateMovablePoses(const RequestSpec& request) const {
        ValidateSphereInsideWorld(
            request.detector_pose,
            scene_->detector.crystal_radius_m
                + scene_->detector.housing_thickness_m,
            "Detector"
        );
        if (place_shields_ && scene_->fe_shield.has_value()) {
            ValidateSphereInsideWorld(
                request.fe_pose,
                scene_->fe_shield->outer_radius_m,
                "Fe shield"
            );
        }
        if (place_shields_ && scene_->pb_shield.has_value()) {
            ValidateSphereInsideWorld(
                request.pb_pose,
                scene_->pb_shield->outer_radius_m,
                "Pb shield"
            );
        }
    }

    void RegisterForceCollisionShieldLeaf(G4VPhysicalVolume* physical) {
        if (!mean_calibration_forced_collision_ || physical == nullptr) {
            return;
        }
        auto* logical = physical->GetLogicalVolume();
        if (
            logical != nullptr
            && logical->GetMaterial() != nullptr
            && logical->GetMaterial() != world_material_
        ) {
            force_collision_leaf_logicals_.push_back(logical);
        }
    }

    void AttachCalibrationForceCollisionOperators() {
        if (!mean_calibration_forced_collision_) {
            return;
        }
        if (force_collision_diagnostics_ == nullptr) {
            throw std::runtime_error(
                "Calibration force collision requires diagnostics wiring."
            );
        }
        std::set<G4LogicalVolume*> unique_logicals;
        for (auto* logical : force_collision_leaf_logicals_) {
            if (logical == nullptr || !unique_logicals.insert(logical).second) {
                continue;
            }
            if (
                logical->GetNoDaughters() != 0
                || logical->GetMaterial() == nullptr
                || logical->GetMaterial() == world_material_
                || logical->GetSolid() == nullptr
                || logical->GetName() == "WorldLV"
                || logical->GetName() == "DetectorHousingLV"
                || logical->GetName() == "DetectorCrystalLV"
            ) {
                throw std::runtime_error(
                    "Calibration force collision may attach only to "
                    "daughter-free homogeneous non-air material leaves."
                );
            }
            auto* biasing_operator = new CalibrationFirstCollisionOperator(
                "CalibrationFirstCollision_"
                    + logical->GetName()
                    + "_"
                    + std::to_string(logical->GetInstanceID()),
                force_collision_diagnostics_
            );
            biasing_operator->AttachTo(logical);
            if (
                G4VBiasingOperator::GetBiasingOperator(logical)
                    != biasing_operator
            ) {
                throw std::runtime_error(
                    "Calibration force-collision operator attachment failed."
                );
            }
        }
        if (unique_logicals.empty()) {
            throw std::runtime_error(
                "Calibration force collision found no eligible material "
                "leaf logical volumes."
            );
        }
    }

    std::array<double, 3> ComputeWorldHalfExtentsM() const {
        const std::array<double, 3> legacy_half_extents_m = {
            0.5 * std::max(40.0, scene_->room_x + 20.0),
            0.5 * std::max(40.0, scene_->room_y + 20.0),
            0.5 * std::max(20.0, scene_->room_z + 10.0),
        };
        std::array<double, 3> daughter_max_abs_m = {0.0, 0.0, 0.0};
        const auto include_point = [&daughter_max_abs_m](
            const double x,
            const double y,
            const double z
        ) {
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                throw std::runtime_error(
                    "World bounds require finite daughter coordinates."
                );
            }
            daughter_max_abs_m[0] = std::max(daughter_max_abs_m[0], std::abs(x));
            daughter_max_abs_m[1] = std::max(daughter_max_abs_m[1], std::abs(y));
            daughter_max_abs_m[2] = std::max(daughter_max_abs_m[2], std::abs(z));
        };
        const auto include_sphere = [&include_point](
            const double x,
            const double y,
            const double z,
            const double radius_m
        ) {
            if (!std::isfinite(radius_m) || radius_m < 0.0) {
                throw std::runtime_error(
                    "World bounds require finite nonnegative daughter radii."
                );
            }
            include_point(x - radius_m, y - radius_m, z - radius_m);
            include_point(x + radius_m, y + radius_m, z + radius_m);
        };

        include_point(0.0, 0.0, 0.0);
        include_point(scene_->room_x, scene_->room_y, scene_->room_z);
        for (const auto& volume : scene_->volumes) {
            if (volume.shape == "box") {
                if (
                    !std::isfinite(volume.sx)
                    || !std::isfinite(volume.sy)
                    || !std::isfinite(volume.sz)
                    || volume.sx < 0.0
                    || volume.sy < 0.0
                    || volume.sz < 0.0
                ) {
                    throw std::runtime_error(
                        "World bounds require finite nonnegative box sizes."
                    );
                }
                const auto rotation = QuaternionToRotation(
                    volume.qw,
                    volume.qx,
                    volume.qy,
                    volume.qz
                );
                for (const double sign_x : {-1.0, 1.0}) {
                    for (const double sign_y : {-1.0, 1.0}) {
                        for (const double sign_z : {-1.0, 1.0}) {
                            const G4ThreeVector local_corner(
                                sign_x * 0.5 * volume.sx,
                                sign_y * 0.5 * volume.sy,
                                sign_z * 0.5 * volume.sz
                            );
                            const G4ThreeVector rotated_corner = (
                                *rotation
                            ) * local_corner;
                            include_point(
                                volume.tx + rotated_corner.x(),
                                volume.ty + rotated_corner.y(),
                                volume.tz + rotated_corner.z()
                            );
                        }
                    }
                }
            } else if (volume.shape == "sphere") {
                include_sphere(
                    volume.tx,
                    volume.ty,
                    volume.tz,
                    volume.radius_m
                );
            } else if (volume.shape == "mesh") {
                for (const auto& triangle : volume.triangles) {
                    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                        const std::size_t offset = 3 * vertex;
                        include_point(
                            triangle[offset],
                            triangle[offset + 1],
                            triangle[offset + 2]
                        );
                    }
                }
            }
        }
        for (const auto& source : scene_->sources) {
            include_point(source.x, source.y, source.z);
        }
        include_sphere(
            request_->detector_pose.x,
            request_->detector_pose.y,
            request_->detector_pose.z,
            scene_->detector.crystal_radius_m
                + scene_->detector.housing_thickness_m
        );
        if (place_shields_ && scene_->fe_shield.has_value()) {
            include_sphere(
                request_->fe_pose.x,
                request_->fe_pose.y,
                request_->fe_pose.z,
                scene_->fe_shield->outer_radius_m
            );
        }
        if (place_shields_ && scene_->pb_shield.has_value()) {
            include_sphere(
                request_->pb_pose.x,
                request_->pb_pose.y,
                request_->pb_pose.z,
                scene_->pb_shield->outer_radius_m
            );
        }

        std::array<double, 3> half_extents_m{};
        for (std::size_t axis = 0; axis < half_extents_m.size(); ++axis) {
            half_extents_m[axis] = std::max(
                legacy_half_extents_m[axis],
                daughter_max_abs_m[axis] + kWorldDaughterMarginM
            );
            if (!std::isfinite(half_extents_m[axis]) || half_extents_m[axis] <= 0.0) {
                throw std::runtime_error(
                    "Computed Geant4 world half-extent must be finite and positive."
                );
            }
        }
        return half_extents_m;
    }

    G4VSolid* BuildSolid(const VolumeSpec& volume, const std::string& name) const {
        if (volume.shape == "box") {
            return new G4Box(name, 0.5 * volume.sx * m, 0.5 * volume.sy * m, 0.5 * volume.sz * m);
        }
        if (volume.shape == "sphere") {
            return new G4Sphere(name, 0.0, volume.radius_m * m, 0.0, 360.0 * deg, 0.0, 180.0 * deg);
        }
        if (volume.shape == "mesh") {
            auto* solid = new G4TessellatedSolid(name);
            for (const auto& triangle : volume.triangles) {
                auto* facet = new G4TriangularFacet(
                    G4ThreeVector(triangle[0] * m, triangle[1] * m, triangle[2] * m),
                    G4ThreeVector(triangle[3] * m, triangle[4] * m, triangle[5] * m),
                    G4ThreeVector(triangle[6] * m, triangle[7] * m, triangle[8] * m),
                    ABSOLUTE
                );
                solid->AddFacet(facet);
            }
            solid->SetSolidClosed(true);
            return solid;
        }
        throw std::runtime_error("Unsupported volume shape: " + volume.shape);
    }

    G4VPhysicalVolume* BuildShield(
        const ShieldSpec& shield,
        const PoseSpec& pose,
        G4LogicalVolume* parent_logic,
        int copy_number
    ) {
        if (
            shield.shape != "spherical_octant_shell"
            || !std::isfinite(shield.inner_radius_m)
            || shield.inner_radius_m < 0.0
            || !std::isfinite(shield.thickness_m)
            || shield.thickness_m <= 0.0
            || !std::isfinite(shield.outer_radius_m)
            || shield.outer_radius_m
                != shield.inner_radius_m + shield.thickness_m
        ) {
            throw std::runtime_error(
                "BuildShield requires a validated positive-thickness "
                "spherical-octant shell."
            );
        }
        auto* material = ResolveMaterial(
            shield.material.name,
            shield.material.density_g_cm3,
            shield.material.composition_by_mass,
            shield.material.preset_name
        );
        auto* solid = new G4Sphere(
            shield.kind + "_ShieldSolid",
            shield.inner_radius_m * m,
            shield.outer_radius_m * m,
            0.0 * deg,
            90.0 * deg,
            0.0 * deg,
            90.0 * deg
        );
        auto* logic = new G4LogicalVolume(solid, material, shield.kind + "_ShieldLV");
        auto rotation = QuaternionToPlacementRotation(pose.qw, pose.qx, pose.qy, pose.qz);
        return new G4PVPlacement(
            rotation.release(),
            G4ThreeVector(pose.x * m, pose.y * m, pose.z * m),
            logic,
            shield.path,
            parent_logic,
            false,
            copy_number,
            true
        );
    }

    void UpdatePhysicalPose(G4VPhysicalVolume* physical, const PoseSpec& pose) const {
        if (physical == nullptr) {
            return;
        }
        auto rotation = QuaternionToPlacementRotation(pose.qw, pose.qx, pose.qy, pose.qz);
        physical->SetTranslation(G4ThreeVector(pose.x * m, pose.y * m, pose.z * m));
        auto* existing_rotation = physical->GetRotation();
        if (existing_rotation == nullptr) {
            physical->SetRotation(rotation.release());
        } else {
            *existing_rotation = *rotation;
        }
    }

    void BuildDetector(G4LogicalVolume* parent_logic, G4NistManager*) {
        const auto& detector = scene_->detector;
        auto* housing_material = ResolveMaterial(detector.housing_material, -1.0, {}, detector.housing_material);
        auto* crystal_material = ResolveMaterial(detector.crystal_material, -1.0, {}, detector.crystal_material);
        const double outer_radius_m = detector.crystal_radius_m + detector.housing_thickness_m;
        auto* housing_solid = new G4Sphere(
            "DetectorHousingSolid",
            0.0,
            outer_radius_m * m,
            0.0,
            360.0 * deg,
            0.0,
            180.0 * deg
        );
        auto* housing_logic = new G4LogicalVolume(housing_solid, housing_material, "DetectorHousingLV");
        auto housing_rotation = QuaternionToPlacementRotation(
            request_->detector_pose.qw,
            request_->detector_pose.qx,
            request_->detector_pose.qy,
            request_->detector_pose.qz
        );
        detector_housing_physical_ = new G4PVPlacement(
            housing_rotation.release(),
            G4ThreeVector(
                request_->detector_pose.x * m,
                request_->detector_pose.y * m,
                request_->detector_pose.z * m
            ),
            housing_logic,
            "DetectorHousingPV",
            parent_logic,
            false,
            2001,
            true
        );
        auto* crystal_solid = new G4Sphere(
            "DetectorCrystalSolid",
            0.0,
            detector.crystal_radius_m * m,
            0.0,
            360.0 * deg,
            0.0,
            180.0 * deg
        );
        auto* crystal_logic = new G4LogicalVolume(crystal_solid, crystal_material, "DetectorCrystalLV");
        new G4PVPlacement(
            nullptr,
            G4ThreeVector(0.0, 0.0, 0.0),
            crystal_logic,
            "DetectorCrystalPV",
            housing_logic,
            false,
            2002,
            true
        );
    }

    G4Material* ResolveMaterial(
        const std::string& material_name,
        const double density_g_cm3,
        std::map<std::string, double> composition_by_mass,
        const std::string& preset_name
    ) const {
        MaterialSpec material;
        material.name = material_name;
        material.preset_name = preset_name;
        material.density_g_cm3 = density_g_cm3;
        material.composition_by_mass = std::move(composition_by_mass);
        return ResolveAttenuationMaterial(material, preset_name);
    }

    std::unique_ptr<G4RotationMatrix> QuaternionToRotation(
        const double qw,
        const double qx,
        const double qy,
        const double qz
    ) const {
        auto rotation = std::make_unique<G4RotationMatrix>();
        const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (norm <= 1.0e-12) {
            return rotation;
        }
        const double w = qw / norm;
        const double x = qx / norm;
        const double y = qy / norm;
        const double z = qz / norm;
        const double r00 = 1.0 - 2.0 * (y * y + z * z);
        const double r01 = 2.0 * (x * y - z * w);
        const double r02 = 2.0 * (x * z + y * w);
        const double r10 = 2.0 * (x * y + z * w);
        const double r11 = 1.0 - 2.0 * (x * x + z * z);
        const double r12 = 2.0 * (y * z - x * w);
        const double r20 = 2.0 * (x * z - y * w);
        const double r21 = 2.0 * (y * z + x * w);
        const double r22 = 1.0 - 2.0 * (x * x + y * y);
        rotation->setRows(
            G4ThreeVector(r00, r01, r02),
            G4ThreeVector(r10, r11, r12),
            G4ThreeVector(r20, r21, r22)
        );
        return rotation;
    }

    std::unique_ptr<G4RotationMatrix> QuaternionToPlacementRotation(
        const double qw,
        const double qx,
        const double qy,
        const double qz
    ) const {
        auto rotation = QuaternionToRotation(qw, qx, qy, qz);
        // G4PVPlacement expects the daughter-to-mother transform inverse of the
        // active quaternion exported from Isaac/PF geometry.
        rotation->invert();
        return rotation;
    }

    const SceneSpec* scene_ = nullptr;
    const RequestSpec* request_ = nullptr;
    EventStore* event_store_ = nullptr;
    TransportDiagnostics* diagnostics_ = nullptr;
    ForceCollisionDiagnostics* force_collision_diagnostics_ = nullptr;
    std::string detector_scoring_mode_ = "full_transport";
    bool place_shields_ = true;
    bool mean_calibration_forced_collision_ = false;
    G4Material* world_material_ = nullptr;
    std::vector<G4LogicalVolume*> force_collision_leaf_logicals_;
    G4VPhysicalVolume* fe_shield_physical_ = nullptr;
    G4VPhysicalVolume* pb_shield_physical_ = nullptr;
    G4VPhysicalVolume* detector_housing_physical_ = nullptr;
    std::array<double, 3> world_half_extents_m_ = {0.0, 0.0, 0.0};
    PoseSpec current_detector_pose_;
    PoseSpec current_fe_pose_;
    PoseSpec current_pb_pose_;
    bool movable_poses_initialized_ = false;
};

struct PrimarySourceSnapshot {
    G4ThreeVector position;
    double energy_keV = 662.0;
    std::string source_bias_mode = "analog";
    std::string source_rate_model = "detector_cps_1m";
    G4ThreeVector detector_center;
    double cone_half_angle_rad = CLHEP::pi;
    double isotropic_fraction = 1.0;
    double primary_history_weight = 1.0;
    double acquisition_duration_s = 0.0;
};

struct PrimaryDirectionSample {
    G4ThreeVector direction;
    double weight = 1.0;
};

struct PrimaryHistoryBatch {
    PrimarySourceSnapshot source;
    std::size_t source_index = 0;
    std::string isotope;
    std::string source_token;
    std::string line_token;
    double expected_unthinned_histories = 0.0;
    long long sampled_histories = 0;
    int angle_stratum_index = -1;
    int angle_stratum_count = 0;
    int angle_mu_stratum_count = 1;
    int angle_phi_stratum_count = 1;
    bool radioactive_decay_event = false;
    int atomic_number = 0;
    int mass_number = 0;
    double excitation_keV = 0.0;
};

struct PrimaryHistorySelection {
    const PrimaryHistoryBatch* batch = nullptr;
    std::size_t batch_index = std::numeric_limits<std::size_t>::max();
    long long primary_history_index = -1;
};

struct WorkerPrimaryContext {
    std::size_t primary_batch_index =
        std::numeric_limits<std::size_t>::max();
    std::size_t source_index = std::numeric_limits<std::size_t>::max();
    long long primary_history_index = -1;
    int angle_stratum_index = -1;
    int angle_stratum_count = 0;
    bool radioactive_decay_event = false;
    double primary_event_time_s = 0.0;
};

class PrimarySourceState {
public:
    void ConfigureSchedule(std::vector<PrimaryHistoryBatch> schedule) {
        schedule_ = std::move(schedule);
        cumulative_histories_.clear();
        cumulative_histories_.reserve(schedule_.size());
        long long total_histories = 0;
        for (auto& batch : schedule_) {
            if (batch.sampled_histories <= 0) {
                throw std::runtime_error(
                    "Primary history schedule requires positive batch sizes."
                );
            }
            if (
                !std::isfinite(batch.expected_unthinned_histories)
                || batch.expected_unthinned_histories <= 0.0
            ) {
                throw std::runtime_error(
                    "Primary history schedule requires positive finite "
                    "unthinned expectations."
                );
            }
            if (
                !std::isfinite(batch.source.primary_history_weight)
                || batch.source.primary_history_weight <= 0.0
            ) {
                throw std::runtime_error(
                    "Primary history schedule requires positive finite "
                    "history weights."
                );
            }
            if (
                (batch.angle_stratum_count == 0
                    && batch.angle_stratum_index != -1)
                || (
                    batch.angle_stratum_count > 0
                    && (
                        batch.angle_stratum_index < 0
                        || batch.angle_stratum_index
                            >= batch.angle_stratum_count
                    )
                )
            ) {
                throw std::runtime_error(
                    "Primary history angle stratum index/count are invalid."
                );
            }
            if (
                batch.angle_mu_stratum_count <= 0
                || batch.angle_phi_stratum_count <= 0
                || (
                    batch.angle_stratum_count > 0
                    && batch.angle_stratum_count
                        != batch.angle_mu_stratum_count
                            * batch.angle_phi_stratum_count
                )
            ) {
                throw std::runtime_error(
                    "Primary history mu/phi stratum counts are invalid."
                );
            }
            if (
                batch.sampled_histories
                    > std::numeric_limits<long long>::max() - total_histories
            ) {
                throw std::runtime_error(
                    "Primary history schedule exceeds the supported event "
                    "count."
                );
            }
            total_histories += batch.sampled_histories;
            cumulative_histories_.push_back(total_histories);
        }
        beam_on_event_offset_.store(0, std::memory_order_release);
    }

    PrimaryHistorySelection SelectForEvent(const int event_id) const {
        if (event_id < 0) {
            throw std::runtime_error(
                "Geant4 produced a negative event identifier."
            );
        }
        const long long absolute_event_id = (
            beam_on_event_offset_.load(std::memory_order_acquire)
            + static_cast<long long>(event_id)
        );
        const auto selected = std::upper_bound(
            cumulative_histories_.begin(),
            cumulative_histories_.end(),
            absolute_event_id
        );
        if (selected == cumulative_histories_.end()) {
            throw std::runtime_error(
                "Geant4 event identifier exceeded the primary history "
                "schedule."
            );
        }
        const auto batch_index = static_cast<std::size_t>(
            std::distance(cumulative_histories_.begin(), selected)
        );
        return {
            &schedule_[batch_index],
            batch_index,
            absolute_event_id,
        };
    }

    void SetBeamOnEventOffset(const long long event_offset) {
        if (event_offset < 0 || event_offset > TotalScheduledHistories()) {
            throw std::runtime_error(
                "Primary history BeamOn offset is outside its schedule."
            );
        }
        beam_on_event_offset_.store(event_offset, std::memory_order_release);
    }

    long long TotalScheduledHistories() const {
        return cumulative_histories_.empty()
            ? 0
            : cumulative_histories_.back();
    }

    const std::vector<PrimaryHistoryBatch>& Schedule() const {
        return schedule_;
    }

private:
    std::vector<PrimaryHistoryBatch> schedule_;
    std::vector<long long> cumulative_histories_;
    std::atomic<long long> beam_on_event_offset_{0};
};

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    PrimaryGeneratorAction(
        const PrimarySourceState* state,
        std::shared_ptr<WorkerPrimaryContext> worker_context
    ) : state_(state), worker_context_(std::move(worker_context)) {
        particle_gun_ = std::make_unique<G4ParticleGun>(1);
        particle_gun_->SetParticleDefinition(G4Gamma::Definition());
    }

    void GeneratePrimaries(G4Event* event) override {
        if (event == nullptr || state_ == nullptr || worker_context_ == nullptr) {
            throw std::runtime_error(
                "Primary generation requires event, schedule, and worker "
                "context."
            );
        }
        const auto selection = state_->SelectForEvent(event->GetEventID());
        if (selection.batch == nullptr) {
            throw std::runtime_error(
                "Primary history schedule returned no batch."
            );
        }
        const auto& batch = *selection.batch;
        const auto& source = batch.source;
        worker_context_->primary_batch_index = selection.batch_index;
        worker_context_->source_index = batch.source_index;
        worker_context_->primary_history_index =
            selection.primary_history_index;
        worker_context_->angle_stratum_index = batch.angle_stratum_index;
        worker_context_->angle_stratum_count = batch.angle_stratum_count;
        worker_context_->radioactive_decay_event = (
            batch.radioactive_decay_event
        );
        particle_gun_->SetParticlePosition(source.position);
        PrimaryDirectionSample sample;
        if (batch.radioactive_decay_event) {
            auto* ion = G4IonTable::GetIonTable()->GetIon(
                batch.atomic_number,
                batch.mass_number,
                batch.excitation_keV * keV
            );
            if (ion == nullptr) {
                throw std::runtime_error(
                    "Geant4 could not construct the configured radioactive ion."
                );
            }
            particle_gun_->SetParticleDefinition(ion);
            particle_gun_->SetParticleEnergy(0.0);
            particle_gun_->SetParticleMomentumDirection(
                G4ThreeVector(0.0, 0.0, 1.0)
            );
            if (
                !std::isfinite(source.acquisition_duration_s)
                || source.acquisition_duration_s <= 0.0
            ) {
                throw std::runtime_error(
                    "Radioactive parent events require a positive acquisition "
                    "duration."
                );
            }
            worker_context_->primary_event_time_s = (
                G4UniformRand() * source.acquisition_duration_s
            );
            particle_gun_->SetParticleTime(
                worker_context_->primary_event_time_s * s
            );
            sample = {G4ThreeVector(0.0, 0.0, 1.0), 1.0};
        } else {
            particle_gun_->SetParticleTime(0.0);
            worker_context_->primary_event_time_s = 0.0;
            particle_gun_->SetParticleDefinition(G4Gamma::Definition());
            sample = SampleDirection(
                source,
                batch.angle_stratum_index,
                batch.angle_stratum_count,
                batch.angle_mu_stratum_count,
                batch.angle_phi_stratum_count
            );
            particle_gun_->SetParticleEnergy(source.energy_keV * keV);
            particle_gun_->SetParticleMomentumDirection(sample.direction);
        }
        particle_gun_->GeneratePrimaryVertex(event);
        ApplyPrimaryWeight(
            event,
            sample.weight * source.primary_history_weight
        );
    }

private:
    void ApplyPrimaryWeight(G4Event* event, const double weight) const {
        if (event == nullptr) {
            return;
        }
        const int vertex_count = event->GetNumberOfPrimaryVertex();
        if (vertex_count <= 0) {
            return;
        }
        auto* vertex = event->GetPrimaryVertex(vertex_count - 1);
        if (vertex == nullptr) {
            return;
        }
        auto* particle = vertex->GetPrimary();
        while (particle != nullptr) {
            particle->SetWeight(weight);
            particle = particle->GetNext();
        }
    }

    PrimaryDirectionSample SampleDirection(
        const PrimarySourceSnapshot& source,
        const int angle_stratum_index,
        const int angle_stratum_count,
        const int angle_mu_stratum_count,
        const int angle_phi_stratum_count
    ) const {
        if (NormalizeSourceBiasMode(source.source_bias_mode) == "detector_cone") {
            const G4ThreeVector axis_vector = source.detector_center - source.position;
            if (axis_vector.mag2() <= 1.0e-18) {
                if (angle_stratum_count > 0) {
                    throw std::runtime_error(
                        "Stratified detector-cone sampling requires distinct "
                        "source and detector positions."
                    );
                }
                return {RandomIsotropicDirection(), 1.0};
            }
            const G4ThreeVector axis = axis_vector.unit();
            const double theta = std::clamp(source.cone_half_angle_rad, 1.0e-9, CLHEP::pi);
            if (angle_stratum_count > 0) {
                if (
                    angle_stratum_index < 0
                    || angle_stratum_index >= angle_stratum_count
                ) {
                    throw std::runtime_error(
                        "Detector-cone angle stratum is outside its support."
                    );
                }
                return {
                    RandomConeDirectionStratum(
                        axis,
                        std::cos(theta),
                        angle_stratum_index,
                        angle_stratum_count,
                        angle_mu_stratum_count,
                        angle_phi_stratum_count
                    ),
                    1.0,
                };
            }
            return {RandomConeDirection(axis, std::cos(theta)), 1.0};
        }
        if (angle_stratum_count > 0) {
            throw std::runtime_error(
                "Angle strata are supported only by detector-cone mean "
                "calibration."
            );
        }
        if (NormalizeSourceBiasMode(source.source_bias_mode) != "mixture_cone_isotropic") {
            return {RandomIsotropicDirection(), 1.0};
        }
        const double f_iso = std::clamp(source.isotropic_fraction, 1.0e-6, 1.0);
        if (f_iso >= 1.0 - 1.0e-12) {
            return {RandomIsotropicDirection(), 1.0};
        }
        const G4ThreeVector axis_vector = source.detector_center - source.position;
        if (axis_vector.mag2() <= 1.0e-18) {
            return {RandomIsotropicDirection(), 1.0};
        }
        const G4ThreeVector axis = axis_vector.unit();
        const double theta = std::clamp(source.cone_half_angle_rad, 1.0e-9, CLHEP::pi);
        const double cos_theta = std::cos(theta);
        const bool sample_isotropic = G4UniformRand() < f_iso;
        const G4ThreeVector direction = sample_isotropic
            ? RandomIsotropicDirection()
            : RandomConeDirection(axis, cos_theta);
        const bool in_cone = direction.dot(axis) >= cos_theta - 1.0e-12;
        const double iso_pdf = 1.0 / (4.0 * CLHEP::pi);
        const double cone_pdf = 1.0 / ConeSolidAngleSr(theta);
        const double sample_pdf = f_iso * iso_pdf + (in_cone ? (1.0 - f_iso) * cone_pdf : 0.0);
        const double weight = sample_pdf > 0.0 ? iso_pdf / sample_pdf : 1.0;
        return {direction, std::max(0.0, weight)};
    }

    G4ThreeVector RandomIsotropicDirection() const {
        const double u = 2.0 * G4UniformRand() - 1.0;
        const double phi = 2.0 * CLHEP::pi * G4UniformRand();
        const double scale = std::sqrt(std::max(0.0, 1.0 - u * u));
        return G4ThreeVector(scale * std::cos(phi), scale * std::sin(phi), u);
    }

    G4ThreeVector RandomConeDirection(const G4ThreeVector& axis, const double cos_theta) const {
        const double u = cos_theta + (1.0 - cos_theta) * G4UniformRand();
        const double phi = 2.0 * CLHEP::pi * G4UniformRand();
        const double transverse = std::sqrt(std::max(0.0, 1.0 - u * u));
        const G4ThreeVector basis_x = axis.orthogonal().unit();
        const G4ThreeVector basis_y = axis.cross(basis_x).unit();
        return (u * axis + transverse * std::cos(phi) * basis_x + transverse * std::sin(phi) * basis_y).unit();
    }

    G4ThreeVector RandomConeDirectionStratum(
        const G4ThreeVector& axis,
        const double cos_theta,
        const int stratum_index,
        const int stratum_count,
        const int mu_stratum_count,
        const int phi_stratum_count
    ) const {
        if (
            stratum_count <= 0
            || stratum_index < 0
            || stratum_index >= stratum_count
            || mu_stratum_count <= 0
            || phi_stratum_count <= 0
            || stratum_count != mu_stratum_count * phi_stratum_count
        ) {
            throw std::runtime_error(
                "Cone stratum index/count must define a nonempty partition."
            );
        }
        const int mu_index = stratum_index / phi_stratum_count;
        const int phi_index = stratum_index % phi_stratum_count;
        const double lower_fraction = (
            static_cast<double>(mu_index)
            / static_cast<double>(mu_stratum_count)
        );
        const double upper_fraction = (
            static_cast<double>(mu_index + 1)
            / static_cast<double>(mu_stratum_count)
        );
        const double fraction = lower_fraction
            + (upper_fraction - lower_fraction) * G4UniformRand();
        const double u = cos_theta + (1.0 - cos_theta) * fraction;
        const double phi = (
            2.0
            * CLHEP::pi
            * (
                static_cast<double>(phi_index) + G4UniformRand()
            )
            / static_cast<double>(phi_stratum_count)
        );
        const double transverse = std::sqrt(std::max(0.0, 1.0 - u * u));
        const G4ThreeVector basis_x = axis.orthogonal().unit();
        const G4ThreeVector basis_y = axis.cross(basis_x).unit();
        return (
            u * axis
            + transverse * std::cos(phi) * basis_x
            + transverse * std::sin(phi) * basis_y
        ).unit();
    }

    std::unique_ptr<G4ParticleGun> particle_gun_;
    const PrimarySourceState* state_ = nullptr;
    std::shared_ptr<WorkerPrimaryContext> worker_context_;
};

class TransportTrackingAction : public G4UserTrackingAction {
public:
    explicit TransportTrackingAction(
        std::shared_ptr<WorkerPrimaryContext> worker_context
    ) : worker_context_(std::move(worker_context)) {}

    void PreUserTrackingAction(const G4Track* track) override {
        if (
            track == nullptr
            || track->GetUserInformation() != nullptr
            || worker_context_ == nullptr
        ) {
            return;
        }
        track->SetUserInformation(new TransportTrackInformation(
            worker_context_->primary_batch_index,
            worker_context_->source_index,
            worker_context_->primary_history_index,
            worker_context_->angle_stratum_index,
            worker_context_->angle_stratum_count,
            false,
            track->GetParentID() > 0,
            0,
            false,
            worker_context_->primary_event_time_s
        ));
    }

    void PostUserTrackingAction(const G4Track* track) override {
        const auto* parent_information = TrackInformation(track);
        if (parent_information == nullptr || fpTrackingManager == nullptr) {
            return;
        }
        auto* secondaries = fpTrackingManager->GimmeSecondaries();
        if (secondaries == nullptr) {
            return;
        }
        for (auto* secondary : *secondaries) {
            if (
                secondary == nullptr
                || secondary->GetUserInformation() != nullptr
            ) {
                continue;
            }
            secondary->SetUserInformation(new TransportTrackInformation(
                parent_information->PrimaryBatchIndex(),
                parent_information->SourceIndex(),
                parent_information->PrimaryHistoryIndex(),
                parent_information->AngleStratumIndex(),
                parent_information->AngleStratumCount(),
                parent_information->GammaInteracted(),
                true,
                parent_information->BiasBranchLineageId(),
                false,
                parent_information->PrimaryEventTimeS()
            ));
        }
    }

private:
    std::shared_ptr<WorkerPrimaryContext> worker_context_;
};

class ScheduledPrimaryRadioactiveDecay final : public G4RadioactiveDecay {
public:
    ScheduledPrimaryRadioactiveDecay() = default;

    G4VParticleChange* DecayIt(
        const G4Track& track,
        const G4Step& step
    ) override {
        if (track.GetParentID() != 0) {
            return G4RadioactiveDecay::DecayIt(track, step);
        }
        auto* scheduled_track = const_cast<G4Track*>(&track);
        const auto original_status = scheduled_track->GetTrackStatus();
        scheduled_track->SetTrackStatus(fAlive);
        auto* change = G4RadioactiveDecay::DecayIt(track, step);
        scheduled_track->SetTrackStatus(original_status);
        return change;
    }

protected:
    G4double GetMeanLifeTime(
        const G4Track& track,
        G4ForceCondition* condition
    ) override {
        if (track.GetParentID() == 0) {
            return std::numeric_limits<G4double>::min();
        }
        return G4RadioactiveDecay::GetMeanLifeTime(track, condition);
    }
};

class ScheduledPrimaryRadioactiveDecayPhysics final
    : public G4RadioactiveDecayPhysics {
public:
    ScheduledPrimaryRadioactiveDecayPhysics()
        : G4RadioactiveDecayPhysics(0) {}

    void ConstructProcess() override {
        G4EmParameters::Instance()->SetAuger(true);
        G4EmParameters::Instance()->SetDeexcitationIgnoreCut(true);
        auto* manager = G4LossTableManager::Instance();
        auto* deexcitation = manager->AtomDeexcitation();
        if (deexcitation == nullptr) {
            deexcitation = new G4UAtomicDeexcitation();
            manager->SetAtomDeexcitation(deexcitation);
            manager->ResetParameters();
        }
        auto* helper = G4PhysicsListHelper::GetPhysicsListHelper();
        helper->RegisterProcess(
            new ScheduledPrimaryRadioactiveDecay(),
            G4GenericIon::GenericIon()
        );
        helper->RegisterProcess(
            new ScheduledPrimaryRadioactiveDecay(),
            G4Triton::Triton()
        );
    }
};

class SecondaryTransportStackingAction : public G4UserStackingAction {
public:
    SecondaryTransportStackingAction(
        std::string secondary_transport_mode,
        std::string detector_scoring_mode,
        TransportDiagnostics* diagnostics,
        const RuntimeDetectorState* detector_state,
        const double detector_radius
    ) : secondary_transport_mode_(NormalizeSecondaryTransportMode(secondary_transport_mode)),
        detector_scoring_mode_(NormalizeDetectorScoringMode(detector_scoring_mode)),
        diagnostics_state_(
            diagnostics == nullptr ? nullptr : diagnostics->AcquireLocal()
        ),
        detector_state_(detector_state),
        detector_radius_(std::max(0.0, detector_radius)) {}

    G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override {
        if (secondary_transport_mode_ != "gamma_only" || track == nullptr) {
            return fUrgent;
        }
        if (track->GetParentID() <= 0) {
            return fUrgent;
        }
        if (track->GetDefinition() == G4Gamma::Definition()) {
            return fUrgent;
        }
        if (
            detector_scoring_mode_ != "incident_gamma_energy"
            && IsInsideDetectorAssembly(track->GetPosition())
        ) {
            return fUrgent;
        }
        TransportDiagnostics::AddKilledNonGammaSecondary(
            diagnostics_state_.get()
        );
        return fKill;
    }

private:
    bool IsInsideDetectorAssembly(const G4ThreeVector& position) const {
        constexpr double kTolerance = 1.0 * mm;
        if (detector_state_ == nullptr) {
            throw std::runtime_error(
                "Gamma-only transport requires runtime detector state."
            );
        }
        return (
            position - detector_state_->Center()
        ).mag() <= detector_radius_ + kTolerance;
    }

    std::string secondary_transport_mode_ = "full_transport";
    std::string detector_scoring_mode_ = "full_transport";
    TransportDiagnostics::LocalHandle diagnostics_state_;
    const RuntimeDetectorState* detector_state_ = nullptr;
    double detector_radius_ = 0.0;
};

class TransportSteppingAction : public G4UserSteppingAction {
public:
    TransportSteppingAction(
        std::set<std::string> absorbing_volume_names,
        TransportDiagnostics* diagnostics,
        ForceCollisionDiagnostics* force_collision_diagnostics,
        EventStore* event_store,
        const std::string& detector_scoring_mode,
        const std::string& secondary_transport_mode,
        const RuntimeDetectorState* detector_state,
        const double detector_radius
    ) : absorbing_volume_names_(std::move(absorbing_volume_names)),
        diagnostics_state_(
            diagnostics == nullptr ? nullptr : diagnostics->AcquireLocal()
        ),
        force_collision_diagnostics_state_(
            force_collision_diagnostics == nullptr
                ? nullptr
                : force_collision_diagnostics->AcquireLocal()
        ),
        event_state_(
            event_store == nullptr ? nullptr : event_store->AcquireLocal()
        ),
        detector_scoring_mode_(NormalizeDetectorScoringMode(detector_scoring_mode)),
        secondary_transport_mode_(NormalizeSecondaryTransportMode(secondary_transport_mode)),
        detector_state_(detector_state),
        detector_radius_(std::max(0.0, detector_radius)) {}

    void UserSteppingAction(const G4Step* step) override {
        if (step == nullptr) {
            return;
        }
        TransportDiagnostics::AddStep(diagnostics_state_.get(), step);
        MarkGammaInteraction(step);
        RecordCompletedForceCollisionBranch(step);
        if (ScoreFastDetectorEntry(step)) {
            return;
        }
        if (KillNonGammaOutsideDetector(step)) {
            return;
        }
        if (absorbing_volume_names_.empty()) {
            return;
        }
        auto* track = step->GetTrack();
        if (track == nullptr) {
            return;
        }
        if (IsAbsorbingVolume(step->GetPreStepPoint()->GetPhysicalVolume())) {
            track->SetTrackStatus(fStopAndKill);
            return;
        }
        if (IsAbsorbingVolume(step->GetPostStepPoint()->GetPhysicalVolume())) {
            track->SetTrackStatus(fStopAndKill);
        }
    }

private:
    void RecordCompletedForceCollisionBranch(const G4Step* step) const {
        auto* track = step == nullptr ? nullptr : step->GetTrack();
        auto* information = MutableTrackInformation(track);
        if (
            track == nullptr
            || information == nullptr
            || information->ActiveForceCollisionSplitId() < 0
        ) {
            return;
        }
        const auto* force_data = ForceCollisionTrackData(track);
        if (force_data == nullptr || !force_data->IsFreeFromBiasing()) {
            return;
        }
        ForceCollisionDiagnostics::RecordTerminalBranch(
            force_collision_diagnostics_state_.get(),
            information->ActiveForceCollisionSplitId(),
            information->BiasBranchLineageId(),
            track->GetWeight()
        );
        information->EndForceCollisionSplit();
    }

    void MarkGammaInteraction(const G4Step* step) const {
        auto* track = step == nullptr ? nullptr : step->GetTrack();
        if (
            track == nullptr
            || track->GetDefinition() != G4Gamma::Definition()
        ) {
            return;
        }
        auto* information = MutableTrackInformation(track);
        if (
            information != nullptr
            && information->ActiveForceCollisionSplitId() >= 0
            && information->ForceCollisionClone()
            && HasActiveForceCollisionScheme(track)
        ) {
            // Geant4's clone is deliberately transported by its
            // force-free-flight biasing operations. Their wrapped EM process
            // can appear as ProcessDefinedStep even though no physical
            // interaction occurred. Its restored boundary weight is recorded
            // when the auxiliary force-collision state becomes free.
            return;
        }
        const auto* post_point = step->GetPostStepPoint();
        const auto* process = post_point == nullptr
            ? nullptr
            : post_point->GetProcessDefinedStep();
        if (
            process == nullptr
            || process->GetProcessType() == fTransportation
        ) {
            return;
        }
        if (information != nullptr) {
            information->MarkGammaInteracted();
        }
    }

    bool IsDetectorCrystalVolume(const G4VPhysicalVolume* volume) const {
        if (volume == nullptr) {
            return false;
        }
        return volume->GetName() == "DetectorCrystalPV";
    }

    bool IsDetectorVolume(const G4VPhysicalVolume* volume) const {
        if (volume == nullptr) {
            return false;
        }
        const auto name = volume->GetName();
        return name == "DetectorCrystalPV" || name == "DetectorHousingPV";
    }

    bool ScoreFastDetectorEntry(const G4Step* step) const {
        if (detector_scoring_mode_ != "incident_gamma_energy") {
            return false;
        }
        auto* track = step->GetTrack();
        if (track == nullptr) {
            return false;
        }
        const auto* pre_point = step->GetPreStepPoint();
        const auto* post_point = step->GetPostStepPoint();
        if (pre_point == nullptr || post_point == nullptr) {
            return false;
        }
        if (IsDetectorVolume(pre_point->GetPhysicalVolume())) {
            track->SetTrackStatus(fStopAndKill);
            return true;
        }
        if (!IsDetectorVolume(post_point->GetPhysicalVolume())) {
            return false;
        }
        if (track->GetDefinition() == G4Gamma::Definition()) {
            double energy_mev = post_point->GetKineticEnergy() / MeV;
            if (!(std::isfinite(energy_mev) && energy_mev > 0.0)) {
                energy_mev = track->GetKineticEnergy() / MeV;
            }
            if (std::isfinite(energy_mev) && energy_mev > 0.0) {
                if (detector_state_ == nullptr || detector_radius_ <= 0.0) {
                    throw std::runtime_error(
                        "Incident-gamma scoring requires detector geometry."
                    );
                }
                const auto direction = track->GetMomentumDirection().unit();
                const auto relative = (
                    post_point->GetPosition() - detector_state_->Center()
                );
                const double raw_impact_parameter_fraction = (
                    relative.cross(direction).mag() / detector_radius_
                );
                if (
                    !std::isfinite(raw_impact_parameter_fraction)
                    || raw_impact_parameter_fraction < -1.0e-9
                    || raw_impact_parameter_fraction > 1.0 + 1.0e-6
                ) {
                    throw std::runtime_error(
                        "Detector-entry impact parameter is outside its "
                        "physical support."
                    );
                }
                const double impact_parameter_fraction = std::clamp(
                    raw_impact_parameter_fraction,
                    0.0,
                    1.0
                );
                EventStore::AddEnergyDeposit(
                    event_state_.get(),
                    energy_mev,
                    post_point->GetGlobalTime() / s,
                    std::max(0.0, track->GetWeight()),
                    ClassifyDetectorEntryTrack(track),
                    PrimaryBatchIndexForTrack(track),
                    PrimaryHistoryIndexForTrack(track),
                    BiasBranchLineageIdForTrack(track),
                    PrimaryEventTimeForTrack(track),
                    impact_parameter_fraction
                );
                TransportDiagnostics::AddDetectorHitStep(
                    diagnostics_state_.get()
                );
            }
        }
        track->SetTrackStatus(fStopAndKill);
        return true;
    }

    bool KillNonGammaOutsideDetector(const G4Step* step) const {
        if (secondary_transport_mode_ != "gamma_only") {
            return false;
        }
        auto* track = step->GetTrack();
        if (track == nullptr || track->GetParentID() <= 0) {
            return false;
        }
        if (track->GetDefinition() == G4Gamma::Definition()) {
            return false;
        }
        const auto* pre_point = step->GetPreStepPoint();
        if (
            detector_scoring_mode_ != "incident_gamma_energy"
            && pre_point != nullptr
            && IsDetectorVolume(pre_point->GetPhysicalVolume())
        ) {
            return false;
        }
        TransportDiagnostics::AddKilledNonGammaSecondary(
            diagnostics_state_.get()
        );
        track->SetTrackStatus(fStopAndKill);
        return true;
    }

    bool IsAbsorbingVolume(const G4VPhysicalVolume* volume) const {
        if (volume == nullptr) {
            return false;
        }
        return absorbing_volume_names_.count(volume->GetName()) > 0;
    }

    std::set<std::string> absorbing_volume_names_;
    TransportDiagnostics::LocalHandle diagnostics_state_;
    ForceCollisionDiagnostics::LocalHandle
        force_collision_diagnostics_state_;
    EventStore::LocalHandle event_state_;
    std::string detector_scoring_mode_ = "full_transport";
    std::string secondary_transport_mode_ = "full_transport";
    const RuntimeDetectorState* detector_state_ = nullptr;
    double detector_radius_ = 0.0;
};

class EventAction : public G4UserEventAction {
public:
    explicit EventAction(EventStore* store)
        : event_state_(store == nullptr ? nullptr : store->AcquireLocal()),
          coincidence_window_s_(
              store == nullptr
                  ? kDefaultDetectorCoincidenceWindowS
                  : store->CoincidenceWindowS()
          ) {}

    void BeginOfEventAction(const G4Event*) override {
        EventStore::BeginEvent(event_state_.get());
    }

    void EndOfEventAction(const G4Event*) override {
        EventStore::EndEvent(
            event_state_.get(),
            coincidence_window_s_
        );
    }

private:
    EventStore::LocalHandle event_state_;
    double coincidence_window_s_ = kDefaultDetectorCoincidenceWindowS;
};

class SidecarActionInitialization : public G4VUserActionInitialization {
public:
    SidecarActionInitialization(
        const PrimarySourceState* source_state,
        std::set<std::string> absorbing_volume_names,
        TransportDiagnostics* diagnostics,
        ForceCollisionDiagnostics* force_collision_diagnostics,
        EventStore* event_store,
        std::string detector_scoring_mode,
        std::string secondary_transport_mode,
        const RuntimeDetectorState* detector_state,
        double detector_radius
    ) : source_state_(source_state),
        absorbing_volume_names_(std::move(absorbing_volume_names)),
        diagnostics_(diagnostics),
        force_collision_diagnostics_(force_collision_diagnostics),
        event_store_(event_store),
        detector_scoring_mode_(std::move(detector_scoring_mode)),
        secondary_transport_mode_(std::move(secondary_transport_mode)),
        detector_state_(detector_state),
        detector_radius_(std::max(0.0, detector_radius)) {}

    void Build() const override {
        auto worker_context = std::make_shared<WorkerPrimaryContext>();
        SetUserAction(new PrimaryGeneratorAction(
            source_state_,
            worker_context
        ));
        SetUserAction(new EventAction(event_store_));
        SetUserAction(new TransportTrackingAction(worker_context));
        SetUserAction(new SecondaryTransportStackingAction(
            secondary_transport_mode_,
            detector_scoring_mode_,
            diagnostics_,
            detector_state_,
            detector_radius_
        ));
        SetUserAction(new TransportSteppingAction(
            absorbing_volume_names_,
            diagnostics_,
            force_collision_diagnostics_,
            event_store_,
            detector_scoring_mode_,
            secondary_transport_mode_,
            detector_state_,
            detector_radius_
        ));
    }

private:
    const PrimarySourceState* source_state_ = nullptr;
    std::set<std::string> absorbing_volume_names_;
    TransportDiagnostics* diagnostics_ = nullptr;
    ForceCollisionDiagnostics* force_collision_diagnostics_ = nullptr;
    EventStore* event_store_ = nullptr;
    std::string detector_scoring_mode_ = "full_transport";
    std::string secondary_transport_mode_ = "full_transport";
    const RuntimeDetectorState* detector_state_ = nullptr;
    double detector_radius_ = 0.0;
};

G4RunManager* CreateConfiguredRunManager(const int thread_count, bool* use_multithreaded) {
    const int requested_threads = std::max(1, thread_count);
    if (use_multithreaded != nullptr) {
        *use_multithreaded = false;
    }
#ifdef G4MULTITHREADED
    if (requested_threads > 1) {
        auto* run_manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::MTOnly);
        if (auto* mt_manager = dynamic_cast<G4MTRunManager*>(run_manager)) {
            mt_manager->SetNumberOfThreads(requested_threads);
            if (use_multithreaded != nullptr) {
                *use_multithreaded = true;
            }
            return run_manager;
        }
        delete run_manager;
        throw std::runtime_error(
            "Requested multiple Geant4 threads, but the MT run manager could not be created."
        );
    }
#else
    if (requested_threads > 1) {
        throw std::runtime_error(
            "Requested multiple Geant4 threads, but this binary has no multithreaded Geant4 support."
        );
    }
#endif
    return G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);
}

int BeamOnScheduledHistories(
    G4RunManager* run_manager,
    PrimarySourceState* source_state
) {
    if (run_manager == nullptr || source_state == nullptr) {
        throw std::runtime_error(
            "Scheduled BeamOn requires a run manager and primary state."
        );
    }
    long long remaining = source_state->TotalScheduledHistories();
    long long event_offset = 0;
    int beam_on_calls = 0;
    constexpr int kMaxBeamOnEvents = 1000000000;
    while (remaining > 0) {
        const int chunk = static_cast<int>(
            std::min<long long>(remaining, kMaxBeamOnEvents)
        );
        source_state->SetBeamOnEventOffset(event_offset);
        run_manager->BeamOn(chunk);
        remaining -= chunk;
        event_offset += chunk;
        beam_on_calls += 1;
    }
    return beam_on_calls;
}

SceneSpec ReadSceneFile(const std::string& scene_path) {
    std::ifstream input(scene_path);
    if (!input) {
        throw std::runtime_error("Failed to open scene file: " + scene_path);
    }
    SceneSpec scene;
    std::unordered_map<std::string, std::size_t> volume_index_by_path;
    std::set<std::string> parsed_shield_kinds;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto tokens = Split(line);
        if (tokens.empty()) {
            continue;
        }
        const auto fields = ParseFields(tokens);
        if (tokens[0] == "SCENE") {
            scene.scene_hash = ParseString(fields, "scene_hash");
            scene.surface_source_contract_sha256 = ParseString(
                fields,
                "surface_source_contract_sha256"
            );
            scene.nuclide_catalog_sha256 = ParseString(
                fields,
                "nuclide_catalog_sha256"
            );
            scene.usd_path = ParseString(fields, "usd_path");
            scene.room_x = ParseDouble(fields, "room_x", 10.0);
            scene.room_y = ParseDouble(fields, "room_y", 20.0);
            scene.room_z = ParseDouble(fields, "room_z", 10.0);
        } else if (tokens[0] == "DETECTOR") {
            scene.detector.crystal_radius_m = ParseDouble(
                fields,
                "crystal_radius_m",
                kDefaultCrystalRadiusM
            );
            scene.detector.crystal_length_m = ParseDouble(
                fields,
                "crystal_length_m",
                kDefaultCrystalLengthM
            );
            scene.detector.housing_thickness_m = ParseDouble(
                fields,
                "housing_thickness_m",
                kDefaultHousingThicknessM
            );
            scene.detector.coincidence_window_s = ParseDouble(
                fields,
                "coincidence_window_s",
                kDefaultDetectorCoincidenceWindowS
            );
            scene.detector.crystal_shape = ParseString(fields, "crystal_shape", "sphere");
            scene.detector.crystal_material = ParseString(fields, "crystal_material", "cebr3");
            scene.detector.housing_material = ParseString(fields, "housing_material", "aluminum");
        } else if (tokens[0] == "SHIELD") {
            const std::array<std::string, 8> required_fields = {
                "kind",
                "path",
                "shape",
                "inner_radius_m",
                "outer_radius_m",
                "thickness_m",
                "use_angle_attenuation",
                "material_name",
            };
            for (const auto& required_field : required_fields) {
                if (
                    fields.find(required_field) == fields.end()
                    || fields.at(required_field) == "-"
                ) {
                    throw std::runtime_error(
                        "SHIELD is missing required field "
                        + required_field + "."
                    );
                }
            }
            const auto shield_kind = ParseString(fields, "kind");
            if (shield_kind != "fe" && shield_kind != "pb") {
                throw std::runtime_error(
                    "SHIELD kind must be exactly fe or pb."
                );
            }
            if (!parsed_shield_kinds.insert(shield_kind).second) {
                throw std::runtime_error(
                    "Scene contains a duplicate SHIELD kind " + shield_kind + "."
                );
            }
            ShieldSpec shield = shield_kind == "pb" ? DefaultPbShieldSpec() : DefaultFeShieldSpec();
            shield.kind = shield_kind;
            shield.path = ParseString(fields, "path");
            shield.shape = ParseString(fields, "shape", "spherical_octant_shell");
            shield.inner_radius_m = ParseDouble(fields, "inner_radius_m", shield.inner_radius_m);
            shield.outer_radius_m = ParseDouble(fields, "outer_radius_m", shield.outer_radius_m);
            shield.thickness_m = ParseDouble(fields, "thickness_m", shield.thickness_m);
            shield.sx = ParseDouble(fields, "sx", 0.25);
            shield.sy = ParseDouble(fields, "sy", 0.08);
            shield.sz = ParseDouble(fields, "sz", 0.25);
            shield.material.name = ParseString(fields, "material_name");
            shield.material.density_g_cm3 = ParseDouble(fields, "density_g_cm3", -1.0);
            shield.material.preset_name = ParseString(fields, "preset_name");
            auto validated_shield = ValidateParsedShield(
                std::move(shield),
                ParseString(fields, "use_angle_attenuation")
            );
            if (shield_kind == "fe") {
                scene.fe_shield = std::move(validated_shield);
            } else if (shield_kind == "pb") {
                scene.pb_shield = std::move(validated_shield);
            }
        } else if (tokens[0] == "NUCLIDE") {
            NuclideSpec nuclide;
            nuclide.isotope = ParseString(fields, "isotope");
            nuclide.atomic_number = static_cast<int>(
                ParseLong(fields, "atomic_number", 0)
            );
            nuclide.mass_number = static_cast<int>(
                ParseLong(fields, "mass_number", 0)
            );
            nuclide.geant4_excitation_keV = ParseDouble(
                fields,
                "geant4_excitation_keV",
                0.0
            );
            nuclide.half_life_s = ParseDouble(fields, "half_life_s", 0.0);
            nuclide.prompt_cascade_model = ParseString(
                fields,
                "prompt_cascade_model"
            );
            const bool monoenergetic_probe = (
                nuclide.prompt_cascade_model
                    == "independent_monoenergetic_probe"
                && nuclide.atomic_number == 0
                && nuclide.mass_number == 0
                && nuclide.geant4_excitation_keV == 0.0
                && nuclide.half_life_s == 0.0
            );
            const bool evaluated_nuclide = (
                nuclide.atomic_number > 0
                && nuclide.mass_number > 0
                && std::isfinite(nuclide.geant4_excitation_keV)
                && nuclide.geant4_excitation_keV >= 0.0
                && std::isfinite(nuclide.half_life_s)
                && nuclide.half_life_s > 0.0
                && nuclide.prompt_cascade_model
                    == "geant4_radioactive_decay"
            );
            if (
                nuclide.isotope.empty()
                || (!evaluated_nuclide && !monoenergetic_probe)
            ) {
                throw std::runtime_error(
                    "NUCLIDE contains invalid evaluated decay metadata."
                );
            }
            if (!scene.nuclides.emplace(nuclide.isotope, nuclide).second) {
                throw std::runtime_error(
                    "Scene contains duplicate NUCLIDE " + nuclide.isotope + "."
                );
            }
        } else if (tokens[0] == "GAMMA") {
            const auto isotope = ParseString(fields, "isotope");
            const auto nuclide = scene.nuclides.find(isotope);
            if (nuclide == scene.nuclides.end()) {
                throw std::runtime_error(
                    "GAMMA must follow its parent NUCLIDE entry."
                );
            }
            const auto line_index = ParseLong(fields, "line_index", -1);
            if (
                line_index < 0
                || static_cast<std::size_t>(line_index)
                    != nuclide->second.gamma_lines.size()
            ) {
                throw std::runtime_error(
                    "GAMMA line_index must be contiguous and zero-based."
                );
            }
            const double energy_keV = ParseDouble(fields, "energy_keV", -1.0);
            const double photons_per_decay = ParseDouble(
                fields,
                "photons_per_decay",
                -1.0
            );
            if (
                !std::isfinite(energy_keV)
                || energy_keV <= 0.0
                || !std::isfinite(photons_per_decay)
                || photons_per_decay <= 0.0
                || photons_per_decay > 1.0
            ) {
                throw std::runtime_error(
                    "GAMMA requires a positive energy and probability in (0,1]."
                );
            }
            nuclide->second.gamma_lines.push_back({
                energy_keV,
                photons_per_decay,
            });
        } else if (tokens[0] == "TRANSPORT_GAMMA") {
            const auto isotope = ParseString(fields, "isotope");
            const auto nuclide = scene.nuclides.find(isotope);
            if (nuclide == scene.nuclides.end()) {
                throw std::runtime_error(
                    "TRANSPORT_GAMMA must follow its parent NUCLIDE entry."
                );
            }
            const auto line_index = ParseLong(fields, "line_index", -1);
            if (
                line_index < 0
                || static_cast<std::size_t>(line_index)
                    != nuclide->second.transport_gamma_lines.size()
            ) {
                throw std::runtime_error(
                    "TRANSPORT_GAMMA line_index must be contiguous and "
                    "zero-based."
                );
            }
            const double energy_keV = ParseDouble(
                fields,
                "energy_keV",
                -1.0
            );
            const double relative_weight = ParseDouble(
                fields,
                "relative_weight",
                -1.0
            );
            if (
                !std::isfinite(energy_keV)
                || energy_keV <= 0.0
                || !std::isfinite(relative_weight)
                || relative_weight <= 0.0
            ) {
                throw std::runtime_error(
                    "TRANSPORT_GAMMA requires positive finite energy and "
                    "relative weight."
                );
            }
            nuclide->second.transport_gamma_lines.push_back({
                energy_keV,
                relative_weight,
            });
        } else if (tokens[0] == "SOURCE") {
            SourceSpec source;
            source.isotope = ParseString(fields, "isotope");
            source.x = ParseDouble(fields, "x");
            source.y = ParseDouble(fields, "y");
            source.z = ParseDouble(fields, "z");
            source.intensity_cps_1m = ParseDouble(
                fields,
                "intensity_cps_1m",
                std::numeric_limits<double>::quiet_NaN()
            );
            source.activity_bq = ParseDouble(
                fields,
                "activity_bq",
                std::numeric_limits<double>::quiet_NaN()
            );
            const bool has_detector_cps = (
                std::isfinite(source.intensity_cps_1m)
                && source.intensity_cps_1m > 0.0
            );
            const bool has_activity = (
                std::isfinite(source.activity_bq)
                && source.activity_bq > 0.0
            );
            if (has_detector_cps == has_activity) {
                throw std::runtime_error(
                    "SOURCE requires exactly one positive intensity_cps_1m "
                    "or activity_bq."
                );
            }
            source.anchor_x = ParseDouble(fields, "anchor_x", source.x);
            source.anchor_y = ParseDouble(fields, "anchor_y", source.y);
            source.anchor_z = ParseDouble(fields, "anchor_z", source.z);
            source.surface_chart_id = ParseLong(fields, "surface_chart_id", -1);
            source.surface_u = ParseDouble(fields, "surface_u", -1.0);
            source.surface_v = ParseDouble(fields, "surface_v", -1.0);
            source.surface_normal_x = ParseDouble(fields, "surface_normal_x", 0.0);
            source.surface_normal_y = ParseDouble(fields, "surface_normal_y", 0.0);
            source.surface_normal_z = ParseDouble(fields, "surface_normal_z", 0.0);
            source.surface_emission_epsilon_m = ParseDouble(
                fields,
                "surface_emission_epsilon_m",
                0.0
            );
            source.surface_emission_policy_sha256 = ParseString(
                fields,
                "surface_emission_policy_sha256"
            );
            scene.sources.push_back(source);
        } else if (tokens[0] == "VOLUME") {
            VolumeSpec volume;
            volume.path = ParseString(fields, "path");
            volume.shape = ParseString(fields, "shape");
            volume.tx = ParseDouble(fields, "tx");
            volume.ty = ParseDouble(fields, "ty");
            volume.tz = ParseDouble(fields, "tz");
            volume.qw = ParseDouble(fields, "qw", 1.0);
            volume.qx = ParseDouble(fields, "qx", 0.0);
            volume.qy = ParseDouble(fields, "qy", 0.0);
            volume.qz = ParseDouble(fields, "qz", 0.0);
            volume.sx = ParseDouble(fields, "sx", -1.0);
            volume.sy = ParseDouble(fields, "sy", -1.0);
            volume.sz = ParseDouble(fields, "sz", -1.0);
            volume.radius_m = ParseDouble(fields, "radius_m", -1.0);
            volume.material.name = ParseString(fields, "material_name");
            volume.material.density_g_cm3 = ParseDouble(fields, "density_g_cm3", -1.0);
            volume.material.preset_name = ParseString(fields, "preset_name");
            volume.transport_group = ToLower(ParseString(fields, "transport_group"));
            volume.transport_mode = ToLower(ParseString(fields, "transport_mode", "geant4"));
            volume_index_by_path[volume.path] = scene.volumes.size();
            scene.volumes.push_back(volume);
        } else if (tokens[0] == "COMP") {
            const auto path = ParseString(fields, "path");
            const auto element = ParseString(fields, "element");
            const auto fraction = ParseDouble(fields, "fraction");
            const auto it = volume_index_by_path.find(path);
            if (it != volume_index_by_path.end()) {
                scene.volumes[it->second].material.composition_by_mass[element] = fraction;
            } else if (
                scene.fe_shield.has_value()
                && scene.fe_shield->path == path
            ) {
                scene.fe_shield->material.composition_by_mass[element] = fraction;
            } else if (
                scene.pb_shield.has_value()
                && scene.pb_shield->path == path
            ) {
                scene.pb_shield->material.composition_by_mass[element] = fraction;
            } else {
                throw std::runtime_error(
                    "COMP references an unknown or disabled material path."
                );
            }
        } else if (tokens[0] == "TRI") {
            const auto path = ParseString(fields, "path");
            const auto it = volume_index_by_path.find(path);
            if (it == volume_index_by_path.end()) {
                continue;
            }
            scene.volumes[it->second].triangles.push_back({
                ParseDouble(fields, "ax"), ParseDouble(fields, "ay"), ParseDouble(fields, "az"),
                ParseDouble(fields, "bx"), ParseDouble(fields, "by"), ParseDouble(fields, "bz"),
                ParseDouble(fields, "cx"), ParseDouble(fields, "cy"), ParseDouble(fields, "cz")
            });
        }
    }
    if (!scene.nuclide_catalog_sha256.empty()) {
        for (const auto& source : scene.sources) {
            const auto nuclide = scene.nuclides.find(source.isotope);
            if (
                nuclide == scene.nuclides.end()
                || nuclide->second.gamma_lines.empty()
                || nuclide->second.transport_gamma_lines.empty()
            ) {
                throw std::runtime_error(
                    "Authenticated scene source lacks evaluated NUCLIDE/GAMMA "
                    "metadata: " + source.isotope
                );
            }
        }
    }
    return scene;
}

RequestSpec ReadRequestFile(const std::string& request_path) {
    std::ifstream input(request_path);
    if (!input) {
        throw std::runtime_error("Failed to open request file: " + request_path);
    }
    RequestSpec request;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto tokens = Split(line);
        if (tokens.empty()) {
            continue;
        }
        const auto fields = ParseFields(tokens);
        if (tokens[0] == "STEP") {
            RequireExactFields(
                fields,
                {
                    "step_id",
                    "dwell_time_s",
                    "seed",
                    "shield_pose_contract_id",
                    "shield_pose_contract_sha256",
                    "native_action_contract_id",
                    "native_action_sha256",
                    "fe_orientation_index",
                    "pb_orientation_index",
                },
                "STEP"
            );
            if (request.has_step) {
                throw std::runtime_error("Request contains duplicate STEP records.");
            }
            request.has_step = true;
            request.step_id = static_cast<int>(ParseLong(fields, "step_id", 0));
            request.dwell_time_s = ParseDouble(fields, "dwell_time_s", 1.0);
            request.seed = ParseLong(fields, "seed", 123);
            request.shield_pose_contract_id = ParseString(
                fields,
                "shield_pose_contract_id"
            );
            request.shield_pose_contract_sha256 = ParseString(
                fields,
                "shield_pose_contract_sha256"
            );
            request.native_action_contract_id = ParseString(
                fields,
                "native_action_contract_id"
            );
            request.native_action_sha256 = ParseString(
                fields,
                "native_action_sha256"
            );
            request.fe_orientation_index = static_cast<int>(ParseLong(
                fields,
                "fe_orientation_index",
                -1
            ));
            request.pb_orientation_index = static_cast<int>(ParseLong(
                fields,
                "pb_orientation_index",
                -1
            ));
        } else if (tokens[0] == "POSE") {
            RequireExactFields(
                fields,
                {"kind", "x", "y", "z", "qw", "qx", "qy", "qz"},
                "POSE"
            );
            PoseSpec pose;
            pose.x = ParseDouble(fields, "x");
            pose.y = ParseDouble(fields, "y");
            pose.z = ParseDouble(fields, "z");
            pose.qw = ParseDouble(fields, "qw", 1.0);
            pose.qx = ParseDouble(fields, "qx", 0.0);
            pose.qy = ParseDouble(fields, "qy", 0.0);
            pose.qz = ParseDouble(fields, "qz", 0.0);
            const auto kind = ParseString(fields, "kind");
            if (kind == "detector") {
                if (request.has_detector_pose) {
                    throw std::runtime_error(
                        "Request contains duplicate detector POSE records."
                    );
                }
                request.has_detector_pose = true;
                request.detector_pose = pose;
            } else if (kind == "fe") {
                if (request.has_fe_pose) {
                    throw std::runtime_error(
                        "Request contains duplicate Fe POSE records."
                    );
                }
                request.has_fe_pose = true;
                request.fe_pose = pose;
            } else if (kind == "pb") {
                if (request.has_pb_pose) {
                    throw std::runtime_error(
                        "Request contains duplicate Pb POSE records."
                    );
                }
                request.has_pb_pose = true;
                request.pb_pose = pose;
            } else {
                throw std::runtime_error("Request contains an unknown POSE kind.");
            }
        } else {
            throw std::runtime_error("Request contains an unknown record type.");
        }
    }
    ValidateNativeActionIdentityContract(request);
    ValidateShieldPoseContract(request);
    return request;
}

std::string GeometryCacheKey(
    const SceneSpec& scene,
    const std::string& physics_profile,
    const int thread_count,
    const std::string& detector_scoring_mode,
    const std::string& secondary_transport_mode,
    const bool mean_calibration_forced_collision,
    const std::string& primary_emission_model,
    const bool sample_detector_response,
    const std::string& detector_green_operator_binary_sha256,
    const std::string& detector_green_operator_contract_sha256
) {
    std::ostringstream stream;
    stream << std::setprecision(17)
           << scene.scene_hash << "|"
           << physics_profile << "|"
           << std::max(1, thread_count) << "|"
           << NormalizeDetectorScoringMode(detector_scoring_mode) << "|"
           << NormalizeSecondaryTransportMode(secondary_transport_mode) << "|"
           << NormalizePrimaryEmissionModel(primary_emission_model) << "|"
           << (sample_detector_response ? "green_operator" : "raw_detector")
           << "|"
           << detector_green_operator_binary_sha256 << "|"
           << detector_green_operator_contract_sha256 << "|"
           << (
                mean_calibration_forced_collision
                    ? "forced_first_collision"
                    : "analog_transport"
              );
    return stream.str();
}

class TransportSession {
public:
    TransportSession(
        SceneSpec scene,
        RequestSpec geometry_request,
        std::string physics_profile,
        const int thread_count,
        std::string detector_scoring_mode,
        std::string secondary_transport_mode,
        const bool mean_calibration_forced_collision,
        std::string primary_emission_model,
        const bool sample_detector_response,
        std::string detector_green_operator_path,
        std::string detector_green_operator_binary_sha256,
        std::string detector_green_operator_contract_sha256
    ) : scene_(std::move(scene)),
        geometry_request_(geometry_request),
        physics_profile_(std::move(physics_profile)),
        detector_scoring_mode_(NormalizeDetectorScoringMode(detector_scoring_mode)),
        secondary_transport_mode_(NormalizeSecondaryTransportMode(secondary_transport_mode)),
        primary_emission_model_(
            NormalizePrimaryEmissionModel(primary_emission_model)
        ),
        thread_count_(std::max(1, thread_count)),
        mean_calibration_forced_collision_(
            mean_calibration_forced_collision
        ),
        sample_detector_response_(sample_detector_response),
        detector_green_operator_path_(
            std::move(detector_green_operator_path)
        ),
        detector_green_operator_binary_sha256_(
            std::move(detector_green_operator_binary_sha256)
        ),
        detector_green_operator_contract_sha256_(
            std::move(detector_green_operator_contract_sha256)
        ),
        use_theory_tvl_(UseTheoryTvlProfile(physics_profile_)),
        event_store_(scene_.detector.coincidence_window_s) {
        const bool green_contract_complete = (
            !detector_green_operator_path_.empty()
            && IsLowercaseSha256(
                detector_green_operator_binary_sha256_
            )
            && IsLowercaseSha256(
                detector_green_operator_contract_sha256_
            )
        );
        const bool green_contract_empty = (
            detector_green_operator_path_.empty()
            && detector_green_operator_binary_sha256_.empty()
            && detector_green_operator_contract_sha256_.empty()
        );
        if (
            (sample_detector_response_ && !green_contract_complete)
            || (!sample_detector_response_ && !green_contract_empty)
        ) {
            throw std::runtime_error(
                "Detector Green path and hashes are required exactly when "
                "detector-response sampling is enabled."
            );
        }
        if (sample_detector_response_) {
            detector_green_operator_ = std::make_unique<
                DetectorGreenOperator
            >(detector_green_operator_path_);
        }
        for (const auto& volume : scene_.volumes) {
            if (ToLower(volume.transport_mode) != "absorber") {
                continue;
            }
            absorbing_volume_names_.insert(volume.path);
            if (!volume.transport_group.empty()) {
                absorbing_transport_groups_.insert(volume.transport_group);
            }
        }
        run_manager_ = CreateConfiguredRunManager(thread_count_, &run_manager_multithreaded_);
        auto detector = std::make_unique<Geant4SceneConstruction>(
            &scene_,
            &geometry_request_,
            &event_store_,
            &diagnostics_,
            mean_calibration_forced_collision_
                ? &force_collision_diagnostics_
                : nullptr,
            detector_scoring_mode_,
            !use_theory_tvl_,
            mean_calibration_forced_collision_
        );
        detector_construction_ = detector.get();
        run_manager_->SetUserInitialization(detector.release());
        G4PhysListFactory factory;
        auto* physics_list = factory.GetReferencePhysList(
            kReferencePhysicsListName
        );
        if (physics_list == nullptr) {
            throw std::runtime_error(
                "Geant4 did not provide the required FTFP_BERT physics list."
            );
        }
        physics_list->ReplacePhysics(new G4EmStandardPhysics_option4());
        physics_list->SetDefaultCutValue(kProductionCutRangeMm * mm);
        if (primary_emission_model_ == "geant4_radioactive_decay") {
            // Parent-decay events have already been sampled from the source
            // activity for the requested dwell interval.  Geant4 11.2+
            // otherwise discards sampled decay times beyond its one-year
            // default, biasing long-lived nuclides toward no emission.
            G4HadronicParameters::Instance()
                ->SetTimeThresholdForRadioactiveDecay(
                    std::numeric_limits<G4double>::max()
                );
            physics_list->RegisterPhysics(
                new ScheduledPrimaryRadioactiveDecayPhysics()
            );
        }
        if (mean_calibration_forced_collision_) {
            auto* biasing_physics = new G4GenericBiasingPhysics();
            biasing_physics->Bias("gamma");
            physics_list->RegisterPhysics(biasing_physics);
        }
        run_manager_->SetUserInitialization(physics_list);
        detector_runtime_state_.Update(geometry_request_.detector_pose);
        const double detector_radius = DetectorTargetRadiusM(scene_.detector) * m;
        auto action_initialization = std::make_unique<SidecarActionInitialization>(
            &primary_state_,
            absorbing_volume_names_,
            &diagnostics_,
            mean_calibration_forced_collision_
                ? &force_collision_diagnostics_
                : nullptr,
            &event_store_,
            detector_scoring_mode_,
            secondary_transport_mode_,
            &detector_runtime_state_,
            detector_radius
        );
        run_manager_->SetUserInitialization(action_initialization.release());
        run_manager_->Initialize();
        auto* gamma_process_manager = (
            G4Gamma::GammaDefinition()->GetProcessManager()
        );
        if (gamma_process_manager == nullptr) {
            throw std::runtime_error(
                "Initialized Geant4 gamma has no process manager."
            );
        }
        const auto* gamma_processes = gamma_process_manager->GetProcessList();
        if (gamma_processes == nullptr) {
            throw std::runtime_error(
                "Initialized Geant4 gamma has no process list."
            );
        }
        for (
            G4int index = 0;
            index < gamma_process_manager->GetProcessListLength();
            ++index
        ) {
            auto* process = (*gamma_processes)[index];
            if (process != nullptr) {
                gamma_process_names_.insert(process->GetProcessName());
            }
            auto* general = dynamic_cast<G4GammaGeneralProcess*>(process);
            if (general == nullptr) {
                continue;
            }
            for (const auto* name : {"compt", "Rayl", "phot", "conv"}) {
                const auto* sub_process = general->GetEmProcess(name);
                if (sub_process != nullptr) {
                    gamma_em_subprocess_names_.insert(
                        sub_process->GetProcessName()
                    );
                }
            }
        }
        for (const auto* required : {"compt", "Rayl", "phot"}) {
            if (
                gamma_process_names_.count(required) == 0
                && gamma_em_subprocess_names_.count(required) == 0
            ) {
                throw std::runtime_error(
                    std::string("Required option4 gamma process is absent: ")
                    + required
                );
            }
        }
    }

    ~TransportSession() {
        delete run_manager_;
    }

    TransportSession(const TransportSession&) = delete;
    TransportSession& operator=(const TransportSession&) = delete;

    SimulationResult Run(
        const RequestSpec& request,
        const double dead_time_tau_s,
        const TransportOptions& options,
        const bool geometry_cache_hit,
        const bool persistent_process
    ) {
        if (
            options.mean_calibration_forced_collision
                != mean_calibration_forced_collision_
        ) {
            throw std::runtime_error(
                "Transport session force-collision mode disagrees with the "
                "request options."
            );
        }
        if (
            options.sample_detector_response
                != sample_detector_response_
            || options.detector_green_operator_path
                != detector_green_operator_path_
            || options.detector_green_operator_binary_sha256
                != detector_green_operator_binary_sha256_
            || options.detector_green_operator_contract_sha256
                != detector_green_operator_contract_sha256_
            || (
                options.sample_detector_response
                != (detector_green_operator_ != nullptr)
            )
        ) {
            throw std::runtime_error(
                "Transport session detector Green contract changed after "
                "initialization."
            );
        }
        CLHEP::HepRandom::setTheSeed(request.seed);
        bool movable_geometry_updated = false;
        if (detector_construction_ != nullptr) {
            movable_geometry_updated = (
                detector_construction_->UpdateMovablePoses(request)
            );
        }
        geometry_request_ = request;
        detector_runtime_state_.Update(request.detector_pose);
        event_store_.ClearDeposits();
        diagnostics_.Clear();
        if (mean_calibration_forced_collision_) {
            force_collision_diagnostics_.Clear();
        }
        std::mt19937_64 rng(static_cast<std::uint64_t>(request.seed));
        std::vector<EnergyDeposit> energy_deposits;
        const auto start_time = std::chrono::steady_clock::now();
        long total_primaries = 0;
        double expected_unthinned_primaries = 0.0;
        double expected_sampled_primaries = 0.0;
        std::map<std::string, double> source_equivalent_counts;
        std::map<std::string, double> transport_detected_counts;
        std::map<std::string, double> transport_uncollided_primary_counts;
        std::map<std::string, double> transport_interacted_primary_counts;
        std::map<std::string, double> transport_secondary_counts;
        const double reference_acceptance = DetectorReferenceAcceptance(scene_.detector);
        const std::string source_rate_model = NormalizeSourceRateModel(options.source_rate_model);
        const bool detector_cps_rate_model = source_rate_model == "detector_cps_1m";
        const bool parent_activity_rate_model = (
            source_rate_model == "parent_decay_activity_bq"
        );
        const std::string primary_emission_model = (
            NormalizePrimaryEmissionModel(options.primary_emission_model)
        );
        const bool radioactive_decay_emission = (
            primary_emission_model == "geant4_radioactive_decay"
        );
        if (primary_emission_model != primary_emission_model_) {
            throw std::runtime_error(
                "Primary emission model changed after Geant4 physics-list "
                "construction."
            );
        }
        const bool source_bias_weighted_transport = (
            !detector_cps_rate_model && UsesSourceBias(options)
        );
        const std::string source_bias_mode = NormalizeSourceBiasMode(options.source_bias_mode);
        const std::string effective_source_bias_mode = detector_cps_rate_model
            ? "detector_cone"
            : source_bias_mode;
        const bool cone_sampled_transport = (
            source_bias_weighted_transport || effective_source_bias_mode == "detector_cone"
        );
        const double requested_primary_sampling_fraction = std::clamp(
            options.primary_sampling_fraction,
            1.0e-6,
            1.0
        );
        const bool primary_sampling_budget_enabled = options.target_sampled_primaries > 0;
        const bool mean_calibration_enabled = (
            options.mean_calibration_histories_per_source_line > 0
        );
        if (radioactive_decay_emission) {
            if (
                !parent_activity_rate_model
                || source_bias_mode != "analog"
                || detector_scoring_mode_ != "full_transport"
                || secondary_transport_mode_ != "full_transport"
                || requested_primary_sampling_fraction != 1.0
                || primary_sampling_budget_enabled
                || mean_calibration_enabled
                || options.mean_calibration_forced_collision
                || options.sample_detector_response
                || use_theory_tvl_
            ) {
                throw std::runtime_error(
                    "Geant4 radioactive-decay emission requires parent-decay "
                    "activity in Bq, analog source sampling, full "
                    "detector and secondary transport, unit-weight histories, "
                    "and no response sampling, theory attenuation, or mean "
                    "calibration."
                );
            }
            for (const auto& source : scene_.sources) {
                const auto nuclide = scene_.nuclides.find(source.isotope);
                if (nuclide == scene_.nuclides.end()) {
                    throw std::runtime_error(
                        "Radioactive-decay emission requires evaluated NUCLIDE "
                        "metadata for every source."
                    );
                }
                if (
                    !std::isfinite(source.activity_bq)
                    || source.activity_bq <= 0.0
                    || std::isfinite(source.intensity_cps_1m)
                    || nuclide->second.atomic_number <= 0
                    || nuclide->second.mass_number <= 0
                    || nuclide->second.prompt_cascade_model
                        != "geant4_radioactive_decay"
                ) {
                    throw std::runtime_error(
                        "Radioactive-decay sources require only activity_bq."
                    );
                }
            }
        } else {
            for (const auto& source : scene_.sources) {
                if (
                    !std::isfinite(source.intensity_cps_1m)
                    || source.intensity_cps_1m <= 0.0
                    || std::isfinite(source.activity_bq)
                ) {
                    throw std::runtime_error(
                        "Independent-gamma sources require only "
                        "intensity_cps_1m."
                    );
                }
            }
        }
        if (options.decay_comparison_diagnostic) {
            const double diagnostic_max = (
                options.decay_comparison_energy_max_keV
            );
            const double diagnostic_bin_count = diagnostic_max / 2.0;
            if (
                !std::isfinite(diagnostic_max)
                || diagnostic_max < 1700.0
                || diagnostic_max > 10000.0
                || std::abs(
                    diagnostic_bin_count
                    - std::round(diagnostic_bin_count)
                ) > 1.0e-12
                || options.background_cps != 0.0
                || dead_time_tau_s != 0.0
                || detector_scoring_mode_ != "full_transport"
                || secondary_transport_mode_ != "full_transport"
                || requested_primary_sampling_fraction != 1.0
                || primary_sampling_budget_enabled
                || mean_calibration_enabled
                || options.mean_calibration_forced_collision
                || options.sample_detector_response
                || use_theory_tvl_
            ) {
                throw std::runtime_error(
                    "Decay-cascade comparison requires a 2-keV-aligned "
                    "1700--10000 keV diagnostic range, zero background and "
                    "dead time, full analog detector/secondary transport, "
                    "unit histories, and no runtime shortcuts."
                );
            }
        }
        const long long angle_stratum_count_ll = (
            static_cast<long long>(
                options.mean_calibration_angle_strata_mu
            )
            * static_cast<long long>(
                options.mean_calibration_angle_strata_phi
            )
        );
        if (
            options.mean_calibration_angle_strata_mu <= 0
            || options.mean_calibration_angle_strata_phi <= 0
            || angle_stratum_count_ll <= 0
            || angle_stratum_count_ll
                > static_cast<long long>(std::numeric_limits<int>::max())
        ) {
            throw std::runtime_error(
                "Mean-calibration angle stratum counts are invalid."
            );
        }
        const int angle_stratum_count = static_cast<int>(
            angle_stratum_count_ll
        );
        if (primary_sampling_budget_enabled && !detector_cps_rate_model) {
            throw std::runtime_error(
                "target_sampled_primaries requires source_rate_model=detector_cps_1m"
            );
        }
        if (mean_calibration_enabled) {
            if (
                !detector_cps_rate_model
                || (
                    detector_scoring_mode_ != "incident_gamma_energy"
                    && detector_scoring_mode_ != "full_transport"
                )
                || secondary_transport_mode_ != "full_transport"
                || options.sample_detector_response
                || !options.validation_entry_class_spectra
                || options.background_cps != 0.0
                || dead_time_tau_s != 0.0
                || requested_primary_sampling_fraction != 1.0
                || primary_sampling_budget_enabled
                || source_bias_weighted_transport
                || use_theory_tvl_
                || (
                    detector_scoring_mode_ == "full_transport"
                    && options.mean_calibration_forced_collision
                )
            ) {
                throw std::runtime_error(
                    "Fixed source-line sampling requires detector_cps_1m, "
                    "implemented detector scoring, full secondary transport, "
                    "raw detector entries, zero background/dead-time, unit "
                    "primary sampling, and non-theory Geant4 transport; "
                    "full-detector scoring also forbids forced collision."
                );
            }
            if (
                options.mean_calibration_histories_per_source_line
                    % angle_stratum_count
                != 0
            ) {
                throw std::runtime_error(
                    "Mean-calibration histories per source-line must be "
                    "divisible by the mu/phi stratum count."
                );
            }
            if (
                options.mean_calibration_histories_per_source_line
                    / angle_stratum_count
                < 2
            ) {
                throw std::runtime_error(
                    "Mean calibration requires at least two histories per "
                    "mu/phi stratum for covariance estimation."
                );
            }
        } else if (
            options.mean_calibration_angle_strata_mu != 1
            || options.mean_calibration_angle_strata_phi != 1
            || options.mean_calibration_forced_collision
        ) {
            throw std::runtime_error(
                "Mean-calibration strata/forced collision require a fixed "
                "source-line history quota."
            );
        }
        std::map<std::string, double>
            detector_green_reference_efficiency_by_isotope;
        if (detector_cps_rate_model && sample_detector_response_) {
            if (detector_green_operator_ == nullptr) {
                throw std::runtime_error(
                    "Detector-cps Green normalization requires its operator."
                );
            }
            const double target_radius_m = DetectorTargetRadiusM(
                scene_.detector
            );
            for (const auto& source : scene_.sources) {
                if (
                    detector_green_reference_efficiency_by_isotope.count(
                        source.isotope
                    ) != 0U
                ) {
                    continue;
                }
                const auto lines = GammaLinesForIsotope(
                    scene_,
                    source.isotope
                );
                double total_line_intensity = 0.0;
                double weighted_efficiency = 0.0;
                for (const auto& line : lines) {
                    total_line_intensity += std::max(0.0, line.intensity);
                    weighted_efficiency += std::max(0.0, line.intensity)
                        * detector_green_operator_
                            ->ReferencePulseDetectionProbability(
                                line.energy_keV,
                                target_radius_m
                            );
                }
                if (
                    !std::isfinite(total_line_intensity)
                    || total_line_intensity <= 0.0
                ) {
                    throw std::runtime_error(
                        "Detector Green source-rate normalization requires "
                        "positive catalog line intensity."
                    );
                }
                const double reference_efficiency = (
                    weighted_efficiency / total_line_intensity
                );
                if (
                    !std::isfinite(reference_efficiency)
                    || reference_efficiency <= 0.0
                    || reference_efficiency > 1.0 + 1.0e-12
                ) {
                    throw std::runtime_error(
                        "Detector Green catalog-weighted reference efficiency "
                        "is invalid."
                    );
                }
                detector_green_reference_efficiency_by_isotope[
                    source.isotope
                ] = std::min(reference_efficiency, 1.0);
            }
        }
        for (const auto& source : scene_.sources) {
            if (radioactive_decay_emission) {
                const double mean_events = source.activity_bq
                    * request.dwell_time_s;
                if (mean_events > 0.0) {
                    expected_unthinned_primaries += mean_events;
                }
                continue;
            }
            const double point_geom_scale = InverseSquareScale(
                source.x,
                source.y,
                source.z,
                request.detector_pose.x,
                request.detector_pose.y,
                request.detector_pose.z
            );
            const double detector_cps_geom_scale = DetectorCpsGeometryScale(
                source.x,
                source.y,
                source.z,
                request.detector_pose.x,
                request.detector_pose.y,
                request.detector_pose.z,
                scene_.detector
            );
            const double geom_scale = detector_cps_rate_model
                ? detector_cps_geom_scale
                : point_geom_scale;
            const auto lines = GammaLinesForIsotope(scene_, source.isotope);
            const double shield_transmission = use_theory_tvl_
                ? TheoryTvlTransmission(source, scene_, request)
                : 1.0;
            double total_line_intensity = 0.0;
            for (const auto& line : lines) {
                total_line_intensity += std::max(0.0, line.intensity);
            }
            for (const auto& line : lines) {
                const double source_rate_scale = detector_cps_rate_model
                    ? geom_scale
                    : 1.0 / reference_acceptance;
                const double line_weight = (
                    detector_cps_rate_model && total_line_intensity > 0.0
                )
                    ? line.intensity / total_line_intensity
                    : line.intensity;
                const double detector_green_reference_efficiency = (
                    detector_cps_rate_model && sample_detector_response_
                )
                    ? detector_green_reference_efficiency_by_isotope.at(
                        source.isotope
                    )
                    : 1.0;
                const double mean_events = source.intensity_cps_1m
                    * request.dwell_time_s
                    * shield_transmission
                    * line_weight
                    * source_rate_scale
                    / detector_green_reference_efficiency;
                if (mean_events > 0.0) {
                    expected_unthinned_primaries += mean_events;
                }
            }
        }
        if (
            !std::isfinite(expected_unthinned_primaries)
            || expected_unthinned_primaries < 0.0
        ) {
            throw std::runtime_error(
                "Expected detector-equivalent primary count is invalid"
            );
        }
        double primary_sampling_fraction = requested_primary_sampling_fraction;
        std::string primary_sampling_fraction_resolution = "fixed_fraction";
        if (primary_sampling_budget_enabled) {
            const double budget_fraction = expected_unthinned_primaries > 0.0
                ? static_cast<double>(options.target_sampled_primaries)
                    / expected_unthinned_primaries
                : std::numeric_limits<double>::infinity();
            if (budget_fraction < requested_primary_sampling_fraction) {
                primary_sampling_fraction = std::clamp(budget_fraction, 1.0e-6, 1.0);
                primary_sampling_fraction_resolution = "target_budget_limited";
            } else {
                primary_sampling_fraction_resolution = "maximum_fraction_limited";
            }
        }
        if (mean_calibration_enabled) {
            primary_sampling_fraction = 1.0;
            primary_sampling_fraction_resolution =
                "fixed_source_line_mean_calibration";
        }
        const double primary_history_weight = 1.0 / primary_sampling_fraction;
        const bool history_thinning_enabled = primary_sampling_fraction < 1.0;
        const bool transport_tally_weighted = (
            source_bias_weighted_transport
            || history_thinning_enabled
            || mean_calibration_enabled
        );
        double effective_cone_min_deg = std::numeric_limits<double>::infinity();
        double effective_cone_max_deg = 0.0;
        const G4ThreeVector detector_center(
            request.detector_pose.x * m,
            request.detector_pose.y * m,
            request.detector_pose.z * m
        );
        std::map<std::string, double> source_equivalent_counts_by_source;
        std::map<std::string, double> scheduled_incident_gamma_counts_by_line;
        std::map<std::string, double> transport_detected_counts_by_source;
        std::map<std::string, double> transport_uncollided_primary_counts_by_source;
        std::map<std::string, double> transport_interacted_primary_counts_by_source;
        std::map<std::string, double> transport_secondary_counts_by_source;
        std::map<std::string, double> transport_detected_counts_by_line;
        std::map<std::string, double> transport_uncollided_primary_counts_by_line;
        std::map<std::string, double> transport_interacted_primary_counts_by_line;
        std::map<std::string, double> transport_secondary_counts_by_line;
        double scheduled_unthinned_primaries = 0.0;
        double mean_calibration_history_weight_min =
            std::numeric_limits<double>::infinity();
        double mean_calibration_history_weight_max = 0.0;
        std::vector<PrimaryHistoryBatch> primary_schedule;
        for (std::size_t source_index = 0; source_index < scene_.sources.size(); ++source_index) {
            const auto& source = scene_.sources[source_index];
            const std::string source_token = "src" + std::to_string(source_index)
                + "_" + SanitizeMetadataToken(source.isotope);
            const double point_geom_scale = InverseSquareScale(
                source.x,
                source.y,
                source.z,
                request.detector_pose.x,
                request.detector_pose.y,
                request.detector_pose.z
            );
            const double detector_cps_geom_scale = DetectorCpsGeometryScale(
                source.x,
                source.y,
                source.z,
                request.detector_pose.x,
                request.detector_pose.y,
                request.detector_pose.z,
                scene_.detector
            );
            const double geom_scale = detector_cps_rate_model
                ? detector_cps_geom_scale
                : point_geom_scale;
            const auto lines = GammaLinesForIsotope(scene_, source.isotope);
            const double shield_transmission = use_theory_tvl_
                ? TheoryTvlTransmission(source, scene_, request)
                : 1.0;
            const double source_strength = radioactive_decay_emission
                ? source.activity_bq
                : source.intensity_cps_1m;
            source_equivalent_counts[source.isotope] += source_strength
                * request.dwell_time_s
                * (radioactive_decay_emission ? 1.0 : geom_scale)
                * shield_transmission;
            source_equivalent_counts_by_source[source_token] += source_strength
                * request.dwell_time_s
                * (radioactive_decay_emission ? 1.0 : geom_scale)
                * shield_transmission;
            if (radioactive_decay_emission) {
                const auto nuclide = scene_.nuclides.find(source.isotope);
                if (nuclide == scene_.nuclides.end()) {
                    throw std::runtime_error(
                        "Radioactive source is absent from the evaluated "
                        "nuclide catalog."
                    );
                }
                const double mean_events = source.activity_bq
                    * request.dwell_time_s;
                const std::string line_token = source_token + "_decay";
                scheduled_incident_gamma_counts_by_line[line_token] += mean_events;
                if (mean_events <= 0.0) {
                    continue;
                }
                scheduled_unthinned_primaries += mean_events;
                expected_sampled_primaries += mean_events;
                std::poisson_distribution<long long> distribution(mean_events);
                const long long histories = std::max(0LL, distribution(rng));
                if (histories <= 0) {
                    continue;
                }
                if (
                    histories
                    > static_cast<long long>(
                        std::numeric_limits<long>::max() - total_primaries
                    )
                ) {
                    throw std::runtime_error(
                        "Radioactive decay-event count exceeds the supported "
                        "range."
                    );
                }
                total_primaries += static_cast<long>(histories);
                PrimarySourceSnapshot source_snapshot;
                source_snapshot.position = G4ThreeVector(
                    source.x * m,
                    source.y * m,
                    source.z * m
                );
                source_snapshot.energy_keV = 0.0;
                source_snapshot.source_bias_mode = "analog";
                source_snapshot.source_rate_model = source_rate_model;
                source_snapshot.detector_center = detector_center;
                source_snapshot.cone_half_angle_rad = CLHEP::pi;
                source_snapshot.isotropic_fraction = 1.0;
                source_snapshot.primary_history_weight = 1.0;
                source_snapshot.acquisition_duration_s = request.dwell_time_s;
                primary_schedule.push_back({
                    std::move(source_snapshot),
                    source_index,
                    source.isotope,
                    source_token,
                    line_token,
                    mean_events,
                    histories,
                    -1,
                    0,
                    1,
                    1,
                    true,
                    nuclide->second.atomic_number,
                    nuclide->second.mass_number,
                    nuclide->second.geant4_excitation_keV,
                });
                continue;
            }
            double total_line_intensity = 0.0;
            for (const auto& line : lines) {
                total_line_intensity += std::max(0.0, line.intensity);
            }
            for (const auto& line : lines) {
                const double source_rate_scale = detector_cps_rate_model
                    ? geom_scale
                    : 1.0 / reference_acceptance;
                const double line_weight = (
                    detector_cps_rate_model && total_line_intensity > 0.0
                )
                    ? line.intensity / total_line_intensity
                    : line.intensity;
                const double detector_green_reference_efficiency = (
                    detector_cps_rate_model && sample_detector_response_
                )
                    ? detector_green_reference_efficiency_by_isotope.at(
                        source.isotope
                    )
                    : 1.0;
                const double mean_events = source.intensity_cps_1m
                    * request.dwell_time_s
                    * shield_transmission
                    * line_weight
                    * source_rate_scale
                    / detector_green_reference_efficiency;
                const std::string line_token = source_token
                    + "_" + EnergyMetadataToken(line.energy_keV);
                scheduled_incident_gamma_counts_by_line[line_token] += mean_events;
                if (mean_events <= 0.0) {
                    continue;
                }
                scheduled_unthinned_primaries += mean_events;
                long long histories = 0;
                double per_history_weight = primary_history_weight;
                if (mean_calibration_enabled) {
                    histories = (
                        options
                            .mean_calibration_histories_per_source_line
                    );
                    expected_sampled_primaries += static_cast<double>(
                        histories
                    );
                    per_history_weight = (
                        mean_events / static_cast<double>(histories)
                    );
                    mean_calibration_history_weight_min = std::min(
                        mean_calibration_history_weight_min,
                        per_history_weight
                    );
                    mean_calibration_history_weight_max = std::max(
                        mean_calibration_history_weight_max,
                        per_history_weight
                    );
                } else {
                    const double sampled_mean_events = (
                        mean_events * primary_sampling_fraction
                    );
                    expected_sampled_primaries += sampled_mean_events;
                    std::poisson_distribution<long long> distribution(
                        sampled_mean_events
                    );
                    histories = std::max(0LL, distribution(rng));
                    if (histories <= 0) {
                        continue;
                    }
                }
                if (
                    histories
                    > static_cast<long long>(
                        std::numeric_limits<long>::max() - total_primaries
                    )
                ) {
                    throw std::runtime_error(
                        "Primary history count exceeds the supported range."
                    );
                }
                total_primaries += static_cast<long>(histories);
                const double cone_half_angle_rad = EffectiveConeHalfAngleRad(
                    source,
                    scene_,
                    request,
                    options
                );
                if (cone_sampled_transport) {
                    const double cone_half_angle_deg = cone_half_angle_rad * 180.0 / CLHEP::pi;
                    effective_cone_min_deg = std::min(effective_cone_min_deg, cone_half_angle_deg);
                    effective_cone_max_deg = std::max(effective_cone_max_deg, cone_half_angle_deg);
                }
                PrimarySourceSnapshot source_snapshot;
                source_snapshot.position = G4ThreeVector(
                    source.x * m,
                    source.y * m,
                    source.z * m
                );
                source_snapshot.energy_keV = line.energy_keV;
                source_snapshot.source_bias_mode =
                    effective_source_bias_mode;
                source_snapshot.source_rate_model = source_rate_model;
                source_snapshot.detector_center = detector_center;
                source_snapshot.cone_half_angle_rad =
                    cone_half_angle_rad;
                source_snapshot.isotropic_fraction =
                    source_bias_weighted_transport
                        ? options.source_bias_isotropic_fraction
                        : 1.0;
                source_snapshot.primary_history_weight =
                    per_history_weight;
                if (mean_calibration_enabled) {
                    const long long histories_per_stratum = (
                        histories / angle_stratum_count
                    );
                    for (
                        int stratum_index = 0;
                        stratum_index < angle_stratum_count;
                        ++stratum_index
                    ) {
                        primary_schedule.push_back({
                            source_snapshot,
                            source_index,
                            source.isotope,
                            source_token,
                            line_token,
                            mean_events
                                / static_cast<double>(
                                    angle_stratum_count
                                ),
                            histories_per_stratum,
                            stratum_index,
                            angle_stratum_count,
                            options.mean_calibration_angle_strata_mu,
                            options.mean_calibration_angle_strata_phi,
                        });
                    }
                } else {
                    primary_schedule.push_back({
                        std::move(source_snapshot),
                        source_index,
                        source.isotope,
                        source_token,
                        line_token,
                        mean_events,
                        histories,
                        -1,
                        0,
                        1,
                        1,
                    });
                }
            }
        }
        primary_state_.ConfigureSchedule(std::move(primary_schedule));
        const int primary_beam_on_calls = BeamOnScheduledHistories(
            run_manager_,
            &primary_state_
        );
        const ForceCollisionRunSummary force_collision_summary = (
            mean_calibration_forced_collision_
                ? force_collision_diagnostics_.ValidateAndSummarize()
                : ForceCollisionRunSummary{}
        );
        auto deposits = event_store_.TakeEventDepositsMeV();
        const auto& completed_schedule = primary_state_.Schedule();
        std::vector<std::array<std::map<int, long long>, 3>>
            mean_calibration_entry_histograms(
                completed_schedule.size()
            );
        std::vector<std::map<long long, std::map<int, double>>>
            mean_calibration_cluster_scores(
                completed_schedule.size()
            );
        std::vector<std::map<int, double>>
            mean_calibration_cluster_first_sums(
                completed_schedule.size()
            );
        std::vector<std::map<std::pair<int, int>, double>>
            mean_calibration_cluster_sum_outers(
                completed_schedule.size()
            );
        std::vector<std::map<int, double>>
            mean_calibration_combined_first_sums(
                completed_schedule.size()
            );
        std::vector<std::map<std::pair<int, int>, double>>
            mean_calibration_combined_sum_outers(
                completed_schedule.size()
            );
        std::set<long long> mean_calibration_detected_histories;
        long long acquisition_window_rejected_pulses = 0;
        long long acquisition_window_rejected_prompt_pulses = 0;
        long long acquisition_window_rejected_delayed_pulses = 0;
        long long prompt_coincidence_pulses = 0;
        long long delayed_pulses = 0;
        std::map<std::pair<std::size_t, long long>, long long>
            pulses_per_parent_history;
        std::map<std::string, long long> pulse_deposit_multiplicity;
        energy_deposits.reserve(deposits.size());
        for (const auto& deposit : deposits) {
            if (deposit.edep_mev <= 0.0) {
                continue;
            }
            if (deposit.primary_batch_index >= completed_schedule.size()) {
                throw std::runtime_error(
                    "Detector deposit references an invalid primary history "
                    "batch."
                );
            }
            const auto& batch = completed_schedule[
                deposit.primary_batch_index
            ];
            if (
                radioactive_decay_emission
                && deposit.global_time_s
                    > request.dwell_time_s + 1.0e-15
            ) {
                ++acquisition_window_rejected_pulses;
                const double delay_s = (
                    deposit.global_time_s - deposit.primary_event_time_s
                );
                if (delay_s <= scene_.detector.coincidence_window_s) {
                    ++acquisition_window_rejected_prompt_pulses;
                } else {
                    ++acquisition_window_rejected_delayed_pulses;
                }
                continue;
            }
            if (
                radioactive_decay_emission
                && deposit.global_time_s + 1.0e-15
                    < deposit.primary_event_time_s
            ) {
                throw std::runtime_error(
                    "A radioactive detector pulse predates its parent event."
                );
            }
            if (radioactive_decay_emission) {
                ++pulses_per_parent_history[
                    {
                        deposit.primary_batch_index,
                        deposit.primary_history_index,
                    }
                ];
                ++pulse_deposit_multiplicity[
                    std::to_string(deposit.step_deposit_count)
                ];
                const double delay_s = (
                    deposit.global_time_s - deposit.primary_event_time_s
                );
                if (delay_s <= scene_.detector.coincidence_window_s) {
                    ++prompt_coincidence_pulses;
                } else {
                    ++delayed_pulses;
                }
            }
            const double detected_weight = std::max(0.0, deposit.weight);
            transport_detected_counts[batch.isotope] += detected_weight;
            transport_detected_counts_by_source[batch.source_token] +=
                detected_weight;
            transport_detected_counts_by_line[batch.line_token] +=
                detected_weight;
            if (
                deposit.entry_class
                    == DetectorEntryClass::kUncollidedPrimary
            ) {
                transport_uncollided_primary_counts[batch.isotope] +=
                    detected_weight;
                transport_uncollided_primary_counts_by_source[
                    batch.source_token
                ] += detected_weight;
                transport_uncollided_primary_counts_by_line[
                    batch.line_token
                ] += detected_weight;
            } else if (
                deposit.entry_class
                    == DetectorEntryClass::kInteractedPrimary
            ) {
                transport_interacted_primary_counts[batch.isotope] +=
                    detected_weight;
                transport_interacted_primary_counts_by_source[
                    batch.source_token
                ] += detected_weight;
                transport_interacted_primary_counts_by_line[
                    batch.line_token
                ] += detected_weight;
            } else {
                transport_secondary_counts[batch.isotope] += detected_weight;
                transport_secondary_counts_by_source[batch.source_token] +=
                    detected_weight;
                transport_secondary_counts_by_line[batch.line_token] +=
                    detected_weight;
            }
            energy_deposits.push_back({
                deposit.edep_mev * 1000.0,
                detected_weight,
                deposit.entry_class,
                batch.isotope,
                batch.source_token,
                batch.line_token,
                deposit.primary_batch_index,
                deposit.primary_history_index,
                deposit.bias_branch_lineage_id,
                deposit.step_deposit_count,
                deposit.global_time_s,
                deposit.impact_parameter_fraction,
            });
        }
        std::map<std::string, long long> parent_pulse_multiplicity;
        for (const auto& item : pulses_per_parent_history) {
            ++parent_pulse_multiplicity[std::to_string(item.second)];
        }
        if (radioactive_decay_emission) {
            const auto detected_parent_count = static_cast<long long>(
                pulses_per_parent_history.size()
            );
            if (detected_parent_count > total_primaries) {
                throw std::runtime_error(
                    "Detected parent-history count exceeds generated decays."
                );
            }
            parent_pulse_multiplicity["0"] += (
                static_cast<long long>(total_primaries)
                - detected_parent_count
            );
        }
        const double primary_expectation_tolerance = 1.0e-9
            + 1.0e-12 * std::abs(expected_unthinned_primaries);
        if (
            !std::isfinite(scheduled_unthinned_primaries)
            || std::abs(
                scheduled_unthinned_primaries - expected_unthinned_primaries
            ) > primary_expectation_tolerance
        ) {
            throw std::runtime_error(
                "Precomputed and scheduled unthinned primary expectations disagree"
            );
        }
        constexpr double kBinWidthKeV = 2.0;
        const double kEnergyMaxKeV = options.decay_comparison_diagnostic
            ? options.decay_comparison_energy_max_keV
            : 1700.0;
        const int num_bins = static_cast<int>(kEnergyMaxKeV / kBinWidthKeV) + 1;
        std::vector<double> spectrum(num_bins, 0.0);
        std::vector<double> spectrum_variance(num_bins, 0.0);
        std::map<std::string, std::vector<double>> validation_entry_spectra;
        std::map<std::string, std::vector<double>>
            validation_observed_entry_spectra;
        std::map<std::string, std::vector<double>>
            mean_calibration_entry_variance;
        using DetectorResponsePulseKey = std::tuple<
            std::size_t,
            long long,
            long long
        >;
        struct SampledDetectorResponseEntry {
            double global_time_s = 0.0;
            int bin_index = -1;
        };
        std::map<
            DetectorResponsePulseKey,
            std::vector<SampledDetectorResponseEntry>
        > sampled_detector_response_entries;
        long long detector_response_registered_entry_count = 0;
        long long detector_response_coincidence_pulse_count = 0;
        long long detector_response_multi_entry_pulse_count = 0;
        std::normal_distribution<double> gaussian(0.0, 1.0);
        const bool incident_gamma_scoring = detector_scoring_mode_ == "incident_gamma_energy";
        if (options.sample_detector_response && !incident_gamma_scoring) {
            throw std::runtime_error(
                "sample_detector_response requires incident_gamma_energy scoring"
            );
        }
        for (const auto& deposit : energy_deposits) {
            const double energy_keV = deposit.energy_keV;
            if (energy_keV < 0.0 || energy_keV > kEnergyMaxKeV) {
                continue;
            }
            const int raw_index = static_cast<int>(
                std::floor(energy_keV / kBinWidthKeV)
            );
            if (raw_index < 0 || raw_index >= num_bins) {
                continue;
            }
            if (mean_calibration_enabled) {
                if (
                    deposit.primary_batch_index
                        >= completed_schedule.size()
                    || deposit.primary_history_index < 0
                ) {
                    throw std::runtime_error(
                        "Mean-calibration deposit is missing primary-history "
                        "provenance."
                    );
                }
                const auto& batch = completed_schedule[
                    deposit.primary_batch_index
                ];
                const double base_history_weight = (
                    batch.source.primary_history_weight
                );
                if (
                    !std::isfinite(base_history_weight)
                    || base_history_weight <= 0.0
                    || !std::isfinite(deposit.weight)
                    || deposit.weight < 0.0
                ) {
                    throw std::runtime_error(
                        "Mean-calibration history or branch weight is "
                        "invalid."
                    );
                }
                const double relative_branch_weight = (
                    deposit.weight / base_history_weight
                );
                if (
                    !std::isfinite(relative_branch_weight)
                    || relative_branch_weight < 0.0
                    || relative_branch_weight > 1.0 + 1.0e-9
                ) {
                    throw std::runtime_error(
                        "Mean-calibration relative branch weight is outside "
                        "its support."
                    );
                }
                const int cluster_coordinate = (
                    static_cast<int>(
                        DetectorEntryClassIndex(deposit.entry_class)
                    ) * num_bins
                    + raw_index
                );
                mean_calibration_cluster_scores[
                    deposit.primary_batch_index
                ][deposit.primary_history_index][cluster_coordinate]
                    += relative_branch_weight;
                if (!options.mean_calibration_forced_collision) {
                    if (
                        std::abs(relative_branch_weight - 1.0) > 1.0e-12
                    ) {
                        throw std::runtime_error(
                            "Mean-calibration analog history has an "
                            "unexpected track weight."
                        );
                    }
                    if (
                        !mean_calibration_detected_histories.insert(
                            deposit.primary_history_index
                        ).second
                    ) {
                        throw std::runtime_error(
                            "Mean-calibration analog history produced more "
                            "than one merged detector pulse."
                        );
                    }
                    mean_calibration_entry_histograms[
                        deposit.primary_batch_index
                    ][DetectorEntryClassIndex(
                        deposit.entry_class
                    )][raw_index] += 1;
                }
            }
            if (options.validation_entry_class_spectra) {
                const std::string entry_class = DetectorEntryClassToken(
                    deposit.entry_class
                );
                const std::string key = deposit.line_token + "_" + entry_class;
                auto& classified_spectrum = validation_entry_spectra[key];
                if (classified_spectrum.empty()) {
                    classified_spectrum.assign(num_bins, 0.0);
                }
                classified_spectrum[static_cast<std::size_t>(raw_index)]
                    += deposit.weight;
            }
            int index = raw_index;
            if (options.sample_detector_response) {
                if (std::abs(deposit.weight - 1.0) > 1.0e-12) {
                    throw std::runtime_error(
                        "Detector-response marking requires unit-weight histories"
                    );
                }
                index = detector_green_operator_->SampleBin(
                    energy_keV,
                    deposit.impact_parameter_fraction,
                    rng
                );
            } else if (!incident_gamma_scoring) {
                const double smeared = (
                    energy_keV + SigmaEnergyKeV(energy_keV) * gaussian(rng)
                );
                if (smeared < 0.0 || smeared > kEnergyMaxKeV) {
                    continue;
                }
                index = static_cast<int>(
                    std::floor(smeared / kBinWidthKeV)
                );
            }
            if (index >= 0 && index < num_bins) {
                if (options.validation_entry_class_spectra) {
                    const std::string entry_class = DetectorEntryClassToken(
                        deposit.entry_class
                    );
                    const std::string key = (
                        deposit.line_token + "_" + entry_class
                    );
                    auto& classified_spectrum = (
                        validation_observed_entry_spectra[key]
                    );
                    if (classified_spectrum.empty()) {
                        classified_spectrum.assign(num_bins, 0.0);
                    }
                    classified_spectrum[
                        static_cast<std::size_t>(index)
                    ] += deposit.weight;
                }
                if (options.sample_detector_response) {
                    sampled_detector_response_entries[{
                        deposit.primary_batch_index,
                        deposit.primary_history_index,
                        deposit.bias_branch_lineage_id,
                    }].push_back({
                        deposit.global_time_s,
                        index,
                    });
                    ++detector_response_registered_entry_count;
                } else {
                    spectrum[index] += deposit.weight;
                    if (!mean_calibration_enabled) {
                        spectrum_variance[index] += (
                            deposit.weight * deposit.weight
                        );
                    }
                }
            }
        }
        if (options.sample_detector_response) {
            for (auto& pulse_item : sampled_detector_response_entries) {
                auto& entries = pulse_item.second;
                std::sort(
                    entries.begin(),
                    entries.end(),
                    [](const auto& lhs, const auto& rhs) {
                        return lhs.global_time_s < rhs.global_time_s;
                    }
                );
                double pulse_start_time_s = 0.0;
                int pulse_bin_index = -1;
                long long pulse_entry_count = 0;
                const auto flush_response_pulse = [&]() {
                    if (pulse_entry_count <= 0) {
                        return;
                    }
                    if (
                        pulse_bin_index < 0
                        || pulse_bin_index >= num_bins
                    ) {
                        throw std::runtime_error(
                            "Coincident detector-response pulse is outside "
                            "the approved observed-energy domain."
                        );
                    }
                    spectrum[static_cast<std::size_t>(pulse_bin_index)] += 1.0;
                    spectrum_variance[
                        static_cast<std::size_t>(pulse_bin_index)
                    ] += 1.0;
                    ++detector_response_coincidence_pulse_count;
                    if (pulse_entry_count > 1) {
                        ++detector_response_multi_entry_pulse_count;
                    }
                };
                for (const auto& entry : entries) {
                    if (
                        pulse_entry_count == 0
                        || entry.global_time_s - pulse_start_time_s
                            > scene_.detector.coincidence_window_s
                    ) {
                        flush_response_pulse();
                        pulse_start_time_s = entry.global_time_s;
                        pulse_bin_index = entry.bin_index;
                        pulse_entry_count = 1;
                    } else {
                        pulse_bin_index += entry.bin_index;
                        ++pulse_entry_count;
                    }
                }
                flush_response_pulse();
            }
        }
        if (mean_calibration_enabled) {
            if (options.mean_calibration_forced_collision) {
                for (
                    std::size_t batch_index = 0;
                    batch_index < completed_schedule.size();
                    ++batch_index
                ) {
                    const auto& batch = completed_schedule[batch_index];
                    const double sample_count = static_cast<double>(
                        batch.sampled_histories
                    );
                    if (batch.sampled_histories <= 1) {
                        throw std::runtime_error(
                            "Mean-calibration covariance requires at least "
                            "two histories per angular stratum."
                        );
                    }
                    auto& first_sums = (
                        mean_calibration_cluster_first_sums[batch_index]
                    );
                    auto& sum_outers = (
                        mean_calibration_cluster_sum_outers[batch_index]
                    );
                    auto& combined_first_sums = (
                        mean_calibration_combined_first_sums[batch_index]
                    );
                    auto& combined_sum_outers = (
                        mean_calibration_combined_sum_outers[batch_index]
                    );
                    for (
                        const auto& history_item
                        : mean_calibration_cluster_scores[batch_index]
                    ) {
                        const auto& score = history_item.second;
                        std::map<int, double> combined_score;
                        for (const auto& item : score) {
                            if (
                                item.first < 0
                                || item.first >= 3 * num_bins
                                || !std::isfinite(item.second)
                                || item.second < 0.0
                            ) {
                                throw std::runtime_error(
                                    "Mean-calibration cluster score is "
                                    "invalid."
                                );
                            }
                            first_sums[item.first] += item.second;
                            combined_score[item.first % num_bins]
                                += item.second;
                        }
                        for (
                            auto left = score.begin();
                            left != score.end();
                            ++left
                        ) {
                            for (
                                auto right = left;
                                right != score.end();
                                ++right
                            ) {
                                sum_outers[{
                                    left->first,
                                    right->first,
                                }] += left->second * right->second;
                            }
                        }
                        for (const auto& item : combined_score) {
                            combined_first_sums[item.first] += item.second;
                        }
                        for (
                            auto left = combined_score.begin();
                            left != combined_score.end();
                            ++left
                        ) {
                            for (
                                auto right = left;
                                right != combined_score.end();
                                ++right
                            ) {
                                combined_sum_outers[{
                                    left->first,
                                    right->first,
                                }] += left->second * right->second;
                            }
                        }
                    }
                    const double correction = (
                        batch.source.primary_history_weight
                        * batch.source.primary_history_weight
                        * sample_count
                        / (sample_count - 1.0)
                    );
                    const auto unbiased_variance = [
                        sample_count,
                        correction
                    ](
                        const double first_sum,
                        const double second_sum
                    ) {
                        double centered = (
                            second_sum
                            - first_sum * first_sum / sample_count
                        );
                        const double tolerance = (
                            1.0e-12
                            * std::max(
                                1.0,
                                std::abs(second_sum)
                                    + first_sum * first_sum / sample_count
                            )
                        );
                        if (centered < -tolerance) {
                            throw std::runtime_error(
                                "Mean-calibration cluster covariance is "
                                "materially negative."
                            );
                        }
                        centered = std::max(0.0, centered);
                        return correction * centered;
                    };
                    for (
                        std::size_t entry_index = 0;
                        entry_index < 3;
                        ++entry_index
                    ) {
                        const auto entry_class = static_cast<
                            DetectorEntryClass
                        >(entry_index);
                        const std::string variance_key = (
                            batch.line_token
                            + "_"
                            + DetectorEntryClassToken(entry_class)
                        );
                        auto& class_variance = (
                            mean_calibration_entry_variance[variance_key]
                        );
                        if (class_variance.empty()) {
                            class_variance.assign(num_bins, 0.0);
                        }
                    }
                    for (const auto& item : first_sums) {
                        const int coordinate = item.first;
                        const int entry_index = coordinate / num_bins;
                        const int bin_index = coordinate % num_bins;
                        const auto second = sum_outers.find({
                            coordinate,
                            coordinate,
                        });
                        const double second_sum = second == sum_outers.end()
                            ? 0.0
                            : second->second;
                        const auto entry_class = static_cast<
                            DetectorEntryClass
                        >(entry_index);
                        const std::string variance_key = (
                            batch.line_token
                            + "_"
                            + DetectorEntryClassToken(entry_class)
                        );
                        auto& class_variance = (
                            mean_calibration_entry_variance[variance_key]
                        );
                        if (class_variance.empty()) {
                            class_variance.assign(num_bins, 0.0);
                        }
                        class_variance[
                            static_cast<std::size_t>(bin_index)
                        ] += unbiased_variance(item.second, second_sum);
                    }
                    for (const auto& item : combined_first_sums) {
                        const auto second = combined_sum_outers.find({
                            item.first,
                            item.first,
                        });
                        const double second_sum = (
                            second == combined_sum_outers.end()
                                ? 0.0
                                : second->second
                        );
                        spectrum_variance[
                            static_cast<std::size_t>(item.first)
                        ] += unbiased_variance(item.second, second_sum);
                    }
                }
            } else {
                for (
                    std::size_t batch_index = 0;
                    batch_index < completed_schedule.size();
                    ++batch_index
                ) {
                    const auto& batch = completed_schedule[batch_index];
                    const double sample_count = static_cast<double>(
                        batch.sampled_histories
                    );
                    if (batch.sampled_histories <= 1) {
                        throw std::runtime_error(
                            "Mean-calibration covariance requires at least "
                            "two histories per angular stratum."
                        );
                    }
                    const double correction = (
                        batch.source.primary_history_weight
                        * batch.source.primary_history_weight
                        * sample_count
                        / (sample_count - 1.0)
                    );
                    std::map<int, long long> combined_histogram;
                    for (
                        std::size_t entry_index = 0;
                        entry_index < 3;
                        ++entry_index
                    ) {
                        const auto entry_class = static_cast<
                            DetectorEntryClass
                        >(entry_index);
                        const std::string variance_key = (
                            batch.line_token
                            + "_"
                            + DetectorEntryClassToken(entry_class)
                        );
                        auto& class_variance = (
                            mean_calibration_entry_variance[variance_key]
                        );
                        if (class_variance.empty()) {
                            class_variance.assign(num_bins, 0.0);
                        }
                        for (
                            const auto& item
                            : mean_calibration_entry_histograms[
                                batch_index
                            ][entry_index]
                        ) {
                            const double count = static_cast<double>(
                                item.second
                            );
                            if (count > sample_count) {
                                throw std::runtime_error(
                                    "Mean-calibration stratum hit count "
                                    "exceeds its history count."
                                );
                            }
                            const double variance = correction * (
                                count - count * count / sample_count
                            );
                            class_variance[
                                static_cast<std::size_t>(item.first)
                            ] += variance;
                            combined_histogram[item.first] += item.second;
                        }
                    }
                    for (const auto& item : combined_histogram) {
                        const double count = static_cast<double>(
                            item.second
                        );
                        if (count > sample_count) {
                            throw std::runtime_error(
                                "Mean-calibration spectrum hit count exceeds "
                                "its history count."
                            );
                        }
                        spectrum_variance[
                            static_cast<std::size_t>(item.first)
                        ] += correction * (
                            count - count * count / sample_count
                        );
                    }
                }
            }
        }
        const std::vector<double> source_only_spectrum = spectrum;
        AddBackgroundSpectrum(spectrum, &spectrum_variance, kBinWidthKeV, request.dwell_time_s, options, rng);
        std::vector<double> validation_background_spectrum;
        if (options.validation_entry_class_spectra) {
            validation_background_spectrum.resize(
                spectrum.size(),
                0.0
            );
            for (std::size_t index = 0; index < spectrum.size(); ++index) {
                validation_background_spectrum[index] = std::max(
                    0.0,
                    spectrum[index] - source_only_spectrum[index]
                );
            }
        }
        const double total_counts = std::accumulate(spectrum.begin(), spectrum.end(), 0.0);
        const double pre_dead_time_total_variance = std::accumulate(
            spectrum_variance.begin(),
            spectrum_variance.end(),
            0.0
        );
        const double dwell_time_s = std::max(1.0e-6, request.dwell_time_s);
        const double true_rate = total_counts / dwell_time_s;
        double observed_scale = 1.0 / (
            1.0 + std::max(0.0, true_rate * dead_time_tau_s)
        );
        if (options.sample_detector_response) {
            const auto integer_total = static_cast<long long>(
                std::llround(total_counts)
            );
            std::vector<long long> incident_histogram(
                spectrum.size(),
                0LL
            );
            for (std::size_t index = 0; index < spectrum.size(); ++index) {
                const double value = spectrum[index];
                const auto integer_count = static_cast<long long>(
                    std::llround(value)
                );
                if (
                    integer_count < 0
                    || std::abs(value - static_cast<double>(integer_count))
                        > 1.0e-9
                ) {
                    throw std::runtime_error(
                        "Detector-response marking produced noninteger counts"
                    );
                }
                incident_histogram[index] = integer_count;
            }
            if (
                std::accumulate(
                    incident_histogram.begin(),
                    incident_histogram.end(),
                    0LL
                ) != integer_total
            ) {
                throw std::runtime_error(
                    "Detector-response pulse total disagrees with its histogram"
                );
            }
            const long long accepted_count = (
                SampleNonparalyzableAcceptedCount(
                    integer_total,
                    dwell_time_s,
                    dead_time_tau_s,
                    rng
                )
            );
            const auto accepted_histogram = SampleUniformHistogramSubset(
                incident_histogram,
                accepted_count,
                rng
            );
            for (std::size_t index = 0; index < spectrum.size(); ++index) {
                spectrum[index] = static_cast<double>(
                    accepted_histogram[index]
                );
                spectrum_variance[index] = spectrum[index];
            }
            observed_scale = total_counts > 0.0
                ? static_cast<double>(accepted_count) / total_counts
                : 1.0;
        } else {
            for (std::size_t index = 0; index < spectrum.size(); ++index) {
                spectrum[index] *= observed_scale;
                spectrum_variance[index] *= observed_scale * observed_scale;
            }
        }
        if (
            options.validation_entry_class_spectra
            && !options.sample_detector_response
        ) {
            for (auto& item : validation_entry_spectra) {
                for (double& value : item.second) {
                    value *= observed_scale;
                }
            }
            for (double& value : validation_background_spectrum) {
                value *= observed_scale;
            }
        }
        for (auto& item : transport_detected_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_uncollided_primary_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_interacted_primary_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_secondary_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_detected_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_uncollided_primary_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_interacted_primary_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_secondary_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_detected_counts_by_line) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_uncollided_primary_counts_by_line) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_interacted_primary_counts_by_line) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_secondary_counts_by_line) {
            item.second *= observed_scale;
        }
        const double total_variance = std::accumulate(
            spectrum_variance.begin(),
            spectrum_variance.end(),
            0.0
        );
        const double observed_total_counts = std::accumulate(spectrum.begin(), spectrum.end(), 0.0);
        const double effective_spectrum_entries = total_variance > 0.0
            ? (observed_total_counts * observed_total_counts) / total_variance
            : 0.0;
        const auto end_time = std::chrono::steady_clock::now();
        const auto runtime_s = std::chrono::duration<double>(end_time - start_time).count();
        const double safe_runtime_s = std::max(1.0e-12, runtime_s);
        const auto process_counts = diagnostics_.ProcessCounts();
        const auto volume_step_counts = diagnostics_.VolumeStepCounts();
        const long long compton_count = diagnostics_.ProcessCountForAliases({"compt", "compton"});
        const long long rayleigh_count = diagnostics_.ProcessCountForAliases({"rayl", "rayleigh"});
        const long long photoelectric_count = diagnostics_.ProcessCountForAliases({"phot", "photoelectric"});
        event_store_.ClearDeposits();

        SimulationResult result;
        result.spectrum_counts = std::move(spectrum);
        result.spectrum_count_variance = std::move(spectrum_variance);
        result.metadata["backend"] = "geant4";
        result.metadata["engine_mode"] = "external";
        result.metadata["primary_emission_model"] = primary_emission_model;
        result.metadata["emission_model"] = radioactive_decay_emission
            ? "isotropic_parent_radioactive_decay"
            : (
                detector_cps_rate_model
                    ? "detector_equivalent_cone"
                    : (
                        source_bias_weighted_transport
                            ? "weighted_isotropic"
                            : "isotropic"
                    )
            );
        result.metadata["source_rate_model"] = source_rate_model;
        result.metadata["source_strength_field"] = radioactive_decay_emission
            ? "activity_bq"
            : "intensity_cps_1m";
        if (radioactive_decay_emission) {
            result.metadata["activity_bq_definition"] =
                "parent_decays_per_second_at_acquisition_start";
            result.metadata["radioactive_source_state_semantics"] =
                "scheduled_parent_decays_no_preexisting_daughter_inventory";
        } else {
            result.metadata["intensity_cps_1m_definition"] =
                "pre_dead_time_detector_pulse_rate_at_1m";
        }
        result.metadata["detector_cps_green_reference_normalization"] = (
            detector_cps_rate_model && sample_detector_response_
                ? "catalog_branching_weighted_absolute_detection_"
                    "efficiency_at_1m_v1"
                : "disabled"
        );
        result.metadata["line_intensities_normalized"] = (
            detector_cps_rate_model && !radioactive_decay_emission
        ) ? "true" : "false";
        result.metadata["prompt_decay_cascade_transport"] = (
            radioactive_decay_emission ? "true" : "false"
        );
        result.metadata["true_coincidence_summing"] = (
            radioactive_decay_emission
                ? "global_time_window_energy_deposit_sum"
                : "disabled"
        );
        result.metadata["detector_coincidence_window_s"] = SerializeDouble(
            scene_.detector.coincidence_window_s
        );
        result.metadata["delayed_decay_pulse_separation"] = (
            radioactive_decay_emission ? "true" : "false"
        );
        result.metadata["radioactive_decay_time_window"] = (
            radioactive_decay_emission
                ? "parent_events_uniform_in_acquisition_prompt_parent_forced_"
                    "daughters_geant4_timed_out_of_window_rejected"
                : "disabled"
        );
        result.metadata["acquisition_window_s"] = SerializeDouble(
            request.dwell_time_s
        );
        result.metadata["acquisition_window_rejected_pulses"] = (
            std::to_string(acquisition_window_rejected_pulses)
        );
        result.metadata["acquisition_window_rejected_prompt_pulses"] = (
            std::to_string(acquisition_window_rejected_prompt_pulses)
        );
        result.metadata["acquisition_window_rejected_delayed_pulses"] = (
            std::to_string(acquisition_window_rejected_delayed_pulses)
        );
        result.metadata["prompt_coincidence_pulses"] = std::to_string(
            prompt_coincidence_pulses
        );
        result.metadata["delayed_pulses"] = std::to_string(delayed_pulses);
        result.metadata["parent_pulse_multiplicity_histogram"] = (
            SerializeCounterMap(parent_pulse_multiplicity)
        );
        result.metadata["pulse_step_deposit_multiplicity_histogram"] = (
            SerializeCounterMap(pulse_deposit_multiplicity)
        );
        bool all_sources_surface_bound = !scene_.sources.empty();
        std::string surface_emission_policy_sha256;
        double surface_emission_epsilon_m = 0.0;
        result.metadata["native_surface_source_count"] = std::to_string(
            scene_.sources.size()
        );
        for (
            std::size_t source_index = 0;
            source_index < scene_.sources.size();
            ++source_index
        ) {
            const auto& source = scene_.sources[source_index];
            const bool has_contract = (
                source.surface_chart_id >= 0
                && source.surface_u >= 0.0
                && source.surface_u <= 1.0
                && source.surface_v >= 0.0
                && source.surface_v <= 1.0
                && !source.surface_emission_policy_sha256.empty()
                && source.surface_emission_epsilon_m > 0.0
            );
            if (!has_contract) {
                all_sources_surface_bound = false;
                continue;
            }
            const double dx = source.x - source.anchor_x;
            const double dy = source.y - source.anchor_y;
            const double dz = source.z - source.anchor_z;
            const double normal_norm = std::sqrt(
                source.surface_normal_x * source.surface_normal_x
                + source.surface_normal_y * source.surface_normal_y
                + source.surface_normal_z * source.surface_normal_z
            );
            const double epsilon = source.surface_emission_epsilon_m;
            const double contract_error = std::sqrt(
                std::pow(dx - epsilon * source.surface_normal_x, 2)
                + std::pow(dy - epsilon * source.surface_normal_y, 2)
                + std::pow(dz - epsilon * source.surface_normal_z, 2)
            );
            if (
                std::abs(normal_norm - 1.0) > 1.0e-12
                || contract_error > 1.0e-12
            ) {
                throw std::runtime_error(
                    "Native source emission XYZ violates the declared "
                    "surface-anchor epsilon contract."
                );
            }
            if (surface_emission_policy_sha256.empty()) {
                surface_emission_policy_sha256 =
                    source.surface_emission_policy_sha256;
                surface_emission_epsilon_m = epsilon;
            } else if (
                surface_emission_policy_sha256
                    != source.surface_emission_policy_sha256
                || std::abs(surface_emission_epsilon_m - epsilon) > 1.0e-15
            ) {
                throw std::runtime_error(
                    "Native sources declare inconsistent surface-emission "
                    "contracts."
                );
            }
            const std::string prefix = (
                "native_surface_source_"
                + std::to_string(source_index)
                + "_"
            );
            result.metadata[prefix + "isotope"] = source.isotope;
            if (radioactive_decay_emission) {
                result.metadata[prefix + "activity_bq"] = SerializeDouble(
                    source.activity_bq
                );
            } else {
                result.metadata[prefix + "intensity_cps_1m"] = SerializeDouble(
                    source.intensity_cps_1m
                );
                if (sample_detector_response_) {
                    result.metadata[
                        prefix + "detector_green_reference_efficiency"
                    ] = SerializeDouble(
                        detector_green_reference_efficiency_by_isotope.at(
                            source.isotope
                        )
                    );
                }
            }
            result.metadata[prefix + "surface_chart_id"] = std::to_string(
                source.surface_chart_id
            );
            result.metadata[prefix + "surface_emission_policy_sha256"] = (
                source.surface_emission_policy_sha256
            );
            result.metadata[prefix + "anchor_x"] = SerializeDouble(
                source.anchor_x
            );
            result.metadata[prefix + "anchor_y"] = SerializeDouble(
                source.anchor_y
            );
            result.metadata[prefix + "anchor_z"] = SerializeDouble(
                source.anchor_z
            );
            result.metadata[prefix + "transport_x"] = SerializeDouble(
                source.x
            );
            result.metadata[prefix + "transport_y"] = SerializeDouble(
                source.y
            );
            result.metadata[prefix + "transport_z"] = SerializeDouble(
                source.z
            );
            result.metadata[prefix + "surface_u"] = SerializeDouble(
                source.surface_u
            );
            result.metadata[prefix + "surface_v"] = SerializeDouble(
                source.surface_v
            );
            result.metadata[prefix + "surface_normal_x"] = SerializeDouble(
                source.surface_normal_x
            );
            result.metadata[prefix + "surface_normal_y"] = SerializeDouble(
                source.surface_normal_y
            );
            result.metadata[prefix + "surface_normal_z"] = SerializeDouble(
                source.surface_normal_z
            );
        }
        result.metadata["source_position_semantics"] = (
            "air_side_native_emission_xyz"
        );
        result.metadata["source_anchor_semantics"] = (
            "exact_surface_chart_uv_evaluation_truth"
        );
        result.metadata["all_sources_surface_bound"] = (
            all_sources_surface_bound ? "true" : "false"
        );
        result.metadata["surface_emission_policy_sha256"] = (
            surface_emission_policy_sha256
        );
        result.metadata["surface_emission_epsilon_m"] = SerializeDouble(
            surface_emission_epsilon_m
        );
        result.metadata["physics_profile"] = physics_profile_;
        result.metadata["detector_scoring_mode"] = detector_scoring_mode_;
        result.metadata["detector_fast_scoring"] = detector_scoring_mode_ == "incident_gamma_energy" ? "true" : "false";
        result.metadata["detector_fast_scoring_volume"] = detector_scoring_mode_ == "incident_gamma_energy"
            ? "detector_assembly_entry"
            : "";
        result.metadata["detector_response_applied_in_native"] = (
            detector_scoring_mode_ != "incident_gamma_energy"
            || options.sample_detector_response
        ) ? "true" : "false";
        result.metadata["detector_response_sampling_mode"] = (
            options.sample_detector_response
                ? "independent_green_mark_then_same_history_coincidence_"
                    "sum_nonparalyzable_v1"
                : "disabled"
        );
        result.metadata["detector_response_coincidence_semantics"] = (
            options.sample_detector_response
                ? "sample_each_housing_incident_gamma_then_sum_registered_"
                    "energy_within_same_history_branch_and_detector_window_v1"
                : "disabled"
        );
        result.metadata["detector_response_incident_entry_count"] = (
            std::to_string(energy_deposits.size())
        );
        result.metadata["detector_response_registered_entry_count"] = (
            std::to_string(detector_response_registered_entry_count)
        );
        result.metadata["detector_response_coincidence_pulse_count"] = (
            std::to_string(detector_response_coincidence_pulse_count)
        );
        result.metadata["detector_response_multi_entry_pulse_count"] = (
            std::to_string(detector_response_multi_entry_pulse_count)
        );
        result.metadata["detector_response_sampling_model"] = (
            options.sample_detector_response
                ? "isotope_independent_full_detector_green_operator_v3"
                : ""
        );
        result.metadata[
            "detector_response_sampling_contract_sha256"
        ] = (
            options.sample_detector_response
                ? options.detector_green_operator_contract_sha256
                : ""
        );
        result.metadata["detector_response_operator_binary_sha256"] = (
            options.sample_detector_response
                ? options.detector_green_operator_binary_sha256
                : ""
        );
        result.metadata["detector_response_boundary_state"] = (
            options.sample_detector_response
                ? "normalized_impact_parameter_at_detector_housing_entry_v1"
                : ""
        );
        result.metadata["detector_response_conditioning"] = (
            options.sample_detector_response
                ? "registered_pulse_subprobability_given_housing_incident_gamma_v1"
                : ""
        );
        result.metadata["detector_response_sampling_seed"] = (
            options.sample_detector_response
                ? std::to_string(request.seed)
                : ""
        );
        result.metadata["spectrum_energy_min_keV"] = "0";
        result.metadata["spectrum_energy_max_keV"] = SerializeDouble(
            kEnergyMaxKeV
        );
        result.metadata["spectrum_bin_width_keV"] = SerializeDouble(
            kBinWidthKeV
        );
        result.metadata["spectrum_bin_count"] = std::to_string(num_bins);
        if (options.decay_comparison_diagnostic) {
            result.metadata["decay_comparison_diagnostic"] = "true";
        }
        result.metadata["background_spectrum_model_id"] = (
            "native_geant4_background_shape_v1_bin_centres"
        );
        AddGeant4MaterialMuMetadata(result, scene_);
        result.metadata["secondary_transport_mode"] = secondary_transport_mode_;
        result.metadata["geant4_version_number"] = std::to_string(
            G4VERSION_NUMBER
        );
        result.metadata["geant4_physics_contract_id"] = (
            kGeant4PhysicsContractId
        );
        result.metadata["geant4_physics_contract_sha256"] = (
            kGeant4PhysicsContractSha256
        );
        result.metadata["material_resolution_contract_id"] = (
            kMaterialResolutionContractId
        );
        result.metadata["geant4_version_tag"] = kGeant4VersionTag;
        result.metadata["reference_physics_list"] = (
            kReferencePhysicsListName
        );
        result.metadata["electromagnetic_physics_constructor"] = (
            kElectromagneticPhysicsConstructorName
        );
        result.metadata["production_cut_range_mm"] = SerializeDouble(
            kProductionCutRangeMm
        );
        result.metadata["gamma_process_names"] = JoinSet(
            gamma_process_names_,
            ","
        );
        result.metadata["gamma_em_subprocess_names"] = JoinSet(
            gamma_em_subprocess_names_,
            ","
        );
        result.metadata["gamma_only_secondary_transport"] = secondary_transport_mode_ == "gamma_only" ? "true" : "false";
        result.metadata["theory_tvl_attenuation"] = use_theory_tvl_ ? "true" : "false";
        result.metadata["scene_hash"] = scene_.scene_hash;
        result.metadata["surface_source_contract_sha256"] = (
            scene_.surface_source_contract_sha256
        );
        result.metadata["nuclide_catalog_sha256"] = (
            scene_.nuclide_catalog_sha256
        );
        result.metadata["num_primaries"] = std::to_string(total_primaries);
        result.metadata["primary_history_batch_count"] = std::to_string(
            primary_state_.Schedule().size()
        );
        result.metadata["primary_beam_on_calls"] = std::to_string(
            primary_beam_on_calls
        );
        result.metadata["primary_schedule_mode"] = radioactive_decay_emission
            ? "batched_parent_decay_event_schedule"
            : (
                mean_calibration_enabled
                    ? "fixed_source_line_stratified_mean_calibration"
                    : "batched_source_line_event_schedule"
            );
        result.metadata["mean_calibration_enabled"] = (
            mean_calibration_enabled ? "true" : "false"
        );
        result.metadata[
            "mean_calibration_histories_per_source_line"
        ] = std::to_string(
            options.mean_calibration_histories_per_source_line
        );
        result.metadata["mean_calibration_angle_strata_mu"] = (
            std::to_string(options.mean_calibration_angle_strata_mu)
        );
        result.metadata["mean_calibration_angle_strata_phi"] = (
            std::to_string(options.mean_calibration_angle_strata_phi)
        );
        result.metadata["mean_calibration_angle_stratum_count"] = (
            std::to_string(angle_stratum_count)
        );
        result.metadata["mean_calibration_forced_collision"] = (
            options.mean_calibration_forced_collision ? "true" : "false"
        );
        result.metadata["mean_calibration_history_weight_min"] = (
            mean_calibration_enabled
                && std::isfinite(mean_calibration_history_weight_min)
                ? SerializeDouble(mean_calibration_history_weight_min)
                : "0"
        );
        result.metadata["mean_calibration_history_weight_max"] = (
            mean_calibration_enabled
                ? SerializeDouble(mean_calibration_history_weight_max)
                : "0"
        );
        result.metadata["mean_calibration_history_weight_semantics"] = (
            mean_calibration_enabled
                ? "expected_source_line_mean_divided_by_fixed_quota"
                : "disabled"
        );
        result.metadata["mean_calibration_covariance_semantics"] = (
            mean_calibration_enabled
                ? (
                    options.mean_calibration_forced_collision
                        ? (
                            "independent_mu_phi_stratum_original_history_"
                            "branch_cluster_sufficient_statistics_v2"
                        )
                        : (
                            "independent_mu_phi_stratum_sample_mean_cluster_"
                            "sufficient_statistics_v1"
                        )
                )
                : "disabled"
        );
        result.metadata["mean_calibration_force_collision_leaf_count"] = (
            detector_construction_ == nullptr
                ? "0"
                : std::to_string(
                    detector_construction_->ForceCollisionLeafCount()
                )
        );
        result.metadata["mean_calibration_force_collision_split_count"] = (
            std::to_string(force_collision_summary.split_count)
        );
        result.metadata[
            "mean_calibration_force_collision_max_abs_weight_error"
        ] = SerializeDouble(
            force_collision_summary.maximum_absolute_weight_error
        );
        result.metadata[
            "mean_calibration_force_collision_max_rel_weight_error"
        ] = SerializeDouble(
            force_collision_summary.maximum_relative_weight_error
        );
        if (mean_calibration_enabled) {
            for (
                std::size_t batch_index = 0;
                batch_index < completed_schedule.size();
                ++batch_index
            ) {
                const auto& batch = completed_schedule[batch_index];
                const std::string prefix = (
                    "mean_calibration_batch_"
                    + std::to_string(batch_index)
                    + "_"
                );
                result.metadata[prefix + "source_token"] = (
                    SanitizeMetadataToken(batch.source_token)
                );
                result.metadata[prefix + "line_token"] = (
                    SanitizeMetadataToken(batch.line_token)
                );
                result.metadata[prefix + "expected_unthinned_histories"] = (
                    SerializeDouble(batch.expected_unthinned_histories)
                );
                result.metadata[prefix + "sampled_histories"] = (
                    std::to_string(batch.sampled_histories)
                );
                result.metadata[prefix + "history_weight"] = (
                    SerializeDouble(
                        batch.source.primary_history_weight
                    )
                );
                result.metadata[prefix + "angle_stratum_index"] = (
                    std::to_string(batch.angle_stratum_index)
                );
                if (options.mean_calibration_forced_collision) {
                    result.metadata[
                        prefix + "cluster_coordinate_semantics"
                    ] = "entry_class_major_then_energy_bin";
                    result.metadata[
                        prefix + "cluster_score_semantics"
                    ] = (
                        "sum_branch_relative_bias_weight_one_hot_per_"
                        "original_history"
                    );
                    result.metadata[
                        prefix + "sparse_cluster_first_sum"
                    ] = SerializeSparseDoubleMap(
                        mean_calibration_cluster_first_sums[batch_index]
                    );
                    result.metadata[
                        prefix + "sparse_cluster_sum_outer"
                    ] = SerializeSparseSecondMoments(
                        mean_calibration_cluster_sum_outers[batch_index]
                    );
                    result.metadata[
                        prefix + "sparse_combined_bin_first_sum"
                    ] = SerializeSparseDoubleMap(
                        mean_calibration_combined_first_sums[batch_index]
                    );
                    result.metadata[
                        prefix + "sparse_combined_bin_sum_outer"
                    ] = SerializeSparseSecondMoments(
                        mean_calibration_combined_sum_outers[batch_index]
                    );
                    for (
                        std::size_t entry_index = 0;
                        entry_index < 3;
                        ++entry_index
                    ) {
                        std::map<int, double> class_first_sum;
                        for (
                            const auto& item
                            : mean_calibration_cluster_first_sums[
                                batch_index
                            ]
                        ) {
                            if (
                                item.first / num_bins
                                    == static_cast<int>(entry_index)
                            ) {
                                class_first_sum[item.first % num_bins]
                                    += item.second;
                            }
                        }
                        const auto entry_class = static_cast<
                            DetectorEntryClass
                        >(entry_index);
                        result.metadata[
                            prefix
                            + "sparse_cluster_first_sum_"
                            + DetectorEntryClassToken(entry_class)
                        ] = SerializeSparseDoubleMap(class_first_sum);
                    }
                } else {
                    for (
                        std::size_t entry_index = 0;
                        entry_index < 3;
                        ++entry_index
                    ) {
                        const auto entry_class = static_cast<
                            DetectorEntryClass
                        >(entry_index);
                        result.metadata[
                            prefix
                            + "sparse_entry_histogram_"
                            + DetectorEntryClassToken(entry_class)
                        ] = SerializeSparseBinCounts(
                            mean_calibration_entry_histograms[
                                batch_index
                            ][entry_index]
                        );
                    }
                }
            }
        }
        result.metadata["expected_primary_semantics"] = radioactive_decay_emission
            ? "parent_activity_bq_times_live_time"
            : (
                detector_cps_rate_model
                    ? "detector_equivalent_histories"
                    : "isotropic_physical_histories"
            );
        if (detector_cps_rate_model) {
            result.metadata["expected_detector_equivalent_primaries"] = SerializeDouble(
                expected_unthinned_primaries
            );
        } else {
            result.metadata["expected_physical_primaries"] = SerializeDouble(
                expected_unthinned_primaries
            );
        }
        result.metadata["expected_unthinned_primaries"] = SerializeDouble(
            expected_unthinned_primaries
        );
        result.metadata["expected_sampled_primaries"] = SerializeDouble(expected_sampled_primaries);
        result.metadata["primary_sampling_fraction"] = SerializeDouble(primary_sampling_fraction);
        result.metadata["primary_history_weight"] = SerializeDouble(primary_history_weight);
        result.metadata["requested_primary_sampling_fraction"] = SerializeDouble(
            requested_primary_sampling_fraction
        );
        result.metadata["target_sampled_primaries"] = std::to_string(
            options.target_sampled_primaries
        );
        result.metadata["primary_sampling_budget_enabled"] = (
            primary_sampling_budget_enabled ? "true" : "false"
        );
        result.metadata["primary_sampling_fraction_resolution"] = (
            primary_sampling_fraction_resolution
        );
        result.metadata["history_thinning_enabled"] = history_thinning_enabled ? "true" : "false";
        result.metadata["transport_history_mode"] = (
            mean_calibration_enabled
                ? "fixed_source_line_stratified_weighted_mean"
                : (
                    history_thinning_enabled
                        ? "weighted_thinning"
                        : "full_unit_weight"
                )
        );
        result.metadata["transport_tally_weighted"] = transport_tally_weighted ? "true" : "false";
        result.metadata["reference_detector_acceptance"] = std::to_string(reference_acceptance);
        result.metadata["detector_crystal_radius_m"] = std::to_string(scene_.detector.crystal_radius_m);
        result.metadata["detector_housing_thickness_m"] = std::to_string(scene_.detector.housing_thickness_m);
        result.metadata["detector_target_radius_m"] = std::to_string(DetectorTargetRadiusM(scene_.detector));
        result.metadata["fe_shield_present"] = (
            scene_.fe_shield.has_value() ? "true" : "false"
        );
        result.metadata["pb_shield_present"] = (
            scene_.pb_shield.has_value() ? "true" : "false"
        );
        result.metadata["fe_shield_thickness_m"] = SerializeDouble(
            scene_.fe_shield.has_value()
                ? scene_.fe_shield->thickness_m
                : 0.0
        );
        result.metadata["pb_shield_thickness_m"] = SerializeDouble(
            scene_.pb_shield.has_value()
                ? scene_.pb_shield->thickness_m
                : 0.0
        );
        const auto fe_shield_normal = ShieldNormalFromPose(request.fe_pose);
        const auto pb_shield_normal = ShieldNormalFromPose(request.pb_pose);
        result.metadata["native_action_contract_id"] =
            kNativeActionIdentityContractId;
        result.metadata["native_action_sha256"] = request.native_action_sha256;
        result.metadata["native_action_step_id"] = std::to_string(request.step_id);
        result.metadata["native_action_seed"] = std::to_string(request.seed);
        result.metadata["native_action_dwell_time_s"] =
            SerializeDouble(request.dwell_time_s);
        result.metadata["native_action_fe_orientation_index"] = std::to_string(
            request.fe_orientation_index
        );
        result.metadata["native_action_pb_orientation_index"] = std::to_string(
            request.pb_orientation_index
        );
        AddNativeActionPoseMetadata(result, "detector", request.detector_pose);
        AddNativeActionPoseMetadata(result, "fe_shield", request.fe_pose);
        AddNativeActionPoseMetadata(result, "pb_shield", request.pb_pose);
        result.metadata["shield_pose_contract_id"] = kShieldPoseContractId;
        result.metadata["shield_pose_contract_sha256"] =
            kShieldPoseContractSha256;
        result.metadata["fe_orientation_index"] = std::to_string(
            request.fe_orientation_index
        );
        result.metadata["pb_orientation_index"] = std::to_string(
            request.pb_orientation_index
        );
        result.metadata["fe_shield_normal_x"] = std::to_string(fe_shield_normal[0]);
        result.metadata["fe_shield_normal_y"] = std::to_string(fe_shield_normal[1]);
        result.metadata["fe_shield_normal_z"] = std::to_string(fe_shield_normal[2]);
        result.metadata["pb_shield_normal_x"] = std::to_string(pb_shield_normal[0]);
        result.metadata["pb_shield_normal_y"] = std::to_string(pb_shield_normal[1]);
        result.metadata["pb_shield_normal_z"] = std::to_string(pb_shield_normal[2]);
        AddShieldAxisMetadata(result, "fe", request.fe_pose);
        AddShieldAxisMetadata(result, "pb", request.pb_pose);
        result.metadata["total_spectrum_counts"] = std::to_string(observed_total_counts);
        result.metadata["primaries_per_sec"] = std::to_string(static_cast<double>(total_primaries) / safe_runtime_s);
        result.metadata["effective_entries_per_sec"] = std::to_string(effective_spectrum_entries / safe_runtime_s);
        result.metadata["total_track_steps"] = std::to_string(diagnostics_.TotalTrackSteps());
        result.metadata["detector_hit_events"] = std::to_string(energy_deposits.size());
        result.metadata["detector_hit_steps"] = std::to_string(diagnostics_.DetectorHitSteps());
        result.metadata["secondary_count"] = std::to_string(diagnostics_.SecondaryCount());
        result.metadata["killed_non_gamma_secondary_count"] = std::to_string(
            diagnostics_.KilledNonGammaSecondaryCount()
        );
        result.metadata["process_count_compton"] = std::to_string(compton_count);
        result.metadata["process_count_rayleigh"] = std::to_string(rayleigh_count);
        result.metadata["process_count_photoelectric"] = std::to_string(photoelectric_count);
        result.metadata["transport_process_counts"] = SerializeCounterMap(process_counts);
        result.metadata["volume_step_counts"] = SerializeCounterMap(volume_step_counts);
        result.metadata["volume_step_counts_top"] = SerializeTopCounterMap(volume_step_counts, 20);
        result.metadata["requested_threads"] = std::to_string(thread_count_);
        result.metadata["multithreaded_run_manager"] = run_manager_multithreaded_ ? "true" : "false";
        result.metadata["background_cps"] = std::to_string(options.background_cps);
        result.metadata["poisson_background"] = (
            mean_calibration_enabled ? "false" : "true"
        );
        result.metadata["validation_entry_class_spectra"] = (
            options.validation_entry_class_spectra ? "true" : "false"
        );
        if (options.validation_entry_class_spectra) {
            result.metadata["validation_entry_spectrum_space"] = (
                mean_calibration_enabled
                    ? (
                        incident_gamma_scoring
                            ? "pre_dead_time_raw_incident_gamma_weighted_mean"
                            : "pre_dead_time_raw_energy_deposition_weighted_mean"
                    )
                    : (
                        options.sample_detector_response
                    ? "pre_dead_time_raw_incident_gamma"
                    : "observed_native_histogram"
                    )
            );
            result.metadata["validation_entry_spectrum_grouping"] = (
                "source_token_initial_gamma_line_entry_class"
            );
            result.metadata["validation_observed_entry_spectrum_space"] = (
                mean_calibration_enabled
                    ? (
                        incident_gamma_scoring
                            ? "pre_dead_time_sampled_detector_response_weighted_mean"
                            : "pre_dead_time_smeared_energy_deposition_weighted_mean"
                    )
                    : "observed_native_histogram"
            );
            result.metadata[
                options.sample_detector_response
                    ? "validation_only_background_analysis_spectrum"
                    : "validation_only_background_incident_spectrum"
            ] = SerializeDoubleVector(validation_background_spectrum);
            for (const auto& item : validation_entry_spectra) {
                result.metadata[
                    "validation_only_entry_spectrum_" + item.first
                ] = SerializeDoubleVector(item.second);
            }
            for (const auto& item : validation_observed_entry_spectra) {
                result.metadata[
                    "validation_only_observed_entry_spectrum_" + item.first
                ] = SerializeDoubleVector(item.second);
            }
            if (mean_calibration_enabled) {
                for (const auto& item : mean_calibration_entry_variance) {
                    result.metadata[
                        "validation_only_entry_variance_" + item.first
                    ] = SerializeDoubleVector(item.second);
                }
            }
        }
        result.metadata["source_bias_weighted_transport"] = source_bias_weighted_transport
            ? "true"
            : "false";
        result.metadata["weighted_transport"] = transport_tally_weighted ? "true" : "false";
        result.metadata["source_bias"] = effective_source_bias_mode;
        result.metadata["source_bias_mode"] = effective_source_bias_mode;
        result.metadata["source_bias_isotropic_fraction"] = std::to_string(
            source_bias_weighted_transport
                ? std::clamp(options.source_bias_isotropic_fraction, 1.0e-6, 1.0)
                : 1.0
        );
        result.metadata["source_bias_cone_policy"] = options.source_bias_cone_policy;
        result.metadata["source_bias_effective_cone_half_angle_deg_min"] = std::to_string(
            cone_sampled_transport && std::isfinite(effective_cone_min_deg) ? effective_cone_min_deg : 0.0
        );
        result.metadata["source_bias_effective_cone_half_angle_deg_max"] = std::to_string(
            cone_sampled_transport && std::isfinite(effective_cone_max_deg) ? effective_cone_max_deg : 0.0
        );
        result.metadata["weighted_spectrum_sumw2"] = SerializeDouble(total_variance);
        result.metadata["weighted_spectrum_effective_entries"] = SerializeDouble(effective_spectrum_entries);
        result.metadata["spectrum_variance_semantics"] = (
            mean_calibration_enabled
                ? "stratified_fixed_quota_sample_mean_covariance"
                : (
                    options.sample_detector_response
                ? "renewal_total_conditional_multinomial_marks"
                : "compound_poisson_sumw2_includes_counting"
                )
        );
        result.metadata["spectrum_variance_dead_time_propagation"] = (
            mean_calibration_enabled
                ? "disabled_zero_dead_time_mean_calibration"
                : (
                    options.sample_detector_response
                        ? "event_time_nonparalyzable_global_stream"
                        : "fixed_observed_scale"
                )
        );
        result.metadata["dead_time_scale_semantics"] = (
            mean_calibration_enabled
                ? "identity_zero_dead_time_mean_calibration"
                : (
                    options.sample_detector_response
                        ? "realized_global_acceptance_fraction"
                        : "fixed_scale_from_realized_pre_dead_time_rate"
                )
        );
        result.metadata["dead_time_tau_s"] = SerializeDouble(dead_time_tau_s);
        result.metadata["dead_time_observed_scale"] = SerializeDouble(observed_scale);
        result.metadata["dwell_time_s"] = SerializeDouble(dwell_time_s);
        result.metadata["pre_dead_time_total_spectrum_counts"] = SerializeDouble(total_counts);
        result.metadata["pre_dead_time_weighted_spectrum_sumw2"] = SerializeDouble(
            pre_dead_time_total_variance
        );
        result.metadata["absorbing_volume_count"] = std::to_string(absorbing_volume_names_.size());
        result.metadata["absorbing_transport_groups"] = JoinSet(absorbing_transport_groups_, ",");
        result.metadata["persistent_process"] = persistent_process ? "true" : "false";
        result.metadata["geometry_cache_hit"] = geometry_cache_hit ? "true" : "false";
        result.metadata["movable_geometry_updated"] = (
            movable_geometry_updated ? "true" : "false"
        );
        result.metadata["run_time_s"] = std::to_string(runtime_s);
        for (const auto& item : source_equivalent_counts) {
            result.metadata["source_equivalent_counts_" + item.first] = std::to_string(item.second);
        }
        for (const auto& item : source_equivalent_counts_by_source) {
            result.metadata["source_equivalent_counts_" + item.first] = std::to_string(item.second);
        }
        for (const auto& item : scheduled_incident_gamma_counts_by_line) {
            result.metadata["scheduled_incident_gamma_counts_" + item.first] = std::to_string(item.second);
        }
        for (const auto& item : transport_detected_counts) {
            result.metadata["transport_detected_counts_" + item.first] = std::to_string(item.second);
            const double total_detected = std::max(0.0, item.second);
            const double uncollided = std::max(
                0.0,
                transport_uncollided_primary_counts[item.first]
            );
            const double interacted = std::max(
                0.0,
                transport_interacted_primary_counts[item.first]
            );
            const double secondary = std::max(0.0, transport_secondary_counts[item.first]);
            result.metadata["transport_uncollided_primary_counts_" + item.first] = std::to_string(uncollided);
            result.metadata["transport_interacted_primary_counts_" + item.first] = std::to_string(interacted);
            result.metadata["transport_secondary_counts_" + item.first] = std::to_string(secondary);
            result.metadata["transport_non_uncollided_fraction_" + item.first] = std::to_string(
                total_detected > 0.0 ? std::clamp((interacted + secondary) / total_detected, 0.0, 1.0) : 0.0
            );
        }
        for (const auto& item : transport_detected_counts_by_source) {
            result.metadata["transport_detected_counts_" + item.first] = std::to_string(item.second);
            const double total_detected = std::max(0.0, item.second);
            const double uncollided = std::max(0.0, transport_uncollided_primary_counts_by_source[item.first]);
            const double interacted = std::max(0.0, transport_interacted_primary_counts_by_source[item.first]);
            const double secondary = std::max(0.0, transport_secondary_counts_by_source[item.first]);
            result.metadata["transport_uncollided_primary_counts_" + item.first] = std::to_string(uncollided);
            result.metadata["transport_interacted_primary_counts_" + item.first] = std::to_string(interacted);
            result.metadata["transport_secondary_counts_" + item.first] = std::to_string(secondary);
            result.metadata["transport_non_uncollided_fraction_" + item.first] = std::to_string(
                total_detected > 0.0 ? std::clamp((interacted + secondary) / total_detected, 0.0, 1.0) : 0.0
            );
        }
        for (const auto& item : transport_detected_counts_by_line) {
            result.metadata["transport_detected_counts_" + item.first] = std::to_string(item.second);
            const double total_detected = std::max(0.0, item.second);
            const double uncollided = std::max(0.0, transport_uncollided_primary_counts_by_line[item.first]);
            const double interacted = std::max(0.0, transport_interacted_primary_counts_by_line[item.first]);
            const double secondary = std::max(0.0, transport_secondary_counts_by_line[item.first]);
            result.metadata["transport_uncollided_primary_counts_" + item.first] = std::to_string(uncollided);
            result.metadata["transport_interacted_primary_counts_" + item.first] = std::to_string(interacted);
            result.metadata["transport_secondary_counts_" + item.first] = std::to_string(secondary);
            result.metadata["transport_non_uncollided_fraction_" + item.first] = std::to_string(
                total_detected > 0.0 ? std::clamp((interacted + secondary) / total_detected, 0.0, 1.0) : 0.0
            );
        }
        if (!scene_.usd_path.empty()) {
            result.metadata["usd_path"] = scene_.usd_path;
        }
        return result;
    }

private:
    SceneSpec scene_;
    RequestSpec geometry_request_;
    std::string physics_profile_;
    std::string detector_scoring_mode_ = "full_transport";
    std::string secondary_transport_mode_ = "full_transport";
    std::string primary_emission_model_ = "independent_gamma_lines";
    int thread_count_ = 1;
    bool mean_calibration_forced_collision_ = false;
    bool sample_detector_response_ = false;
    std::string detector_green_operator_path_;
    std::string detector_green_operator_binary_sha256_;
    std::string detector_green_operator_contract_sha256_;
    std::unique_ptr<DetectorGreenOperator> detector_green_operator_;
    bool use_theory_tvl_ = false;
    bool run_manager_multithreaded_ = false;
    EventStore event_store_;
    TransportDiagnostics diagnostics_;
    ForceCollisionDiagnostics force_collision_diagnostics_;
    PrimarySourceState primary_state_;
    RuntimeDetectorState detector_runtime_state_;
    Geant4SceneConstruction* detector_construction_ = nullptr;
    G4RunManager* run_manager_ = nullptr;
    std::set<std::string> absorbing_volume_names_;
    std::set<std::string> absorbing_transport_groups_;
    std::set<std::string> gamma_process_names_;
    std::set<std::string> gamma_em_subprocess_names_;
};

SimulationResult RunTransport(
    const SceneSpec& scene,
    const RequestSpec& request,
    const std::string& physics_profile,
    const int thread_count,
    const double dead_time_tau_s,
    const TransportOptions& options
) {
    TransportSession session(
        scene,
        request,
        physics_profile,
        thread_count,
        options.detector_scoring_mode,
        options.secondary_transport_mode,
        options.mean_calibration_forced_collision,
        options.primary_emission_model,
        options.sample_detector_response,
        options.detector_green_operator_path,
        options.detector_green_operator_binary_sha256,
        options.detector_green_operator_contract_sha256
    );
    return session.Run(request, dead_time_tau_s, options, false, false);
}

void WriteResponseFile(const SimulationResult& result, const std::string& response_path) {
    std::ofstream output(response_path);
    if (!output) {
        throw std::runtime_error("Failed to open response file: " + response_path);
    }
    for (const auto& item : result.metadata) {
        output << "META " << item.first << "=" << item.second << "\n";
    }
    output << std::setprecision(12);
    output << "SPECTRUM ";
    for (std::size_t index = 0; index < result.spectrum_counts.size(); ++index) {
        if (index > 0) {
            output << ",";
        }
        output << result.spectrum_counts[index];
    }
    output << "\n";
    if (!result.spectrum_count_variance.empty()) {
        output << "SPECTRUM_VARIANCE ";
        for (std::size_t index = 0; index < result.spectrum_count_variance.size(); ++index) {
            if (index > 0) {
                output << ",";
            }
            output << result.spectrum_count_variance[index];
        }
        output << "\n";
    }
}

void RunPersistentServer(
    const std::string& physics_profile,
    const int thread_count,
    const double dead_time_tau_s,
    const TransportOptions& options
) {
    std::unique_ptr<TransportSession> session;
    std::string session_key;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            const auto tokens = Split(line);
            if (tokens.empty()) {
                continue;
            }
            if (tokens[0] == "SHUTDOWN" || tokens[0] == "QUIT") {
                std::cout << "SIMBRIDGE_OK shutdown" << std::endl;
                break;
            }
            if (tokens[0] != "RUN") {
                throw std::runtime_error("Unsupported persistent command: " + tokens[0]);
            }
            const auto fields = ParseFields(tokens);
            const auto scene_path = ParseString(fields, "scene");
            const auto request_path = ParseString(fields, "request");
            const auto response_path = ParseString(fields, "response");
            if (scene_path.empty() || request_path.empty() || response_path.empty()) {
                throw std::runtime_error("RUN requires scene, request, and response fields.");
            }
            const auto scene = ReadSceneFile(scene_path);
            const auto request = ReadRequestFile(request_path);
            const auto key = GeometryCacheKey(
                scene,
                physics_profile,
                thread_count,
                options.detector_scoring_mode,
                options.secondary_transport_mode,
                options.mean_calibration_forced_collision,
                options.primary_emission_model,
                options.sample_detector_response,
                options.detector_green_operator_binary_sha256,
                options.detector_green_operator_contract_sha256
            );
            const bool geometry_cache_hit = session != nullptr && key == session_key;
            if (!geometry_cache_hit) {
                session = std::make_unique<TransportSession>(
                    scene,
                    request,
                    physics_profile,
                    thread_count,
                    options.detector_scoring_mode,
                    options.secondary_transport_mode,
                    options.mean_calibration_forced_collision,
                    options.primary_emission_model,
                    options.sample_detector_response,
                    options.detector_green_operator_path,
                    options.detector_green_operator_binary_sha256,
                    options.detector_green_operator_contract_sha256
                );
                session_key = key;
            }
            auto result = session->Run(
                request,
                dead_time_tau_s,
                options,
                geometry_cache_hit,
                true
            );
            WriteResponseFile(result, response_path);
            std::cout << "SIMBRIDGE_OK response=" << response_path << std::endl;
        } catch (const std::exception& exc) {
            std::cout << "SIMBRIDGE_ERR " << exc.what() << std::endl;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string scene_path;
        std::string request_path;
        std::string response_path;
        std::string physics_profile = "balanced";
        int thread_count = 1;
        double dead_time_tau_s = 5.813e-9;
        bool persistent = false;
        TransportOptions transport_options;
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--scene" && index + 1 < argc) {
                scene_path = argv[++index];
            } else if (arg == "--request" && index + 1 < argc) {
                request_path = argv[++index];
            } else if (arg == "--response" && index + 1 < argc) {
                response_path = argv[++index];
            } else if (arg == "--physics-profile" && index + 1 < argc) {
                physics_profile = argv[++index];
            } else if (arg == "--threads" && index + 1 < argc) {
                thread_count = std::stoi(argv[++index]);
            } else if (arg == "--dead-time-tau-s" && index + 1 < argc) {
                dead_time_tau_s = std::stod(argv[++index]);
            } else if (arg == "--background-cps" && index + 1 < argc) {
                transport_options.background_cps = std::max(0.0, std::stod(argv[++index]));
            } else if (arg == "--source-rate-model" && index + 1 < argc) {
                transport_options.source_rate_model = NormalizeSourceRateModel(argv[++index]);
            } else if (
                arg == "--primary-emission-model" && index + 1 < argc
            ) {
                transport_options.primary_emission_model = (
                    NormalizePrimaryEmissionModel(argv[++index])
                );
            } else if (arg == "--source-bias-mode" && index + 1 < argc) {
                transport_options.source_bias_mode = NormalizeSourceBiasMode(argv[++index]);
            } else if (arg == "--source-bias-cone-policy" && index + 1 < argc) {
                transport_options.source_bias_cone_policy = argv[++index];
                if (transport_options.source_bias_cone_policy != "detector_covering") {
                    throw std::runtime_error(
                        "--source-bias-cone-policy must be detector_covering"
                    );
                }
            } else if (arg == "--source-bias-isotropic-fraction" && index + 1 < argc) {
                transport_options.source_bias_isotropic_fraction = std::clamp(std::stod(argv[++index]), 1.0e-6, 1.0);
            } else if (arg == "--detector-scoring-mode" && index + 1 < argc) {
                transport_options.detector_scoring_mode = NormalizeDetectorScoringMode(argv[++index]);
            } else if (arg == "--secondary-transport-mode" && index + 1 < argc) {
                transport_options.secondary_transport_mode = NormalizeSecondaryTransportMode(argv[++index]);
            } else if (arg == "--primary-sampling-fraction" && index + 1 < argc) {
                transport_options.primary_sampling_fraction = std::clamp(
                    std::stod(argv[++index]),
                    1.0e-6,
                    1.0
                );
            } else if (arg == "--target-sampled-primaries" && index + 1 < argc) {
                transport_options.target_sampled_primaries = std::stoll(argv[++index]);
                if (transport_options.target_sampled_primaries <= 0) {
                    throw std::runtime_error(
                        "--target-sampled-primaries requires a positive integer"
                    );
                }
            } else if (
                arg
                    == "--mean-calibration-histories-per-source-line"
                && index + 1 < argc
            ) {
                transport_options
                    .mean_calibration_histories_per_source_line = (
                        std::stoll(argv[++index])
                    );
                if (
                    transport_options
                        .mean_calibration_histories_per_source_line
                    <= 0
                ) {
                    throw std::runtime_error(
                        "--mean-calibration-histories-per-source-line "
                        "requires a positive integer"
                    );
                }
            } else if (
                arg == "--mean-calibration-angle-strata-mu"
                && index + 1 < argc
            ) {
                transport_options.mean_calibration_angle_strata_mu = (
                    std::stoi(argv[++index])
                );
                if (
                    transport_options.mean_calibration_angle_strata_mu
                    <= 0
                ) {
                    throw std::runtime_error(
                        "--mean-calibration-angle-strata-mu requires a "
                        "positive integer"
                    );
                }
            } else if (
                arg == "--mean-calibration-angle-strata-phi"
                && index + 1 < argc
            ) {
                transport_options.mean_calibration_angle_strata_phi = (
                    std::stoi(argv[++index])
                );
                if (
                    transport_options.mean_calibration_angle_strata_phi
                    <= 0
                ) {
                    throw std::runtime_error(
                        "--mean-calibration-angle-strata-phi requires a "
                        "positive integer"
                    );
                }
            } else if (arg == "--mean-calibration-forced-collision") {
                throw std::runtime_error(
                    "--mean-calibration-forced-collision is unavailable: "
                    "the complex-scene Geant4 branch estimator failed its "
                    "analog-mean exactness test; use fixed-quota stratified "
                    "analog calibration."
                );
            } else if (arg == "--validation-entry-class-spectra") {
                transport_options.validation_entry_class_spectra = true;
            } else if (arg == "--sample-detector-response") {
                transport_options.sample_detector_response = true;
            } else if (
                arg == "--detector-green-operator-path"
                && index + 1 < argc
            ) {
                transport_options.detector_green_operator_path = argv[++index];
            } else if (
                arg == "--detector-green-operator-binary-sha256"
                && index + 1 < argc
            ) {
                transport_options.detector_green_operator_binary_sha256 = (
                    argv[++index]
                );
            } else if (
                arg == "--detector-green-operator-contract-sha256"
                && index + 1 < argc
            ) {
                transport_options.detector_green_operator_contract_sha256 = (
                    argv[++index]
                );
            } else if (arg == "--decay-comparison-diagnostic") {
                transport_options.decay_comparison_diagnostic = true;
            } else if (
                arg == "--decay-comparison-energy-max-kev"
                && index + 1 < argc
            ) {
                transport_options.decay_comparison_energy_max_keV = (
                    std::stod(argv[++index])
                );
                transport_options.decay_comparison_energy_max_overridden = (
                    true
                );
            } else if (arg == "--persistent") {
                persistent = true;
            } else {
                throw std::runtime_error(
                    "Unsupported Geant4 sidecar option: " + arg
                );
            }
        }
        const auto normalized_source_bias_mode = NormalizeSourceBiasMode(transport_options.source_bias_mode);
        if (
            normalized_source_bias_mode != "analog"
            && normalized_source_bias_mode != "mixture_cone_isotropic"
            && normalized_source_bias_mode != "detector_cone"
        ) {
            throw std::runtime_error("Unsupported source bias mode: " + transport_options.source_bias_mode);
        }
        transport_options.source_bias_mode = normalized_source_bias_mode;
        const auto normalized_source_rate_model = NormalizeSourceRateModel(transport_options.source_rate_model);
        if (
            normalized_source_rate_model != "detector_cps_1m"
            && normalized_source_rate_model != "isotropic_emission_equivalent"
            && normalized_source_rate_model != "parent_decay_activity_bq"
        ) {
            throw std::runtime_error("Unsupported source rate model: " + transport_options.source_rate_model);
        }
        transport_options.source_rate_model = normalized_source_rate_model;
        const auto normalized_primary_emission_model = (
            NormalizePrimaryEmissionModel(
                transport_options.primary_emission_model
            )
        );
        if (
            normalized_primary_emission_model != "independent_gamma_lines"
            && normalized_primary_emission_model
                != "geant4_radioactive_decay"
        ) {
            throw std::runtime_error(
                "Unsupported primary emission model: "
                + transport_options.primary_emission_model
            );
        }
        transport_options.primary_emission_model = (
            normalized_primary_emission_model
        );
        if (
            transport_options.decay_comparison_energy_max_overridden
            && !transport_options.decay_comparison_diagnostic
        ) {
            throw std::runtime_error(
                "--decay-comparison-energy-max-kev requires "
                "--decay-comparison-diagnostic."
            );
        }
        const auto normalized_detector_scoring_mode = NormalizeDetectorScoringMode(
            transport_options.detector_scoring_mode
        );
        if (
            normalized_detector_scoring_mode != "full_transport"
            && normalized_detector_scoring_mode != "incident_gamma_energy"
        ) {
            throw std::runtime_error(
                "Unsupported detector scoring mode: "
                + transport_options.detector_scoring_mode
            );
        }
        transport_options.detector_scoring_mode = normalized_detector_scoring_mode;
        const auto normalized_secondary_transport_mode = NormalizeSecondaryTransportMode(
            transport_options.secondary_transport_mode
        );
        if (
            normalized_secondary_transport_mode != "full_transport"
            && normalized_secondary_transport_mode != "gamma_only"
        ) {
            throw std::runtime_error(
                "Unsupported secondary transport mode: "
                + transport_options.secondary_transport_mode
            );
        }
        transport_options.secondary_transport_mode = normalized_secondary_transport_mode;
        const bool detector_green_contract_complete = (
            !transport_options.detector_green_operator_path.empty()
            && IsLowercaseSha256(
                transport_options.detector_green_operator_binary_sha256
            )
            && IsLowercaseSha256(
                transport_options.detector_green_operator_contract_sha256
            )
        );
        const bool detector_green_contract_empty = (
            transport_options.detector_green_operator_path.empty()
            && transport_options.detector_green_operator_binary_sha256.empty()
            && transport_options.detector_green_operator_contract_sha256.empty()
        );
        if (
            (
                transport_options.sample_detector_response
                && !detector_green_contract_complete
            )
            || (
                !transport_options.sample_detector_response
                && !detector_green_contract_empty
            )
        ) {
            throw std::runtime_error(
                "Detector Green path and lowercase SHA-256 contracts are "
                "required exactly with --sample-detector-response."
            );
        }
        if (persistent) {
            RunPersistentServer(
                physics_profile,
                thread_count,
                dead_time_tau_s,
                transport_options
            );
            return 0;
        }
        if (scene_path.empty() || request_path.empty() || response_path.empty()) {
            throw std::runtime_error(
                "Usage: geant4_sidecar --scene <path> --request <path> --response <path> "
                "or geant4_sidecar --persistent "
                "[--validation-entry-class-spectra]"
            );
        }
        const auto scene = ReadSceneFile(scene_path);
        const auto request = ReadRequestFile(request_path);
        const auto result = RunTransport(
            scene,
            request,
            physics_profile,
            thread_count,
            dead_time_tau_s,
            transport_options
        );
        WriteResponseFile(result, response_path);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
