#include "paired_all64_phase_space.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <istream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace rotating_shield::paired_all64 {
namespace {

constexpr std::array<std::uint8_t, 8> kBankMagic = {
    'R', 'S', 'P', 'F', 'P', 'S', 'B', '3'
};
constexpr std::array<std::uint8_t, 8> kCovarianceMagic = {
    'R', 'S', 'P', 'C', 'O', 'V', '0', '2'
};
constexpr std::size_t kSha256Size = 32;
constexpr std::uint64_t kMaximumSerializedRecords = 1ULL << 34U;
constexpr const char* kReplaySeedDomain =
    "paired_all64_phase_space_replay_seed_v2";
constexpr const char* kHistoryReplaySeedDomain =
    "paired_all64_history_replay_seed_v2";
constexpr const char* kBlockDomain =
    "paired_all64_cross_pair_block_v1";
constexpr const char* kStratumAssignmentDomain =
    "paired_all64_stratum_assignment_v1";

std::uint32_t RotateRight(const std::uint32_t value, const unsigned shift) {
    return (value >> shift) | (value << (32U - shift));
}

class Sha256 {
public:
    Sha256() = default;

    void Update(const std::uint8_t* data, const std::size_t size) {
        if (finalized_) {
            throw std::logic_error("Cannot update a finalized SHA-256.");
        }
        for (std::size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == buffer_.size()) {
                Transform(buffer_.data());
                bit_count_ += 512U;
                buffer_size_ = 0;
            }
        }
    }

    void Update(const std::vector<std::uint8_t>& value) {
        Update(value.data(), value.size());
    }

    void Update(const std::string& value) {
        Update(
            reinterpret_cast<const std::uint8_t*>(value.data()),
            value.size()
        );
    }

    std::array<std::uint8_t, kSha256Size> Finalize() {
        if (finalized_) {
            return digest_;
        }
        const std::uint64_t total_bits =
            bit_count_ + static_cast<std::uint64_t>(buffer_size_) * 8U;
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56U) {
            while (buffer_size_ < buffer_.size()) {
                buffer_[buffer_size_++] = 0U;
            }
            Transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56U) {
            buffer_[buffer_size_++] = 0U;
        }
        for (int byte = 7; byte >= 0; --byte) {
            buffer_[buffer_size_++] = static_cast<std::uint8_t>(
                (total_bits >> static_cast<unsigned>(byte * 8)) & 0xffU
            );
        }
        Transform(buffer_.data());
        for (std::size_t index = 0; index < state_.size(); ++index) {
            digest_[4U * index] =
                static_cast<std::uint8_t>((state_[index] >> 24U) & 0xffU);
            digest_[4U * index + 1U] =
                static_cast<std::uint8_t>((state_[index] >> 16U) & 0xffU);
            digest_[4U * index + 2U] =
                static_cast<std::uint8_t>((state_[index] >> 8U) & 0xffU);
            digest_[4U * index + 3U] =
                static_cast<std::uint8_t>(state_[index] & 0xffU);
        }
        finalized_ = true;
        return digest_;
    }

private:
    void Transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            words[index] =
                (static_cast<std::uint32_t>(block[4U * index]) << 24U)
                | (static_cast<std::uint32_t>(block[4U * index + 1U]) << 16U)
                | (static_cast<std::uint32_t>(block[4U * index + 2U]) << 8U)
                | static_cast<std::uint32_t>(block[4U * index + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t before_15 = words[index - 15U];
            const std::uint32_t before_2 = words[index - 2U];
            const std::uint32_t sigma0 =
                RotateRight(before_15, 7U)
                ^ RotateRight(before_15, 18U)
                ^ (before_15 >> 3U);
            const std::uint32_t sigma1 =
                RotateRight(before_2, 17U)
                ^ RotateRight(before_2, 19U)
                ^ (before_2 >> 10U);
            words[index] = words[index - 16U] + sigma0
                + words[index - 7U] + sigma1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t sum1 =
                RotateRight(e, 6U) ^ RotateRight(e, 11U)
                ^ RotateRight(e, 25U);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 =
                h + sum1 + choice + constants[index] + words[index];
            const std::uint32_t sum0 =
                RotateRight(a, 2U) ^ RotateRight(a, 13U)
                ^ RotateRight(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t bit_count_ = 0;
    bool finalized_ = false;
    std::array<std::uint8_t, kSha256Size> digest_{};
};

std::array<std::uint8_t, kSha256Size> Digest(
    const std::uint8_t* data,
    const std::size_t size
) {
    Sha256 hasher;
    hasher.Update(data, size);
    return hasher.Finalize();
}

std::array<std::uint8_t, kSha256Size> Digest(const std::string& value) {
    Sha256 hasher;
    hasher.Update(value);
    return hasher.Finalize();
}

std::string HexDigest(
    const std::array<std::uint8_t, kSha256Size>& digest
) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : digest) {
        stream << std::setw(2) << static_cast<unsigned>(byte);
    }
    return stream.str();
}

bool IsLowerSha256(const std::string& value) {
    return value.size() == 64U
        && std::all_of(value.begin(), value.end(), [](const char character) {
            return (character >= '0' && character <= '9')
                || (character >= 'a' && character <= 'f');
        });
}

bool IsNonemptyAscii(const std::string& value) {
    return !value.empty()
        && std::all_of(
            value.begin(),
            value.end(),
            [](const unsigned char character) {
                return character < 0x80U;
            }
        );
}

void RequireProfile(const DedicatedProfile& profile) {
    if (profile.name != kDedicatedProfile) {
        throw std::invalid_argument(
            "Paired replay requires its dedicated calibration profile."
        );
    }
}

bool IsFiniteVector(const std::array<double, 3>& value) {
    return std::all_of(value.begin(), value.end(), [](const double component) {
        return std::isfinite(component);
    });
}

double Dot(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

double Norm(const std::array<double, 3>& value) {
    return std::sqrt(Dot(value, value));
}

std::array<double, 3> Subtract(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

void ValidateBoundary(const Boundary& boundary) {
    if (!IsFiniteVector(boundary.center_m)) {
        throw std::invalid_argument("Boundary center must be finite.");
    }
    if (!std::isfinite(boundary.radius_m) || boundary.radius_m <= 0.0) {
        throw std::invalid_argument("Boundary radius must be finite and positive.");
    }
}

void ValidateCrossing(
    const Boundary& boundary,
    const Crossing& crossing
) {
    if (crossing.branch_id == 0U) {
        throw std::invalid_argument("Crossing branch_id must be positive.");
    }
    if (crossing.particle_name.empty()) {
        throw std::invalid_argument(
            "Crossing particle_name must identify the Geant4 species."
        );
    }
    if (
        crossing.particle_name.size() > 128U
        || !std::all_of(
            crossing.particle_name.begin(),
            crossing.particle_name.end(),
            [](const unsigned char character) {
                return character >= 0x21U && character <= 0x7eU;
            }
        )
    ) {
        throw std::invalid_argument(
            "Crossing particle_name must be short printable ASCII."
        );
    }
    if (crossing.weight != 1.0) {
        throw std::invalid_argument(
            "Exact paired replay rejects weighted boundary crossings."
        );
    }
    if (
        !IsFiniteVector(crossing.position_m)
        || !IsFiniteVector(crossing.direction)
        || !IsFiniteVector(crossing.polarization)
        || !std::isfinite(crossing.kinetic_energy_mev)
        || !std::isfinite(crossing.mass_mev)
        || !std::isfinite(crossing.charge_eplus)
        || !std::isfinite(crossing.global_time_s)
        || !std::isfinite(crossing.proper_time_s)
    ) {
        throw std::invalid_argument("Crossing state must be finite.");
    }
    if (
        crossing.kinetic_energy_mev <= 0.0
        || crossing.mass_mev < 0.0
        || crossing.global_time_s < 0.0
        || crossing.proper_time_s < 0.0
    ) {
        throw std::invalid_argument(
            "Crossing energy must be positive; mass and times must be "
            "nonnegative."
        );
    }
    const double direction_norm = Norm(crossing.direction);
    if (std::abs(direction_norm - 1.0) > 1.0e-9) {
        throw std::invalid_argument("Crossing direction must be unit length.");
    }
    if (Norm(crossing.polarization) > 1.0 + 1.0e-9) {
        throw std::invalid_argument(
            "Crossing polarization magnitude cannot exceed one."
        );
    }
    const auto radial = Subtract(crossing.position_m, boundary.center_m);
    const double radial_norm = Norm(radial);
    const double boundary_tolerance =
        std::max(1.0e-8, boundary.radius_m * 1.0e-7);
    if (std::abs(radial_norm - boundary.radius_m) > boundary_tolerance) {
        throw std::invalid_argument(
            "Crossing position is not on the capture boundary."
        );
    }
    const double radial_projection = Dot(radial, crossing.direction);
    if (!(radial_projection < -boundary.radius_m * 1.0e-12)) {
        throw std::invalid_argument(
            "Capture accepts only inward boundary crossings."
        );
    }
    const std::uint32_t known_flags =
        static_cast<std::uint32_t>(InteractionFlags::kInteracted)
        | static_cast<std::uint32_t>(InteractionFlags::kSecondaryLineage);
    const auto flags = static_cast<std::uint32_t>(
        crossing.interaction_flags
    );
    if ((flags & ~known_flags) != 0U) {
        throw std::invalid_argument("Crossing interaction flags are unknown.");
    }
    const bool interacted = HasInteractionFlag(
        crossing.interaction_flags,
        InteractionFlags::kInteracted
    );
    if (interacted != (crossing.gamma_interaction_count > 0U)) {
        throw std::invalid_argument(
            "Interaction flag disagrees with the interaction count."
        );
    }
    const bool secondary = HasInteractionFlag(
        crossing.interaction_flags,
        InteractionFlags::kSecondaryLineage
    );
    if (crossing.generation == 0U) {
        if (crossing.parent_branch_id != 0U || secondary) {
            throw std::invalid_argument(
                "Primary branch lineage is internally inconsistent."
            );
        }
    } else if (crossing.parent_branch_id == 0U || !secondary) {
        throw std::invalid_argument(
            "Secondary branch lineage is internally inconsistent."
        );
    }
}

void ValidateCanonicalBank(const Bank& bank, const bool require_nonempty) {
    ValidateBoundary(bank.boundary);
    if (require_nonempty && bank.histories.empty()) {
        throw std::invalid_argument(
            "Paired phase-space bank must contain at least one history."
        );
    }
    std::uint64_t prior_history = 0U;
    bool first_history = true;
    std::uint64_t crossing_count = 0U;
    using GroupKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    using SourceLineKey = std::pair<std::uint32_t, std::uint32_t>;
    struct GroupValidation {
        std::uint32_t angle_stratum_count = 0U;
        std::size_t history_count = 0U;
        double estimator_coefficient = 0.0;
    };
    std::map<GroupKey, GroupValidation> groups;
    std::map<SourceLineKey, std::uint32_t> stratum_count_by_source_line;
    std::map<SourceLineKey, std::set<std::uint32_t>>
        strata_by_source_line;
    for (const auto& history : bank.histories) {
        if (
            !first_history
            && history.original_history_id <= prior_history
        ) {
            throw std::invalid_argument(
                "Bank histories must be uniquely sorted by original ID."
            );
        }
        first_history = false;
        prior_history = history.original_history_id;
        if (
            history.angle_stratum_count == 0U
            || history.angle_stratum_index >= history.angle_stratum_count
            || !std::isfinite(history.estimator_coefficient)
            || history.estimator_coefficient <= 0.0
        ) {
            throw std::invalid_argument(
                "History stratum identity or estimator coefficient is "
                "invalid."
            );
        }
        const SourceLineKey source_line{
            history.source_index,
            history.line_index,
        };
        const auto [count_entry, inserted_count] =
            stratum_count_by_source_line.emplace(
                source_line,
                history.angle_stratum_count
            );
        if (
            !inserted_count
            && count_entry->second != history.angle_stratum_count
        ) {
            throw std::invalid_argument(
                "One source-line schedule declares inconsistent angle "
                "stratum counts."
            );
        }
        strata_by_source_line[source_line].insert(
            history.angle_stratum_index
        );
        const GroupKey group_key{
            history.source_index,
            history.line_index,
            history.angle_stratum_index,
        };
        const auto [group_entry, inserted_group] = groups.emplace(
            group_key,
            GroupValidation{
                history.angle_stratum_count,
                0U,
                history.estimator_coefficient,
            }
        );
        if (
            !inserted_group
            && (
                group_entry->second.angle_stratum_count
                    != history.angle_stratum_count
                || group_entry->second.estimator_coefficient
                    != history.estimator_coefficient
            )
        ) {
            throw std::invalid_argument(
                "Histories in one source-line-stratum group must share the "
                "same authenticated estimator coefficient."
            );
        }
        group_entry->second.history_count += 1U;
        std::uint64_t prior_branch = 0U;
        bool first_branch = true;
        for (const auto& crossing : history.crossings) {
            ValidateCrossing(bank.boundary, crossing);
            if (
                crossing.source_index != history.source_index
                || crossing.line_index != history.line_index
            ) {
                throw std::invalid_argument(
                    "Crossing source-line label differs from its history."
                );
            }
            if (!first_branch && crossing.branch_id <= prior_branch) {
                throw std::invalid_argument(
                    "Crossings must be uniquely sorted by branch ID."
                );
            }
            first_branch = false;
            prior_branch = crossing.branch_id;
            ++crossing_count;
            if (crossing_count > kMaximumSerializedRecords) {
                throw std::invalid_argument(
                    "Bank crossing count exceeds the supported bound."
                );
            }
        }
    }
    if (require_nonempty) {
        for (const auto& [source_line, declared_count]
             : stratum_count_by_source_line) {
            const auto& observed = strata_by_source_line.at(source_line);
            if (observed.size() != declared_count) {
                throw std::invalid_argument(
                    "A complete bank must contain every declared angle "
                    "stratum for each source-line schedule."
                );
            }
            std::optional<std::size_t> quota;
            for (std::uint32_t stratum = 0U; stratum < declared_count;
                 ++stratum) {
                if (observed.count(stratum) == 0U) {
                    throw std::invalid_argument(
                        "A complete bank has a gap in its angle strata."
                    );
                }
                const auto group = groups.find(GroupKey{
                    source_line.first,
                    source_line.second,
                    stratum,
                });
                if (group == groups.end()) {
                    throw std::logic_error(
                        "Validated stratum has no group record."
                    );
                }
                if (!quota.has_value()) {
                    quota = group->second.history_count;
                } else if (*quota != group->second.history_count) {
                    throw std::invalid_argument(
                        "Fixed-quota capture requires equal history counts "
                        "across angle strata of one source line."
                    );
                }
            }
        }
    }
}

void SortCanonical(Bank& bank) {
    std::sort(
        bank.histories.begin(),
        bank.histories.end(),
        [](const History& left, const History& right) {
            return left.original_history_id < right.original_history_id;
        }
    );
    for (auto& history : bank.histories) {
        std::sort(
            history.crossings.begin(),
            history.crossings.end(),
            [](const Crossing& left, const Crossing& right) {
                return left.branch_id < right.branch_id;
            }
        );
    }
}

template <typename Integer>
void AppendLittleEndian(
    std::vector<std::uint8_t>& output,
    const Integer value
) {
    static_assert(std::is_integral_v<Integer>, "Integer type required.");
    using Unsigned = std::make_unsigned_t<Integer>;
    const Unsigned converted = static_cast<Unsigned>(value);
    for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
        output.push_back(static_cast<std::uint8_t>(
            (converted >> (8U * byte)) & static_cast<Unsigned>(0xffU)
        ));
    }
}

void AppendDouble(
    std::vector<std::uint8_t>& output,
    const double value
) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    AppendLittleEndian(output, bits);
}

void AppendString(
    std::vector<std::uint8_t>& output,
    const std::string& value
) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("String is too large to serialize.");
    }
    AppendLittleEndian(output, static_cast<std::uint32_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
}

class Reader {
public:
    Reader(
        const std::vector<std::uint8_t>& payload,
        const std::size_t limit
    ) : payload_(payload), limit_(limit) {}

    template <typename Integer>
    Integer ReadInteger() {
        static_assert(std::is_integral_v<Integer>, "Integer type required.");
        Require(sizeof(Integer));
        using Unsigned = std::make_unsigned_t<Integer>;
        Unsigned value = 0U;
        for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
            value |= static_cast<Unsigned>(payload_[position_++])
                << static_cast<unsigned>(8U * byte);
        }
        return static_cast<Integer>(value);
    }

    double ReadDouble() {
        const std::uint64_t bits = ReadInteger<std::uint64_t>();
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::array<double, 3> ReadVector() {
        return {ReadDouble(), ReadDouble(), ReadDouble()};
    }

    std::string ReadString() {
        const auto size = ReadInteger<std::uint32_t>();
        Require(size);
        const auto begin = payload_.begin()
            + static_cast<std::ptrdiff_t>(position_);
        position_ += size;
        return std::string(
            begin,
            begin + static_cast<std::ptrdiff_t>(size)
        );
    }

    std::size_t Position() const noexcept {
        return position_;
    }

    void Require(const std::size_t size) const {
        if (size > limit_ - std::min(position_, limit_)) {
            throw std::invalid_argument("Phase-space bank payload is truncated.");
        }
    }

private:
    const std::vector<std::uint8_t>& payload_;
    std::size_t limit_;
    std::size_t position_ = 0U;
};

std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output << "\\u00" << std::hex << std::setw(2)
                           << std::setfill('0')
                           << static_cast<unsigned>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string TypedHistoryIdentity(const std::uint64_t history_id) {
    return "int:" + std::to_string(history_id);
}

std::uint64_t First63Bits(
    const std::array<std::uint8_t, kSha256Size>& digest
) {
    std::uint64_t value = 0U;
    for (std::size_t index = 0; index < 8U; ++index) {
        value = (value << 8U) | static_cast<std::uint64_t>(digest[index]);
    }
    return value & ((1ULL << 63U) - 1ULL);
}

std::string ReplaySeedIdentityJson(
    const std::uint64_t root_seed,
    const std::string& bank_payload_sha256,
    const std::uint32_t shield_pair_id
) {
    std::ostringstream output;
    output
        << "{\"bank_payload_sha256\":\"" << bank_payload_sha256
        << "\",\"domain\":\"" << kReplaySeedDomain
        << "\",\"profile\":\"" << kDedicatedProfile
        << "\",\"root_seed\":" << root_seed
        << ",\"schema_version\":" << kBankSchemaVersion
        << ",\"shield_pair_id\":" << shield_pair_id << "}\n";
    return output.str();
}

std::size_t ScoreIndex(
    const std::size_t pair,
    const std::size_t history,
    const std::size_t feature,
    const std::size_t history_count,
    const std::size_t feature_count
) {
    return (pair * history_count + history) * feature_count + feature;
}

std::size_t PairFeatureIndex(
    const std::size_t pair,
    const std::size_t feature,
    const std::size_t feature_count
) {
    return pair * feature_count + feature;
}

std::size_t FactorIndex(
    const std::size_t block,
    const std::size_t pair,
    const std::size_t feature,
    const std::size_t feature_count
) {
    return (
        block * kShieldPairCount * feature_count
        + pair * feature_count
        + feature
    );
}

std::size_t GroupPairFeatureIndex(
    const std::size_t group,
    const std::size_t pair,
    const std::size_t feature,
    const std::size_t feature_count
) {
    return (
        group * kShieldPairCount * feature_count
        + pair * feature_count
        + feature
    );
}

std::size_t HistoryFactorIndex(
    const std::size_t history,
    const std::size_t pair,
    const std::size_t feature,
    const std::size_t feature_count
) {
    return (
        history * kShieldPairCount * feature_count
        + pair * feature_count
        + feature
    );
}

std::string CovarianceHeaderJson(
    const CrossPairStratifiedCovariance& artifact
) {
    std::ostringstream output;
    output
        << "{\"centered_factor_shape\":["
        << artifact.history_count << ",64," << artifact.feature_count << "]"
        << ",\"estimate_shape\":[64," << artifact.feature_count << "]"
        << ",\"feature_count\":" << artifact.feature_count
        << ",\"first_sum_shape\":["
        << artifact.group_count << ",64," << artifact.feature_count << "]"
        << ",\"group_assignment_sha256\":\""
        << artifact.group_assignment_sha256
        << "\",\"group_count\":" << artifact.group_count
        << ",\"history_count\":" << artifact.history_count
        << ",\"pair_ids\":[";
    for (std::size_t pair = 0; pair < kShieldPairCount; ++pair) {
        if (pair != 0U) {
            output << ",";
        }
        output << pair;
    }
    output
        << "],\"schema_version\":2"
        << ",\"score_semantics\":\""
        << JsonEscape(artifact.score_semantics)
        << "\",\"semantics\":\"" << kCovarianceSemantics
        << "\",\"total_cross_pair_covariance_shape\":[64,64]}\n";
    return output.str();
}

std::string CovarianceArtifactHash(
    const CrossPairStratifiedCovariance& artifact
) {
    Sha256 hasher;
    hasher.Update(CovarianceHeaderJson(artifact));
    std::vector<std::uint8_t> integer_bytes;
    integer_bytes.reserve(
        artifact.original_history_ids.size() * sizeof(std::uint64_t)
        + artifact.history_group_indices.size() * sizeof(std::uint32_t)
        + artifact.groups.size() * 6U * sizeof(std::uint64_t)
    );
    for (const auto history_id : artifact.original_history_ids) {
        AppendLittleEndian(integer_bytes, history_id);
    }
    for (const auto group_index : artifact.history_group_indices) {
        AppendLittleEndian(integer_bytes, group_index);
    }
    for (const auto& group : artifact.groups) {
        AppendLittleEndian(integer_bytes, group.source_index);
        AppendLittleEndian(integer_bytes, group.line_index);
        AppendLittleEndian(integer_bytes, group.angle_stratum_index);
        AppendLittleEndian(integer_bytes, group.angle_stratum_count);
        AppendLittleEndian(
            integer_bytes,
            static_cast<std::uint64_t>(group.history_count)
        );
        AppendDouble(integer_bytes, group.estimator_coefficient);
    }
    hasher.Update(integer_bytes);
    auto update_doubles = [&hasher](const auto& values) {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(values.size() * sizeof(double));
        for (const double value : values) {
            AppendDouble(bytes, value);
        }
        hasher.Update(bytes);
    };
    update_doubles(artifact.estimate_by_pair_feature);
    update_doubles(artifact.first_sum_by_group_pair_feature);
    update_doubles(artifact.centered_factor_by_history);
    update_doubles(artifact.total_cross_pair_covariance);
    return HexDigest(hasher.Finalize());
}

}  // namespace

InteractionFlags operator|(
    const InteractionFlags left,
    const InteractionFlags right
) {
    return static_cast<InteractionFlags>(
        static_cast<std::uint32_t>(left)
        | static_cast<std::uint32_t>(right)
    );
}

bool HasInteractionFlag(
    const InteractionFlags value,
    const InteractionFlags flag
) {
    return (
        static_cast<std::uint32_t>(value)
        & static_cast<std::uint32_t>(flag)
    ) != 0U;
}

DedicatedProfile RequireDedicatedProfile(
    const std::string& profile,
    const bool standard_runtime
) {
    if (standard_runtime) {
        throw std::invalid_argument(
            "Paired all-64 replay is not selectable by standard runtime."
        );
    }
    if (profile != kDedicatedProfile) {
        throw std::invalid_argument(
            "Unknown paired all-64 calibration profile."
        );
    }
    return DedicatedProfile{profile};
}

CaptureAccumulator::CaptureAccumulator(const Boundary boundary)
    : boundary_(boundary) {
    ValidateBoundary(boundary_);
}

void CaptureAccumulator::RegisterHistory(
    const std::uint64_t original_history_id,
    const std::uint32_t source_index,
    const std::uint32_t line_index,
    const std::uint32_t angle_stratum_index,
    const std::uint32_t angle_stratum_count,
    const double estimator_coefficient
) {
    if (
        angle_stratum_count == 0U
        || angle_stratum_index >= angle_stratum_count
        || !std::isfinite(estimator_coefficient)
        || estimator_coefficient <= 0.0
    ) {
        throw std::invalid_argument(
            "History registration requires a valid stratum and positive "
            "external estimator coefficient."
        );
    }
    if (history_index_.find(original_history_id) != history_index_.end()) {
        throw std::invalid_argument(
            "Original history was registered more than once in one worker."
        );
    }
    history_index_.emplace(original_history_id, histories_.size());
    captured_branches_.emplace(
        original_history_id,
        std::unordered_set<std::uint64_t>{}
    );
    histories_.push_back(History{
        original_history_id,
        source_index,
        line_index,
        angle_stratum_index,
        angle_stratum_count,
        estimator_coefficient,
        {},
    });
}

CaptureResult CaptureAccumulator::RecordFirstInwardCrossing(
    const std::uint64_t original_history_id,
    const Crossing& crossing
) {
    const auto history_entry = history_index_.find(original_history_id);
    if (history_entry == history_index_.end()) {
        throw std::invalid_argument(
            "Crossing belongs to an unregistered original history."
        );
    }
    ValidateCrossing(boundary_, crossing);
    const auto& history = histories_.at(history_entry->second);
    if (
        crossing.source_index != history.source_index
        || crossing.line_index != history.line_index
    ) {
        throw std::invalid_argument(
            "Crossing source-line identity differs from its history."
        );
    }
    auto& branches = captured_branches_.at(original_history_id);
    if (!branches.insert(crossing.branch_id).second) {
        return CaptureResult::kAlreadyCaptured;
    }
    histories_.at(history_entry->second).crossings.push_back(crossing);
    return CaptureResult::kCaptured;
}

const Boundary& CaptureAccumulator::GetBoundary() const noexcept {
    return boundary_;
}

std::size_t CaptureAccumulator::HistoryCount() const noexcept {
    return histories_.size();
}

std::size_t CaptureAccumulator::CrossingCount() const noexcept {
    return std::accumulate(
        histories_.begin(),
        histories_.end(),
        std::size_t{0},
        [](const std::size_t total, const History& history) {
            return total + history.crossings.size();
        }
    );
}

Bank CaptureAccumulator::Finalize() const {
    Bank bank{boundary_, histories_};
    SortCanonical(bank);
    ValidateCanonicalBank(bank, false);
    return bank;
}

Bank MergeWorkerBanks(
    const DedicatedProfile& profile,
    const std::vector<Bank>& worker_banks
) {
    RequireProfile(profile);
    if (worker_banks.empty()) {
        throw std::invalid_argument("At least one worker bank is required.");
    }
    const Boundary boundary = worker_banks.front().boundary;
    ValidateBoundary(boundary);
    Bank merged{boundary, {}};
    std::size_t total_histories = 0U;
    for (const auto& worker : worker_banks) {
        ValidateCanonicalBank(worker, false);
        if (
            worker.boundary.center_m != boundary.center_m
            || worker.boundary.radius_m != boundary.radius_m
        ) {
            throw std::invalid_argument(
                "Worker phase-space bank boundaries differ."
            );
        }
        total_histories += worker.histories.size();
    }
    merged.histories.reserve(total_histories);
    for (const auto& worker : worker_banks) {
        merged.histories.insert(
            merged.histories.end(),
            worker.histories.begin(),
            worker.histories.end()
        );
    }
    SortCanonical(merged);
    ValidateCanonicalBank(merged, true);
    return merged;
}

std::vector<std::uint8_t> SerializeBank(
    const DedicatedProfile& profile,
    const Bank& bank
) {
    RequireProfile(profile);
    ValidateCanonicalBank(bank, true);
    std::uint64_t crossing_count = 0U;
    for (const auto& history : bank.histories) {
        crossing_count += history.crossings.size();
    }
    std::vector<std::uint8_t> payload;
    payload.reserve(
        80U + bank.histories.size() * 40U
        + static_cast<std::size_t>(crossing_count) * 160U
    );
    payload.insert(payload.end(), kBankMagic.begin(), kBankMagic.end());
    AppendLittleEndian(payload, kBankSchemaVersion);
    AppendLittleEndian(payload, std::uint32_t{0});
    for (const double component : bank.boundary.center_m) {
        AppendDouble(payload, component);
    }
    AppendDouble(payload, bank.boundary.radius_m);
    AppendLittleEndian(
        payload,
        static_cast<std::uint64_t>(bank.histories.size())
    );
    AppendLittleEndian(payload, crossing_count);
    for (const auto& history : bank.histories) {
        AppendLittleEndian(payload, history.original_history_id);
        AppendLittleEndian(payload, history.source_index);
        AppendLittleEndian(payload, history.line_index);
        AppendLittleEndian(payload, history.angle_stratum_index);
        AppendLittleEndian(payload, history.angle_stratum_count);
        AppendDouble(payload, history.estimator_coefficient);
        AppendLittleEndian(
            payload,
            static_cast<std::uint64_t>(history.crossings.size())
        );
        for (const auto& crossing : history.crossings) {
            AppendLittleEndian(payload, crossing.branch_id);
            AppendLittleEndian(payload, crossing.parent_branch_id);
            AppendLittleEndian(payload, crossing.source_index);
            AppendLittleEndian(payload, crossing.line_index);
            AppendLittleEndian(payload, crossing.pdg_code);
            AppendString(payload, crossing.particle_name);
            AppendLittleEndian(payload, crossing.generation);
            AppendLittleEndian(payload, crossing.gamma_interaction_count);
            AppendLittleEndian(
                payload,
                static_cast<std::uint32_t>(crossing.interaction_flags)
            );
            for (const double value : crossing.position_m) {
                AppendDouble(payload, value);
            }
            for (const double value : crossing.direction) {
                AppendDouble(payload, value);
            }
            for (const double value : crossing.polarization) {
                AppendDouble(payload, value);
            }
            AppendDouble(payload, crossing.kinetic_energy_mev);
            AppendDouble(payload, crossing.mass_mev);
            AppendDouble(payload, crossing.charge_eplus);
            AppendDouble(payload, crossing.global_time_s);
            AppendDouble(payload, crossing.proper_time_s);
            AppendDouble(payload, crossing.weight);
        }
    }
    const auto checksum = Digest(payload.data(), payload.size());
    payload.insert(payload.end(), checksum.begin(), checksum.end());
    return payload;
}

Bank DeserializeBank(
    const DedicatedProfile& profile,
    const std::vector<std::uint8_t>& payload
) {
    RequireProfile(profile);
    constexpr std::size_t minimum_size =
        8U + 4U + 4U + 4U * 8U + 8U + 8U + kSha256Size;
    if (payload.size() < minimum_size) {
        throw std::invalid_argument("Phase-space bank payload is truncated.");
    }
    const std::size_t body_size = payload.size() - kSha256Size;
    const auto checksum = Digest(payload.data(), body_size);
    if (!std::equal(
        checksum.begin(),
        checksum.end(),
        payload.begin() + static_cast<std::ptrdiff_t>(body_size)
    )) {
        throw std::invalid_argument(
            "Phase-space bank payload checksum does not match."
        );
    }
    if (!std::equal(kBankMagic.begin(), kBankMagic.end(), payload.begin())) {
        throw std::invalid_argument("Phase-space bank magic is invalid.");
    }
    Reader reader(payload, body_size);
    for (std::size_t index = 0; index < kBankMagic.size(); ++index) {
        if (reader.ReadInteger<std::uint8_t>() != kBankMagic[index]) {
            throw std::invalid_argument("Phase-space bank magic is invalid.");
        }
    }
    if (reader.ReadInteger<std::uint32_t>() != kBankSchemaVersion) {
        throw std::invalid_argument(
            "Unsupported phase-space bank schema version."
        );
    }
    if (reader.ReadInteger<std::uint32_t>() != 0U) {
        throw std::invalid_argument(
            "Phase-space bank reserved flags must be zero."
        );
    }
    Bank bank;
    bank.boundary.center_m = reader.ReadVector();
    bank.boundary.radius_m = reader.ReadDouble();
    const auto history_count = reader.ReadInteger<std::uint64_t>();
    const auto expected_crossing_count = reader.ReadInteger<std::uint64_t>();
    if (
        history_count == 0U
        || history_count > kMaximumSerializedRecords
        || expected_crossing_count > kMaximumSerializedRecords
    ) {
        throw std::invalid_argument(
            "Phase-space bank record counts are outside supported bounds."
        );
    }
    constexpr std::uint64_t history_record_size = 40U;
    constexpr std::uint64_t crossing_record_size = 164U;
    const std::uint64_t remaining_bytes = static_cast<std::uint64_t>(
        body_size - reader.Position()
    );
    if (
        history_count > remaining_bytes / history_record_size
        || expected_crossing_count
            > (
                remaining_bytes
                - history_count * history_record_size
            ) / crossing_record_size
    ) {
        throw std::invalid_argument(
            "Phase-space bank counts exceed the payload byte length."
        );
    }
    bank.histories.reserve(static_cast<std::size_t>(history_count));
    std::uint64_t crossing_count = 0U;
    for (std::uint64_t history_index = 0; history_index < history_count;
         ++history_index) {
        History history;
        history.original_history_id =
            reader.ReadInteger<std::uint64_t>();
        history.source_index = reader.ReadInteger<std::uint32_t>();
        history.line_index = reader.ReadInteger<std::uint32_t>();
        history.angle_stratum_index =
            reader.ReadInteger<std::uint32_t>();
        history.angle_stratum_count =
            reader.ReadInteger<std::uint32_t>();
        history.estimator_coefficient = reader.ReadDouble();
        const auto event_crossing_count =
            reader.ReadInteger<std::uint64_t>();
        if (
            event_crossing_count > expected_crossing_count - crossing_count
        ) {
            throw std::invalid_argument(
                "Event crossing count exceeds the bank total."
            );
        }
        history.crossings.reserve(
            static_cast<std::size_t>(event_crossing_count)
        );
        for (std::uint64_t crossing_index = 0;
             crossing_index < event_crossing_count;
             ++crossing_index) {
            Crossing crossing;
            crossing.branch_id = reader.ReadInteger<std::uint64_t>();
            crossing.parent_branch_id =
                reader.ReadInteger<std::uint64_t>();
            crossing.source_index = reader.ReadInteger<std::uint32_t>();
            crossing.line_index = reader.ReadInteger<std::uint32_t>();
            crossing.pdg_code = reader.ReadInteger<std::int32_t>();
            crossing.particle_name = reader.ReadString();
            crossing.generation = reader.ReadInteger<std::uint32_t>();
            crossing.gamma_interaction_count =
                reader.ReadInteger<std::uint32_t>();
            crossing.interaction_flags = static_cast<InteractionFlags>(
                reader.ReadInteger<std::uint32_t>()
            );
            crossing.position_m = reader.ReadVector();
            crossing.direction = reader.ReadVector();
            crossing.polarization = reader.ReadVector();
            crossing.kinetic_energy_mev = reader.ReadDouble();
            crossing.mass_mev = reader.ReadDouble();
            crossing.charge_eplus = reader.ReadDouble();
            crossing.global_time_s = reader.ReadDouble();
            crossing.proper_time_s = reader.ReadDouble();
            crossing.weight = reader.ReadDouble();
            history.crossings.push_back(crossing);
            ++crossing_count;
        }
        bank.histories.push_back(std::move(history));
    }
    if (crossing_count != expected_crossing_count) {
        throw std::invalid_argument(
            "Parsed crossing count differs from the bank header."
        );
    }
    if (reader.Position() != body_size) {
        throw std::invalid_argument(
            "Phase-space bank contains unknown trailing fields."
        );
    }
    ValidateCanonicalBank(bank, true);
    return bank;
}

void WriteBank(
    const DedicatedProfile& profile,
    const Bank& bank,
    std::ostream& output
) {
    const auto payload = SerializeBank(profile, bank);
    output.write(
        reinterpret_cast<const char*>(payload.data()),
        static_cast<std::streamsize>(payload.size())
    );
    if (!output) {
        throw std::runtime_error("Failed to write phase-space bank payload.");
    }
}

Bank ReadBank(
    const DedicatedProfile& profile,
    std::istream& input
) {
    const std::vector<char> raw_payload(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>{}
    );
    if (!input.eof() && input.fail()) {
        throw std::runtime_error("Failed to read phase-space bank payload.");
    }
    const std::vector<std::uint8_t> payload(
        raw_payload.begin(),
        raw_payload.end()
    );
    return DeserializeBank(profile, payload);
}

std::string BankPayloadSha256(
    const DedicatedProfile& profile,
    const Bank& bank
) {
    const auto payload = SerializeBank(profile, bank);
    return HexDigest(Digest(payload.data(), payload.size()));
}

std::uint64_t DeriveReplaySeed(
    const std::uint64_t root_seed,
    const std::string& bank_payload_sha256,
    const std::uint32_t shield_pair_id
) {
    if (!IsLowerSha256(bank_payload_sha256)) {
        throw std::invalid_argument(
            "Bank payload identity must be a lowercase SHA-256."
        );
    }
    if (root_seed >= (1ULL << 63U)) {
        throw std::invalid_argument("Root seed must fit signed 63 bits.");
    }
    if (shield_pair_id >= kShieldPairCount) {
        throw std::invalid_argument("Shield pair ID must be in [0, 63].");
    }
    return First63Bits(Digest(ReplaySeedIdentityJson(
        root_seed,
        bank_payload_sha256,
        shield_pair_id
    )));
}

std::uint64_t DeriveHistoryReplaySeed(
    const std::uint64_t replay_seed,
    const std::uint64_t original_history_id
) {
    if (replay_seed >= (1ULL << 63U)) {
        throw std::invalid_argument("Replay seed must fit signed 63 bits.");
    }
    const std::string identity =
        std::string(kHistoryReplaySeedDomain)
        + "|" + std::to_string(replay_seed)
        + "|int:" + std::to_string(original_history_id);
    const std::uint64_t derived = First63Bits(Digest(identity));
    return derived == 0U ? 1U : derived;
}

ReplaySchedule::ReplaySchedule(
    const DedicatedProfile& profile,
    const Bank& bank,
    const std::uint32_t shield_pair_id,
    const std::uint64_t replay_seed
) : shield_pair_id_(shield_pair_id) {
    RequireProfile(profile);
    ValidateCanonicalBank(bank, true);
    if (shield_pair_id >= kShieldPairCount) {
        throw std::invalid_argument("Shield pair ID must be in [0, 63].");
    }
    events_.reserve(bank.histories.size());
    for (const auto& history : bank.histories) {
        events_.push_back(ReplayEvent{
            history.original_history_id,
            history.source_index,
            history.line_index,
            history.angle_stratum_index,
            history.angle_stratum_count,
            history.estimator_coefficient,
            DeriveHistoryReplaySeed(
                replay_seed,
                history.original_history_id
            ),
            history.crossings,
        });
    }
}

std::size_t ReplaySchedule::EventCount() const noexcept {
    return events_.size();
}

const ReplayEvent& ReplaySchedule::Event(const std::size_t index) const {
    return events_.at(index);
}

std::uint32_t ReplaySchedule::ShieldPairId() const noexcept {
    return shield_pair_id_;
}

PairedScoreAccumulator::PairedScoreAccumulator(
    const DedicatedProfile& profile,
    std::vector<HistoryEstimatorIdentity> history_identities,
    const std::size_t feature_count
) : history_identities_(std::move(history_identities)),
    feature_count_(feature_count) {
    RequireProfile(profile);
    if (history_identities_.empty()) {
        throw std::invalid_argument("At least one history score is required.");
    }
    if (feature_count_ == 0U) {
        throw std::invalid_argument("Feature count must be positive.");
    }
    if (!std::is_sorted(
        history_identities_.begin(),
        history_identities_.end(),
        [](const auto& left, const auto& right) {
            return left.original_history_id < right.original_history_id;
        }
    )) {
        throw std::invalid_argument(
            "Original history IDs must be in canonical sorted order."
        );
    }
    using GroupKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    using SourceLineKey = std::pair<std::uint32_t, std::uint32_t>;
    std::map<GroupKey, std::pair<std::size_t, double>> group_counts;
    std::map<SourceLineKey, std::uint32_t> stratum_counts;
    std::map<SourceLineKey, std::set<std::uint32_t>> observed_strata;
    for (std::size_t index = 0U; index < history_identities_.size(); ++index) {
        const auto& identity = history_identities_[index];
        if (
            (index > 0U
                && identity.original_history_id
                    == history_identities_[index - 1U].original_history_id)
            || identity.angle_stratum_count == 0U
            || identity.angle_stratum_index >= identity.angle_stratum_count
            || !std::isfinite(identity.estimator_coefficient)
            || identity.estimator_coefficient <= 0.0
        ) {
            throw std::invalid_argument(
                "History estimator identities are duplicated or invalid."
            );
        }
        const GroupKey key{
            identity.source_index,
            identity.line_index,
            identity.angle_stratum_index,
        };
        const SourceLineKey source_line{
            identity.source_index,
            identity.line_index,
        };
        const auto [stratum_count_it, stratum_count_inserted] =
            stratum_counts.emplace(
                source_line,
                identity.angle_stratum_count
            );
        if (
            !stratum_count_inserted
            && stratum_count_it->second != identity.angle_stratum_count
        ) {
            throw std::invalid_argument(
                "One source-line schedule cannot declare inconsistent "
                "angle stratum counts."
            );
        }
        observed_strata[source_line].insert(
            identity.angle_stratum_index
        );
        const auto [entry, inserted] = group_counts.emplace(
            key,
            std::make_pair(
                std::size_t{0},
                identity.estimator_coefficient
            )
        );
        if (
            !inserted
            && entry->second.second != identity.estimator_coefficient
        ) {
            throw std::invalid_argument(
                "One stratum cannot mix estimator coefficients."
            );
        }
        entry->second.first += 1U;
    }
    if (std::any_of(
        group_counts.begin(),
        group_counts.end(),
        [](const auto& item) {
            return item.second.first < 2U;
        }
    )) {
        throw std::invalid_argument(
            "Exact stratified covariance requires at least two original "
            "histories in every source-line-angle stratum."
        );
    }
    for (const auto& [source_line, stratum_count] : stratum_counts) {
        const auto& strata = observed_strata.at(source_line);
        if (strata.size() != stratum_count) {
            throw std::invalid_argument(
                "Exact covariance requires every declared angle stratum."
            );
        }
        std::optional<std::size_t> fixed_quota;
        for (std::uint32_t stratum = 0U; stratum < stratum_count; ++stratum) {
            const auto group = group_counts.find(GroupKey{
                source_line.first,
                source_line.second,
                stratum,
            });
            if (group == group_counts.end()) {
                throw std::invalid_argument(
                    "Exact covariance angle strata must be contiguous."
                );
            }
            if (!fixed_quota.has_value()) {
                fixed_quota = group->second.first;
            } else if (*fixed_quota != group->second.first) {
                throw std::invalid_argument(
                    "Exact covariance requires fixed history quota across "
                    "angle strata of each source line."
                );
            }
        }
    }
    if (
        feature_count_
        > std::numeric_limits<std::size_t>::max()
            / history_identities_.size() / kShieldPairCount
    ) {
        throw std::overflow_error("Paired score matrix is too large.");
    }
    scores_.assign(
        kShieldPairCount * history_identities_.size() * feature_count_,
        0.0
    );
}

void PairedScoreAccumulator::SubmitPairScores(
    const std::uint32_t shield_pair_id,
    const std::vector<double>& history_feature_scores
) {
    if (shield_pair_id >= kShieldPairCount) {
        throw std::invalid_argument("Shield pair ID must be in [0, 63].");
    }
    if (submitted_.at(shield_pair_id)) {
        throw std::invalid_argument(
            "Shield pair scores were submitted more than once."
        );
    }
    const std::size_t expected =
        history_identities_.size() * feature_count_;
    if (history_feature_scores.size() != expected) {
        throw std::invalid_argument(
            "Pair score matrix shape differs from the declared history/feature axes."
        );
    }
    if (!std::all_of(
        history_feature_scores.begin(),
        history_feature_scores.end(),
        [](const double value) {
            return std::isfinite(value);
        }
    )) {
        throw std::invalid_argument("Pair scores must all be finite.");
    }
    const std::size_t offset =
        static_cast<std::size_t>(shield_pair_id) * expected;
    std::copy(
        history_feature_scores.begin(),
        history_feature_scores.end(),
        scores_.begin() + static_cast<std::ptrdiff_t>(offset)
    );
    submitted_.at(shield_pair_id) = true;
}

bool PairedScoreAccumulator::Complete() const noexcept {
    return std::all_of(
        submitted_.begin(),
        submitted_.end(),
        [](const bool value) {
            return value;
        }
    );
}

CrossPairStratifiedCovariance PairedScoreAccumulator::FinalizeExact(
    const std::string& score_semantics
) const {
    if (!Complete()) {
        throw std::invalid_argument(
            "All 64 pair score matrices are required before covariance."
        );
    }
    if (!IsNonemptyAscii(score_semantics)) {
        throw std::invalid_argument(
            "Score semantics must be nonempty ASCII."
        );
    }
    using GroupKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    std::map<GroupKey, std::size_t> group_index_by_key;
    for (const auto& identity : history_identities_) {
        group_index_by_key.emplace(
            GroupKey{
                identity.source_index,
                identity.line_index,
                identity.angle_stratum_index,
            },
            0U
        );
    }
    std::size_t next_group = 0U;
    for (auto& item : group_index_by_key) {
        item.second = next_group++;
    }

    CrossPairStratifiedCovariance artifact;
    artifact.history_count = history_identities_.size();
    artifact.group_count = group_index_by_key.size();
    artifact.feature_count = feature_count_;
    artifact.score_semantics = score_semantics;
    artifact.original_history_ids.reserve(artifact.history_count);
    artifact.history_group_indices.reserve(artifact.history_count);
    artifact.groups.resize(artifact.group_count);
    for (const auto& identity : history_identities_) {
        const GroupKey key{
            identity.source_index,
            identity.line_index,
            identity.angle_stratum_index,
        };
        const std::size_t group_index = group_index_by_key.at(key);
        artifact.original_history_ids.push_back(
            identity.original_history_id
        );
        artifact.history_group_indices.push_back(
            static_cast<std::uint32_t>(group_index)
        );
        auto& group = artifact.groups.at(group_index);
        if (group.history_count == 0U) {
            group.source_index = identity.source_index;
            group.line_index = identity.line_index;
            group.angle_stratum_index = identity.angle_stratum_index;
            group.angle_stratum_count = identity.angle_stratum_count;
            group.estimator_coefficient =
                identity.estimator_coefficient;
        } else if (
            group.angle_stratum_count != identity.angle_stratum_count
            || group.estimator_coefficient
                != identity.estimator_coefficient
        ) {
            throw std::logic_error(
                "Validated stratum identity changed during covariance."
            );
        }
        group.history_count += 1U;
    }
    std::ostringstream assignment_json;
    assignment_json
        << "{\"domain\":\"" << kStratumAssignmentDomain
        << "\",\"histories\":[";
    for (std::size_t history = 0; history < artifact.history_count;
         ++history) {
        if (history != 0U) {
            assignment_json << ",";
        }
        assignment_json
            << "{\"group_index\":"
            << artifact.history_group_indices.at(history)
            << ",\"history_id\":\""
            << TypedHistoryIdentity(
                artifact.original_history_ids.at(history)
            )
            << "\"}";
    }
    assignment_json << "]}\n";
    artifact.group_assignment_sha256 =
        HexDigest(Digest(assignment_json.str()));

    const std::size_t pair_feature_count =
        kShieldPairCount * feature_count_;
    if (
        artifact.group_count
            > std::numeric_limits<std::size_t>::max()
                / pair_feature_count
        || artifact.history_count
            > std::numeric_limits<std::size_t>::max()
                / pair_feature_count
    ) {
        throw std::overflow_error(
            "Exact stratified covariance artifact is too large."
        );
    }
    artifact.estimate_by_pair_feature.assign(
        pair_feature_count,
        0.0
    );
    artifact.first_sum_by_group_pair_feature.assign(
        artifact.group_count * pair_feature_count,
        0.0
    );
    artifact.centered_factor_by_history.assign(
        artifact.history_count * pair_feature_count,
        0.0
    );

    for (std::size_t history = 0U; history < artifact.history_count;
         ++history) {
        const std::size_t group =
            artifact.history_group_indices.at(history);
        for (std::size_t pair = 0U; pair < kShieldPairCount; ++pair) {
            for (std::size_t feature = 0U; feature < feature_count_;
                 ++feature) {
                artifact.first_sum_by_group_pair_feature.at(
                    GroupPairFeatureIndex(
                        group,
                        pair,
                        feature,
                        feature_count_
                    )
                ) += scores_.at(ScoreIndex(
                    pair,
                    history,
                    feature,
                    artifact.history_count,
                    feature_count_
                ));
            }
        }
    }
    for (std::size_t group = 0U; group < artifact.group_count; ++group) {
        const auto& descriptor = artifact.groups.at(group);
        if (descriptor.history_count < 2U) {
            throw std::logic_error(
                "Exact covariance group lost its minimum history count."
            );
        }
        for (std::size_t pair = 0U; pair < kShieldPairCount; ++pair) {
            for (std::size_t feature = 0U; feature < feature_count_;
                 ++feature) {
                artifact.estimate_by_pair_feature.at(PairFeatureIndex(
                    pair,
                    feature,
                    feature_count_
                )) += descriptor.estimator_coefficient
                    * artifact.first_sum_by_group_pair_feature.at(
                        GroupPairFeatureIndex(
                            group,
                            pair,
                            feature,
                            feature_count_
                        )
                    );
            }
        }
    }
    for (std::size_t history = 0U; history < artifact.history_count;
         ++history) {
        const std::size_t group =
            artifact.history_group_indices.at(history);
        const auto& descriptor = artifact.groups.at(group);
        const double sample_count =
            static_cast<double>(descriptor.history_count);
        const double scale = descriptor.estimator_coefficient
            * std::sqrt(sample_count / (sample_count - 1.0));
        for (std::size_t pair = 0U; pair < kShieldPairCount; ++pair) {
            for (std::size_t feature = 0U; feature < feature_count_;
                 ++feature) {
                const double group_mean =
                    artifact.first_sum_by_group_pair_feature.at(
                        GroupPairFeatureIndex(
                            group,
                            pair,
                            feature,
                            feature_count_
                        )
                    ) / sample_count;
                artifact.centered_factor_by_history.at(
                    HistoryFactorIndex(
                        history,
                        pair,
                        feature,
                        feature_count_
                    )
                ) = scale * (
                    scores_.at(ScoreIndex(
                        pair,
                        history,
                        feature,
                        artifact.history_count,
                        feature_count_
                    )) - group_mean
                );
            }
        }
    }
    std::array<double, kShieldPairCount> total_factor_by_pair{};
    for (std::size_t history = 0U; history < artifact.history_count;
         ++history) {
        total_factor_by_pair.fill(0.0);
        for (std::size_t pair = 0U; pair < kShieldPairCount; ++pair) {
            for (std::size_t feature = 0U; feature < feature_count_;
                 ++feature) {
                total_factor_by_pair[pair] +=
                    artifact.centered_factor_by_history.at(
                        HistoryFactorIndex(
                            history,
                            pair,
                            feature,
                            feature_count_
                        )
                    );
            }
        }
        for (std::size_t left = 0U; left < kShieldPairCount; ++left) {
            for (std::size_t right = 0U; right < kShieldPairCount;
                 ++right) {
                artifact.total_cross_pair_covariance.at(
                    left * kShieldPairCount + right
                ) += total_factor_by_pair[left]
                    * total_factor_by_pair[right];
            }
        }
    }
    artifact.artifact_sha256 = CovarianceArtifactHash(artifact);
    return artifact;
}

ApproximateCrossPairBlockDiagnostic
PairedScoreAccumulator::FinalizeApproximateBlockDiagnostic(
    const std::size_t block_count
) const {
    if (!Complete()) {
        throw std::invalid_argument(
            "All 64 pair score matrices are required before diagnostics."
        );
    }
    if (
        block_count < 2U
        || history_identities_.size() < block_count
        || history_identities_.size() % block_count != 0U
    ) {
        throw std::invalid_argument(
            "History count must be divisible by at least two diagnostic "
            "blocks."
        );
    }
    struct RankedHistory {
        std::size_t index = 0U;
        std::string identity;
        std::array<std::uint8_t, kSha256Size> digest{};
    };
    std::vector<RankedHistory> ranked;
    ranked.reserve(history_identities_.size());
    for (std::size_t index = 0U; index < history_identities_.size();
         ++index) {
        const std::string identity = TypedHistoryIdentity(
            history_identities_[index].original_history_id
        );
        ranked.push_back(RankedHistory{
            index,
            identity,
            Digest(std::string(kBlockDomain) + "|" + identity),
        });
    }
    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const RankedHistory& left, const RankedHistory& right) {
            return std::tie(left.digest, left.identity)
                < std::tie(right.digest, right.identity);
        }
    );

    ApproximateCrossPairBlockDiagnostic diagnostic;
    diagnostic.history_count = history_identities_.size();
    diagnostic.block_count = block_count;
    diagnostic.histories_per_block =
        history_identities_.size() / block_count;
    diagnostic.feature_count = feature_count_;
    diagnostic.pooled_mean_by_pair_feature.assign(
        kShieldPairCount * feature_count_,
        0.0
    );
    diagnostic.covariance_factor_by_block.assign(
        block_count * kShieldPairCount * feature_count_,
        0.0
    );
    for (std::size_t pair = 0; pair < kShieldPairCount; ++pair) {
        for (std::size_t feature = 0; feature < feature_count_; ++feature) {
            double total = 0.0;
            for (const auto& item : ranked) {
                total += scores_.at(ScoreIndex(
                    pair,
                    item.index,
                    feature,
                    history_identities_.size(),
                    feature_count_
                ));
            }
            diagnostic.pooled_mean_by_pair_feature.at(PairFeatureIndex(
                pair,
                feature,
                feature_count_
            )) = total / static_cast<double>(history_identities_.size());
        }
    }
    const double factor_scale =
        1.0 / std::sqrt(
            static_cast<double>(block_count * (block_count - 1U))
        );
    for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
        const std::size_t block = rank % block_count;
        const std::size_t history = ranked[rank].index;
        for (std::size_t pair = 0; pair < kShieldPairCount; ++pair) {
            for (std::size_t feature = 0; feature < feature_count_; ++feature) {
                diagnostic.covariance_factor_by_block.at(FactorIndex(
                    block,
                    pair,
                    feature,
                    feature_count_
                )) += scores_.at(ScoreIndex(
                    pair,
                    history,
                    feature,
                    history_identities_.size(),
                    feature_count_
                )) / static_cast<double>(
                    diagnostic.histories_per_block
                );
            }
        }
    }
    for (std::size_t block = 0; block < block_count; ++block) {
        for (std::size_t pair = 0; pair < kShieldPairCount; ++pair) {
            for (std::size_t feature = 0; feature < feature_count_; ++feature) {
                const std::size_t factor_index = FactorIndex(
                    block,
                    pair,
                    feature,
                    feature_count_
                );
                diagnostic.covariance_factor_by_block.at(factor_index) = (
                    diagnostic.covariance_factor_by_block.at(factor_index)
                    - diagnostic.pooled_mean_by_pair_feature.at(
                        PairFeatureIndex(
                        pair,
                        feature,
                        feature_count_
                    ))
                ) * factor_scale;
            }
        }
    }
    return diagnostic;
}

std::vector<std::uint8_t> SerializeCovarianceArtifact(
    const CrossPairStratifiedCovariance& artifact
) {
    using GroupKey =
        std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    using SourceLineKey = std::pair<std::uint32_t, std::uint32_t>;
    const auto checked_product = [](const std::size_t left,
                                    const std::size_t right,
                                    const char* message) {
        if (
            left != 0U
            && right > std::numeric_limits<std::size_t>::max() / left
        ) {
            throw std::invalid_argument(message);
        }
        return left * right;
    };
    if (
        artifact.history_count == 0U
        || artifact.group_count == 0U
        || artifact.group_count > artifact.history_count
        || artifact.feature_count == 0U
        || !IsNonemptyAscii(artifact.score_semantics)
        || !IsLowerSha256(artifact.group_assignment_sha256)
        || artifact.original_history_ids.size() != artifact.history_count
        || artifact.history_group_indices.size() != artifact.history_count
        || artifact.groups.size() != artifact.group_count
    ) {
        throw std::invalid_argument(
            "Exact stratified covariance artifact is structurally invalid."
        );
    }
    const std::size_t pair_feature_count = checked_product(
        kShieldPairCount,
        artifact.feature_count,
        "Exact covariance pair-feature axis is too large."
    );
    const std::size_t first_sum_count = checked_product(
        artifact.group_count,
        pair_feature_count,
        "Exact covariance group-score array is too large."
    );
    const std::size_t factor_count = checked_product(
        artifact.history_count,
        pair_feature_count,
        "Exact covariance factor array is too large."
    );
    if (
        artifact.estimate_by_pair_feature.size() != pair_feature_count
        || artifact.first_sum_by_group_pair_feature.size()
            != first_sum_count
        || artifact.centered_factor_by_history.size() != factor_count
    ) {
        throw std::invalid_argument(
            "Exact stratified covariance array shape is invalid."
        );
    }
    if (!std::is_sorted(
        artifact.original_history_ids.begin(),
        artifact.original_history_ids.end()
    )) {
        throw std::invalid_argument(
            "Exact covariance history IDs must be canonically sorted."
        );
    }
    if (
        std::adjacent_find(
            artifact.original_history_ids.begin(),
            artifact.original_history_ids.end()
        ) != artifact.original_history_ids.end()
    ) {
        throw std::invalid_argument(
            "Exact covariance history IDs must be unique."
        );
    }

    std::vector<std::size_t> observed_group_counts(
        artifact.group_count,
        0U
    );
    for (const auto group_index : artifact.history_group_indices) {
        if (group_index >= artifact.group_count) {
            throw std::invalid_argument(
                "Exact covariance history group index is out of range."
            );
        }
        observed_group_counts.at(group_index) += 1U;
    }
    std::optional<GroupKey> previous_group;
    std::map<SourceLineKey, std::uint32_t> stratum_counts;
    std::map<SourceLineKey, std::set<std::uint32_t>> observed_strata;
    std::map<SourceLineKey, std::size_t> fixed_quotas;
    for (std::size_t index = 0U; index < artifact.group_count; ++index) {
        const auto& group = artifact.groups.at(index);
        const GroupKey key{
            group.source_index,
            group.line_index,
            group.angle_stratum_index,
        };
        if (
            previous_group.has_value()
            && !(previous_group.value() < key)
        ) {
            throw std::invalid_argument(
                "Exact covariance groups are not in canonical order."
            );
        }
        previous_group = key;
        if (
            group.angle_stratum_count == 0U
            || group.angle_stratum_index >= group.angle_stratum_count
            || group.history_count < 2U
            || group.history_count != observed_group_counts.at(index)
            || !std::isfinite(group.estimator_coefficient)
            || group.estimator_coefficient <= 0.0
        ) {
            throw std::invalid_argument(
                "Exact covariance group descriptor is invalid."
            );
        }
        const SourceLineKey source_line{
            group.source_index,
            group.line_index,
        };
        const auto [count_it, count_inserted] = stratum_counts.emplace(
            source_line,
            group.angle_stratum_count
        );
        if (
            !count_inserted
            && count_it->second != group.angle_stratum_count
        ) {
            throw std::invalid_argument(
                "One source-line has inconsistent stratum counts."
            );
        }
        observed_strata[source_line].insert(group.angle_stratum_index);
        const auto [quota_it, quota_inserted] = fixed_quotas.emplace(
            source_line,
            group.history_count
        );
        if (!quota_inserted && quota_it->second != group.history_count) {
            throw std::invalid_argument(
                "Exact covariance violates fixed quota across angle strata."
            );
        }
    }
    for (const auto& [source_line, stratum_count] : stratum_counts) {
        const auto& strata = observed_strata.at(source_line);
        if (strata.size() != stratum_count) {
            throw std::invalid_argument(
                "Exact covariance is missing a declared angle stratum."
            );
        }
        for (std::uint32_t stratum = 0U; stratum < stratum_count; ++stratum) {
            if (strata.count(stratum) == 0U) {
                throw std::invalid_argument(
                    "Exact covariance angle strata are not contiguous."
                );
            }
        }
    }

    std::ostringstream assignment_json;
    assignment_json
        << "{\"domain\":\"" << kStratumAssignmentDomain
        << "\",\"histories\":[";
    for (std::size_t history = 0U; history < artifact.history_count;
         ++history) {
        if (history != 0U) {
            assignment_json << ",";
        }
        assignment_json
            << "{\"group_index\":"
            << artifact.history_group_indices.at(history)
            << ",\"history_id\":\""
            << TypedHistoryIdentity(
                artifact.original_history_ids.at(history)
            )
            << "\"}";
    }
    assignment_json << "]}\n";
    if (
        artifact.group_assignment_sha256
        != HexDigest(Digest(assignment_json.str()))
    ) {
        throw std::invalid_argument(
            "Exact covariance group assignment hash does not match."
        );
    }

    const auto require_finite = [](const auto& values) {
        if (!std::all_of(
            values.begin(),
            values.end(),
            [](const double value) {
                return std::isfinite(value);
            }
        )) {
            throw std::invalid_argument(
                "Exact covariance numeric values must be finite."
            );
        }
    };
    require_finite(artifact.estimate_by_pair_feature);
    require_finite(artifact.first_sum_by_group_pair_feature);
    require_finite(artifact.centered_factor_by_history);
    require_finite(artifact.total_cross_pair_covariance);

    std::vector<double> expected_estimate(pair_feature_count, 0.0);
    for (std::size_t group = 0U; group < artifact.group_count; ++group) {
        const double coefficient =
            artifact.groups.at(group).estimator_coefficient;
        for (std::size_t item = 0U; item < pair_feature_count; ++item) {
            expected_estimate.at(item) += coefficient
                * artifact.first_sum_by_group_pair_feature.at(
                    group * pair_feature_count + item
                );
        }
    }
    if (expected_estimate != artifact.estimate_by_pair_feature) {
        throw std::invalid_argument(
            "Exact covariance estimate disagrees with grouped first sums."
        );
    }
    for (std::size_t group = 0U; group < artifact.group_count; ++group) {
        for (std::size_t item = 0U; item < pair_feature_count; ++item) {
            double centered_sum = 0.0;
            double absolute_sum = 0.0;
            for (std::size_t history = 0U;
                 history < artifact.history_count; ++history) {
                if (artifact.history_group_indices.at(history) != group) {
                    continue;
                }
                const double value =
                    artifact.centered_factor_by_history.at(
                        history * pair_feature_count + item
                    );
                centered_sum += value;
                absolute_sum += std::abs(value);
            }
            const double tolerance = 4.0
                * static_cast<double>(
                    artifact.groups.at(group).history_count + 1U
                )
                * std::numeric_limits<double>::epsilon()
                * std::max(1.0, absolute_sum);
            if (std::abs(centered_sum) > tolerance) {
                throw std::invalid_argument(
                    "Exact covariance factors are not centered within a "
                    "source-line-angle stratum."
                );
            }
        }
    }
    std::array<double, kShieldPairCount * kShieldPairCount>
        expected_covariance{};
    std::array<double, kShieldPairCount> history_pair_factors{};
    for (std::size_t history = 0U; history < artifact.history_count;
         ++history) {
        history_pair_factors.fill(0.0);
        for (std::size_t pair = 0U; pair < kShieldPairCount; ++pair) {
            for (std::size_t feature = 0U;
                 feature < artifact.feature_count; ++feature) {
                history_pair_factors.at(pair) +=
                    artifact.centered_factor_by_history.at(
                        HistoryFactorIndex(
                            history,
                            pair,
                            feature,
                            artifact.feature_count
                        )
                    );
            }
        }
        for (std::size_t left = 0U; left < kShieldPairCount; ++left) {
            for (std::size_t right = 0U; right < kShieldPairCount; ++right) {
                expected_covariance.at(left * kShieldPairCount + right)
                    += history_pair_factors.at(left)
                        * history_pair_factors.at(right);
            }
        }
    }
    if (expected_covariance != artifact.total_cross_pair_covariance) {
        throw std::invalid_argument(
            "Exact covariance outer product disagrees with factor rows."
        );
    }
    const std::string expected_hash = CovarianceArtifactHash(artifact);
    if (artifact.artifact_sha256 != expected_hash) {
        throw std::invalid_argument(
            "Exact covariance artifact hash does not match."
        );
    }

    std::vector<std::uint8_t> payload;
    payload.insert(
        payload.end(),
        kCovarianceMagic.begin(),
        kCovarianceMagic.end()
    );
    AppendLittleEndian(payload, std::uint32_t{2});
    AppendLittleEndian(payload, std::uint32_t{0});
    AppendLittleEndian(
        payload,
        static_cast<std::uint64_t>(artifact.history_count)
    );
    AppendLittleEndian(
        payload,
        static_cast<std::uint64_t>(artifact.group_count)
    );
    AppendLittleEndian(
        payload,
        static_cast<std::uint64_t>(artifact.feature_count)
    );
    AppendString(payload, artifact.score_semantics);
    AppendString(payload, artifact.group_assignment_sha256);
    for (const auto history_id : artifact.original_history_ids) {
        AppendLittleEndian(payload, history_id);
    }
    for (const auto group_index : artifact.history_group_indices) {
        AppendLittleEndian(payload, group_index);
    }
    for (const auto& group : artifact.groups) {
        AppendLittleEndian(payload, group.source_index);
        AppendLittleEndian(payload, group.line_index);
        AppendLittleEndian(payload, group.angle_stratum_index);
        AppendLittleEndian(payload, group.angle_stratum_count);
        AppendLittleEndian(
            payload,
            static_cast<std::uint64_t>(group.history_count)
        );
        AppendDouble(payload, group.estimator_coefficient);
    }
    for (const double value : artifact.estimate_by_pair_feature) {
        AppendDouble(payload, value);
    }
    for (const double value : artifact.first_sum_by_group_pair_feature) {
        AppendDouble(payload, value);
    }
    for (const double value : artifact.centered_factor_by_history) {
        AppendDouble(payload, value);
    }
    for (const double value : artifact.total_cross_pair_covariance) {
        AppendDouble(payload, value);
    }
    const auto checksum = Digest(payload.data(), payload.size());
    payload.insert(payload.end(), checksum.begin(), checksum.end());
    return payload;
}

CrossPairStratifiedCovariance DeserializeCovarianceArtifact(
    const std::vector<std::uint8_t>& payload
) {
    constexpr std::size_t fixed_prefix_size =
        8U + 4U + 4U + 3U * sizeof(std::uint64_t);
    if (payload.size() < fixed_prefix_size + kSha256Size) {
        throw std::invalid_argument(
            "Exact covariance payload is truncated."
        );
    }
    const std::size_t body_size = payload.size() - kSha256Size;
    const auto checksum = Digest(payload.data(), body_size);
    if (!std::equal(
        checksum.begin(),
        checksum.end(),
        payload.begin() + static_cast<std::ptrdiff_t>(body_size)
    )) {
        throw std::invalid_argument(
            "Exact covariance payload checksum does not match."
        );
    }
    Reader reader(payload, body_size);
    for (std::size_t index = 0U; index < kCovarianceMagic.size(); ++index) {
        if (reader.ReadInteger<std::uint8_t>() != kCovarianceMagic[index]) {
            throw std::invalid_argument(
                "Exact covariance payload magic is invalid."
            );
        }
    }
    if (reader.ReadInteger<std::uint32_t>() != 2U) {
        throw std::invalid_argument(
            "Unsupported exact covariance schema version."
        );
    }
    if (reader.ReadInteger<std::uint32_t>() != 0U) {
        throw std::invalid_argument(
            "Exact covariance reserved flags must be zero."
        );
    }

    const auto raw_history_count = reader.ReadInteger<std::uint64_t>();
    const auto raw_group_count = reader.ReadInteger<std::uint64_t>();
    const auto raw_feature_count = reader.ReadInteger<std::uint64_t>();
    for (const auto value : {
        raw_history_count,
        raw_group_count,
        raw_feature_count,
    }) {
        if (
            value > std::numeric_limits<std::size_t>::max()
            || value > kMaximumSerializedRecords
        ) {
            throw std::invalid_argument(
                "Exact covariance dimension exceeds supported bounds."
            );
        }
    }
    CrossPairStratifiedCovariance artifact;
    artifact.history_count =
        static_cast<std::size_t>(raw_history_count);
    artifact.group_count = static_cast<std::size_t>(raw_group_count);
    artifact.feature_count = static_cast<std::size_t>(raw_feature_count);
    artifact.score_semantics = reader.ReadString();
    artifact.group_assignment_sha256 = reader.ReadString();
    if (
        artifact.history_count == 0U
        || artifact.group_count == 0U
        || artifact.group_count > artifact.history_count
        || artifact.feature_count == 0U
    ) {
        throw std::invalid_argument(
            "Exact covariance dimensions are invalid."
        );
    }
    constexpr std::size_t identity_bytes =
        sizeof(std::uint64_t) + sizeof(std::uint32_t);
    constexpr std::size_t group_bytes =
        4U * sizeof(std::uint32_t)
        + sizeof(std::uint64_t) + sizeof(double);
    if (
        artifact.history_count
            > (body_size - reader.Position()) / identity_bytes
        || artifact.group_count
            > (
                body_size - reader.Position()
                - artifact.history_count * identity_bytes
            ) / group_bytes
    ) {
        throw std::invalid_argument(
            "Exact covariance identity axes exceed payload length."
        );
    }
    artifact.original_history_ids.reserve(artifact.history_count);
    artifact.history_group_indices.reserve(artifact.history_count);
    for (std::size_t index = 0U; index < artifact.history_count; ++index) {
        artifact.original_history_ids.push_back(
            reader.ReadInteger<std::uint64_t>()
        );
    }
    for (std::size_t index = 0U; index < artifact.history_count; ++index) {
        artifact.history_group_indices.push_back(
            reader.ReadInteger<std::uint32_t>()
        );
    }
    artifact.groups.reserve(artifact.group_count);
    for (std::size_t index = 0U; index < artifact.group_count; ++index) {
        StratumCovarianceDescriptor group;
        group.source_index = reader.ReadInteger<std::uint32_t>();
        group.line_index = reader.ReadInteger<std::uint32_t>();
        group.angle_stratum_index =
            reader.ReadInteger<std::uint32_t>();
        group.angle_stratum_count =
            reader.ReadInteger<std::uint32_t>();
        const auto raw_group_history_count =
            reader.ReadInteger<std::uint64_t>();
        if (
            raw_group_history_count
            > std::numeric_limits<std::size_t>::max()
        ) {
            throw std::invalid_argument(
                "Exact covariance group size exceeds this platform."
            );
        }
        group.history_count =
            static_cast<std::size_t>(raw_group_history_count);
        group.estimator_coefficient = reader.ReadDouble();
        artifact.groups.push_back(group);
    }
    if (
        artifact.feature_count
        > std::numeric_limits<std::size_t>::max() / kShieldPairCount
    ) {
        throw std::invalid_argument(
            "Exact covariance feature axis is too large."
        );
    }
    const std::size_t pair_feature_count =
        kShieldPairCount * artifact.feature_count;
    if (
        artifact.group_count
            > std::numeric_limits<std::size_t>::max()
                / pair_feature_count
        || artifact.history_count
            > std::numeric_limits<std::size_t>::max()
                / pair_feature_count
    ) {
        throw std::invalid_argument(
            "Exact covariance arrays exceed this platform."
        );
    }
    const std::size_t estimate_count = pair_feature_count;
    const std::size_t first_sum_count =
        artifact.group_count * pair_feature_count;
    const std::size_t factor_count =
        artifact.history_count * pair_feature_count;
    constexpr std::size_t covariance_count =
        kShieldPairCount * kShieldPairCount;
    if (
        estimate_count
            > std::numeric_limits<std::size_t>::max() - first_sum_count
        || estimate_count + first_sum_count
            > std::numeric_limits<std::size_t>::max() - factor_count
        || estimate_count + first_sum_count + factor_count
            > std::numeric_limits<std::size_t>::max() - covariance_count
    ) {
        throw std::invalid_argument(
            "Exact covariance serialized arrays are too large."
        );
    }
    const std::size_t numeric_count =
        estimate_count + first_sum_count + factor_count + covariance_count;
    if (
        numeric_count
        != (body_size - reader.Position()) / sizeof(double)
        || (body_size - reader.Position()) % sizeof(double) != 0U
    ) {
        throw std::invalid_argument(
            "Exact covariance numeric arrays differ from payload length."
        );
    }
    artifact.estimate_by_pair_feature.reserve(estimate_count);
    artifact.first_sum_by_group_pair_feature.reserve(first_sum_count);
    artifact.centered_factor_by_history.reserve(factor_count);
    for (std::size_t index = 0U; index < estimate_count; ++index) {
        artifact.estimate_by_pair_feature.push_back(reader.ReadDouble());
    }
    for (std::size_t index = 0U; index < first_sum_count; ++index) {
        artifact.first_sum_by_group_pair_feature.push_back(
            reader.ReadDouble()
        );
    }
    for (std::size_t index = 0U; index < factor_count; ++index) {
        artifact.centered_factor_by_history.push_back(
            reader.ReadDouble()
        );
    }
    for (double& value : artifact.total_cross_pair_covariance) {
        value = reader.ReadDouble();
    }
    if (reader.Position() != body_size) {
        throw std::invalid_argument(
            "Exact covariance payload contains unknown trailing fields."
        );
    }
    artifact.artifact_sha256 = CovarianceArtifactHash(artifact);
    const auto canonical = SerializeCovarianceArtifact(artifact);
    if (canonical != payload) {
        throw std::invalid_argument(
            "Exact covariance payload is not canonical."
        );
    }
    return artifact;
}

}  // namespace rotating_shield::paired_all64
