#pragma once

// On-disk index of retained-session snapshots, and the sidecar metadata that makes the index
// rebuildable without reading any snapshot payload.
//
// A snapshot file alone cannot answer "which saved session continues this prompt": the bytes
// are an opaque host image. Each snapshot therefore gets a small text sidecar
// (`<snapshot>.meta`) that records the identity facts the Engine already knows at save time:
// the storage frontier N, the full-ledger digest, and every restorable checkpoint
// (frontier K, digest of ledger[0:K]). Startup reads only the sidecars, so rebuilding the
// index of a multi-GB slot directory costs milliseconds.
//
// Identity here narrows the candidate set; it never proves a hit. The Engine still runs the
// Program's exact verification after materializing a candidate, and a mismatch simply falls
// back to the ordinary cold path.
//
// Lookup is a single pass over the prompt: FNV-1a is a byte-rolling hash, so every requested
// frontier's digest falls out of one walk instead of one hash per candidate frontier.

#include "ninfer/types.h"
#include "runtime/contract/ledger_digest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime {

inline constexpr std::uint32_t kSessionSidecarVersion = 1;
inline constexpr std::string_view kSessionSnapshotSuffix = ".slot";
inline constexpr std::string_view kSessionSidecarSuffix  = ".meta";
inline constexpr std::string_view kSessionTemporarySuffix = ".tmp";

// One restorable frontier inside a snapshot: the ledger depth and the digest of the ledger
// prefix it covers. The Engine rewinds a restored continuation to the deepest frontier that
// matches the incoming prompt.
struct SessionCheckpointMeta {
    std::uint32_t frontier = 0;
    std::string digest;

    [[nodiscard]] friend bool operator==(const SessionCheckpointMeta&,
                                         const SessionCheckpointMeta&) noexcept = default;
};

struct SessionRecord {
    std::string path;                    // snapshot file (write-once)
    std::string model_binding;
    std::uint32_t snapshot_frontier = 0; // N: depth of the stored session
    std::string session_digest;          // digest(ledger[0:N])
    std::vector<SessionCheckpointMeta> checkpoints;
    std::uint64_t bytes       = 0;
    std::uint64_t created_ms  = 0;
    std::uint64_t accessed_ms = 0;
    std::uint32_t generation  = 0;
    std::string binding_key;             // explicit session key, empty when unbound

    // The deepest checkpoint at or below `frontier`, if any.
    [[nodiscard]] const SessionCheckpointMeta* checkpoint_at_or_below(
        std::uint32_t frontier) const noexcept {
        const SessionCheckpointMeta* best = nullptr;
        for (const SessionCheckpointMeta& checkpoint : checkpoints) {
            if (checkpoint.frontier > frontier) { continue; }
            if (best == nullptr || checkpoint.frontier > best->frontier) { best = &checkpoint; }
        }
        return best;
    }

    [[nodiscard]] friend bool operator==(const SessionRecord&,
                                         const SessionRecord&) noexcept = default;
};

namespace detail {

inline std::uint64_t session_now_ms() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

// Text sidecars are line oriented. A model binding may contain newlines (chat templates carry
// them), so it is escaped on the way out and unescaped on the way in.
inline std::string escape_sidecar_value(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char c : value) {
        if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else { out += c; }
    }
    return out;
}

inline std::string unescape_sidecar_value(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] != '\\' || index + 1 >= value.size()) {
            out += value[index];
            continue;
        }
        ++index;
        out += value[index] == 'n' ? '\n' : value[index];
    }
    return out;
}

} // namespace detail

[[nodiscard]] inline std::string session_sidecar_path_for(std::string_view snapshot_path) {
    return std::string(snapshot_path) + std::string(kSessionSidecarSuffix);
}

[[nodiscard]] inline std::string serialize_session_sidecar(const SessionRecord& record) {
    std::ostringstream out;
    out << "meta_version=" << kSessionSidecarVersion << '\n';
    out << "snapshot_path=" << detail::escape_sidecar_value(record.path) << '\n';
    out << "model_binding=" << detail::escape_sidecar_value(record.model_binding) << '\n';
    out << "snapshot_frontier=" << record.snapshot_frontier << '\n';
    out << "session_digest=" << record.session_digest << '\n';
    out << "snapshot_bytes=" << record.bytes << '\n';
    out << "created_ms=" << record.created_ms << '\n';
    out << "accessed_ms=" << record.accessed_ms << '\n';
    out << "generation=" << record.generation << '\n';
    if (!record.binding_key.empty()) {
        out << "binding_key=" << detail::escape_sidecar_value(record.binding_key) << '\n';
    }
    for (const SessionCheckpointMeta& checkpoint : record.checkpoints) {
        out << "checkpoint=" << checkpoint.frontier << ':' << checkpoint.digest << '\n';
    }
    return out.str();
}

// Tolerant parse: unknown keys are ignored so a newer writer's extra fields do not make an
// older reader drop a valid snapshot. Returns nullopt when the required identity is absent.
[[nodiscard]] inline std::optional<SessionRecord>
parse_session_sidecar(std::string_view content) {
    SessionRecord record;
    bool has_path     = false;
    bool has_binding  = false;
    bool has_digest   = false;
    std::istringstream stream{std::string(content)};
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line.front() == '#') { continue; }
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) { continue; }
        const std::string key   = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        try {
            if (key == "meta_version") {
                if (std::stoul(value) != kSessionSidecarVersion) { return std::nullopt; }
            } else if (key == "snapshot_path") {
                record.path = detail::unescape_sidecar_value(value);
                has_path    = true;
            } else if (key == "model_binding") {
                record.model_binding = detail::unescape_sidecar_value(value);
                has_binding          = true;
            } else if (key == "snapshot_frontier") {
                record.snapshot_frontier = static_cast<std::uint32_t>(std::stoul(value));
            } else if (key == "session_digest") {
                record.session_digest = value;
                has_digest            = !value.empty();
            } else if (key == "snapshot_bytes") {
                record.bytes = std::stoull(value);
            } else if (key == "created_ms") {
                record.created_ms = std::stoull(value);
            } else if (key == "accessed_ms") {
                record.accessed_ms = std::stoull(value);
            } else if (key == "generation") {
                record.generation = static_cast<std::uint32_t>(std::stoul(value));
            } else if (key == "binding_key") {
                record.binding_key = detail::unescape_sidecar_value(value);
            } else if (key == "checkpoint") {
                const std::size_t colon = value.find(':');
                if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size()) {
                    return std::nullopt;
                }
                SessionCheckpointMeta checkpoint;
                checkpoint.frontier = static_cast<std::uint32_t>(std::stoul(value.substr(0, colon)));
                checkpoint.digest   = value.substr(colon + 1);
                if (checkpoint.frontier == 0 || checkpoint.digest.empty()) {
                    return std::nullopt;
                }
                record.checkpoints.push_back(std::move(checkpoint));
            }
        } catch (const std::exception&) {
            // A malformed numeric field makes the sidecar untrustworthy.
            return std::nullopt;
        }
    }
    if (!has_path || !has_binding || !has_digest || record.snapshot_frontier == 0) {
        return std::nullopt;
    }
    std::sort(record.checkpoints.begin(), record.checkpoints.end(),
              [](const SessionCheckpointMeta& left, const SessionCheckpointMeta& right) {
                  return left.frontier < right.frontier;
              });
    record.checkpoints.erase(
        std::unique(record.checkpoints.begin(), record.checkpoints.end(),
                    [](const SessionCheckpointMeta& left, const SessionCheckpointMeta& right) {
                        return left.frontier == right.frontier;
                    }),
        record.checkpoints.end());
    // Every snapshot can always be resumed at its own frontier, so the storage frontier is a
    // checkpoint even when the writer recorded no others.
    const bool endpoint_recorded =
        std::any_of(record.checkpoints.begin(), record.checkpoints.end(),
                    [&](const SessionCheckpointMeta& checkpoint) {
                        return checkpoint.frontier == record.snapshot_frontier;
                    });
    if (!endpoint_recorded) {
        record.checkpoints.push_back(
            SessionCheckpointMeta{record.snapshot_frontier, record.session_digest});
    }
    if (record.created_ms == 0) { record.created_ms = record.accessed_ms; }
    if (record.accessed_ms == 0) { record.accessed_ms = record.created_ms; }
    return record;
}

// A lookup hit: the snapshot to restore and the frontier K the incoming prompt matched. The
// Engine rewinds the restored continuation to K; everything above it is re-prefilled.
struct SessionMatch {
    SessionRecord record;
    std::uint32_t frontier = 0;
};

// Persistent session index over one snapshot directory. All entry points are thread safe; the
// Engine calls rebuild() at construction and record()/lookup() from request and writer threads.
class SessionStore {
public:
    explicit SessionStore(std::filesystem::path directory, std::uint64_t max_bytes = 0)
        : directory_(std::move(directory)), max_bytes_(max_bytes) {}

    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }

    // Write-once snapshot path. The timestamp and a process-local sequence keep two writers
    // from ever targeting the same file, so a crash mid-write cannot corrupt an indexed
    // snapshot.
    [[nodiscard]] std::filesystem::path allocate_snapshot_path() const {
        const std::uint32_t sequence = sequence_.fetch_add(1, std::memory_order_relaxed);
        char name[64];
        std::snprintf(name, sizeof(name), "snap_%llx_%04x%.*s",
                      static_cast<unsigned long long>(detail::session_now_ms()), sequence,
                      static_cast<int>(kSessionSnapshotSuffix.size()),
                      kSessionSnapshotSuffix.data());
        return directory_ / name;
    }

    // Startup reconstruction: index every sidecar whose snapshot file still exists, delete
    // stale sidecars and orphaned temporary files, then prune to the byte budget. Returns the
    // number of indexed sessions.
    std::size_t rebuild() {
        std::scoped_lock lock(mutex_);
        index_.clear();
        std::error_code error;
        if (!std::filesystem::exists(directory_, error)) { return 0; }
        std::vector<std::filesystem::path> temporary_files;
        std::vector<std::filesystem::path> sidecars;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(directory_, error)) {
            if (error) { break; }
            if (!entry.is_regular_file()) { continue; }
            const std::string name = entry.path().filename().string();
            if (name.size() >= kSessionTemporarySuffix.size() &&
                name.compare(name.size() - kSessionTemporarySuffix.size(),
                             kSessionTemporarySuffix.size(), kSessionTemporarySuffix) == 0) {
                temporary_files.push_back(entry.path());
            } else if (name.size() >= kSessionSidecarSuffix.size() &&
                       name.compare(name.size() - kSessionSidecarSuffix.size(),
                                    kSessionSidecarSuffix.size(), kSessionSidecarSuffix) == 0) {
                sidecars.push_back(entry.path());
            }
        }
        for (const std::filesystem::path& path : temporary_files) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
        for (const std::filesystem::path& sidecar : sidecars) {
            std::optional<SessionRecord> record = read_sidecar(sidecar);
            if (!record) {
                std::error_code ignored;
                std::filesystem::remove(sidecar, ignored);
                continue;
            }
            std::error_code missing;
            if (!std::filesystem::exists(record->path, missing)) {
                std::error_code ignored;
                std::filesystem::remove(sidecar, ignored);
                continue;
            }
            index_.emplace(record->path, std::move(*record));
        }
        prune_locked();
        return index_.size();
    }

    // Publish the sidecar for a snapshot the Engine just wrote. The snapshot file itself is
    // never rewritten; a later save of the same session allocates a new path.
    void record(SessionRecord record) {
        if (record.path.empty() || record.snapshot_frontier == 0 ||
            record.session_digest.empty()) {
            return;
        }
        if (record.created_ms == 0) { record.created_ms = detail::session_now_ms(); }
        if (record.accessed_ms == 0) { record.accessed_ms = record.created_ms; }
        const std::string sidecar = session_sidecar_path_for(record.path);
        const std::string text    = serialize_session_sidecar(record);
        {
            // Write-then-rename: a crash mid-write leaves a temporary file that rebuild()
            // discards, never a half-parsed sidecar pointing at a valid snapshot.
            const std::string temporary = sidecar + std::string(kSessionTemporarySuffix);
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            if (!out) { return; }
            out.write(text.data(), static_cast<std::streamsize>(text.size()));
            out.close();
            if (!out.good()) {
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                return;
            }
            std::error_code error;
            std::filesystem::rename(temporary, sidecar, error);
            if (error) {
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                return;
            }
        }
        std::scoped_lock lock(mutex_);
        index_[record.path] = std::move(record);
        prune_locked();
    }

    // Forget one snapshot: its sidecar is removed here; the caller owns the payload file.
    void erase(std::string_view snapshot_path) {
        std::scoped_lock lock(mutex_);
        forget_locked(std::string(snapshot_path));
    }

    // Deepest checkpoint whose digest matches a prefix of `prompt`, within `model_binding`.
    // Ties prefer the newest generation, then the most recently accessed snapshot.
    [[nodiscard]] std::optional<SessionMatch>
    lookup(std::span<const TokenId> prompt, std::string_view model_binding) {
        std::vector<std::uint32_t> frontiers;
        {
            std::scoped_lock lock(mutex_);
            for (const auto& [path, record] : index_) {
                (void)path;
                if (record.model_binding != model_binding) { continue; }
                for (const SessionCheckpointMeta& checkpoint : record.checkpoints) {
                    if (checkpoint.frontier == 0 || checkpoint.frontier > prompt.size()) {
                        continue;
                    }
                    frontiers.push_back(checkpoint.frontier);
                }
            }
        }
        if (frontiers.empty()) { return std::nullopt; }
        std::sort(frontiers.begin(), frontiers.end());
        frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());

        // One rolling pass produces the digest at every requested frontier.
        std::vector<std::pair<std::uint32_t, std::string>> prefix_digests;
        prefix_digests.reserve(frontiers.size());
        std::uint64_t hash      = 1469598103934665603ULL;
        const auto* bytes       = reinterpret_cast<const unsigned char*>(prompt.data());
        const std::size_t total = prompt.size() * sizeof(TokenId);
        std::size_t next        = 0;
        for (std::size_t index = 0; index < total; ++index) {
            hash = (hash ^ bytes[index]) * 1099511628211ULL;
            if (next < frontiers.size() &&
                (index + 1) == static_cast<std::size_t>(frontiers[next]) * sizeof(TokenId)) {
                char text[17];
                std::snprintf(text, sizeof(text), "%016llx",
                              static_cast<unsigned long long>(hash));
                prefix_digests.emplace_back(frontiers[next], std::string(text, 16));
                ++next;
            }
        }

        std::scoped_lock lock(mutex_);
        std::optional<SessionRecord> best;
        std::uint32_t best_frontier = 0;
        // Deepest frontier first: the first frontier with any match is the longest reusable
        // prefix, so the scan can stop there.
        for (auto candidate = prefix_digests.rbegin(); candidate != prefix_digests.rend();
             ++candidate) {
            const auto& [frontier, digest] = *candidate;
            for (const auto& [path, record] : index_) {
                (void)path;
                if (record.model_binding != model_binding) { continue; }
                const auto match =
                    std::find_if(record.checkpoints.begin(), record.checkpoints.end(),
                                 [&](const SessionCheckpointMeta& checkpoint) {
                                     return checkpoint.frontier == frontier &&
                                            checkpoint.digest == digest;
                                 });
                if (match == record.checkpoints.end()) { continue; }
                const bool better =
                    !best || record.generation > best->generation ||
                    (record.generation == best->generation &&
                     record.accessed_ms > best->accessed_ms);
                if (better) {
                    best          = record;
                    best_frontier = frontier;
                }
            }
            if (best) { break; }
        }
        if (best) {
            auto stored = index_.find(best->path);
            if (stored != index_.end()) {
                stored->second.accessed_ms = detail::session_now_ms();
                best->accessed_ms          = stored->second.accessed_ms;
            }
            return SessionMatch{std::move(*best), best_frontier};
        }
        return std::nullopt;
    }

    // Drop least-recently-accessed snapshots until the directory fits the byte budget. The
    // newest snapshot always survives, so a budget smaller than one session still leaves the
    // store usable.
    void prune() {
        std::scoped_lock lock(mutex_);
        prune_locked();
    }

    [[nodiscard]] std::size_t size() const {
        std::scoped_lock lock(mutex_);
        return index_.size();
    }

    [[nodiscard]] std::vector<SessionRecord> entries() const {
        std::scoped_lock lock(mutex_);
        std::vector<SessionRecord> out;
        out.reserve(index_.size());
        for (const auto& [path, record] : index_) {
            (void)path;
            out.push_back(record);
        }
        return out;
    }

private:
    [[nodiscard]] std::optional<SessionRecord>
    read_sidecar(const std::filesystem::path& sidecar) const {
        std::ifstream in(sidecar, std::ios::binary);
        if (!in) { return std::nullopt; }
        std::ostringstream buffer;
        buffer << in.rdbuf();
        if (!in.good() && !in.eof()) { return std::nullopt; }
        std::optional<SessionRecord> record = parse_session_sidecar(buffer.str());
        if (!record) { return std::nullopt; }
        // The sidecar names an absolute path; fall back to the sidecar's directory when the
        // directory moved with the store.
        std::error_code error;
        if (!std::filesystem::exists(record->path, error)) {
            const std::filesystem::path sibling =
                sidecar.parent_path() / std::filesystem::path(record->path).filename();
            if (std::filesystem::exists(sibling, error)) {
                record->path = sibling.string();
            }
        }
        return record;
    }

    void forget_locked(const std::string& snapshot_path) {
        const auto found = index_.find(snapshot_path);
        if (found == index_.end()) { return; }
        index_.erase(found);
        std::error_code ignored;
        std::filesystem::remove(session_sidecar_path_for(snapshot_path), ignored);
    }

    void prune_locked() {
        if (max_bytes_ == 0 || index_.empty()) { return; }
        std::uint64_t total = 0;
        for (const auto& [path, record] : index_) {
            (void)path;
            total += record.bytes;
        }
        while (total > max_bytes_ && index_.size() > 1) {
            auto oldest = index_.begin();
            for (auto it = index_.begin(); it != index_.end(); ++it) {
                if (it->second.accessed_ms < oldest->second.accessed_ms) { oldest = it; }
            }
            total -= std::min(total, oldest->second.bytes);
            std::error_code ignored;
            std::filesystem::remove(oldest->second.path, ignored);
            std::filesystem::remove(session_sidecar_path_for(oldest->second.path), ignored);
            index_.erase(oldest);
        }
    }

    const std::filesystem::path directory_;
    const std::uint64_t max_bytes_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, SessionRecord> index_; // snapshot path -> record
    mutable std::atomic<std::uint32_t> sequence_{0};
};

} // namespace ninfer::runtime
