// On-disk session index: sidecar round-trip, tolerant parsing, deepest-prefix lookup,
// write-once path allocation, rebuild against the real directory, and LRU pruning.

#include "runtime/engine/session_store.h"
#include "runtime/contract/ledger_digest.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::runtime::ledger_prefix_digest;
using ninfer::runtime::parse_session_sidecar;
using ninfer::runtime::serialize_session_sidecar;
using ninfer::runtime::SessionCheckpointMeta;
using ninfer::runtime::SessionRecord;
using ninfer::runtime::SessionStore;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<TokenId> tokens(std::uint32_t count) {
    std::vector<TokenId> out(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        out[index] = static_cast<TokenId>(index + 1);
    }
    return out;
}

void write_file(const std::filesystem::path& path, std::string_view content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
}

SessionRecord make_record(const std::filesystem::path& directory, std::string name,
                          std::string binding, std::uint32_t frontier,
                          std::uint64_t bytes, std::uint64_t accessed_ms,
                          std::uint32_t generation,
                          std::vector<SessionCheckpointMeta> checkpoints) {
    SessionRecord record;
    record.path              = (directory / name).string();
    record.model_binding     = std::move(binding);
    record.snapshot_frontier = frontier;
    record.session_digest    = ledger_prefix_digest(tokens(frontier));
    record.checkpoints       = std::move(checkpoints);
    record.bytes             = bytes;
    record.created_ms        = accessed_ms;
    record.accessed_ms       = accessed_ms;
    record.generation        = generation;
    return record;
}

} // namespace

int main() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "ninfer_session_store_test";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root, error);

    int failures = 0;

    // Digest vectors come from an independent implementation of the engine's hash spec
    // (offset basis 0x14650fb0739d0383, prime 0x100000001b3, little-endian token bytes).
    failures += check(ledger_prefix_digest({}) == "14650fb0739d0383", "empty digest");
    failures += check(ledger_prefix_digest(tokens(1)) == "91599a98306c74a2", "one-token digest");
    const std::vector<TokenId> three{1, 2, 3};
    failures += check(ledger_prefix_digest(three) == "6fdedb25bbb27e53", "three-token digest");
    const std::vector<TokenId> eight{1, 2, 3, 4, 5, 6, 7, 8};
    failures += check(ledger_prefix_digest(eight) == "a1561a40a1947f8b", "eight-token digest");
    failures += check(ledger_prefix_digest(tokens(8)) == ledger_prefix_digest(eight),
                      "digest is a function of the token bytes only");

    // Sidecar round trip, including a binding with newlines and several checkpoints.
    {
        SessionRecord record = make_record(root, "a.slot", "qwen3.8-27b\nbinding", 64, 1024, 111,
                                           3,
                                           {SessionCheckpointMeta{16, ledger_prefix_digest(tokens(16))},
                                            SessionCheckpointMeta{64, ledger_prefix_digest(tokens(64))}});
        const std::string text = serialize_session_sidecar(record);
        const auto parsed      = parse_session_sidecar(text);
        failures += check(parsed.has_value(), "sidecar did not round-trip");
        if (parsed) {
            failures += check(parsed->path == record.path, "round-trip path");
            failures += check(parsed->model_binding == record.model_binding, "round-trip binding");
            failures += check(parsed->snapshot_frontier == 64, "round-trip frontier");
            failures += check(parsed->session_digest == record.session_digest, "round-trip digest");
            failures += check(parsed->bytes == 1024, "round-trip bytes");
            failures += check(parsed->generation == 3, "round-trip generation");
            failures += check(parsed->checkpoints.size() == 2, "round-trip checkpoint count");
            failures += check(parsed->checkpoints[0].frontier == 16 &&
                                  parsed->checkpoints[1].frontier == 64,
                              "round-trip checkpoint order");
            failures += check(parsed->checkpoint_at_or_below(40) != nullptr &&
                                  parsed->checkpoint_at_or_below(40)->frontier == 16,
                              "checkpoint_at_or_below picks the deepest covered frontier");
            failures += check(parsed->checkpoint_at_or_below(15) == nullptr,
                              "checkpoint_at_or_below below every frontier");
        }
    }

    // Endpoint checkpoint synthesis when the writer recorded none.
    {
        SessionRecord record = make_record(root, "b.slot", "m", 32, 1, 1, 1, {});
        const auto parsed = parse_session_sidecar(serialize_session_sidecar(record));
        failures += check(parsed && parsed->checkpoints.size() == 1 &&
                              parsed->checkpoints[0].frontier == 32 &&
                              parsed->checkpoints[0].digest == record.session_digest,
                          "storage frontier became a checkpoint");
    }

    // Tolerant rejection of untrustworthy sidecars.
    {
        failures += check(!parse_session_sidecar("snapshot_path=/x\nmodel_binding=m\n"),
                          "missing digest accepted");
        failures += check(!parse_session_sidecar(
                              "meta_version=99\nsnapshot_path=/x\nmodel_binding=m\n"
                              "snapshot_frontier=1\nsession_digest=00\n"),
                          "future meta version accepted");
        failures += check(!parse_session_sidecar(
                              "snapshot_path=/x\nmodel_binding=m\nsnapshot_frontier=1\n"
                              "session_digest=00\nsnapshot_bytes=not-a-number\n"),
                          "malformed number accepted");
        failures += check(!parse_session_sidecar(
                              "snapshot_path=/x\nmodel_binding=m\nsnapshot_frontier=1\n"
                              "session_digest=00\ncheckpoint=abc\n"),
                          "malformed checkpoint accepted");
        failures += check(parse_session_sidecar(
                              "snapshot_path=/x\nmodel_binding=m\nsnapshot_frontier=4\n"
                              "session_digest=00\nunknown_key=value\n")
                              .has_value(),
                          "unknown key rejected");
    }

    // Write-once paths are unique and carry the snapshot suffix.
    {
        SessionStore store(root / "allocate");
        const std::filesystem::path first  = store.allocate_snapshot_path();
        const std::filesystem::path second = store.allocate_snapshot_path();
        failures += check(first != second, "allocated paths collided");
        failures += check(first.extension() == ".slot", "allocated suffix");
    }

    // Index behaviour over a real directory.
    {
        const std::filesystem::path directory = root / "index";
        SessionStore store(directory);

        SessionRecord short_session =
            make_record(directory, "short.slot", "model-a", 32, 100, 10, 1,
                        {SessionCheckpointMeta{32, ledger_prefix_digest(tokens(32))}});
        SessionRecord long_session =
            make_record(directory, "long.slot", "model-a", 64, 200, 20, 1,
                        {SessionCheckpointMeta{32, ledger_prefix_digest(tokens(32))},
                         SessionCheckpointMeta{64, ledger_prefix_digest(tokens(64))}});
        SessionRecord other_model =
            make_record(directory, "other.slot", "model-b", 64, 200, 30, 1,
                        {SessionCheckpointMeta{64, ledger_prefix_digest(tokens(64))}});
        for (const SessionRecord& record : {short_session, long_session, other_model}) {
            write_file(record.path, "payload");
            store.record(record);
        }
        failures += check(store.size() == 3, "record did not index three sessions");

        // A 64-token prompt matches only the long session's deepest checkpoint.
        auto hit = store.lookup(tokens(64), "model-a");
        failures += check(hit.has_value(), "deepest checkpoint did not match");
        if (hit) {
            failures += check(hit->record.path == long_session.path,
                              "lookup chose the wrong session");
            failures += check(hit->frontier == 64, "lookup reported the wrong matched frontier");
        }
        // A 40-token prompt matches the 32-token checkpoint of the most recently used session.
        auto shallow = store.lookup(tokens(40), "model-a");
        failures += check(shallow.has_value() && shallow->record.path == long_session.path &&
                              shallow->frontier == 32,
                          "shallow lookup did not report the covered checkpoint");
        // A different model binding never matches.
        failures += check(!store.lookup(tokens(64), "model-c").has_value(),
                          "lookup ignored the model binding");
        // A prompt shorter than every checkpoint misses.
        failures += check(!store.lookup(tokens(8), "model-a").has_value(),
                          "short prompt matched a checkpoint it cannot cover");
        // A prompt that diverges in its last token still matches the 32-token checkpoint, but
        // not the 64-token one.
        {
            std::vector<TokenId> diverging = tokens(64);
            diverging[63] += 1;
            auto partial                   = store.lookup(diverging, "model-a");
            failures += check(partial.has_value() && partial->frontier == 32,
                              "diverging prompt did not fall back to the covered checkpoint");
        }

        // Newer generation wins at an equal frontier.
        SessionRecord newer = make_record(directory, "newer.slot", "model-a", 64, 200, 40, 7,
                                          {SessionCheckpointMeta{
                                              64, ledger_prefix_digest(tokens(64))}});
        write_file(newer.path, "payload");
        store.record(newer);
        auto newest = store.lookup(tokens(64), "model-a");
        failures += check(newest.has_value() && newest->record.path == newer.path,
                          "newest generation did not win the tie");

        // Rebuild from sidecars only, then a stale sidecar and an orphan temporary file vanish.
        write_file(directory / "stale.slot.meta",
                   serialize_session_sidecar(make_record(directory, "gone.slot", "model-a", 16, 1,
                                                         1, 1, {})));
        write_file(directory / "orphan.slot.tmp", "half-written");
        const std::size_t indexed = store.rebuild();
        failures += check(indexed == 4, "rebuild indexed the wrong number of sessions");
        failures += check(!std::filesystem::exists(directory / "orphan.slot.tmp"),
                          "orphaned temporary file survived rebuild");
        failures += check(!std::filesystem::exists(directory / "stale.slot.meta"),
                          "stale sidecar survived rebuild");
        auto after_rebuild = store.lookup(tokens(64), "model-a");
        failures += check(after_rebuild.has_value() && after_rebuild->record.path == newer.path,
                          "rebuilt index did not reproduce the lookup");
    }

    // LRU pruning honours the byte budget and keeps the newest snapshot.
    {
        const std::filesystem::path directory = root / "prune";
        SessionStore store(directory, 250);
        SessionRecord oldest = make_record(directory, "oldest.slot", "model-a", 16, 100, 10, 1,
                                           {});
        SessionRecord middle = make_record(directory, "middle.slot", "model-a", 32, 100, 20, 2,
                                           {});
        SessionRecord newest = make_record(directory, "newest.slot", "model-a", 64, 100, 30, 3,
                                           {});
        for (const SessionRecord& record : {oldest, middle, newest}) {
            write_file(record.path, "payload");
            store.record(record);
        }
        // 300 bytes over a 250-byte budget: exactly one eviction, and it must be the LRU one.
        failures += check(store.size() == 2, "prune did not evict to the byte budget");
        failures += check(!std::filesystem::exists(oldest.path), "prune kept the LRU snapshot");
        failures += check(!std::filesystem::exists(
                              std::filesystem::path(oldest.path + ".meta")),
                          "prune kept the LRU sidecar");
        failures += check(std::filesystem::exists(newest.path), "prune evicted the newest snapshot");

        // A budget smaller than one session still leaves the store usable.
        SessionStore tiny(directory, 1);
        const std::size_t survivors = tiny.rebuild();
        failures += check(survivors == 1, "tiny budget emptied the store");
    }

    std::filesystem::remove_all(root, error);
    if (failures != 0) {
        std::cerr << failures << " session store checks failed\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
