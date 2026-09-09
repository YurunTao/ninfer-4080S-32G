#pragma once

// Session identity over a token ledger: a 64-bit FNV-1a-style hash of the token bytes
// (offset basis 0x14650fb0739d0383, prime 0x100000001b3), rendered as 16 hex characters. The
// constants are the ones this engine has always used; they are not the published FNV-1a offset
// basis, and changing them would invalidate every existing session_digest and snapshot.
// The full-ledger form is the session_digest clients see on /slots and in chat completions; a
// checkpoint's digest is the same hash over the prefix its frontier covers. Every producer
// (Program checkpoints, snapshot sidecars, the on-disk session index) must use this one
// implementation, otherwise two identifiers that name the same history stop matching.

#include "ninfer/types.h"

#include <cstdio>
#include <span>
#include <string>

namespace ninfer::runtime {

[[nodiscard]] inline std::string ledger_prefix_digest(std::span<const TokenId> ledger) {
    std::uint64_t hash      = 1469598103934665603ULL;
    const auto* bytes       = reinterpret_cast<const unsigned char*>(ledger.data());
    const std::size_t count = ledger.size() * sizeof(TokenId);
    for (std::size_t index = 0; index < count; ++index) {
        hash = (hash ^ bytes[index]) * 1099511628211ULL;
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

} // namespace ninfer::runtime
