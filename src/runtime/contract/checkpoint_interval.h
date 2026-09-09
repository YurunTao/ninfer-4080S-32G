#pragma once

// Periodic interior capture frontiers.
//
// A long prompt otherwise has a rewindable point only at its rewrite checkpoint (the start of
// the current turn), so an edit anywhere below that point re-prefills the whole prompt. The
// request plan offers a private long anchor every `interval` tokens; the ordinary retention
// policy decides which of those offers survive, so the count stays bounded by
// --max-long-anchors-per-continuation.

#include <cstdint>
#include <vector>

namespace ninfer::runtime {

// Interior frontiers strictly below `prompt_tokens`, at multiples of `interval`. The prompt
// frontier itself is never returned: it is already the session endpoint. A zero interval
// disables periodic capture.
[[nodiscard]] inline std::vector<std::uint32_t> periodic_capture_frontiers(
    std::uint32_t prompt_tokens, std::uint32_t interval) {
    std::vector<std::uint32_t> frontiers;
    if (interval == 0 || prompt_tokens <= interval) { return frontiers; }
    frontiers.reserve((prompt_tokens / interval) + 1U);
    for (std::uint32_t frontier = interval; frontier < prompt_tokens;) {
        frontiers.push_back(frontier);
        if (frontier > prompt_tokens - interval) { break; }
        frontier += interval;
    }
    return frontiers;
}

} // namespace ninfer::runtime
