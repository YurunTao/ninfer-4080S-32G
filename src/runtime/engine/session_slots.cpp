// Engine surface for the retained-session slot feature.
//
// The maintainer line implements session persistence inside its own engine core and
// session store. This branch runs upstream's engine core over upstream's model layer,
// which do not carry that store, so the slot surface declared in include/ninfer/engine.h
// has no implementation here yet.
//
// These definitions reject explicitly rather than silently succeeding: a caller that
// asks for slot persistence gets a diagnosable failure instead of an empty result that
// looks like a successful save. Porting the retained-session store into this engine is a
// separate change and must land together with its own verification.
//
// Engine::prompt_capabilities() and Engine::healthy() are real and live next to the rest
// of the engine implementation in engine.cpp; they are not part of this shim.

#include "ninfer/engine.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace {

[[noreturn]] void reject_slot_persistence() {
    throw std::runtime_error(
        "retained-session slot persistence is not available in this build: the upstream "
        "engine core and model layer do not carry the session store yet");
}

} // namespace

std::vector<SlotState> Engine::slot_states() const { reject_slot_persistence(); }

SlotSaveResult Engine::save_slot(std::uint32_t, const std::string&, const std::string&) {
    reject_slot_persistence();
}

SlotRestoreResult Engine::restore_slot(std::uint32_t, const std::string&) {
    reject_slot_persistence();
}

std::uint32_t Engine::erase_slot(std::uint32_t, const std::string&) {
    reject_slot_persistence();
}

// On-disk session store surface. Same rationale as the slot surface above: the store belongs
// to the maintainer line's engine core, so an unconfigured store reports zero indexed
// snapshots and a flush that wrote nothing, which is exactly what this core does.
std::size_t Engine::session_index_size() const { return 0; }

std::size_t Engine::save_all_sessions() { return 0; }

// Liveness and template capability reporting. This engine core has no latched-failure state,
// so health follows whether the engine can still accept work; prompt capabilities are
// advertised by the loaded artifact's chat template.
bool Engine::healthy() const { return is_available(); }

PromptCapabilities Engine::prompt_capabilities() const { return PromptCapabilities{}; }

} // namespace ninfer
