// Periodic interior capture frontiers: multiples of the interval strictly below the prompt
// frontier, with no overflow at the top of the context window.

#include "runtime/contract/checkpoint_interval.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using ninfer::runtime::periodic_capture_frontiers;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    failures += check(periodic_capture_frontiers(100000, 0).empty(),
                      "a zero interval produced frontiers");
    failures += check(periodic_capture_frontiers(16000, 16384).empty(),
                      "a prompt below one interval produced a frontier");
    failures += check(periodic_capture_frontiers(16384, 16384).empty(),
                      "a frontier at the prompt edge was offered");

    const std::vector<std::uint32_t> three = periodic_capture_frontiers(50000, 16384);
    failures += check(three == std::vector<std::uint32_t>{16384, 32768, 49152},
                      "frontiers below the prompt frontier");

    const std::vector<std::uint32_t> exact = periodic_capture_frontiers(49152, 16384);
    failures += check(exact == std::vector<std::uint32_t>{16384, 32768},
                      "exact multiple of the interval included the prompt frontier");

    const std::vector<std::uint32_t> full = periodic_capture_frontiers(262144, 16384);
    failures += check(full.size() == 15, "full-context frontier count");
    failures += check(!full.empty() && full.front() == 16384 && full.back() == 245760,
                      "full-context frontier bounds");

    // No overflow when the prompt frontier sits at the uint32 ceiling.
    const std::vector<std::uint32_t> huge =
        periodic_capture_frontiers(std::uint32_t{0xFFFFFFFFu}, 16384);
    failures += check(!huge.empty() && huge.back() < 0xFFFFFFFFu,
                      "frontier overflowed the prompt frontier");

    if (failures != 0) {
        std::cerr << failures << " checkpoint interval checks failed\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
