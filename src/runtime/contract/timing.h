#pragma once

// Upstream's contract split (request.h / execution.h / resources.h / timing.h)
// overlaps almost entirely with this branch's single contract header. The branch's
// engine, planner, scheduler and generation layer all consume types.h, so types.h
// is the single definition site and this header forwards to it, keeping every
// existing include of this path working.

#include "runtime/contract/types.h"