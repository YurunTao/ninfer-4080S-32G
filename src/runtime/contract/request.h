#pragma once

// Runtime request contracts.
//
// The merge of upstream's contract split (request.h / execution.h / resources.h /
// timing.h) with this branch's single contract header (types.h) left the same
// runtime types declared twice, and the branch's engine, planner and generation
// layer all depend on types.h. The two headers therefore hold one contract with a
// single definition site: types.h owns the declarations and this header forwards,
// so every existing `runtime/contract/request.h` include keeps working.
//
// The engine-side fields this header used to add on top of types.h are declared
// there instead: ResolvedExecutionOptions, ResolvedRequestOptions,
// ContinuationAction, OutputDecision, CancellationFlagView and RequestPlanSummary.

#include "runtime/contract/types.h"
