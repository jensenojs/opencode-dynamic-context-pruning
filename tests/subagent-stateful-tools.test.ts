import assert from "node:assert/strict"
import test from "node:test"
import { deduplicate } from "../lib/strategies/deduplication"
import { createSessionState } from "../lib/state"
import { defaultConfig, type PluginConfig } from "../lib/config"
import { Logger } from "../lib/logger"

/**
 * Regression tests for tool calls into long-lived subagent sessions.
 *
 * Scenario under test: an orchestrator dispatches multiple rounds of work
 * to the same subagent session (task tool + task_id resume).
 *
 * Deduplication groups calls by (tool name + params) signature. For tools
 * whose output depends on session state rather than inputs alone - such as
 * task - identical inputs do not imply redundant outputs, so these tools
 * must stay out of dedup grouping.
 *
 * Note: the extended subagent result injection (which rewrote the
 * <task_result> body of completed task parts with the subagent session's
 * current last assistant text) has been removed entirely: a stored task
 * result is immutable history and must never be rewritten. There is no
 * code path left to regression-test against; the removal is the fix.
 */

function buildConfig(overrides?: {
    dedupProtectedTools?: string[]
}): PluginConfig {
    const config = structuredClone(defaultConfig) as PluginConfig
    if (overrides?.dedupProtectedTools) {
        config.strategies.deduplication.protectedTools = [
            ...new Set([
                ...config.strategies.deduplication.protectedTools,
                ...overrides.dedupProtectedTools,
            ]),
        ]
    }
    return config
}

const logger = new Logger(false)

// ---------------------------------------------------------------------------
// Suite A: deduplication and stateful task calls
// ---------------------------------------------------------------------------

function registerToolCall(
    state: ReturnType<typeof createSessionState>,
    callID: string,
    tool: string,
    parameters: Record<string, unknown>,
    tokenCount: number,
) {
    state.toolIdList.push(callID)
    state.toolParameters.set(callID, { tool, parameters, tokenCount } as any)
}

test("dedup: resumed task calls with identical inputs keep every round's result", () => {
    // The same prompt re-dispatched to the same subagent session (task_id)
    // across rounds. Each round produces a distinct result, so none of the
    // calls may be grouped as duplicates.
    const state = createSessionState()
    const sameParams = { prompt: "[DELTA] adjudicate clause 6", task_id: "ses_sub123" }

    registerToolCall(state, "call-round-1", "task", sameParams, 10154)
    registerToolCall(state, "call-round-2", "task", sameParams, 10053)

    deduplicate(state, logger, buildConfig(), [])

    assert.equal(
        state.prune.tools.has("call-round-1"),
        false,
        "round-1 result was pruned: identical inputs do not imply redundant " +
            "outputs for tools backed by a long-lived session",
    )
    assert.equal(state.prune.tools.has("call-round-2"), false)
})

test("dedup: identical read calls are still deduplicated (control)", () => {
    // For idempotent tools, keeping only the latest output remains the
    // intended behavior.
    const state = createSessionState()
    const sameParams = { filePath: "/repo/src/index.ts" }

    registerToolCall(state, "call-read-1", "read", sameParams, 500)
    registerToolCall(state, "call-read-2", "read", sameParams, 500)

    deduplicate(state, logger, buildConfig(), [])

    assert.equal(state.prune.tools.has("call-read-1"), true)
    assert.equal(state.prune.tools.has("call-read-2"), false)
})

test("dedup: task calls with different parameters are untouched (control)", () => {
    const state = createSessionState()

    registerToolCall(state, "call-a", "task", { prompt: "round A" }, 1000)
    registerToolCall(state, "call-b", "task", { prompt: "round B" }, 1000)

    deduplicate(state, logger, buildConfig(), [])

    assert.equal(state.prune.tools.size, 0)
})

test("dedup: user protectedTools entries still apply on top of defaults (control)", () => {
    const state = createSessionState()
    const sameParams = { command: "npm test" }

    registerToolCall(state, "call-1", "bash", sameParams, 1000)
    registerToolCall(state, "call-2", "bash", sameParams, 1000)

    deduplicate(state, logger, buildConfig({ dedupProtectedTools: ["bash"] }), [])

    assert.equal(state.prune.tools.size, 0)
})
