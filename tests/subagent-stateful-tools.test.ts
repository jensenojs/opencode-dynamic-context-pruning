import assert from "node:assert/strict"
import test from "node:test"
import { deduplicate } from "../lib/strategies/deduplication"
import { injectExtendedSubAgentResults } from "../lib/messages/inject/subagent-results"
import { createSessionState, type WithParts } from "../lib/state"
import type { PluginConfig } from "../lib/config"
import { Logger } from "../lib/logger"

/**
 * Regression tests for stateful subagent tool calls.
 *
 * Incident background: an orchestrator dispatched the same prompt to a
 * resumed subagent session (task tool + task_id) across multiple rounds.
 * Two independent defects made the orchestrator believe the subagent was
 * repeating itself or that its replies were lost:
 *
 *   1. The deduplication strategy groups calls by (tool name + params)
 *      signature. For a *stateful* tool like task-with-task_id, identical
 *      inputs do NOT imply redundant outputs - each round produces a new,
 *      distinct result. Deduplication pruned every round but the last and
 *      replaced their outputs with "information superseded", actively
 *      misinforming the model.
 *
 *   2. injectExtendedSubAgentResults rewrites the <task_result> body of
 *      every historical task part with the subagent session's CURRENT last
 *      assistant text. Its correctness relies on an in-memory cache
 *      (state.subAgentResultCache) that is not persisted. After a server
 *      restart, all historical task results pointing at the same subagent
 *      session are uniformly rewritten to the subagent's latest reply -
 *      the orchestrator literally sees N identical "results".
 */

function buildConfig(overrides?: {
    dedupProtectedTools?: string[]
    allowSubAgents?: boolean
}): PluginConfig {
    return {
        enabled: true,
        debug: false,
        pruneNotification: "off",
        pruneNotificationType: "chat",
        commands: { enabled: true, protectedTools: [] },
        manualMode: { enabled: false, automaticStrategies: true },
        turnProtection: { enabled: false, turns: 4 },
        experimental: {
            allowSubAgents: overrides?.allowSubAgents ?? false,
            customPrompts: false,
        },
        protectedFilePatterns: [],
        compress: {
            mode: "message",
            permission: "allow",
            showCompression: false,
            maxContextLimit: 150000,
            minContextLimit: 50000,
            nudgeFrequency: 5,
            iterationNudgeThreshold: 15,
            nudgeForce: "soft",
            protectedTools: ["task"],
            protectTags: false,
            protectUserMessages: false,
        },
        strategies: {
            deduplication: {
                enabled: true,
                protectedTools: overrides?.dedupProtectedTools ?? [],
            },
            purgeErrors: { enabled: true, turns: 4, protectedTools: [] },
        },
    } as PluginConfig
}

const logger = new Logger(false)

// ---------------------------------------------------------------------------
// Suite A: deduplication must not destroy results of stateful task calls
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

test("dedup: resumed task calls with identical inputs must keep every round's result", () => {
    // Reproduces the incident: the SAME prompt re-dispatched to the SAME
    // subagent session (task_id) across rounds. Each round produced a
    // different result (blind-review round vs adjudication round).
    // Deduplication must not treat them as redundant.
    const state = createSessionState()
    const sameParams = { prompt: "[DELTA] adjudicate clause 6", task_id: "ses_sub123" }

    registerToolCall(state, "call-round-1", "task", sameParams, 10154)
    registerToolCall(state, "call-round-2", "task", sameParams, 10053)

    deduplicate(state, logger, buildConfig(), [])

    assert.equal(
        state.prune.tools.has("call-round-1"),
        false,
        "round-1 result was pruned: dedup-by-input is unsound for stateful tools " +
            "(identical prompt + task_id resume => distinct outputs)",
    )
    assert.equal(state.prune.tools.has("call-round-2"), false)
})

test("dedup: identical read calls are still deduplicated (control)", () => {
    // Guards against overcorrection: for idempotent tools, keeping only the
    // latest output remains the intended behavior.
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

test("dedup: protectedTools [task] config workaround keeps resumed results", () => {
    // Documents the user-side mitigation shipped in dcp.jsonc during the
    // incident. This must keep passing regardless of the default fix.
    const state = createSessionState()
    const sameParams = { prompt: "same", task_id: "ses_sub123" }

    registerToolCall(state, "call-1", "task", sameParams, 1000)
    registerToolCall(state, "call-2", "task", sameParams, 1000)

    deduplicate(state, logger, buildConfig({ dedupProtectedTools: ["task"] }), [])

    assert.equal(state.prune.tools.size, 0)
})

// ---------------------------------------------------------------------------
// Suite B: subagent result injection must not rewrite historical results
// ---------------------------------------------------------------------------

function assistantMessage(id: string, text: string): WithParts {
    return {
        info: {
            id,
            role: "assistant",
            sessionID: "ses_sub123",
            agent: "explore",
            time: { created: 1 },
        } as WithParts["info"],
        parts: [
            {
                id: `${id}-p1`,
                messageID: id,
                sessionID: "ses_sub123",
                type: "text",
                text,
            } as any,
        ],
    }
}

function parentTaskPart(callID: string, roundResult: string) {
    return {
        id: `${callID}-part`,
        messageID: `msg-${callID}`,
        sessionID: "ses_parent",
        type: "tool" as const,
        tool: "task",
        callID,
        state: {
            status: "completed" as const,
            input: { prompt: "same prompt", task_id: "ses_sub123" },
            output: `<task_result>\n${roundResult}\n</task_result>`,
            metadata: { sessionId: "ses_sub123" },
        },
    }
}

function parentMessage(callID: string, roundResult: string): WithParts {
    return {
        info: {
            id: `msg-${callID}`,
            role: "assistant",
            sessionID: "ses_parent",
            agent: "assistant",
            time: { created: 1 },
        } as WithParts["info"],
        parts: [parentTaskPart(callID, roundResult) as any],
    }
}

function mockClientReturning(subAgentMessages: WithParts[]) {
    return {
        session: {
            messages: async () => ({ data: subAgentMessages }),
        },
    }
}

test("inject: cold cache must not rewrite historical task results to the subagent's latest reply", async () => {
    // Exact reproduction of the "subagent is repeating itself" symptom:
    // two completed task calls into the same resumed subagent session,
    // each with its OWN result stored. The subagent session has since
    // advanced to round 2. A fresh SessionState simulates a server restart
    // (subAgentResultCache is in-memory only and not persisted).
    //
    // Expected invariant: each historical result is preserved.
    // Current behavior: BOTH outputs become "ROUND-2 result" - the
    // orchestrator sees two identical results and concludes the subagent
    // is parroting its previous message.
    const state = createSessionState()
    const messages = [
        parentMessage("call-round-1", "ROUND-1 blind-review result"),
        parentMessage("call-round-2", "ROUND-2 adjudication result"),
    ]

    const subAgentNow = [
        assistantMessage("sub-msg-1", "ROUND-1 blind-review result"),
        assistantMessage("sub-msg-2", "ROUND-2 adjudication result"),
    ]

    await injectExtendedSubAgentResults(
        mockClientReturning(subAgentNow),
        state,
        logger,
        messages,
        true,
    )

    const round1Output = (messages[0].parts[0] as any).state.output as string
    const round2Output = (messages[1].parts[0] as any).state.output as string

    assert.ok(
        round1Output.includes("ROUND-1 blind-review result"),
        `historical round-1 result was destroyed by injection; got:\n${round1Output}`,
    )
    assert.ok(
        round2Output.includes("ROUND-2 adjudication result"),
        `round-2 result missing; got:\n${round2Output}`,
    )
})

test("inject: just-completed task result survives a cold cache (control)", async () => {
    // When the subagent session has NOT advanced beyond this call, the
    // injected text is identical to the stored body and the output must
    // still contain the original result.
    const state = createSessionState()
    const messages = [parentMessage("call-only", "THE final result")]

    const subAgentNow = [assistantMessage("sub-msg-1", "THE final result")]

    await injectExtendedSubAgentResults(
        mockClientReturning(subAgentNow),
        state,
        logger,
        messages,
        true,
    )

    const output = (messages[0].parts[0] as any).state.output as string
    assert.ok(output.includes("THE final result"))
})

test("inject: disabled via allowSubAgents=false leaves outputs untouched (control)", async () => {
    const state = createSessionState()
    const messages = [parentMessage("call-1", "original body")]

    const subAgentNow = [assistantMessage("sub-msg-1", "newer reply")]

    await injectExtendedSubAgentResults(
        mockClientReturning(subAgentNow),
        state,
        logger,
        messages,
        false,
    )

    const output = (messages[0].parts[0] as any).state.output as string
    assert.ok(output.includes("original body"))
    assert.ok(!output.includes("newer reply"))
})
