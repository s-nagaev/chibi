# Context compaction, explained

This document describes how Chibi keeps a conversation inside the model's context window. It is written from the business perspective: what happens, in which order, and what the agent and the user see at each step. No code, no configuration file references — just the behavior an operator or a curious user can observe.

## Why compaction exists

Every model has a finite context window: the total amount of text (measured in tokens) it can consider at once — system prompt, tool descriptions, conversation history and tool results together. A long working session eventually outgrows that window. Without intervention the next request is simply rejected by the provider and the user sees an error instead of an answer.

Chibi solves this in layers. Each layer is less careful than the previous one, and each fires only when the previous one has not been enough. The layers, in order:

1. The agent itself compacts on request, guided by per-turn warnings.
2. The system compacts on its own after an answer has been delivered (forced post-answer compaction).
3. A last-resort trim guard silently hides the oldest material for a single request.
4. After an actual provider overflow, a reactive recovery pass repairs the history and retries.

## Context measurement and the per-turn context note

Before every request Chibi measures how large the conversation actually is. When the provider reported the size of the last request, that real number is used; otherwise Chibi falls back to a local estimate. The measured figure drives everything below.

The agent learns the current state through a small `<context_note>` block appended to the latest user message on every turn. It contains:

- **The approximate context size** — always, whenever a figure is available: roughly how many tokens the conversation occupies and what percentage that is of the limit. This is an observation, not an instruction.
- **A compaction advisory** — once the context passes half of the budget (50% by default, configurable): a statement that the context is getting full and which compaction tools exist. It does not order the agent to do anything; the standing instruction to compact lives in the system prompt.
- **A tool-results share warning** — when bulky background tool results (large JSON blobs returned by tools earlier in the conversation) occupy more than a quarter of the context, the note names the share and lists the largest offenders by tool name, mentioning that stale tool results can be dropped. This targets the most common form of silent bloat: a few huge tool answers nobody needs anymore.

The note is attached only to the outgoing request, never persisted into stored history, and never shown to the user. The agent is also instructed never to describe the note or the compaction mechanics to the user — compaction is plumbing, not content.

## Agent-initiated compaction

The agent has two compaction tools:

- **`summarize_history`** — replaces the conversation history with an exhaustive summary exchange. The summary fully replaces what came before: nothing of the old dialogue survives except what the summary captured. This is the right tool when the dialogue itself is large.
- **`clear_tool_call_history`** — drops past tool-call traffic and replaces it with a short summary of what the calls produced, keeping the rest of the dialogue intact. This is the right tool when stale tool results dominate.

The system prompt carries a standing rule, repeated in both tool descriptions: **compact FIRST, answer SECOND**. When the context note reports high usage or a warning — or the agent notices on its own that the conversation has grown long — it must call the appropriate tool before producing the substantive answer, then deliver the full answer as usual. A compaction may never cost the user their answer.

## Forced post-answer compaction

Warnings can be ignored: a model absorbed in a long task may keep working past every advisory until the context sits right at the edge of the window. The next request would then be trimmed by the last-resort guard (see below), losing real content without the model ever being asked.

To prevent that, Chibi re-checks every thread after an answer has been delivered. The check is fire-and-forget: it runs in the background, never delays the answer, never changes what the user receives, and any failure inside it is logged and swallowed. It runs at most once per delivered answer.

The trigger threshold is deliberately conservative: compaction fires only once the measured context has reached the point where the *next* request would be trimmed — the model's context window minus the space reserved for the summarizer's own output. (For models whose window is unknown, the configured `MAX_HISTORY_TOKENS` threshold is used instead.) In other words, forced compaction steps in exactly at the doorstep of content loss, not earlier — the agent's own tools remain the preferred mechanism, and this layer only catches what they missed.

When the trigger fires, the system compacts the conversation unconditionally in the background, replacing the stored history with a **seed**: the new, much smaller conversation state. The seed's summary message starts with a visible marker line — `[SYSTEM] Context was forcibly compacted on <date>; older turns were replaced by this summary` — so the agent can always tell that it is reading a compaction result rather than the original dialogue. After the marker the seed contains:

- a **detailed summary** of the earlier conversation — what was decided, what was learned, what was built;
- a **"Current work in progress"** section — the task that was interrupted mid-flight, so the agent continues it coherently instead of restarting;
- a **"Background tasks"** section — delegated work still outstanding at the moment of compaction.

The prompt instructs the agent that, on seeing the marker, it must treat the summary as the complete record of everything before it, read the current-work and background-task sections first, and never ask the user to repeat what the summary already covers. Agent-initiated compactions (the two tools above) never carry the marker — the agent asked for those itself, so a "this was done behind your back" banner would be false.

Only threads that talk to a user are covered; one-shot sub-agent conversations have no persisted history and no delivered answers, so there is nothing to re-check.

## The trim guard: last resort before an overflow

Before each request — and again if the conversation grows mid-turn — a guard compares the outgoing request against the fit threshold. If it does not fit, the guard performs a **lossy, transient trim**: the oldest conversation units are dropped, and messages that are still too large are truncated in place, until the request fits. This is not compaction: nothing is summarized, the stored history is untouched, and the dropped content is gone only for this one request.

Because the trim hides real content, the agent is told about it: the next context note carries a `guard_trim_warning` stating what was hidden (how many units, how many messages, how many truncated) and reminding the agent that the compaction tools can reduce future trims. The warning is read-once — it appears once, on the request that follows the trim, and then disappears.

Known nuance: an in-loop trim (one that happens after the request was already assembled) is recorded too late for the current request's note, so its disclosure lands on the **next** request — one request late. The warning always describes the trims since the last disclosure, so nothing is lost, only delayed.

## Reactive recovery after a provider overflow

If, despite everything, the provider rejects a request because the context window is exceeded, Chibi reacts: the conversation is compacted, the request is retried exactly once, and the user receives a normal answer (with a short note that the context was compressed). If the retry still does not fit, a synchronous last-resort shrink drops the oldest material in memory until the retried request fits — a deliberate, accepted loss, because the alternative is an error bubble for the user. The shrink only replaces the outgoing request once it is proven to fit; the user never sees a broken turn. This recovery is the safety net behind all preventive layers and can be disabled per deployment.

## When compaction fails: back-off

A compaction attempt can fail: the summarizer returns nothing, the result does not actually reduce the context, or the machinery crashes. Failures are contained — the history is left untouched, the delivered answer is unaffected, and nothing is shown to the user.

After a failed forced compaction the thread **backs off**: the post-answer check skips that thread for the next three delivered turns or thirty minutes, whichever comes first, so a persistently failing thread does not burn a compaction attempt on every single turn. The thread's normal operation continues throughout; only the automatic re-compaction is suspended, and the agent's own tools and the guard still work.

## Configuration

The whole forced post-answer layer is governed by one switch, `CONTEXT_FORCED_COMPACTION_ENABLED` (default: `true`). Setting it to `false` disables the post-answer check entirely; the agent's own compaction tools, the guard and the reactive recovery remain in place, so overflow protection degrades to the older behavior rather than disappearing. The reactive recovery has its own separate switch, `REACTIVE_CONTEXT_RECOVERY`.

## Which level fires when

| Situation | What happens | Who notices |
| --- | --- | --- |
| Context passes ~50% of the budget | Compaction advisory appears in the `<context_note>` | The agent, via the note |
| Bulky tool results pass ~25% of the context | Tool-results warning with the top offenders | The agent, via the note |
| Agent decides to compact | `summarize_history` or `clear_tool_call_history` runs before the answer | Nobody visible; the history shrinks |
| Context reaches window minus summarizer reservation after an answer | Forced post-answer compaction replaces history with a seed (marker + summary + current work + background tasks) | The agent, via the seed marker; the user never does |
| Outgoing request would not fit | Trim guard hides oldest units for this request only | The agent, via `guard_trim_warning` on the following request (in-loop trims: one request late) |
| Provider rejects a request as oversized | Reactive recovery: compact, retry once, last-resort shrink if needed | The user gets a normal answer with a short "context compressed" note |
| A forced compaction attempt fails | History untouched; thread backs off 3 turns / 30 minutes | Nobody visible; failure is logged |
