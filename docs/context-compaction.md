# Context compaction, explained

This document describes how Chibi keeps a conversation inside the model's context window. It is written from the behavior perspective: what happens, in which order, and what the agent and the user see at each step. It names the tools, the configuration knobs and the stored formats an operator or a curious reader will meet, but it stays free of code. Where this document and the implementation ever disagree, the code is the source of truth.

## Why compaction exists

Every model has a finite context window: the total amount of text (measured in tokens) it can consider at once. System prompt, tool descriptions, conversation history and tool results all share that budget. A long working session eventually outgrows it. Without intervention the next request is simply rejected by the provider and the user sees an error instead of an answer.

Chibi solves this in layers. Each layer is less careful than the previous one, and each fires only when the previous ones have not been enough. The layers, in order:

1. The agent itself compacts on request, guided by per-turn warnings.
2. The system compacts on its own after an answer has been delivered (forced post-answer compaction).
3. Transient guards trim the outgoing request when it would not fit: once before the provider call, and again between the steps of a multi-step tool loop.
4. After an actual provider overflow, a reactive recovery pass repairs the history and retries.

## How the context is measured

Everything below depends on one number: how large the outgoing request actually is. All of it goes through a single measurement entry point; nothing else in the system computes a request size on its own.

The figure has two parts. The base is the provider-reported prompt token count of the last successful turn, persisted per thread as a durable usage record together with the model it was measured on and the number of conversation messages the measured prompt contained. On Anthropic, tokens served from the cache are added back, so the record reflects the physical prompt size, not the billed one. On top of that base, the messages appended since the measurement are counted locally by a heuristic tokenizer (roughly one token per four characters), which also counts serialized tool-call arguments, because for tool-heavy turns those are the bulk of the volume.

The heuristic is deliberately simple and no tokenizer ships with Chibi. It counts only the conversation content, roles and tool-call arguments, while the provider figure covers the entire request (system prompt, skills, tool schemas, structural overhead), so the estimate under-reports a real request. That is an accepted temporary accuracy loss: the durable provider measurement, not the estimate, is the source of truth.

A stored record can be unusable, and in each of these cases the whole list falls back to the pure heuristic estimate:

- no record exists yet (the thread was never measured);
- the record was measured on a different model (a different model means a different tokenizer and a different window);
- the record is malformed (non-positive token count or message count);
- the record is stale (it contains more messages than the history now holds, which means the history was truncated or replaced without a measurement since).

A degraded figure can be wrong, but it can never block a request or trigger a compaction on its own.

The model's context window comes from a built-in table of provider-advertised windows. Lookup is exact first, then through normalization (trailing date and snapshot suffixes removed), then through the longest matching family prefix. Unknown models resolve to no window at all. The configured `max_history_tokens` (default: 200000) is not a universal stand-in for a missing window: when the window cannot be resolved, both transient guards let the request through unchanged and defer to the reactive recovery. `max_history_tokens` is used as the bound in exactly three places: the verification budget for compactions, the forced-compaction trigger threshold, and the summarizer fit fallback (60% of it) for the summarizer call's own input.

## The per-turn context note

Before every request Chibi assembles a small `<context_note>` block and appends it to the last user message of the outgoing request. It contains:

- **The approximate context size**, whenever a figure is available: how many tokens the conversation occupies and what percentage that is of the `max_history_tokens` limit. When no real measurement is cached, the line simply reads `approximate_context_size: n/a`.
- **A compaction advisory** once the context passes half of the budget (50% by default, configurable): a statement that the context is getting full and a reminder of which compaction tools exist. It does not order the agent to do anything; the standing instruction to compact lives in the system prompt.
- **A tool-results share warning** when bulky background tool results (large JSON blobs returned by tools earlier in the conversation) occupy more than a quarter (25%) of the context. The note names the share and lists the largest offenders by tool name with their estimated sizes, mentioning that stale tool results can be dropped with `clear_tool_call_history`. This targets the most common form of silent bloat: a few huge tool answers nobody needs anymore.
- **A guard-trim warning** when one of the transient guards had to hide content from a recent request. It states what was hidden (how many oldest units, how many messages, how many truncated) and reminds the agent that the compaction tools can reduce future trims. The warning is read-once: it appears on the request that follows the trim and then disappears.

The warnings are informational: they report how full the context is and name the tools. Compaction policy and timing live in the system prompt, not in the note.

The note is attached to the outgoing request only. It is never persisted into stored history (it is stripped again at the persistence boundary) and never shown to the user. Requests that have no user interface, such as sub-agent conversations, never carry the note at all.

## The agent's own compaction tools

The agent has two compaction tools, and the system prompt carries a standing rule repeated in both descriptions: **compact FIRST, answer SECOND**. When the context note reports high usage or a warning, or the agent notices on its own that the conversation has grown long, it must call the appropriate tool before producing the substantive answer, then deliver the full answer as usual. A compaction may never cost the user their answer.

**`summarize_history`** replaces the whole dialogue with the summary the agent provides. The replacement is built as a two-message seed: a short user question ("What we were talking about?") and an assistant answer holding the summary. Nothing of the old dialogue survives except what the summary captured, so the description demands an exhaustive one. An empty or whitespace-only summary is rejected with an error rather than wiping the thread. The seed is written through the same replacement helper the automatic path uses; the current turn's own messages are committed by the turn pipeline after the tool returns, so the stored history becomes the seed followed by the current turn.

**`clear_tool_call_history`** drops past tool-call traffic and keeps the rest of the dialogue verbatim. Concretely: every stored tool-result message is removed, and every earlier assistant message that carried tool calls keeps its text but loses its call payloads, remaining as an empty stub. The rest of the dialogue is re-persisted unchanged, with the tool-call arguments of earlier turns gone.

One mechanism deserves an explicit mention, because it is easy to miss: the summary that replaces the dropped traffic is supplied as the `summary` argument of the `clear_tool_call_history` call itself, and that is exactly where it survives. The drop is applied to the history as it existed at call time; the current call and its result are persisted afterwards, so the summary stays visible to the model as the arguments of the most recent tool call. A subsequent second call in the same thread does wipe the previous call's summary, because by then it has become part of the stored history.

The split of labor between the two tools mirrors the two shapes of bloat: prefer `clear_tool_call_history` when stale tool results dominate, use `summarize_history` when the dialogue itself is large.

## Forced post-answer compaction

Warnings can be ignored: a model absorbed in a long task may keep working past every advisory until the context sits right at the edge of the window. The next request would then be trimmed by the transient guard, losing real content without the model ever being asked.

To prevent that, Chibi re-checks every thread after an answer has been delivered. The check is fire-and-forget: it runs in the background, never delays the answer, never changes what the user receives, and any failure inside it is logged and swallowed. It runs exactly once per delivered answer, and only for threads that talk to a user; one-shot sub-agent conversations have no persisted history and no delivered answers, so there is nothing to re-check.

The trigger threshold is `min(max_history_tokens, the model's context window)`, falling back to `max_history_tokens` alone when the window is unknown. Reaching the threshold triggers compaction; the window cap only keeps the trigger inside models whose physical context is smaller than the configured budget. In other words, forced compaction steps in when the configured limit is reached, not earlier: the agent's own tools remain the preferred mechanism, and this layer only catches what they missed.

Before the summarizer is called, several cheap conditions can skip the attempt entirely: the `CONTEXT_FORCED_COMPACTION_ENABLED` switch is off, a compaction for this thread is already in flight, the thread is backing off after a previous failure, the stored history is empty, or the history consists solely of a compaction seed (a compaction just happened and there is nothing new to summarize; a thread that merely contains a seed but regrew past the threshold stays compactable).

When the trigger fires, the conversation is collapsed by a summarizer call into a seed: the same two-message shape the `summarize_history` tool writes, but with one addition. The seed's summary message starts with a visible marker line:

`[SYSTEM] Context was forcibly compacted on <date>; older turns were replaced by this summary`

The date is written as a UTC timestamp. The marker is the durable in-history signal that older turns were replaced without the agent's consent, and the system prompt teaches the model to treat the summary as the complete record of everything before it, to read the sections below first, and to never ask the user to repeat what the summary already covers. Agent-initiated compactions never carry the marker: the agent asked for those itself, so a "this was done behind your back" banner would be false.

The summarizer prompt mandates a fixed structure for the summary text, three sections in English:

- `## Summary`: a detailed account of the topic, key decisions, facts, constraints and outcomes;
- `## Current work in progress`: the task that was interrupted mid-flight, with artifacts and current state, so the agent continues it coherently instead of restarting;
- `## Background tasks`: scheduled jobs, running sub-agents and other outstanding background activity.

An empty summarizer answer is rejected the same way an empty tool summary is: the thread is left untouched instead of being seeded with nothing.

### Verification before the wipe

The decisive safety property of the forced path: the summary must actually reduce the context, and the compacted state must verifiably fit, BEFORE the old history is discarded. The exact predicate is that the predicted post-compaction request size plus the output reservation must not exceed the budget (the model window, or `max_history_tokens` for an unknown model). The predicted size is the local count of the seed itself; because the provider has not seen the compacted history yet, this is explicitly a prediction and can never be recorded as a verified success. The authoritative confirmation arrives later, when the provider reports the prompt token count of the first request served after the compaction.

When the check fails, either because the summary did not reduce the size or because the result still does not fit, nothing is written: the thread stays exactly as it was, the failure is logged, and the already-delivered answer is unaffected. A no-op summarizer call can never masquerade as a compaction.

### Back-off after failures

After a failed forced compaction the thread backs off: the post-answer check skips it for the next three delivered turns or thirty minutes, whichever comes first, so a persistently failing thread does not burn an attempt on every single turn. Normal operation continues throughout; only the automatic re-compaction is suspended, and the agent's own tools and the guards still work. The in-flight flag and the back-off bookkeeping live in a process-local store: a process restart clears them.

## The pre-call trim guard

Before each request, under the same per-thread lock the provider call uses, a guard compares the assembled request against the fit threshold: the model window minus the output reservation (the same reservation the request itself carries). If the request does not fit, the guard performs a lossy, transient trim:

1. The oldest complete units are dropped. A unit is an assistant message together with every tool result answering its calls, so a tool call is never separated from its results. A unit is dropped whole only when its size does not overshoot the remaining excess; units far larger than the gap they would close are left for the truncation step.
2. The largest remaining message contents are progressively halved, tool results first, then other messages, largest first within each class. Only the text shrinks; message identities and tool-call structures are never touched, and the most recent message is never dropped or truncated.
3. If truncation is exhausted and the request still does not fit, the units the overshoot guard protected in step one are dropped whole after all, oldest first. This bounded sweep keeps the promise that the request fits even when the blocker is a non-truncatable unit, and the extra loss is accepted by design.

This is not compaction: nothing is summarized, the stored history is untouched, the next turn re-reads the full history, and the dropped content is gone only for this one request. Because the trim hides real content, the agent is told about it: what was removed is recorded in a per-thread side channel, and the next context note carries a `guard_trim_warning` line describing it.

When the model window cannot be resolved at all, the guard does not guess one. It lets the request through unchanged and leaves the case to the reactive recovery, which reacts to the provider's own verdict. And when even the fully trimmed request does not fit (the most recent message alone exceeds the input space), the best-effort list is sent anyway and the provider's verdict remains the ground truth.

## The in-loop budget guard

During a multi-step tool loop the conversation grows between requests: each tool result is appended before the next recursive call. Before each such request, a second guard re-measures the running size and, when it plus the output reservation exceeds the window, hides the excess older material for that request.

The mechanics are the shared trim machinery in its drop-only mode: oldest complete units are removed under the same pair-integrity rule, but content is never truncated in place, because the loop's request lists hold independently serialized copies that an in-place truncation could not reach. Two units are always protected: the one holding the most recent message (the exchange the model is waiting on) and the one holding the first message (losing the head would trade a budget problem for a protocol-invalid request). Everything between them is droppable, oldest first. The same transient semantics apply: the mask applies to this request only, the stored history is untouched, and the removal is disclosed through the same side channel, so the next context note carries a `guard_trim_warning`.

Known nuance: an in-loop trim happens after the request for the current step was already assembled, so its disclosure lands on the next request, one request late. The warning always describes the trims since the last disclosure, so nothing is lost, only delayed.

## Reactive recovery after a provider overflow

If, despite everything, the provider rejects a request because the context is too long, Chibi reacts. The recovery is gated by its own switch, `reactive_context_recovery` (default: on); when it is off, the overflow error is presented as-is.

The recovery is fresh by construction: it re-measures the thread from storage and re-assembles the request rather than reusing the failed one. It then runs the same compaction machine as the forced path (the seed in that case carries the forced marker too, since the agent did not ask for it). The outcomes branch as follows:

- If the request was not reduced, the recovery fails honestly and the original overflow error is presented instead of re-sending an untrimmed request.
- If the request was reduced but still does not fit a resolvable window, the transient last-resort path force-fits it before the retry.
- If the window is unknown, the reduced request is sent as the backstop's last move and the provider's verdict decides.

The request is then retried exactly once, as a single direct provider call. After a successful retry, the provider's reported prompt token count is consumed by the same verification predicate in its authoritative mode: this is the only place in the system where a compaction can be confirmed against a provider-reported post-compaction figure. The user receives a normal answer with a short appended note that the context was compressed. If the retry itself overflows, the original error is re-raised and shown.

## How the summarizer itself stays within budget

Compaction summarizes the conversation with a model call of its own, and that call has an input too. When producing a forced summary, the summarizer input is trimmed in two tiers so that summarization can never itself overflow:

1. Background tool-response JSON blobs are dropped first, largest first. They are the bulk of the volume and the least valuable as summarization context. The most recent message is protected even here, so the summarizer always keeps the latest context anchor.
2. The oldest remaining complete units are dropped next, under the same pair-integrity rule.

The fit budget is the space the summarizer call actually has: the context window of the active model minus the output reservation, resolved through the same provider path the failed request used. When the window cannot be resolved, a documented fallback (60% of `max_history_tokens`) keeps the budget bounded instead of unbounded. Tool-call pairs and hidden messages are excluded from the summarizer input altogether. On a resolvable large-window model the whole history reaches the summarizer intact; full collapse is the intended normal case, and the fit only guards the corner cases.

## What the user sees

Normally nothing: compaction is plumbing. The only visible artifacts are:

- slightly delayed answers right after a heavy turn, when the forced post-answer check runs in the background (it never delays the answer itself, only competes for resources);
- the `[SYSTEM] Context was forcibly compacted...` marker, visible only if the user inspects the stored thread;
- an honest error message in the rare case where the reactive recovery also fails and the overflow error reaches the presentation layer;
- the short "(Context was compressed to stay within limits.)" note appended to an answer that was delivered after a reactive recovery.

The agent is instructed, in the system prompt, never to describe the compaction mechanics or the context note to the user.

## What the model sees

The model observes the machinery through a few small, deliberate surfaces:

- the per-turn `<context_note>` block on the last user message;
- the seed pair after a full compaction: the short compaction question and the summary answer (with the marker line on forced compactions);
- empty stubs where tool calls were dropped: assistant messages that keep their text but no longer carry call payloads, after `clear_tool_call_history`;
- the fact that stable message identities are preserved across selective drops and transient trims: message ids are never renumbered, recreated or re-keyed, so nothing the model refers to shifts underneath it.

## Guarantees and limits

- Forced compaction verifies before it wipes: a summary that does not reduce the size, or whose result does not verifiably fit, leaves the thread untouched.
- Empty summaries are rejected in both paths: the tool refuses with an error, and the automatic machine skips the attempt entirely.
- Agent-supplied summaries have no max-length guard. Unlike working notes, which are capped, a `summarize_history` or `clear_tool_call_history` call with an enormous summary is accepted as-is. This is a known gap.
- The transient guards never mutate stored history: trims apply to one outgoing request only.
- Concurrent compactions are prevented by a per-thread lock, so a compaction can never overlap itself or a chat turn's history mutations within one process.
- The measured-size cache, the in-flight flags and the back-off state are process-local: a restart clears them (the durable measurement record survives, it lives in storage).
- Whole-history replacement is wipe-then-write, and a crash in the milliseconds between the two leaves a thread reading as empty. This window is an explicitly accepted risk.
- Compaction machinery failures are contained: the delivered answer is never affected, and failures are logged, never surfaced to the user.

## Configuration

- `CONTEXT_FORCED_COMPACTION_ENABLED` (default: `true`): the post-answer forced layer. Setting it to `false` disables the post-answer check entirely; the agent's own tools, the guards and the reactive recovery remain in place.
- `reactive_context_recovery` (default: `true`): the overflow backstop. When off, a provider-rejected oversized request is presented as an error.
- `max_history_tokens` (default: 200000): the verification bound and summarizer-fit fallback for models whose window is unknown, the denominator of the context note's percentage, and the cap in the forced-compaction threshold. It is not a stand-in context window: the transient guards never read it.
- `context_size_warning_threshold` (default: 50): the percentage at which the compaction advisory appears.
- The tool-results warning fires at a fixed 25% share of the context; the back-off after a failed forced compaction is fixed at three turns or thirty minutes.

## Which level fires when

| Situation | What happens | Who notices |
| --- | --- | --- |
| Context passes 50% of `max_history_tokens` | Compaction advisory appears in the `<context_note>` | The agent, via the note |
| Bulky tool results pass 25% of the context | Tool-results warning with the top offenders by tool name | The agent, via the note |
| Agent decides to compact | `summarize_history` or `clear_tool_call_history` runs before the answer | Nobody visible; the history shrinks |
| Context reaches `min(max_history_tokens, window)` after an answer | Forced post-answer compaction replaces history with a marked seed (marker + Summary + Current work + Background tasks), after verification | The agent, via the seed marker; the user never does |
| Outgoing request would not fit the window | Pre-call guard drops oldest units, halves largest contents, transiently | The agent, via `guard_trim_warning` on the following request |
| In-loop request would not fit | In-loop guard drops oldest units (drop-only), transiently | The agent, via `guard_trim_warning` on the next request (one request late) |
| Provider rejects a request as oversized | Reactive recovery: compact, retry once, force-fit if needed | The user gets a normal answer with a short compression note |
| A forced compaction attempt fails | Nothing is written; thread backs off 3 turns / 30 minutes | Nobody visible; failure is logged |
