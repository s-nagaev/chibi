# Chibi IDE Protocol Fixtures

Canonical, versioned fixtures for the `chibi stdio --tui` protocol defined in
`/.project/ide_integration/architecture_specification.md`. Consumed by Chibi backend tests **and** by the
`chibi-vscode` extension tests (the extension keeps its own copies and checks compatibility by `protocol_version`).

Protocol version: **1**.

## Layout

Because the protocol has separate stdin (client → server) and stdout (server → client) streams, each scenario is
split into two files:

- `*_input.jsonl` — frames written by the IDE to the Chibi process stdin.
- `*_output.jsonl` — expected frames emitted by Chibi on stdout.

## Valid scenarios

### `valid_session_input.jsonl` / `valid_session_output.jsonl`
Full happy path:
`initialize` → `ready` → `request` → `status(running)` → `result`.

### `valid_cancel_input.jsonl` / `valid_cancel_output.jsonl`
Request cancellation:
`initialize` → `ready` → `request` → `status(running)` → `cancel` → `error(cancelled)`.

### `valid_shutdown_input.jsonl` / `valid_shutdown_output.jsonl`
Graceful shutdown:
`initialize` → `ready` → `shutdown`.

### `valid_subagent_events_input.jsonl` / `valid_subagent_events_output.jsonl`
Subagent lifecycle events (requires the `capabilities.subagents` opt-in):
`initialize(subagents)` → `ready` → `request` → `status(running)` →
`agent_event(started, active 1, total 1)` → `agent_event(finished, active 0, total 1)` → `result`.
The final zero-active `finished` is the client's signal to hide its subagent spinner line; the result frame stays
the implicit end of the request and no agent_event ever follows it.

Each `agent_event` carries an optional `id`: a per-instance identifier unique to the subagent that produced the
event. It is present on both the `started` and the `finished` event of one delegation, which is what lets a client
keep two concurrently running same-name subagents apart. The field is additive: a client that does not know it
ignores it, and an event without an instance context (the authoritative kill-flush `finished`) omits it. `name`
stays the delegated model label and is display-only.

### `valid_tool_events_input.jsonl` / `valid_tool_events_output.jsonl`
Foreground tool-call lifecycle events (requires the `capabilities.tool_events` opt-in):
`initialize(tool_events, subagents)` → `ready` → `request` → `status(running)` →
`tool_event(start, read_file, args)` → `tool_event(end, read_file, ok true, result_preview)` →
`agent_event(started, id)` → `agent_event(finished, id)` → `result`.

A `tool_event` is a correlated, non-terminal frame carrying the request's `request_id` and `phase`:
`start` carries `name` plus `args` (the model-supplied arguments, size-capped), `end` carries `name`, a boolean
`ok` and a `result_preview` truncated to at most 2000 characters. Emission is gated on the
`capabilities.tool_events` opt-in and requires a live request on the thread: background tool runs
(`run_in_background`) and interface-less (sub-agent-internal) calls emit nothing at all. The args size cap
replaces an oversized `args` mapping with `{"_truncated": true}` rather than an arbitrary substring.

The opt-in is independent of `subagents`: a client that declares `tool_events` without `subagents` still receives
its `tool_event` frames (the owning request is resolved from the transport's own active-request bookkeeping, never
from the subagents-gated counters). The fixture declares both flags so it can also show an id-bearing
`agent_event` pair, which is emitted by a delegation (a background-default tool, so the delegation itself emits
no `tool_event`). A session without `capabilities.tool_events` in the `initialize` handshake is exactly the
canonical `valid_session_*` scenario — no `tool_event` frames are ever emitted to a client that did not opt in.

### `valid_streaming_session_input.jsonl` / `valid_streaming_session_output.jsonl`
Live response-delta streaming (requires the `capabilities.streaming` opt-in):
`initialize(streaming)` → `ready` (identical to the canonical one — the server never advertises
`streaming` in `ready.capabilities`; the client opt-in is the only gate) → `request` →
`status(running)` → `delta` (head chunk flushed inline: it reached the ≥512-char
`DELTA_FLUSH_CHARS` threshold) → `delta` (short tail chunk flushed by `drain_deltas`
before the terminal write) → `result`.

The final `result` frame stays authoritative: clients render it over the streamed partial
text, making reconciliation idempotent. A session without `capabilities.streaming` in the
`initialize` handshake is exactly the canonical `valid_session_*` scenario — no `delta`
frames are ever emitted to a client that did not opt in.

### `valid_thread_renamed_input.jsonl` / `valid_thread_renamed_output.jsonl`
Thread rename push events (requires the `capabilities.thread_renames` opt-in):
`initialize(thread_renames)` → `ready` (identical to the canonical one — the server never advertises
`thread_renames` in `ready.capabilities`; the client opt-in is the only gate) → `request` →
`status(running)` → `thread_renamed` → `result`.

The `thread_renamed` frame is non-terminal and thread-scoped: it carries `thread_id` and the new
`name`, and never a `request_id`. Emission is gated on the `thread_renames` opt-in; the frame is
drained before the terminal write of the request it was scheduled during. A session without
`capabilities.thread_renames` in the `initialize` handshake is exactly the canonical
`valid_session_*` scenario — no `thread_renamed` frames are ever emitted to a client that did not
opt in.

### `valid_models_input.jsonl` / `valid_models_output.jsonl`
Structured model listing for a client that sees `ready.capabilities.models`:
`initialize` → `ready` → `models` → `models_result`.

The `models` frame is a correlated request/answer pair: `request_id` is mandatory and a frame without it answers
`malformed_request`. The `models_result` listing is the same listing, in the same order, that the bare `/model`
command renders as text, with the active-selection marker turned into the `active` boolean and the marker
stripped from `display_name`. `thread_id` is optional and defaults to `0`; it only chooses the thread the
`active` flag is computed for. `default` is the account-level fallback (the configured default provider, then
the first chat-ready provider) and never the thread selection, so it stays `null` when no chat provider is
configured.

## Invalid / edge cases

### `invalid_cases.jsonl`
Machine-readable manifest of independent invalid/edge cases. Each line is a JSON object with:

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | Case identifier. |
| `setup` | list of objects | Client input frames to send before the test input (e.g. handshake). |
| `input` | object or string | The test input. A JSON object for structured input; a string for raw malformed input. |
| `expected_output` | list of objects | Expected server output frames, in order, including any `ready` produced by `setup`. |

Tests should run each case in isolation (fresh process) using only the listed `setup` + `input` and assert that
stdout matches `expected_output` exactly.

| Case | Summary |
|------|---------|
| `malformed_json` | Raw non-JSON input → global `malformed_request`. |
| `unknown_message_type` | Unknown `type` after handshake → `unknown_message`. |
| `missing_required_fields` | `request` missing required fields → `malformed_request`. |
| `unsupported_protocol_version` | Handshake with unsupported version → `unsupported_protocol_version`; process shuts down. |
| `request_before_ready` | `request` before handshake → `not_initialized`. |
| `cancel_unknown_request` | `cancel` for a non-existent request → `unknown_request`. |
| `request_missing_request_id` | `request` without `request_id` → `malformed_request`. |
| `models_without_request_id` | `models` without `request_id` → `malformed_request`. |
