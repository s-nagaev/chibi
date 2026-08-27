# Chibi IDE Protocol Fixtures

Canonical, versioned fixtures for the `chibi ide --stdio` protocol defined in
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
