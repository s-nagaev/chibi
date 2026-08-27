# Chibi IDE Runner — Independent Review

## Executive Summary

This document synthesizes three independent reviews of the Chibi IDE runner — the JSONL-over-stdio transport that lets an external IDE interact with the Chibi assistant. The reviews were produced by three different models examining the same codebase from different angles: (1) a **transport/protocol review** focused on wire-protocol correctness, request lifecycle, and the `IDEInterface` adapter; (2) an **integration-layer review** focused on how the IDE runner composes with Chibi's existing services (bot, handlers, storage, command processing); and (3) an **infrastructure/test review** focused on the task-manager subsystem, concurrency primitives, and test coverage.

Methodology: each reviewer read the relevant source files, cross-referenced the broader codebase (`constants.py`, `services/interface.py`, `services/task_manager.py`, `bot.py`, `cli.py`), and produced a structured report. The three reports have been **deduplicated, merged where overlapping, and re-organized by severity** into a single authoritative catalog. No new analysis has been introduced.

Overall assessment: the runner is **functional and architecturally sound at the high level** — the JSONL protocol is well-defined, request lifecycle is well-bounded, and the `IDEInterface` adapter cleanly overrides the `UserInterface` contract. However, the reviews collectively identified **two critical correctness bugs** (`IDEInterface.send_images` emitting Python reprs of `BytesIO`; cancellation ack frames potentially being lost mid-cancellation), several high-severity issues (shared `IDE_STORAGE_ID` collapsing all IDE sessions into one identity, untyped `setattr` smuggling, broken thread-management surface), and a broad pattern of architectural debt: a global `task_manager` singleton, no per-request timeout, single-threaded stdin reading, no schema validation at the protocol boundary, and inadequate test coverage of background-task infrastructure.

The single most actionable finding is the `send_images` bug (§2.1) — any current IDE session that triggers an image-generation path is producing garbage output for the user. The single most impactful architectural concern is the shared `IDE_STORAGE_ID` (§3.1) — running two IDE clients against the same Chibi process will produce surprising cross-contamination of conversation state, model selection, and reset behavior.

---

## 1. Architecture Overview

The Chibi IDE runner is a thin stdio transport that exposes Chibi's assistant to an external IDE. It runs as a separate process invoked by `chibi run-ide --stdio` (CLI entrypoint in `cli.py`). The process owns a single asyncio event loop that drives:

- **A reader thread** that performs blocking `sys.stdin.readline()` via `asyncio.to_thread` and feeds JSONL frames into the loop.
- **A request dispatcher** that validates each frame, dispatches it via a global `task_manager.run_task(...)`, and tracks inflight requests in `_tasks[request_id]` and per-thread concurrency in `_thread_requests[thread_id]`.
- **A writer** that serializes outgoing frames under an `asyncio.Lock` to stdout, ensuring atomic line writes on the JSONL wire.

### 1.1 Wire protocol

All frames are `json.dumps(..., separators=(",", ":")) + "\n"`. Server→client frame types: `ready`, `status` (`queued`/`running`), `result` (with `content`, optional `model`/`provider`), `error` (always with `code` and `message`). Client→server types: `initialize`, `request`, `cancel`, `shutdown`. The protocol is hand-rolled with no schema validation; unknown message types produce an `unknown_message` error and the loop continues.

### 1.2 Request lifecycle

1. Read line → strip → `json.loads` → dispatch to `_handle_message`.
2. On `request`: validate; if `thread_id` already has an outstanding request, emit `status/queued`; dispatch via `task_manager.run_task(self._run_request(...))`; record the task; bump per-thread counter.
3. `_run_request` emits `status/running`, constructs a request-local `IDEInterface` bound to a `responses.append` callback, dispatches on `"/command"` or `handle_user_prompt`, and emits exactly one terminal frame (`result` / `error` / `error/cancelled` / `error/request_failed`) before returning. The terminal-frame invariant is preserved by a top-level `try/finally` in the request coroutine.
4. On `cancel`: `task.cancel()`.
5. On `shutdown`: flip `_stopping = True`; final `task_manager.shutdown()` drains remaining work.

### 1.3 `IDEInterface` adapter

`IDEInterface` is a `UserInterface` subclass that:
- uses a single fixed `IDE_STORAGE_ID` (from `constants.py`, value `-(10**16)`) for chat/user/storage identity across all IDE requests and sessions,
- keeps the per-request `thread_id` and `prompt` in fields,
- routes `send_message` and `send_images` calls into a `_emit` callback (the request's `responses.append`),
- no-ops every other media method (audio/video/document) and every action/attachment method,
- raises `NotImplementedError` for thread create/rename/delete (the IDE owns threads).

The interface exposes `response_model` and `response_provider` as instance attributes that are populated by handlers via `setattr(...)` deep inside the call chain — bypassing type safety.

### 1.4 Task management

The runner uses Chibi's global `task_manager` singleton (from `services/task_manager.py`) to run request coroutines. The manager is implemented as a singleton with `run_task(coro)`, `_discard_task(...)`, and `shutdown()` semantics. Internally it tracks per-user tasks in `_tasks` and per-user-id mappings in `_task_to_user_id`. Default behavior does not impose timeouts, and cancellation handling relies on `task.exception()` after the fact rather than `task.cancelled()` checks.

### 1.5 Integration with Chibi services

The IDE runner reuses Chibi's bot handlers (`handle_user_prompt`, `handle_reset`, `handle_image_generation`, model selection, etc.) by routing requests through them with the `IDEInterface` substituted for `TelegramInterface`. This means IDE sessions share command-processing logic, error-handling patterns, model selection, and storage — but **without per-session isolation** (shared `IDE_STORAGE_ID`).

---

## 2. Bugs & Real Problems

Issues below are deduplicated across the three reviews and organized by severity. Each entry includes description, location, impact, and a suggested fix.

### 2.1 Critical

#### C1. `IDEInterface.send_images()` emits Python object reprs as text
- **Sources:** Integration review (kimi-k2.7-code); Transport review §3.4.
- **Description:** `send_images` constructs content with `"\n".join(str(image) for image in images)`. When `image` is a `BytesIO`, `str(image)` yields `&lt;_io.BytesIO object at 0x...&gt;`. When it's already a string URL, `str` is a no-op (works by accident). Binary image data is therefore rendered as Python repr garbage and streamed to the IDE.
- **Location:** `chibi/runners/ide_transport.py` lines ~117-122 (the `send_images` method on `IDEInterface`).
- **Impact:** Any code path that triggers image generation in an IDE session — `/imagine`, any pipeline that produces images, function calls that return image content — currently produces meaningless text for the user. This is the most concrete and visible runtime bug.
- **Fix:** Add a dedicated `attachment` frame type: `{"type": "attachment", "kind": "image", "mime": "image/png", "data": "<base64>", "request_id": ...}`. Detect `str` URLs and forward them verbatim in a `{"type": "attachment", "kind": "image", "url": ...}` frame. Do not stringify `BytesIO`.

#### C2. Cancellation ack frame may never be emitted
- **Sources:** Transport review §2.6.
- **Description:** `_run_request` does `except asyncio.CancelledError: await self._error("cancelled", ...); raise`. Whether the `await` of `_error` succeeds depends on asyncio version specifics — once a task is cancelled, an immediate subsequent await can re-raise `CancelledError` before the write completes. The cancellation ack is therefore racy.
- **Location:** `chibi/runners/ide_transport.py` lines ~200-204 (the cancellation branch in `_run_request`).
- **Impact:** Clients may not see a terminal `cancelled` frame for cancelled requests. Combined with `task_manager` cancellation semantics, this can produce silent cancellation where neither side has a confirmed handshake.
- **Fix:** Wrap critical terminal-frame writes in `asyncio.shield(...)` so the cancellation delivery is decoupled from the writer's await. Apply the same pattern to `result` and `error` frames to harden the "exactly one terminal frame" invariant.

### 2.2 High

#### H1. `IDE_STORAGE_ID` is a singleton — all IDE sessions share identity
- **Sources:** Transport review §3.3; Integration review (kimi-k2.7-code, "All IDE users share IDE_STORAGE_ID — conversation isolation broken"); Infrastructure review (Gemini 3.1-pro-preview, "IDE_STORAGE_ID keying collisions across sessions").
- **Description:** `IDE_STORAGE_ID = -(10**16)` is used as `user_id`, `storage_id`, and `chat_id` for all IDE requests. The `_thread_requests` dict provides thread-level concurrency isolation, but storage is keyed by `IDE_STORAGE_ID` so conversation history, model selection, `/info`, and reset state are shared across every IDE client connecting to the same Chibi process.
- **Location:** `chibi/constants.py:21`; used in `chibi/runners/ide_transport.py` `IDEInterface.__init__` and throughout handler callsites; database keying in `services/storage` and `services/task_manager`.
- **Impact:** Running two IDE windows or two workspaces against one Chibi process produces surprising cross-contamination of state. The user-visible `/info` output, `/model` selection, and `/reset` semantics all collide.
- **Fix:** Allow IDE to pass a per-session `client_id` (workspace UUID) in the `initialize` frame, then derive `user_id = hash(client_id)` (or similar) and use it as `storage_id`. Document the singleton semantics explicitly until this lands.

#### H2. `setattr` smuggling for `response_model` / `response_provider` bypasses type safety
- **Sources:** Integration review (kimi-k2.7-code); Transport review §2.9, §3.5.
- **Description:** Handlers mutate `interface.response_model` and `interface.response_provider` via `setattr` deep inside the call chain. Nothing enforces types. A misbehaving handler can write a non-string (or anything) and break the `result` frame schema. Subtle ordering bugs are possible if a handler emits `send_message` then sets the model — the result frame is assembled only after the handler returns, so this works today by accident, not by contract.
- **Location:** `chibi/runners/ide_transport.py` `IDEInterface` (response_model/response_provider fields, set via setattr from `bot.handle_user_prompt`); read in `_run_request` lines ~195-199.
- **Impact:** Output contract drift; potential schema corruption of `result` frames; fragile ordering dependencies.
- **Fix:** Type the attributes as `str | None`, validate before assembling the `result` frame. Better: do not mutate interface state at all — return model info via a tuple `(text, meta)` from the handler, or via a `contextvars.ContextVar`.

#### H3. Thread-management tools raise `NotImplementedError`
- **Sources:** Integration review (kimi-k2.7-code).
- **Description:** `IDEInterface` raises `NotImplementedError` for thread create/rename/delete. These tools are nevertheless visible to the LLM via Chibi's tool-call surface. If the model invokes one, the call site raises rather than producing a clean error response.
- **Location:** `chibi/runners/ide_transport.py` `IDEInterface` thread-management methods (create/rename/delete).
- **Impact:** Tool calls fail with an opaque exception rather than a structured "not supported in IDE mode" error. The model has no clear path to recover.
- **Fix:** Either (a) hide these tools from the model in IDE mode (preferred — they are by design IDE-owned), or (b) intercept the calls and return a structured "operation not supported in IDE session" response.

#### H4. `_initialized` / `_stopping` flags are mutated without explicit synchronization
- **Sources:** Transport review §2.1.
- **Description:** These flags are read on the event loop without `await`. Today the single-reader pattern provides implicit atomicity, but the mutation sites are spread across `_handle_message` and `_run_request`, and a future change that adds a second reader (or a code path that reads the flag from a different task context) will hit subtle race conditions.
- **Location:** `chibi/runners/ide_transport.py:131-133` (declaration); mutations in `_handle_message` (lines ~233, ~244, ~271) and `_run_request` (line ~177).
- **Impact:** Today: low (single reader). Future change risk: medium-high.
- **Fix:** Replace booleans with `asyncio.Event` for `_initialized` (handshake semantic) and `_stopping` (shutdown signal). Centralize all mutations and reads through these primitives.

### 2.3 Medium

#### M1. `run_task` has no default timeout — tasks can run indefinitely
- **Sources:** Infrastructure review (Gemini 3.1-pro-preview); also surfaced as transport §3.12 (no per-request timeout).
- **Description:** `task_manager.run_task(coro)` accepts no required `timeout` argument; default is `None`. A handler that hangs (network stall, infinite loop in tool call) blocks forever; `_thread_requests` never decrements; the request_id stays in `_tasks` indefinitely.
- **Location:** `chibi/services/task_manager.py` `run_task` signature; caller at `chibi/runners/ide_transport.py:265-267`.
- **Impact:** Resource leak, stuck transport, eventual `MemoryError` if many requests pile up.
- **Fix:** Add `asyncio.wait_for(timeout, shield=...)` either at the `run_task` layer (with a default negotiated in `ready.capabilities`) or at the IDE request layer (per-call timeout overridable via the `request` frame). Track timeouts distinctly from cancellations.

#### M2. `Coroutine.__name__` is lost after timeout wrapping — `AttributeError` on `.cancel()` and logging
- **Sources:** Infrastructure review (Gemini 3.1-pro-preview).
- **Description:** When `run_task` wraps a coroutine with a timeout or other decorator, the resulting object may not preserve `__name__`. Code paths that introspect `coro.__name__` (for logging, for `add_done_callback` bookkeeping, or for `.cancel()` dispatch) raise `AttributeError`.
- **Location:** `chibi/services/task_manager.py` wrapping code; consumers that introspect `__name__`.
- **Impact:** Crash on introspection; degraded log messages; potential crash on shutdown-time task cancellation.
- **Fix:** Use `functools.wraps` (or equivalent) when wrapping coroutines; expose the original callable's `__name__` via a `_target` attribute if wrapping is unavoidable.

#### M3. `_discard_task` checks cancellation via `task.exception()` — anti-pattern
- **Sources:** Infrastructure review (Gemini 3.1-pro-preview).
- **Description:** The implementation inspects `task.exception()` to determine if a task was cancelled. This consumes the exception and is order-dependent on task state. The canonical approach is `task.cancelled()` first, then `task.exception()` if not cancelled.
- **Location:** `chibi/services/task_manager.py` `_discard_task` (and related cleanup helpers).
- **Impact:** Subtle state-machine bugs around cancellation; potential `InvalidStateError` if `task.exception()` is called before the task is done.
- **Fix:** Use `task.cancelled()` as the primary check; only fall back to `task.exception()` when the task is done and not cancelled.

#### M4. `_discard_task` never deletes empty sets from `_tasks` — memory leak
- **Sources:** Infrastructure review (Gemini 3.1-pro-preview).
- **Description:** After all tasks for a user complete, the set in `_tasks[user_id]` becomes empty but the entry is never removed. `_tasks` grows unboundedly with the number of distinct user IDs over the process lifetime.
- **Location:** `chibi/services/task_manager.py` `_tasks` dict; cleanup paths in `_discard_task`.
- **Impact:** Slow memory growth; long-running IDE sessions or any multi-user process leak dict entries.
- **Fix:** In `_discard_task`, after the last task is removed, `if not _tasks[user_id]: del _tasks[user_id]`. Add the same cleanup for `_task_to_user_id`.

#### M5. `SingletonMeta` is not thread-safe
- **Sources:** Infrastructure review (Gemini 3.1-pro-preview).
- **Description:** `SingletonMeta` metaclass performs the singleton check and instantiation without a lock. Concurrent first-access from multiple threads can produce multiple instances.
- **Location:** `chibi/services/task_manager.py` `SingletonMeta`.
- **Impact:** Theoretical race in test harnesses or any code path that constructs the singleton from a non-main thread (e.g., a worker thread spawned by `asyncio.to_thread`). Today the IDE runner constructs it on the main loop, so impact is low — but other Chibi entrypoints may not.
- **Fix:** Use `threading.Lock` around the existence check and instantiation in `__call__`.

#### M6. `request_id` collision check is too strict
- **Sources:** Transport review §2.2.
- **Description:** `_handle_message` rejects with `malformed_request` whenever `request_id in self._tasks`. This conflates "still in flight" with "id was reused post-completion." Since `_tasks` is popped in the `finally` of `_run_request`, post-completion reuse is safe — only in-flight collisions are real bugs. The error message "request_id is already in use" misleads debugging.
- **Location:** `chibi/runners/ide_transport.py:254`.
- **Impact:** Clients using stable request-id schemes (e.g., `&lt;thread&gt;:&lt;n&gt;`) are rejected unnecessarily.
- **Fix:** Distinguish "still in flight" from "completed." Only reject when the id is currently in `_tasks`.

#### M7. `shutdown` does not drain in-flight requests
- **Sources:** Transport review §2.12.
- **Description:** `shutdown` flips `_stopping = True` and returns to the loop. If the client sends `shutdown` and then keeps stdin open (graceful shutdown pattern), the loop continues reading and dispatching new requests. If stdin is closed, `task_manager.shutdown()` drains existing tasks in the `finally` of `run()`.
- **Location:** `chibi/runners/ide_transport.py:272-274`.
- **Impact:** Ambiguous shutdown semantics; a compromised stdin pipe can both drain and block shutdown.
- **Fix:** Make `shutdown` semantics explicit: "do not accept new requests; drain existing." On receiving `shutdown`, break out of the read loop immediately and `await asyncio.gather(*self._tasks.values(), return_exceptions=True)` before exit.

#### M8. `/model` command duplicates Telegram model-selection logic
- **Sources:** Integration review (kimi-k2.7-code).
- **Description:** The IDE runner re-implements model-selection logic in the `/model` command path rather than delegating to a shared service. Drift between Telegram and IDE code paths is a long-term risk.
- **Location:** `chibi/runners/ide_transport.py` `/model` handling (~lines 172-191).
- **Impact:** Behavioral drift between IDE and Telegram sessions; duplicated code; harder to add per-interface customization.
- **Fix:** Extract model-selection into a service that takes a `UserInterface` and a `storage_id`. Both Telegram and IDE call into the same service.

### 2.4 Low

#### L1. `task_manager.run_task(...)` is fire-and-forget — no done-callback
- **Sources:** Transport review §2.8.
- **Description:** Tasks are dispatched via `run_task` and recorded in `_tasks`, but the runner never installs `add_done_callback`. Unexpected exceptions inside `_run_request` are mostly caught by its top-level `except Exception` (emitting `request_failed`), but `CancelledError` re-raises and may not emit reliably (see C2).
- **Location:** `chibi/runners/ide_transport.py:265-267`.
- **Impact:** Missing observability surface for task outcomes.
- **Fix:** `task.add_done_callback(self._on_task_done)` to remove from `_tasks` and log unexpected exception causes.

#### L2. `_thread_requests` increment/decrement order is fragile
- **Sources:** Transport review §2.3.
- **Description:** The counter is incremented *before* dispatching to `task_manager` and rolled back only if `run_task` returns `None`. Today this is correct; a future edit could remove the rollback and silently leak the counter.
- **Location:** `chibi/runners/ide_transport.py` lines ~264-269.
- **Impact:** Latent bug surface; not currently buggy.
- **Fix:** Move the increment below the `task is None` check so the lifetime of the counter matches the lifetime of the dispatched task.

#### L3. `error_code` / `error_message` are checked after `send_message` calls — order-dependent
- **Sources:** Transport review §2.10.
- **Description:** `/model &lt;bad&gt;` sets `interface.error_code` and emits error. Currently the error path takes precedence over `result`. This works but the contract is implicit.
- **Location:** `chibi/runners/ide_transport.py` lines ~194-198.
- **Impact:** Future handler may set `error_code` *and* call `send_message`, leading to ambiguity.
- **Fix:** Document the contract on `IDEInterface`: "If `error_code` is set after handler completion, emit `error` instead of `result`."

#### L4. Inconsistent error handling: `/imagine` `ValueError` becomes generic `request_failed` rather than `invalid_argument`
- **Sources:** Integration review (kimi-k2.7-code).
- **Description:** Other commands raise structured `invalid_argument` errors for bad input; `/imagine` lets `ValueError` bubble up to the generic `request_failed` path.
- **Location:** `chibi/runners/ide_transport.py` `/imagine` dispatch.
- **Impact:** Inconsistent client experience; IDE cannot programmatically distinguish "bad input" from "internal failure."
- **Fix:** Catch `ValueError` at the command-dispatch boundary and convert to `invalid_argument` with a helpful message.

#### L5. `_set_ide_error` only covers decorated service functions
- **Sources:** Integration review (kimi-k2.7-code).
- **Description:** The IDE error-decoration mechanism is applied only to functions that opt into the decorator. Any code path that constructs an `IDEInterface` and calls a non-decorated handler does not get consistent error-code translation.
- **Location:** `chibi/runners/ide_transport.py` or shared error-decoration module (decorator apply site).
- **Impact:** Coverage gaps in error translation.
- **Fix:** Either decorate the entrypoint functions (so all calls go through the decorator) or apply the decorator universally to handlers that take `UserInterface`.

#### L6. JSON line read has no length cap — DoS via memory exhaustion
- **Sources:** Transport review §2.7.
- **Description:** `sys.stdin.readline()` will read multi-GB lines. `json.loads` may raise `MemoryError` (not currently caught). A malicious or buggy client can OOM the runner.
- **Location:** `chibi/runners/ide_transport.py:152-155`.
- **Impact:** DoS surface.
- **Fix:** Cap input line length (e.g., 1 MB). Treat over-cap as `malformed_request`.

#### L7. Single-threaded stdin reading has no partial-line buffer
- **Sources:** Transport review §3.1.
- **Description:** `readline` returns when it sees `\n`. If a client writes a frame split across two reads of `stdio` (e.g., due to pipe-buffer fragmentation of a large prompt), `readline` returns a partial line, `json.loads` raises, and we send `malformed_request`. The trailing fragment is lost.
- **Location:** `chibi/runners/ide_transport.py` `_read_line`.
- **Impact:** Reliability for large prompts exceeding PIPE_BUF.
- **Fix:** Document the limit (frames must fit in PIPE_BUF), or switch to a length-prefixed protocol.

#### L8. `_write` lock doesn't protect `json.dumps` work
- **Sources:** Transport review §2.5.
- **Description:** `json.dumps` runs before lock acquisition. Today this is fine (lock holds write+flush; libc line-buffers). If stdout is reconfigured (e.g., to non-line-buffered), the lock could be released mid-frame.
- **Location:** `chibi/runners/ide_transport.py:139-145`.
- **Impact:** Latent risk on stdout reconfiguration.
- **Fix:** Document the invariant; acquire the lock earlier if frame atomicity becomes a concern.

#### L9. `_handle_message` for `cancel` doesn't await the cancelled task
- **Sources:** Transport review §2.13.
- **Description:** `self._tasks[request_id].cancel()` is non-blocking. The task removes itself in its `finally`. After cancel returns, `_tasks[request_id]` may still exist for a microsecond.
- **Location:** `chibi/runners/ide_transport.py:267-271`.
- **Impact:** Minor; not a bug.
- **Fix:** Install a done-callback (see L1) and surface exceptions.

#### L10. No `cancel-all` or `stop` message
- **Sources:** Transport review §3.7.
- **Description:** The IDE can only cancel requests individually.
- **Location:** `chibi/runners/ide_transport.py` `_handle_message`.
- **Impact:** Low; UX inconvenience for bulk cancellation.
- **Fix:** Add a `cancel_all` client→server message.

#### L11. No max concurrent requests per thread
- **Sources:** Transport review §3.11.
- **Description:** `_thread_requests` tracks count but only emits "queued" status — no back-pressure. An IDE that floods requests for one thread spawns unbounded concurrent tasks.
- **Location:** `chibi/runners/ide_transport.py:255-269`.
- **Impact:** Resource exhaustion under load.
- **Fix:** Configurable `MAX_PER_THREAD`; queue beyond that.

#### L12. CLI: `ide` without `--stdio` silently does nothing useful
- **Sources:** Transport review §2.15.
- **Description:** `chibi ide` (no flag) prints help. `chibi ide start` errors from click ungracefully. No proper subcommand structure.
- **Location:** `cli.py:30-32`.
- **Impact:** CLI UX.
- **Fix:** Refactor `ide` to be a `@main.command()` with `--stdio`. No group needed.

#### L13. `run_ide()` does side-effect `import chibi.config`
- **Sources:** Transport review §2.16.
- **Description:** Hidden initialization via `# noqa: F401` import.
- **Location:** `chibi/runners/ide.py:9-10`.
- **Impact:** Hidden dependency; conventions silently enforced.
- **Fix:** Explicit `chibi.config.load()` if that's what it does.

#### L14. Logger reconfigured on every `run()`
- **Sources:** Transport review §3.9.
- **Description:** `logger.remove()` strips *all* sinks, then re-adds stderr INFO. Anything that added a sink (e.g., tests) is wiped.
- **Location:** `chibi/runners/ide_transport.py:284`.
- **Impact:** Subtle in tests; low in production.
- **Fix:** Use `logger.add(...)` without `remove` (idempotent), or scope the removal narrowly.

#### L15. `request_id` allowed to be `None` for non-request messages but strict for `request`
- **Sources:** Transport review §2.11.
- **Description:** Mixed validation: `request` requires non-empty str `request_id`; `cancel`/`shutdown` allow `None`. The framework's mental model isn't fully encoded.
- **Location:** `chibi/runners/ide_transport.py:158-168` vs request_id extraction at line ~216.
- **Impact:** Type/schema inconsistency.
- **Fix:** Use typed `TypedDict` or `pydantic` models for each protocol message, validated at boundary.

#### L16. `_stopping` is read but never reset; no "cancel-all and shut down" API
- **Sources:** Transport review §2.14.
- **Description:** Loop exits on EOF or shutdown but never on cancel-all.
- **Location:** `chibi/runners/ide_transport.py:280-288`.
- **Impact:** Minor.
- **Fix:** Optional.

---

## 3. Architectural Concerns

Issues below are non-bugs but represent structural debt that will compound over time. Organized by risk level.

### 3.1 High Risk

#### A1. Shared `IDE_STORAGE_ID` collapses IDE session isolation
See H1. The singleton identity pattern leaks across sessions. Severity compounds because storage keying, task-manager keying, model selection, and reset behavior all collide. This is the single most impactful architectural issue.

#### A2. `task_manager` is a process-global singleton
- **Sources:** Transport review §3.8.
- **Description:** `task_manager` is a free function call into a global. Tests cannot easily inject a fake. The runner's behavior is partly determined by global state.
- **Location:** `chibi/services/task_manager.py`; consumer at `chibi/runners/ide_transport.py:13`.
- **Impact:** Hard to unit-test the runner in isolation; hard to run multiple task managers in the same process (e.g., for isolation in IDE multi-workspace scenarios).
- **Fix:** Define a `TaskManagerProtocol` and accept it via `__init__`. The runner constructs (or receives) a concrete instance per process.

#### A3. No protocol schema validation at the boundary
- **Sources:** Transport review §3.2; §2.11.
- **Description:** Frames are validated ad-hoc in `_handle_message` and `_valid_request`. No `pydantic` models, no schema registry, no message-type dispatch table.
- **Location:** `chibi/runners/ide_transport.py` `_handle_message`, `_valid_request`.
- **Impact:** Adding new message types requires touching multiple functions; drift between server→client and client→server schemas is invisible until runtime; type-safety is manual.
- **Fix:** Introduce `pydantic` models per message type. Replace the `elif`-chain in `_handle_message` with `{type: handler}` dispatch. Reject unknown fields at the boundary.

#### A4. No streaming output — all responses buffered into one `result` frame
- **Sources:** Transport review §3.10.
- **Description:** All `send_message` calls are concatenated into one big `result` frame. For long generations, the user sees nothing until completion.
- **Location:** `chibi/runners/ide_transport.py` `content = "\n".join(responses)` then `await self._write(result)`.
- **Impact:** UX for long generations is unacceptable; user has no progress indication beyond the `running` status.
- **Fix:** Emit `{"type": "delta", "request_id": ..., "content": ...}` from `send_message`; final `result` becomes optional or carries the concatenated string + per-delta offsets.

#### A5. Thread-level concurrency isolated but storage not
- **Sources:** Integration review (kimi-k2.7-code).
- **Description:** `_thread_requests` provides per-thread concurrency isolation in the runner, but storage and conversation history use the shared `IDE_STORAGE_ID`. Half-isolated concurrency is worse than fully-isolated or fully-shared — it creates confusing invariants.
- **Location:** `chibi/runners/ide_transport.py` `_thread_requests`; `IDEInterface` storage fields.
- **Impact:** Surprising cross-contamination: a single thread's history may interleave with another thread's user.
- **Fix:** Derive storage identity per thread (or per session) — see H1.

#### A6. `IDEInterface` mixes input data, output sink, and metadata
- **Sources:** Transport review §4.1 (separation of concerns); §3.5.
- **Description:** `IDEInterface` carries request input (`thread_id`, `prompt`), output sink (`responses`), metadata (`response_model`, `response_provider`), and error state (`error_code`, `error_message`). Three responsibilities, one class.
- **Location:** `chibi/runners/ide_transport.py` `IDEInterface` class.
- **Impact:** Mutation via `setattr` from deep in the handler chain is hard to reason about; type safety is unenforced; testing each concern requires the full interface.
- **Fix:** Split into `RequestContext` (immutable: thread_id, prompt, workspace_root) and `ResponseSink` (mutable: list of messages, model info, error state). Handlers receive both.

### 3.2 Medium Risk

#### A7. Command handling is inlined in `_run_request`
- **Sources:** Transport review §3.6.
- **Description:** The if/elif chain for `/reset`, `/help`, `/model`, etc. lives in the central request coroutine. As commands grow, the function becomes a god method; adding `/foo` requires touching the chain, the `COMMANDS` constant, and the help message.
- **Location:** `chibi/runners/ide_transport.py:172-191`.
- **Impact:** Drift between constant and dispatch; maintenance burden.
- **Fix:** Extract a command registry: `commands.register("/model", handle_model)`. Dispatch becomes a single dict lookup.

#### A8. `setattr` smuggling pattern as architectural smell
- **Sources:** Integration review; Transport review §3.5.
- **Description:** See H2. Beyond the immediate type-safety bug, the pattern of writing response metadata onto the interface from inside a handler is structurally wrong — the interface mixes data sink with side-channel metadata.
- **Impact:** Makes the interface hard to mock, hard to type-check, and easy to misuse.
- **Fix:** Have handlers return `(text, meta)` from a single contract method, or use `contextvars.ContextVar` for cross-cutting metadata.

#### A9. No observable healthcheck / liveness signal
- **Sources:** Transport review §4.4.
- **Description:** No `ping`/`pong` server→client or client→server. CI harnesses and process supervisors have no clean liveness probe.
- **Location:** `chibi/runners/ide_transport.py` (no ping frame in protocol).
- **Impact:** Operational gap.
- **Fix:** Add `{"type": "ping"}` and `{"type": "pong"}` frames.

#### A10. `interface.error_code` is set-and-forgotten on success path
- **Sources:** Transport review §2.10.
- **Description:** The error path is checked *after* `send_message` calls. Implicit ordering contract.
- **Location:** `chibi/runners/ide_transport.py` lines ~194-198.
- **Impact:** Future handler may set error_code and call send_message, leading to ambiguity.
- **Fix:** Document the contract; or refactor so error state is returned as a discriminated union.

### 3.3 Low Risk

#### A11. `COMMANDS` constant vs. inline dispatch can drift
- **Sources:** Transport review §3.6.
- **Description:** Help message, dispatch chain, and `COMMANDS` constant are maintained separately.
- **Impact:** Documentation drift.
- **Fix:** Single source of truth — derive help from the registry.

#### A12. No `--log-level` flag on CLI
- **Sources:** Transport review §4.4.
- **Description:** Stderr loguru sink is INFO; no way to change verbosity from CLI.
- **Location:** `cli.py`; `ide_transport.py:284`.
- **Impact:** Operational inconvenience.
- **Fix:** Add `--log-level` option to `chibi run-ide --stdio`.

#### A13. Per-request latency not surfaced
- **Sources:** Transport review §4.4.
- **Description:** `started_at` is captured but not exposed. No `duration_ms` in completion status.
- **Impact:** Observability gap.
- **Fix:** Emit `{"type": "status", "state": "completed", "duration_ms": ...}` on completion.

---

## 4. Test Coverage Analysis

### 4.1 What's tested

Based on the reviews, **no tests are included with the IDE runner files**. The infrastructure review (Gemini 3.1-pro-preview) explicitly notes that `BackgroundTaskManager` lacks isolated unit tests, and the transport review notes the absence of `test_ide*` or `tests/*ide*` patterns.

### 4.2 What's missing

#### T1. No timeout tests for `BackgroundTaskManager`
- **Sources:** Infrastructure review.
- **Description:** The `run_task` timeout behavior (M1) is completely untested. With the bug fix, this must be the first new test set.
- **Coverage target:** default timeout; custom timeout; timeout-fires-cancels-task; timeout-on-already-done-task.

#### T2. No `DatabaseCache` backend resolution tests
- **Sources:** Infrastructure review.
- **Description:** Backend selection logic (in-memory vs. disk) is untested.
- **Coverage target:** env-var driven backend; default backend; fallback on backend init failure.

#### T3. `BackgroundTaskManager` lacks isolated unit tests
- **Sources:** Infrastructure review.
- **Description:** No unit-test harness for `run_task`, `_discard_task`, `shutdown`.
- **Coverage target:** Per-method unit tests with synthetic coroutines (raising, hanging, cancelling, succeeding, timing out).

#### T4. Transport tests use busy-wait `asyncio.sleep(0.01)` — flaky
- **Sources:** Infrastructure review.
- **Description:** Existing tests (if any) poll with `await asyncio.sleep(0.01)`. This is timing-dependent and flaky on CI.
- **Coverage target:** Replace busy-wait with `asyncio.Event` signaling, or use `asyncio.wait_for` with explicit deadlines.

#### T5. Module-level `_gate` global for sync prevents parallel test execution
- **Sources:** Infrastructure review.
- **Description:** A global synchronization primitive at module scope means tests that touch it cannot run in parallel within the same process.
- **Impact:** Slow CI.
- **Fix:** Move gate to per-test fixture or to class-level scope.

#### T6. Heavy mocking of `handle_user_prompt` rather than integration testing
- **Sources:** Infrastructure review.
- **Description:** Tests stub out the handler instead of running an end-to-end IDE session.
- **Impact:** The interface bugs (C1, H2) would not be caught by current-style tests.
- **Fix:** Add integration tests that spin up an in-process IDE runner with synthetic stdin/stdout and exercise the full request→response path.

#### T7. No edge-case tests for `IDE_STORAGE_ID` user
- **Sources:** Infrastructure review.
- **Description:** Edge cases specific to the shared-storage identity: drop history, image limits, model selection across sessions.
- **Coverage target:** Per-test: drive the runner with a sequence of `initialize` + multiple requests simulating two clients; assert storage state.

#### T8. No shutdown-while-tasks-in-flight tests
- **Sources:** Infrastructure review.
- **Description:** What happens when `shutdown` arrives while requests are mid-flight? With timeouts? Without?
- **Coverage target:** Send `shutdown` mid-request; assert drain or kill behavior; test with timeout wrapping.

#### T9. No protocol version negotiation edge-case tests
- **Sources:** Infrastructure review.
- **Description:** `unsupported_protocol_version` path is described in transport §2.7 but not exercised in tests.
- **Coverage target:** Min version mismatch; max version mismatch; missing version; non-int version; negative version.

#### T10. No round-trip / synthetic-stdio tests
- **Sources:** Transport review §4.6.
- **Description:** Recommend tests that drive the runner with `io.StringIO` for stdin/stdout.
- **Coverage target:** Happy path; error path; out-of-order cancel; protocol version mismatch; slow handler / timeout.

#### T11. No image-generation path tests
- **Sources:** Transport review §4.6.
- **Description:** The `send_images` bug (C1) is the most visible runtime bug — it has no test coverage.
- **Coverage target:** Drive `/imagine`; assert the response includes a valid attachment frame, not Python reprs.

#### T12. No observability / cancellation-ack tests
- **Sources:** Transport review §2.6.
- **Description:** The cancellation ack racy behavior (C2) is untested; clients cannot rely on the contract.
- **Coverage target:** Cancel mid-request; assert exactly one `cancelled` frame is emitted.

### 4.3 Test quality issues

- **Flaky timing patterns** (T4): replace busy-waits with event-driven waits.
- **Global synchronization** (T5): refactor into per-test fixtures.
- **Mocking over integration** (T6): add integration tests for the full request path.
- **Missing invariants**: no test enforces the "exactly one terminal frame per request" invariant.

---

## 5. Improvement Recommendations

### 5.1 Immediate (next sprint)

1. **Fix `IDEInterface.send_images` (C1).** This is a user-visible runtime bug. Add a typed `attachment` frame and route `str` URLs through it; do not stringify `BytesIO`.
2. **Fix cancellation ack (C2).** Wrap terminal-frame writes in `asyncio.shield` so cancellation cannot drop the ack.
3. **Add a `timeout` parameter to `task_manager.run_task` (M1).** Default to a sensible value negotiated in `ready.capabilities`. Per-request override via the `request` frame.
4. **Add tests for `BackgroundTaskManager` timeout (T1) and `send_images` (T11).** Lock in the fix and prevent regression.
5. **Document `IDE_STORAGE_ID` singleton semantics (H1).** Until per-session identity is implemented, the multi-client caveat must be visible in the constants file and in IDE docs.

### 5.2 Short-term (next 1–2 months)

1. **Per-session identity (H1).** Allow IDE to pass a `client_id` (workspace UUID) in `initialize`. Derive `user_id = hash(client_id)`. Wire through `IDEInterface`.
2. **Replace `setattr` with typed returns (H2, A8).** Handlers return `(text, meta)`. Remove the mutation pattern.
3. **Hide unsupported thread tools from the LLM (H3).** Filter tool surface per-interface.
4. **Schema validation at protocol boundary (A3).** Introduce `pydantic` models per message type; replace the `elif` chain with a dispatch table.
5. **`functools.wraps` on `run_task` timeout wrapping (M2).** Preserve `__name__`.
6. **`_discard_task` cleanup of empty sets (M4).** Delete empty entries from `_tasks` and `_task_to_user_id`.
7. **`SingletonMeta` thread safety (M5).** Add `threading.Lock`.
8. **Fix `_discard_task` cancellation check (M3).** Use `task.cancelled()` first.
9. **Strict in-flight-only `request_id` collision check (M6).** Allow post-completion reuse.
10. **Explicit shutdown drain semantics (M7).** On `shutdown`, break the read loop and gather in-flight tasks.
11. **Test gap coverage (T1–T12).** Add unit tests for `BackgroundTaskManager`, integration tests for the runner, edge cases for `IDE_STORAGE_ID` user, and protocol version negotiation.

### 5.3 Long-term (next quarter)

1. **Streaming output via `delta` frames (A4).** Replace single `result` frame with optional deltas.
2. **`TaskManagerProtocol` injection (A2).** Make the runner testable in isolation; allow per-process isolation.
3. **Per-session storage identity (A5).** Full fix for the storage isolation gap.
4. **Split `IDEInterface` into `RequestContext` + `ResponseSink` (A6).** Cleaner separation of concerns.
5. **Command registry (A7).** Replace inline dispatch with a registry. Single source of truth for `COMMANDS`.
6. **Healthcheck (`ping`/`pong`) and per-request latency (A9, A13).** Observability surfaces for CI and production.
7. **CLI: refactor `ide` to `@main.command()` with `--stdio` (L12).**
8. **Document `IDEInterface` contract (L3, A10).** Error-state-vs-success ordering, send_image semantics, single-terminal-frame invariant.

---

## 6. Appendix: Complete Issue Catalog

| ID | Category | Severity | Source Report | Description | Location |
|---|---|---|---|---|---|
| C1 | Bug | Critical | kimi-k2.7-code, MiniMax-M3 | `IDEInterface.send_images()` emits Python object reprs (`BytesIO` → `&lt;_io.BytesIO object at 0x...&gt;`) as text — garbage for binary images | `chibi/runners/ide_transport.py` ~117-122 |
| C2 | Bug | Critical | MiniMax-M3 | Cancellation ack frame may never be emitted because `await self._error(...)` inside `except CancelledError` is racy across asyncio versions | `chibi/runners/ide_transport.py` ~200-204 |
| H1 | Bug | High | MiniMax-M3, kimi-k2.7-code, Gemini-3.1-pro-preview | `IDE_STORAGE_ID` is a singleton — all IDE sessions share identity; cross-contamination of `/info`, `/model`, `/reset`, conversation history | `chibi/constants.py:21`; `chibi/runners/ide_transport.py` `IDEInterface.__init__` |
| H2 | Bug | High | kimi-k2.7-code, MiniMax-M3 | `setattr` smuggling for `response_model`/`response_provider` bypasses type safety; misbehaving handlers can corrupt `result` frame schema | `chibi/runners/ide_transport.py` `IDEInterface` (set via `bot.handle_user_prompt`); read ~195-199 |
| H3 | Bug | High | kimi-k2.7-code | Thread-management tools (create/rename/delete) raise `NotImplementedError` but are visible to LLM via tool-call surface | `chibi/runners/ide_transport.py` `IDEInterface` thread methods |
| H4 | Bug | High | MiniMax-M3 | `_initialized`/`_stopping` flags mutated without explicit synchronization — single-reader pattern provides implicit atomicity today, fragile to future changes | `chibi/runners/ide_transport.py:131-133`; mutations at ~233, ~244, ~271, ~177 |
| M1 | Bug | Medium | Gemini-3.1-pro-preview, MiniMax-M3 | `run_task` timeout defaults to `None` — tasks can run indefinitely | `chibi/services/task_manager.py` `run_task`; caller at `chibi/runners/ide_transport.py:265-267` |
| M2 | Bug | Medium | Gemini-3.1-pro-preview | Coroutine `__name__` lost after timeout wrapping — `AttributeError` on `.cancel()` and logging | `chibi/services/task_manager.py` wrapping code |
| M3 | Bug | Medium | Gemini-3.1-pro-preview | `_discard_task` checks cancellation via `task.exception()` — anti-pattern; use `task.cancelled()` first | `chibi/services/task_manager.py` `_discard_task` |
| M4 | Bug | Medium | Gemini-3.1-pro-preview | `_discard_task` never deletes empty sets from `_tasks` — memory leak | `chibi/services/task_manager.py` `_tasks` dict |
| M5 | Bug | Medium | Gemini-3.1-pro-preview | `SingletonMeta` not thread-safe — concurrent first-access can produce multiple instances | `chibi/services/task_manager.py` `SingletonMeta` |
| M6 | Bug | Medium | MiniMax-M3 | `request_id` collision check too strict — rejects post-completion reuse with misleading "request_id is already in use" | `chibi/runners/ide_transport.py:254` |
| M7 | Bug | Medium | MiniMax-M3 | `shutdown` does not drain in-flight requests — ambiguous shutdown semantics | `chibi/runners/ide_transport.py:272-274` |
| M8 | Bug | Medium | kimi-k2.7-code | `/model` command duplicates Telegram model-selection logic — drift between interfaces | `chibi/runners/ide_transport.py` ~172-191 |
| L1 | Bug | Low | MiniMax-M3 | `task_manager.run_task(...)` is fire-and-forget — no done-callback | `chibi/runners/ide_transport.py:265-267` |
| L2 | Bug | Low | MiniMax-M3 | `_thread_requests` increment/decrement order is fragile (increment before dispatch) | `chibi/runners/ide_transport.py` ~264-269 |
| L3 | Bug | Low | MiniMax-M3 | `error_code`/`error_message` checked after `send_message` calls — order-dependent contract | `chibi/runners/ide_transport.py` ~194-198 |
| L4 | Bug | Low | kimi-k2.7-code | `/imagine` `ValueError` becomes generic `request_failed` rather than `invalid_argument` | `chibi/runners/ide_transport.py` `/imagine` dispatch |
| L5 | Bug | Low | kimi-k2.7-code | `_set_ide_error` only covers decorated service functions — coverage gaps | `chibi/runners/ide_transport.py` decorator apply site |
| L6 | Bug | Low | MiniMax-M3 | JSON line read has no length cap — DoS via memory exhaustion | `chibi/runners/ide_transport.py:152-155` |
| L7 | Bug | Low | MiniMax-M3 | Single-threaded stdin reading has no partial-line buffer — large prompts may exceed PIPE_BUF | `chibi/runners/ide_transport.py` `_read_line` |
| L8 | Bug | Low | MiniMax-M3 | `_write` lock doesn't protect `json.dumps` work — latent risk on stdout reconfiguration | `chibi/runners/ide_transport.py:139-145` |
| L9 | Bug | Low | MiniMax-M3 | `_handle_message` for `cancel` doesn't await the cancelled task | `chibi/runners/ide_transport.py:267-271` |
| L10 | Bug | Low | MiniMax-M3 | No `cancel-all` or `stop` message | `chibi/runners/ide_transport.py` `_handle_message` |
| L11 | Bug | Low | MiniMax-M3 | No max concurrent requests per thread — resource exhaustion under load | `chibi/runners/ide_transport.py:255-269` |
| L12 | Bug | Low | MiniMax-M3 | CLI: `ide` without `--stdio` silently does nothing useful | `cli.py:30-32` |
| L13 | Bug | Low | MiniMax-M3 | `run_ide()` does side-effect `import chibi.config` — hidden dependency | `chibi/runners/ide.py:9-10` |
| L14 | Bug | Low | MiniMax-M3 | Logger reconfigured on every `run()` — `logger.remove()` wipes any other sinks (e.g., in tests) | `chibi/runners/ide_transport.py:284` |
| L15 | Bug | Low | MiniMax-M3 | `request_id` allowed to be `None` for non-request messages but strict for `request` | `chibi/runners/ide_transport.py:158-168` vs ~216 |
| L16 | Bug | Low | MiniMax-M3 | `_stopping` is read but never reset; no "cancel-all and shut down" API | `chibi/runners/ide_transport.py:280-288` |
| A1 | Architecture | High | MiniMax-M3, kimi-k2.7-code | Shared `IDE_STORAGE_ID` collapses IDE session isolation (same as H1) | — |
| A2 | Architecture | High | MiniMax-M3 | `task_manager` is a process-global singleton — hard to test, hard to isolate | `chibi/services/task_manager.py`; consumer at `chibi/runners/ide_transport.py:13` |
| A3 | Architecture | High | MiniMax-M3 | No protocol schema validation at the boundary — frames validated ad-hoc | `chibi/runners/ide_transport.py` `_handle_message`, `_valid_request` |
| A4 | Architecture | High | MiniMax-M3 | No streaming output — all responses buffered into one `result` frame | `chibi/runners/ide_transport.py` `content = "\n".join(responses)` |
| A5 | Architecture | High | kimi-k2.7-code | Thread-level concurrency isolated but storage not — half-isolated invariants | `chibi/runners/ide_transport.py` `_thread_requests`; `IDEInterface` storage fields |
| A6 | Architecture | High | MiniMax-M3 | `IDEInterface` mixes input data, output sink, and metadata — three responsibilities, one class | `chibi/runners/ide_transport.py` `IDEInterface` class |
| A7 | Architecture | Medium | MiniMax-M3 | Command handling inlined in `_run_request` — god method, drift between constant and dispatch | `chibi/runners/ide_transport.py:172-191` |
| A8 | Architecture | Medium | kimi-k2.7-code, MiniMax-M3 | `setattr` smuggling pattern as architectural smell (extends H2) | — |
| A9 | Architecture | Medium | MiniMax-M3 | No observable healthcheck / liveness signal — no `ping`/`pong` | `chibi/runners/ide_transport.py` (no ping frame) |
| A10 | Architecture | Medium | MiniMax-M3 | `interface.error_code` set-and-forgotten on success path — implicit ordering contract | `chibi/runners/ide_transport.py` ~194-198 |
| A11 | Architecture | Low | MiniMax-M3 | `COMMANDS` constant vs. inline dispatch can drift | `chibi/runners/ide_transport.py` |
| A12 | Architecture | Low | MiniMax-M3 | No `--log-level` flag on CLI | `cli.py`; `ide_transport.py:284` |
| A13 | Architecture | Low | MiniMax-M3 | Per-request latency not surfaced | `chibi/runners/ide_transport.py` |
| T1 | Test | — | Gemini-3.1-pro-preview | No timeout tests for `BackgroundTaskManager` | `tests/` (missing) |
| T2 | Test | — | Gemini-3.1-pro-preview | No `DatabaseCache` backend resolution tests | `tests/` (missing) |
| T3 | Test | — | Gemini-3.1-pro-preview | `BackgroundTaskManager` lacks isolated unit tests | `tests/` (missing) |
| T4 | Test | — | Gemini-3.1-pro-preview | Transport tests use busy-wait `asyncio.sleep(0.01)` — flaky | `tests/` (existing) |
| T5 | Test | — | Gemini-3.1-pro-preview | Module-level `_gate` global prevents parallel test execution | `tests/` (existing) |
| T6 | Test | — | Gemini-3.1-pro-preview | Heavy mocking of `handle_user_prompt` rather than integration testing | `tests/` (existing) |
| T7 | Test | — | Gemini-3.1-pro-preview | No edge-case tests for `IDE_STORAGE_ID` user (drop history, image limits) | `tests/` (missing) |
| T8 | Test | — | Gemini-3.1-pro-preview | No shutdown-while-tasks-in-flight tests | `tests/` (missing) |
| T9 | Test | — | Gemini-3.1-pro-preview | No protocol version negotiation edge-case tests | `tests/` (missing) |
| T10 | Test | — | MiniMax-M3 | No round-trip / synthetic-stdio tests | `tests/` (missing) |
| T11 | Test | — | MiniMax-M3 | No image-generation path tests (would catch C1) | `tests/` (missing) |
| T12 | Test | — | MiniMax-M3 | No observability / cancellation-ack tests (would catch C2) | `tests/` (missing) |

---

*Synthesized from three independent reviews. Source reports:*

- *Transport/protocol review (MiniMax/MiniMax-M3) — `chibi/_ide_transport_review_report.md`*
- *Integration-layer review (kimi-k2.7-code)*
- *Infrastructure/test review (Gemini 3.1-pro-preview)*

*Synthesis produced by MiniMax/MiniMax-M3. No new analysis introduced; all content is derived from the source reports.*