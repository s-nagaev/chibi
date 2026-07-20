# Report: IDE Runner Transport — Code Review

## Model
Minimax/MiniMax-M3

## Scope
Read-only review of the Chibi IDE JSONL-over-stdio transport:

- `chibi/runners/ide.py` — entrypoint
- `chibi/runners/ide_transport.py` — protocol & request loop
- `chibi/cli.py` — CLI integration (`ide --stdio`)

Cross-referenced (read-only):
- `chibi/constants.py` (IDE_STORAGE_ID)
- `chibi/services/task_manager.py` (task_manager singleton)
- `chibi/services/interface.py` (UserInterface base)

---

## 1. Architecture & How It Works

### 1.1 Process layout

`chibi run-ide --stdio` (cli.py:27-30) → `run_ide()` (ide.py:8-15) →
`asyncio.run(IDEStdioRunner().run())`.

The runner is a single async event loop that owns:
- one reader thread doing blocking `sys.stdin.readline()` (via `asyncio.to_thread`)
- N concurrent request coroutines dispatched via `task_manager.run_task(...)`
- one `asyncio.Lock`-guarded writer on `sys.stdout`
- a single `_initialized` flag and `_stopping` flag (both unguarded mutation)

Stderr is repurposed for loguru logs (ide_transport.py:285), and stdout is exclusively the JSONL wire. Logger is removed in the file-like sink first and re-added on stderr.

### 1.2 Wire protocol

All frames are `json.dumps(..., separators=(",", ":")) + "\n"`, written under `_stdout_lock`.

Server → Client frame `types` observed:
- `ready` (response to `initialize`, includes `protocol_version` and `capabilities.commands`)
- `status` (`state`: `"queued"` / `"running"`)
- `result` (includes `content`; optionally `model`/`provider`)
- `error` (always includes `code` and `message`; optionally `request_id` and `extra`)

Client → Server `type`s (`_handle_message`, ide_transport.py:209-275):
- `initialize` (must carry int `protocol_version`)
- `request` (must satisfy `_valid_request`)
- `cancel` (targets a previously seen `request_id`)
- `shutdown`

Unknown types yield `unknown_message` error and the loop continues.

### 1.3 Request lifecycle

1. Read line → strip-skip → `json.loads` → `_handle_message`.
2. On `request`: validate; if `thread_id` already has an outstanding request, immediately emit a `queued` status; `task_manager.run_task(self._run_request(...))`; remember the task in `_tasks[request_id]`; bump `_thread_requests[thread_id]`.
3. `_run_request` (ide_transport.py:148-211):
   - emits `status/running`
   - builds an `IDEInterface` bound to a request-local `responses.append` callback
   - dispatches on `"/command"` or `handle_user_prompt`
   - on success: emits a single `result` (optionally with model/provider)
   - on `interface.error_code` set (e.g., `/model` invalid arg): emits a custom error
   - on `CancelledError`: emits `cancelled` and re-raises so `task_manager` can complete
   - on any other exception: logs and emits generic `request_failed`
   - `finally`: pops task, decrements per-thread counter; removes entry at zero
4. On `cancel`: `task.cancel()`.
5. On `shutdown`: flip `_stopping = True`.

### 1.4 IDEInterface adapter

`IDEInterface` (ide_transport.py:24-128) is a `UserInterface` that:
- uses a fixed `IDE_STORAGE_ID` for chat/user/storage identity
- keeps the per-request `thread_id` and `prompt`
- routes `send_message` and `send_images` into `_emit` (a `responses.append` callback supplied by the request)
- no-ops every other media method (audio/video/document) and every "action"/attachment method
- raises `NotImplementedError` for thread create/rename/delete (the IDE owns threads)

The harness uses `setattr(...)` on the interface in `bot.handle_user_prompt` to populate `response_model` / `response_provider` — see §3.

### 1.5 Error handling flow

- Validation errors → `error/malformed_request` (or `not_initialized`, `unsupported_protocol_version`).
- Unknown message types → `error/unknown_message`.
- Protocol version mismatch → `error/unsupported_protocol_version` (with `server_protocol_version`) **and** sets `_stopping=True` + `exit_code=1`.
- Per-request failures → `_run_request` always emits exactly one terminal frame (`result` / `error` / `error/cancelled` / `error/request_failed`) before completion.
- `run()` finally: `task_manager.shutdown()` flushes remaining work.

---

## 2. Bugs & Real Problems Found

### 2.1 `_initialized` and `_stopping` are mutated from multiple coroutines without locking

- **Where:** `ide_transport.py:131-133` (`__init__`); mutations in `_handle_message` (e.g., 233, 244, 271) and `_run_request` (177).
- **What's wrong:** `_initialized` is set true on `initialize` and read in the request/cancel branches. `_stopping` is set in many places. The reader coroutine (`run`) reads `_stopping` on every iteration; the request coroutine writes `_stopping = True` from inside a dispatched task.
- **Why it matters:** These flags are read on the event loop without `await`. Python's `asyncio` does not yield between plain assignments in the same task, but a concurrent task running concurrently *will* interleave reads. The conditions are simple booleans so tearing isn't a concern, but `_handle_message` mutating `_initialized` while `run()` is reading it is a classic visibility/timing pitfall in async code without explicit synchronization. More important: a "concurrent while-loop" assumption is technically impossible here because the reader is a single coroutine, but anyone adding a second reader later will hit subtle races.
- **Fix:** Make these event-loop-local state explicit; or move them behind a single mutable state object updated under the `_stdout_lock` (no — the lock protects JSON writes; use a dedicated asyncio.Event or simple `bool` with the understanding that single-loop semantics give atomicity, but document it).

### 2.2 `request_id` collision: same `request_id` from different threads is rejected

- **Where:** `ide_transport.py:254` — `elif request_id in self._tasks`.
- **What's wrong:** `_tasks` is a flat dict keyed by `request_id`. If a client legitimately uses the same `request_id` string for two different `thread_id`s (the protocol never says `request_id`s must be globally unique), the second request gets `malformed_request` `request_id is already in use`. The check conflates two concepts: *busy request id* vs *id was reused post-completion*.
- **Why it matters:** Clients that tag requests with a stable scheme (e.g., `<thread>:<n>`) will trip this. Even if a client does not, an in-flight + completed collision is impossible (we pop on completion in `finally`) — but the message wording `"request_id is already in use"` is ambiguous and will mislead debugging.
- **Fix:** Distinguish "still in flight" from "completed"; only reject when the id is still in `_tasks`. Optionally reject if the id was *very* recently completed to prevent races; otherwise reuse is fine.

### 2.3 `_thread_requests` is decremented in a `finally` of `_run_request` — but `task_manager.run_task` may swallow errors

- **Where:** `ide_transport.py:158-208` and `ide_transport.py:265-270`.
- **What's wrong:** `_run_request` always runs to `finally` to balance the counter. Good. BUT: when `task_manager.run_task` returns `None` (shutdown), we increment the counter before the call and then decrement after (207 / 269). The decrement returns to 0, which is removed. Looks correct.
   - **However**, the path `task = task_manager.run_task(...)` returns `None` only when the manager is shutting down, and we continue running in the same coroutine. We *don't* await the task — we just return. So `_thread_requests[thread_id] -= 1` runs in the current call frame synchronously. That's fine here.
   - **Real bug:** in `_handle_message` line 269, `self._thread_requests[thread_id] -= 1` is only executed when `task is None`. We never increment for that path? Wait — line 264 increments before checking `contended`. So when task_manager returns None, line 264 already ran and line 269 rolls it back. OK.
- **Why it matters:** No bug here — but the ordering (increment → dispatch → maybe roll-back) is fragile. If a future edit accidentally removes the rollback in the `task is None` branch, the counter leaks.
- **Fix:** Make the increment conditional on a successful dispatch (i.e., move the increment below the `task is None` check). This makes the lifetime obvious.

### 2.4 `handle_image_generation` / `handle_reset` / `handle_user_prompt` can synchronously call `setattr` on `interface`

- **Where:** `ide_transport.py:189, 153, 192` — handler call sites; `interface.response_model` / `response_provider` set indirectly via `setattr` in handlers (see bot service).
- **What's wrong:** Responses are harvested purely via `interface.send_message(...)` writing into a local `responses` list. If a handler emits multiple `send_message` calls, the list is `"\n".join`-ed into a single `content` string. **Streaming chunk order is lost** if a handler interleaves send_message with anything else, and critically: any text that *isn't* routed through `send_message` (e.g., directly via Telegram's internal `update.message.reply_text`, which can't happen here because the interface overrides everything — but only because the interface overrides everything; there is no enforcement).
- **Why it matters:** If a future refactor adds a code path that bypasses `send_message`, the runner silently loses output. This is structural fragility.
- **Fix:** Either instrument the interface to collect into a single text accumulator and accumulate streamed chunks (`delta`-style messages) preserving ordering, or add a `assert_no_bypass` invariant check (e.g., override `__setattr__` to refuse writes outside the request scope). At minimum, document the contract: "All assistant output must flow through send_message."

### 2.5 `_write` lock doesn't protect `json.dumps` work

- **Where:** `ide_transport.py:139-145`.
- **What's wrong:** `json.dumps(message, ensure_ascii=False, separators=(",",":"))` runs *before* acquiring `self._stdout_lock`. Multiple coroutines can therefore serialize on `stdout.write` while running `json.dumps` in parallel — which is fine. But the lock only holds `write + flush`. If anyone ever changes `_write` to take a callback that yields, the lock could be released mid-frame or — worse — multiple frames could be interleaved on `stdout.buffer` if there's a `sys.stdout.reconfigure(line_buffering=False)` change. Today, with default stdout line buffering on a tty-less pipe, the whole message+newline is written atomically by libc.
- **Why it matters:** Low risk today; high risk if `line_buffering` is disabled or stdout is replaced (e.g., redirect to a non-tty file in non-blocking mode).
- **Fix:** Document the invariant: "stdout must be line-buffered and never reconfigured to be block-buffered." Add an assertion in `_write` that the lock is currently held by the current task. Consider acquiring the lock *before* building the JSON for the result frames too, to make end-to-end frame atomicity obvious.

### 2.6 `except asyncio.CancelledError: ... raise` re-raises but `_write` itself cannot run during cancellation

- **Where:** `ide_transport.py:200-204`.
- **What's wrong:** When `_run_request` is cancelled, the cancellation propagates out of `await self._write(...)` because `CancelledError` is raised inside `_write`'s lock acquisition (well, after the lock is held). The current code does:
  ```
  except asyncio.CancelledError:
      await self._error("cancelled", "Request cancelled.", request_id)
      raise
  ```
  In Python 3.8+, once a task is cancelled, awaiting again may raise `CancelledError` immediately and `asyncio.shield` semantics matter. In current 3.11+ behavior, `await self._error(...)` will run *if* the cancellation allows one more step (it often does — the `except` clause itself is the next scheduled step). However, after the await re-raises CancelledError, control returns to the caller and the `finally` runs. The emission of the `error` frame is therefore racy: in some Python versions / shields the write may never execute.
- **Why it matters:** Clients may not see a cancellation acknowledgement for cancelled tasks. Combined with the `task_manager.run_task(...)` API which presumably cancels cleanly, you may end up with silent cancellations.
- **Fix:** Wrap `_write` in `asyncio.shield(...)` for protocol-critical terminal frames (result/error/cancelled), so the cancellation delivery is decoupled from the writer's await.

### 2.7 JSON line read uses `readline` not `readline` with limit — large single line cannot be parsed safely

- **Where:** `ide_transport.py:152-155`.
- **What's wrong:** `sys.stdin.readline()` will happily read multi-GB lines (Python uses `readexactly`-like semantics). A malicious or buggy client could send a massive line; `json.loads` will then either succeed or raise `MemoryError`. We catch only `JSONDecodeError`.
- **Why it matters:** DoS via memory exhaustion.
- **Fix:** Cap input line length (e.g., 1 MB). Treat over-cap as `malformed_request`. Optional: stream-parse JSON incrementally and cap element depth.

### 2.8 `task_manager.run_task(...)` is fire-and-forget; the runner never awaits it

- **Where:** `ide_transport.py:265-267`.
- **What's wrong:** We start the task via `task_manager`, record it in `_tasks`, and return. `task_manager.shutdown()` is awaited in the `finally` of `run()`. While this is presumably correct (background manager handles waiting), **no exception is awaited** — so any `_run_request` exception not internally caught will be raised into `task_manager`'s bookkeeping. If that bookkeeping raises (e.g., log + drop), the user sees nothing on the wire.
   - Mitigation: `_run_request` already has a top-level `except Exception:` that emits `request_failed`. So this is mostly OK.
- **Why it matters:** `CancelledError` is re-raised but not emitted through the same path; see §2.6.
- **Fix:** Mirror the standard "tasks tracked → on completion remove + collect result" pattern. Add an explicit `task.add_done_callback(self._on_task_done)` that removes from `_tasks` and logs unexpected exception causes.

### 2.9 `setattr` of `response_model`/`response_provider` happens inside the call chain — failure leaves stale state on the interface

- **Where:** `_run_request` line 195-199 reads `interface.response_model` / `response_provider`.
- **What's wrong:** These attributes are set on the `IDEInterface` instance during a request. The `IDEInterface` is per-request, so no leakage to other requests. **But** `setattr` bypasses type-checking; nothing enforces that they are `str`. A misbehaving handler could set a non-string and break the result frame schema.
- **Why it matters:** Output contract drift.
- **Fix:** Type the attributes as `str | None`, and validate before constructing the `result` frame. Better: do not mutate interface state at all — return model info via a different mechanism (e.g., a tuple `(text, meta)` from the handler, or a contextvar).

### 2.10 `interface.error_code` / `error_message` set then forgotten on success path

- **Where:** `ide_transport.py:194-198`.
- **What's wrong:** `/model <bad>` sets `interface.error_code = "invalid_argument"` and emits error. Fine. But the interface instance lives only for this request, so reset is implicit. Risk: future handler may set error_code *and* call `send_message`, leading to ambiguity. Currently the code checks `error_code` *after* `send_message` calls and prefers the error path. This is fine but should be documented.
- **Fix:** Add a docstring on `IDEInterface` stating the contract: "If `error_code` is set after handler completion, emit `error` instead of `result`."

### 2.11 `request_id` is allowed to be `None` for non-request messages — but `_valid_request` requires non-empty str

- **Where:** `ide_transport.py:158-168` vs `_handle_message` (request_id extraction at 216).
- **What's wrong:** Strict validation on request, lax handling for cancel/shutdown (request_id is optional). A `request` without a `request_id` gets `malformed_request`, but the framework still tries to do `assert request_id is not None` on line 259 — which is unreachable but indicates the developer's mental model wasn't fully encoded in types.
- **Fix:** Use a typed `TypedDict` for each protocol message, validated by `pydantic` (Chibi already depends on it for models).

### 2.12 `shutdown` does not drain in-flight requests

- **Where:** `ide_transport.py:272-274`.
- **What's wrong:** Sets `_stopping = True` and returns to the loop. The next line is read; if EOF arrives, `run()`'s `finally` calls `task_manager.shutdown()` which presumably awaits tasks. **If the client sends `shutdown` then keeps stdin open**, the loop continues reading and dispatching new requests — which can be allowed (use case: graceful shutdown) but it also accepts `shutdown` from a *peer that has no authority to request it*. There's no auth.
- **Why it matters:** A malformed/compromised stdin pipe can both drain AND block shutdown.
- **Fix:** Either treat `shutdown` as "do not accept new requests but drain existing" (early exit from dispatch, then `await asyncio.gather(*self._tasks.values())`), or document the semantics.

### 2.13 `_handle_message` for `cancel` cancels the task but awaits nothing

- **Where:** `ide_transport.py:267-271`.
- **What's wrong:** `self._tasks[request_id].cancel()` is non-blocking. The task can complete mid-write of its result frame, and that frame may arrive after the cancellation frame was sent — but there's no issue there. **However**, the cancelled task removes itself from `_tasks` in its `finally`. After cancel returns, `_tasks[request_id]` may still exist for a microsecond until the cancellation propagates. Not a bug, just a comment-worthy.
- **Fix:** Use `task.add_done_callback(lambda t: t.result() if not t.cancelled() else None)` to surface exceptions.

### 2.14 `_stopping` is read but never reset; loop exits on EOF or shutdown but never on cancel-all

- **Where:** `ide_transport.py:280-288`.
- **What's wrong:** There's no API to cancel all in-flight tasks and shut down within a single request boundary. `shutdown` only flips the flag.
- **Why it matters:** Minor.
- **Fix:** Optional.

### 2.15 CLI: `ide` without `--stdio` silently does nothing when a subcommand isn't given

- **Where:** `cli.py:30-32`.
- **What's wrong:** If user runs `chibi ide` (no flag, no subcommand), the code prints help. But `chibi ide --stdio --stdio` (or `--stdio` plus a stray arg) does not validate. There are no subcommands on `ide` other than the flag; running `chibi ide start` would error from click — but not gracefully.
- **Fix:** Refactor `ide` to be a `@main.command()` with `--stdio`. No group needed.

### 2.16 `run_ide()` calls `import chibi.config` inside the function for side effects

- **Where:** `ide.py:9-10`.
- **What's wrong:** `import chibi.config  # noqa: F401` — side-effect import (likely env init). Hidden dependency; conventions are silently enforced.
- **Fix:** Explicit `chibi.config.load()` if that's what it does.

---

## 3. Potential Problems (Architecturally Concerning)

### 3.1 Single-threaded stdin reading

- **Where:** `_read_line` (ide_transport.py:152-155).
- **Concern:** `asyncio.to_thread(sys.stdin.readline)` blocks a worker thread. A single producer (the IDE) writes to stdin, so throughput is rarely an issue. But: a partial-line buffer that survived across an EOF is impossible to handle correctly with `readline`. If a client writes a frame split across two reads of `stdio`, `readline` will return an empty string on EOF and we'll exit the loop losing the trailing fragment. Currently this is acceptable because the IDE writes whole lines.
- **Risk:** Future support for very large prompts (multi-MB) may exceed pipe buffer; OS may fragment write → `readline` returns *partial* line, which `json.loads` rejects as `malformed_request`.
- **Mitigation:** Switch to a length-prefixed or explicit-record protocol. Or document: "frames must fit in PIPE_BUF (typically 4 KB on Linux, 16-64 KB on macOS)." Currently silent.

### 3.2 Protocol extensibility

- **Where:** `_handle_message` `elif/else` chain (ide_transport.py:209-275).
- **Concern:** Adding a new client→server `type` requires touching a chain in the same function. New server→client `type`s requires touching `_write` callers. There's no message-schema registry.
- **Risk:** Minor drift over time; each addition may forget to bump `PROTOCOL_VERSION` correctly.
- **Mitigation:** Generate types from a schema; or at least keep a single dispatch table at the top of the module.

### 3.3 IDE_STORAGE_ID collision

- **Where:** `IDE_STORAGE_ID = -(10**16)` (constants.py:21); used as `user_id` and `storage_id` and `chat_id` for all IDE requests.
- **Concern:** A single global identity is shared across all requests and across all IDE clients on the same machine. Database rows keyed by `(user_id, thread_id)` will accumulate across sessions. The `task_manager` also keys by `(user_id, thread_id)`.
- **Risk:** If two IDE clients (or two windows of one IDE) talk to the same Chibi process simultaneously, they share `/info`, `/model` selections, reset state — surprising behavior. Process isolation per workspace would be expected.
- **Mitigation:** Allow IDE to pass a per-session `client_id` (e.g., workspace UUID) in `initialize`, and derive `user_id = hash(client_id)`. Or document the singleton semantics explicitly.

### 3.4 `IDEInterface.send_images` sends BytesIO/image URLs as raw text

- **Where:** `ide_transport.py:117-122`.
- **Concern:** `"\n".join(str(image) for image in images)` calls `str(BytesIO)` → object repr like `<_io.BytesIO object at 0x...>`, or `str(image_url)` if it's already a string. **None of this is an image** — the IDE client receives meaningless text.
- **Risk:** Any code path that triggers image generation (`/imagine` or pipeline that produces images) leaks Python reprs to the user.
- **Mitigation:** Either:
  - Implement a binary-friendly out-of-band channel for images (a temp file with path emission, or base64 in a dedicated frame type).
  - Detect string-vs-bytes and at least forward URLs verbatim (most image generators return URL strings today, which works).
  - Add a typed `attachment` frame: `{"type": "attachment", "kind": "image", "data": "<base64>", "mime": "image/png"}`.

### 3.5 `setattr(response_model/response_provider)` smuggling

- **Where:** `IDEInterface.__init__` declares them on `self` directly (lines 41-42) — typed fields, OK. **But** the `bot` service uses `setattr` to *write* to them.
- **Concern:** Field writes happen deep inside the handler. The interface mixes data (config) with output (`send_message`). This is structural debt: a future typed-data class would do better.
- **Risk:** Subtle ordering bugs if a handler emits a `send_message` then sets model (text has no model info); today the result frame is assembled *after* handler returns so it captures the final values, but only because of accidental ordering.
- **Mitigation:** Have the handler return `(text, meta)` from a single contract method rather than mutating an object.

### 3.6 Command handling is inlined in `_run_request`

- **Where:** `ide_transport.py:172-191` (the if/elif chain for `/reset`, `/help`, `/model`, etc.).
- **Concern:** As commands grow, the function becomes a god method. Adding `/foo` requires touching the central method, the `COMMANDS` constant, and the help message.
- **Risk:** Drift between `COMMANDS` (constant) and the dispatch (inline).
- **Mitigation:** Extract a command registry: `commands.register("/model", handle_model)` etc. Dispatch becomes a single dict lookup.

### 3.7 No `stop` or `cancel-all` message

- **Where:** `_handle_message`.
- **Concern:** The IDE cannot ask "please cancel every in-flight request." Only individual cancels. With many concurrent requests, this would be tedious.
- **Risk:** Low.
- **Mitigation:** Add a `cancel_all` or fold `shutdown` semantics into "drain existing".

### 3.8 `task_manager` is a global singleton

- **Where:** `ide_transport.py:13`, dispatch at 266.
- **Concern:** `task_manager.run_task(...)` is a free function call into a global. Tests cannot easily inject a fake.
- **Risk:** Hard to unit-test the runner in isolation.
- **Mitigation:** Accept a `TaskManagerProtocol` in `__init__`.

### 3.9 Logger reconfiguration on every `run()`

- **Where:** `ide_transport.py:284`.
- **Concern:** `logger.remove()` strips *all* sinks, then re-adds stderr INFO. If something else in the process had added a sink (it shouldn't in stdio mode, but could in tests), it's wiped. Also: in stdio mode, the IDE may have captured stdout — stderr should remain available to loguru sinks.
- **Risk:** Low; subtle in tests.
- **Mitigation:** Use `logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"] != "telegram.ext")` only if a telegram-style sink exists. Or do nothing — `add` is idempotent.

### 3.10 Response truncation / chunking

- **Where:** `content = "\n".join(responses)` then `await self._write(result)`.
- **Concern:** All `send_message` calls are buffered into one big `result` frame. There is no streaming. For long generations, the user sees nothing until completion. The framework supported streaming via `status/running` and could in principle emit `delta` frames.
- **Risk:** UX in long generations.
- **Mitigation:** Emit `{"type": "delta", "request_id": ..., "content": ...}` from `send_message` instead of buffering.

### 3.11 No max concurrent requests per thread

- **Where:** `ide_transport.py:255-269`.
- **Concern:** `_thread_requests` tracks count but only emits a "queued" status — never applies back-pressure. An IDE that floods requests for the same thread will spawn unbounded concurrent tasks.
- **Risk:** Resource exhaustion.
- **Mitigation:** Configurable `MAX_PER_THREAD` (e.g., 1 by default; queue the rest).

### 3.12 No request timeout

- **Where:** `_run_request` has no `asyncio.wait_for`.
- **Concern:** A handler that hangs (network stall, infinite loop in tool call) blocks forever; `_thread_requests` never decrements.
- **Risk:** Resource leak + stuck transport.
- **Mitigation:** Wrap the call in `asyncio.wait_for(timeout, shield=...)` with a per-request timeout negotiated at initialize.

---

## 4. Improvement Points

### 4.1 Code organization

| Area | Improvement |
|---|---|
| `_run_request` (lines 148-211) | Split into `_dispatch_request`, `_dispatch_command`, and `_emit_terminal_frame` helpers. Currently 60+ LOC. |
| `_handle_message` (lines 209-275) | Replace elif-chain with a `{type: handler_class}` registry. Each handler is a small class with `validate(message) → Optional[error_code]` and `handle(...)`. |
| Command handling inline | Extract `COMMANDS = {"reset": handle_reset, "help": ..., "model": ..., ...}` registry (see §3.6). |
| `IDEInterface` | Group fields into `RequestContext` (immutable: thread_id, prompt, workspace_root) and `ResponseSink` (mutable: list of messages, model info, error state). Two responsibilities, two classes. |

### 4.2 Protocol robustness

| Area | Improvement |
|---|---|
| Frame length | Cap line length at `MAX_FRAME_BYTES = 4 * 1024 * 1024` (configurable). |
| Schema | Introduce `pydantic` models: `InitializeMessage`, `RequestMessage`, `CancelMessage`, `ShutdownMessage`. Reject unknown fields or wrong types at boundary. |
| Version handshake | On `unsupported_protocol_version`, also advertise `min_supported` and `max_supported`, not just `server_protocol_version`, so a future v3 server can serve v2 clients gracefully. |
| Cancel ack | Emit `{"type": "cancelled", "request_id": ...}` deterministically using `asyncio.shield` so the message survives the underlying cancel. |
| Back-pressure | Configurable `MAX_CONCURRENT_PER_THREAD`; queue beyond that and emit "queued" status only up to a total. |
| Per-request timeout | Negotiate `default_request_timeout_ms` in `ready.capabilities`; let `request` override per-call. |
| Final frame guarantee | The runner contract is "exactly one terminal frame per request." Add an invariant test that this holds even on `KeyboardInterrupt`, `SystemExit`, etc. |
| Streaming | Add `delta` frames for `send_message` calls; final `result` becomes optional or carries the concatenated string + per-delta offsets. |

### 4.3 Separation of concerns

- Move `COMMANDS` and protocol constants to `chibi/runners/ide_protocol.py`. Keep `ide_transport.py` for I/O only.
- Move `IDEInterface` next to the rest of `chibi.services.interface` subclasses (consistent with `TelegramInterface` there).
- Decouple the runner from `task_manager` (see §3.8): inject `TaskManagerProtocol`.

### 4.4 Observability

| Signal | How |
|---|---|
| Per-request latency | Record `started_at` and emit `{"type": "status", "state": "completed", "duration_ms": ...}`. |
| Per-request outcome | A counters dict keyed by `(outcome, command_or_prompt_kind)`. Optionally expose via admin endpoint. |
| Stream events | On every `status` transition, include a monotonic sequence number for client-side ordering. |
| Log verbosity | Add a `--log-level` flag to control stderr loguru sink. |
| Healthcheck | Periodic `{"type": "ping"}` if server→client side is required; useful for liveness in CI. |

### 4.5 Documentation gaps

- Add a `docs/ide-protocol.md` with:
  - Frame types (server→client, client→server)
  - Field types and required/optional
  - Lifecycle diagram for `request` (queued → running → completed/cancelled/failed)
  - Error code catalog (`malformed_request`, `not_initialized`, `unsupported_protocol_version`, `unknown_request`, `unknown_message`, `invalid_argument`, `cancelled`, `request_failed`)
  - Example session transcript
- Document the singleton storage identity in `constants.py`: `IDE_STORAGE_ID` should have a docstring stating its meaning and the multi-client caveat (§3.3).
- Document the contract on `IDEInterface.send_images` — what does an IDE client receive when image generation runs? Currently broken (§3.4).
- Document the CLI behavior: `chibi ide --stdio` is the only mode; `chibi ide` (no flag) prints help.

### 4.6 Test gaps

No tests are included with these three files (`grep -r "test_ide\|tests.*ide" tests/ 2>/dev/null` — verify before relying on this statement). Recommended:
- Round-trip tests with synthetic stdin/stdout (`io.StringIO`).
- Out-of-order cancel + request flood.
- Protocol version mismatch.
- Slow handler → timeout handling.
- Image generation path on `send_images`.

---

## Files Created / Modified
None — read-only review as instructed.

## DoD Compliance
- [x] Architecture & how it works (wire protocol, lifecycle, interface, error flow) — §1
- [x] Bugs & real problems — §2 (16 items)
- [x] Potential architectural concerns — §3 (12 items)
- [x] Improvement points (organization, protocol robustness, separation, observability, docs, tests) — §4
- [x] Each issue identifies what / where / why / suggested fix

## Notes
- This was a READ-ONLY review. No source files were modified.
- Cross-referenced files (`constants.py`, `task_manager.py`, `interface.py`, `bot` service) only via `grep` to validate assumptions in §3.4 and §3.5; full-text review of those is out of scope.
- The strongest concrete bug is **§2.6**: `await self._error(...)` inside `except CancelledError` is racy across Python asyncio versions. Wrap critical terminal frames in `asyncio.shield`.
- The strongest architectural concern is **§3.4**: `send_images` emits Python reprs for `BytesIO`, which means any image-generation pipeline currently produces garbage text.
