# AGENTS.md — AI Agent Guidelines for Chibi

> **Purpose:** This document defines how AI agents (executors, reviewers, planners) MUST work within the Chibi codebase. It is the single source of truth for code conventions, tooling, and project structure.

> **Precedence:** On conflict, AGENTS.md governs code conventions; workflow skills (`executor_workflow_skill.md`, `reviewer_workflow_skill.md`) govern process. If they disagree on process, the skill wins; on code style and architecture, AGENTS.md wins.

---

## 1. Project Identity

**Chibi** — open-source, self-hosted Telegram bot (MIT License). Orchestrates multiple AI providers, tools, MCP integrations, and sub-agents for autonomous task execution. Built with `python-telegram-bot` v22.7+, Pydantic v2, Redis, DynamoDB.

- Repository: `https://github.com/s-nagaev/chibi`
- Docs: `https://chibi.bot`
- Package: `chibi-bot` on PyPI
- Docker: `pysergio/chibi` on Docker Hub
- Python: `^3.11`

---

## 2. Directory Layout

```
chibi/                    # Main package
├── __main__.py           # `python -m chibi` entry point
├── cli.py                # CLI (Click)
├── config/               # Settings: app, gpt, telegram, logging
│   └── __init__.py       # Exports singletons: application/gpt/telegram_settings
├── constants.py          # Enums, constants, provider metadata
├── exceptions.py         # Custom exceptions
├── models.py             # Pydantic models (User, Message, etc.)
├── interfaces/           # Interface protocols: interface.py (UserInterface,
│   │                     # TelegramInterface, EditorContextProvider),
│   │                     # terminal_interface.py, scheduler_interface.py,
│   │                     # stdio_interface.py (StdioInterface),
│   │                     # stdio_runner.py (StdioRunner)
├── runners/              # Entry points: telegram.py, terminal.py
├── schemas/              # Provider API request/response schemas (Pydantic)
├── service.py            # Core service orchestration
├── services/             # Business logic
│   ├── events/           # Event modules: cwd_events.py, subagent_events.py
│   ├── providers/        # LLM provider adapters
│   │   └── tools/        # ChibiTools exposed to LLMs
│   └── ...
├── storage/              # Persistence: redis.py, dynamodb.py, local.py, database.py (factory)
└── utils/                # Utility functions
data/                     # Runtime data & skills
docs/                     # Documentation
scripts/                  # Operational scripts
tests/                    # pytest suite
.project/                 # PM workflow artifacts (not packaged)
conftest.py               # Shared pytest fixtures
pyproject.toml            # Poetry config + tool settings
```

---

## 3. Python Code Style (Backend)

### 3.1 Docstrings — Google Style, Mandatory

Every function, method, and class MUST have a Google-style docstring. Document `Args` and `Returns`. `Raises:` is required for any function that intentionally raises exceptions. English only.

**`Returns:` — omit when returning `None`:** If the function returns `None` (declared or implicit), do NOT include a `Returns:` section in the docstring. It is noise.

```python
# Correct — no Returns: for None
def save_user(user: User) -> None:
    """Persist the user to the active backend."""

# Correct — Returns: present for non-None
def get_user(user_id: int) -> User | None:
    """Fetch a user by ID.

    Args:
        user_id: Unique user identifier.

    Returns:
        The matching User object, or None if not found.
    """
```

### 3.2 No Comments Outside Functions

- **No block comments** (`# --- Section ---`).
- **No header comments** at the top of functions.
- **No file-level description comments** (use module docstring if truly needed).
- **Inline comments** are allowed ONLY when the code genuinely needs clarification — not as a substitute for clear naming.
- DO NOT use comments to structure code visually. Use empty lines and clear function boundaries.

**TODO/FIXME comments:** Allowed ONLY when linked to a GitHub issue: `# TODO(#1234): Refactor this when PTB v23 is released`. Standalone TODO/FIXME without an issue reference is forbidden.

**`# noqa`:** Forbidden by default — fix the underlying violation instead. The ONLY permitted exception is `# noqa: F401` on intentional side-effect imports (e.g., `import chibi.config.logging  # noqa: F401` to trigger logging setup). Must include the rule code after `noqa:`.

### 3.3 Imports — Module Level Only

- ALL imports go at the top of the file.
- Order: stdlib → third-party → internal (each group separated by a blank line).
- Use `from typing import TYPE_CHECKING` for type-only imports — place them under `if TYPE_CHECKING:`.

**Circular imports — two distinct cases:**

1. **Type-only circular dependency:** Use `TYPE_CHECKING` guard. The import only exists for mypy, never at runtime.
2. **Runtime circular dependency:** A local import inside the function body is permitted, marked with `# Circular import avoidance` comment. This is a last resort — prefer restructuring the modules.

```python
# Good
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from chibi.config import application_settings

if TYPE_CHECKING:
    from chibi.services.bot import ChibiBot
```

### 3.4 No Nested Functions

- Flat is better than nested. Extract to module-level private functions.
- **Exception:** Decorators (e.g., `@contextlib.contextmanager`, `@property`).

```python
# Forbidden
def outer(x: int) -> int:
    def inner(y: int) -> int:  # NO
        return y * 2
    return inner(x)

# Correct
def _double(y: int) -> int:
    """Double a number."""
    return y * 2

def outer(x: int) -> int:
    """Apply doubling to x."""
    return _double(x)
```

### 3.5 No `type: ignore`

- `# type: ignore` is forbidden. Period.
- If mypy complains, fix the root cause — cast types properly, refine signatures, or use explicit `cast()` from `typing`.
- Every existing `type: ignore` in the codebase must eventually be eliminated. New code = zero new ignores.

### 3.6 Type Annotations — Mandatory

ALL function signatures must have explicit type annotations. For variable declarations: annotate any variable mypy cannot infer, plus all module-level and class-level attributes. Local variables with obvious literal RHS (`x = 42`, `name = "Chibi"`) need no annotation — defer to mypy's output, not human judgment.

- **`Any` — avoid in internal logic.** `Any` is permitted ONLY at external I/O boundaries (JSON payloads, third-party untyped APIs, MCP tool inputs, DynamoDB item dicts). Narrow via `cast()` or Pydantic model validation before the value enters business logic.
- **Annotate collections fully:** `list[str]`, `dict[str, int]`, `tuple[int, str]`.
- **`| None` instead of `Optional`:** `str | None`, not `Optional[str]`.
- **`|` instead of `Union`:** `int | str`, not `Union[int, str]`.
- Use `cast()` from `typing` when narrowing types manually (add a comment explaining why the cast is safe).
- Use `TYPE_CHECKING` for imports needed only for type checking.

```python
# Acceptable Any at I/O boundary
def handle_api_payload(raw: dict[str, Any]) -> ProcessedData:  # OK — untrusted external data
    """Parse and validate an external API payload."""
    payload = APIPayloadSchema.model_validate(raw)  # Narrow to Pydantic model
    return ProcessedData(field=payload.key)
```

### 3.7 Pydantic Models

**Every field requires `description`.**

```python
class UserPreferences(BaseModel):
    """Stores user-specific preferences."""

    language: str = Field(default="en", description="Preferred UI language code (ISO 639-1).")
    theme: str | None = Field(default=None, description="Selected visual theme name, or None for default.")
```

**Pydantic v2 ONLY — zero tolerance for v1 API:**
- `model_config = ConfigDict(...)` — never class `Config`.
- `model_validate(...)` / `model_dump(...)` — never `parse_obj()` / `dict()`.
- `field_validator` / `model_validator` — never `@validator` / `@root_validator`.

### 3.8 Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `topic_manager.py` |
| Classes/Pydantic models | `PascalCase` | `UserPreferences` |
| Functions/methods | `snake_case` | `get_user_by_id()` |
| Internal/private functions | `_snake_case` prefix | `_parse_raw_response()` |
| Constants (module-level) | `UPPER_SNAKE_CASE` | `MAX_RETRY_COUNT` |
| Async functions | No special prefix | `fetch_data()` — not `async_fetch_data()` |
| Test files | `test_*.py` | `test_topic_service.py` |

### 3.9 Tests Follow the Same Rules

Tests are first-class code. All rules in sections 3.1–3.8 apply identically to test files. Additionally:

- **Async tests:** Use `pytest-asyncio` (`@pytest.mark.asyncio` or `pytestmark = pytest.mark.asyncio`). Never call `asyncio.run()` in a test.
- **Fixtures:** Shared fixtures in root `conftest.py`, test-suite-local fixtures in `tests/conftest.py`. Scope them explicitly (`function`/`session`/`module`).
- **External mocking:** Mock all external calls — Telegram API, LLM providers, Redis, DynamoDB. Use `fakeredis` for Redis, `freezegun` for time-dependent tests.
- **File naming:** `tests/test_<module>.py`.
- **No coverage threshold specified** — but every new feature must have at least one happy-path test and one edge-case/failure test.

---

## 4. Tooling & Checks

Run from **repo root** (`/Users/sergio/Develop/personal/chibi/`). All checks must pass with **zero errors and zero warnings** before submitting any code change.

**Preferred (Task):**
```bash
task lint && task test
```

`task lint` runs `ruff format .` + `ruff check --fix .` + `mypy .`. `task test` runs `pytest -vv .`.

**Manual (individual tools):**
```bash
ruff format --check . && ruff check . && mypy chibi/ && pytest -q
```

Do NOT override ruff or mypy config from `pyproject.toml`.

---

## 5. Architectural Rules

### 5.1 Provider Pattern

LLM providers live in `chibi/services/providers/`. Each provider extends `Provider` (the abstract base in `provider.py`). Intermediate bases exist: `OpenAIFriendlyProvider`, `RestApiFriendlyProvider`, `AnthropicFriendlyProvider`. Consult `provider.py` for the abstract method contract — implement every abstract method. Additionally:

- Registration is automatic — no manual registry edits: subclassing `Provider` triggers `Provider.__init_subclass__` (`chibi/services/providers/provider.py`), which registers the class in `RegisteredProviders` keyed by the lowercased `name` class attribute. The provider is additionally registered as available when its class attribute `api_key` is truthy.
- Activation happens purely by importing the module: add the provider's import to `chibi/services/providers/__init__.py`.
- API keys are class-attribute snapshots (`api_key = gpt_settings.<provider>_key`); the key fields are declared in `GPTSettings` (`chibi/config/gpt.py`, pydantic-settings with env aliases) and mirrored in `.env` / `.env.example`.
- Include Google-style docstrings on all methods.

### 5.2 ChibiTool Pattern

Tools exposed to LLMs live in `chibi/services/providers/tools/`. Each tool extends `ChibiTool` and registers via the tool registry pattern.

**Declared scope** = the tool's documented inputs and return value. A tool MUST NOT:
- Mutate global state outside the passed-in resources.
- Write to files or storage not explicitly provided as a parameter.
- Produce side effects beyond its return value and explicitly documented state changes.
- Access user data it wasn't granted access to via its parameters.

### 5.3 Config

Configuration is split across multiple modules in `chibi/config/`, exposed as singletons via `chibi/config/__init__.py`:

| Singleton | Source module | Purpose |
|-----------|--------------|---------|
| `application_settings` | `chibi/config/app.py` | App-level settings (environment, paths) |
| `gpt_settings` | `chibi/config/gpt.py` | LLM provider keys, model configs |
| `telegram_settings` | `chibi/config/telegram.py` | Telegram Bot API token, chat configs |

Import from the package: `from chibi.config import application_settings, gpt_settings, telegram_settings`. Never `from chibi.config.settings import ...` — that file does not exist. Environment variables go in `.env`. No hardcoded values. No scattered `os.getenv()`.

**Notable Configuration:**
- `LLM_ROLE`: Available in `gpt_settings`. This environment variable holds free-form text describing the agent's role/persona. It is injected exclusively into the master agent's prompt — NOT sub-agents — with a strict priority hierarchy: INVARIANTS (safety, honesty, privacy, hard rules) > ROLE > default behavior.

### 5.4 Async First

Chibi is async at its core. Strict rules:

- **Use `await`** for all I/O operations.
- **`asyncio.run()` is forbidden inside library code.** Only allowed in runner entry points (`chibi/runners/`).
- **Never `time.sleep()`** — use `await asyncio.sleep()`.
- **Blocking calls** (CPU-heavy, sync libraries) → `await asyncio.to_thread()` or a dedicated executor.
- **Locking:** Use `asyncio.Lock`, not `threading.Lock`.
- **Cancellation safety:** All async context managers (`async with`) and iterators (`async for`) must handle `asyncio.CancelledError` gracefully — clean up resources in `finally` blocks.

### 5.5 Error Handling

- **Raise custom exceptions** from `chibi/exceptions.py` for domain-specific errors. Use built-in exceptions (`ValueError`, `TypeError`) only for programming errors.
- **Never bare `except:`** or `except BaseException:`. Always catch the most specific exception type.
- **Never silent `except ...: pass`.** At minimum, log the suppressed error.
- **`Raises:` section is mandatory** in docstrings for any function that intentionally raises exceptions (see 3.1).
- User-facing error messages: brief, actionable, in the user's language. No stack traces, no internal identifiers.

### 5.6 Logging

- **Use `loguru.logger` exclusively.** Never `logging` stdlib, never `print()`.
- **No secrets in logs.** Never log API keys, tokens, passwords — even partially.
- **Levels:** `logger.info` for system-level events (startup, config changes), `logger.debug` for troubleshooting internals, `logger.warning` for recoverable anomalies, `logger.error` for failures requiring attention.
- **Context:** Use `logger.bind(user_id=...)` or structured format rather than string interpolation when logging in request-context code.

### 5.7 Storage

Three interchangeable backends implement the `Database` ABC (`chibi/storage/abstract.py`). All three store the same data: **users** (`save_user`, `create_user`, `get_user`) and **messages** (`add_message`, `get_messages`, `drop_messages`). The active backend is selected at runtime via `application_settings.storage_backend`:

| Backend | Module | 
|---------|--------|
| Redis | `chibi/storage/redis.py` |
| DynamoDB | `chibi/storage/dynamodb.py` |
| Local (JSON files) | `chibi/storage/local.py` |

- `chibi/storage/database.py` provides `DatabaseCache` — the singleton factory that picks and initializes the active backend. Also exports `@inject_database` decorator for injecting the `Database` instance into async functions.
- **Never instantiate a storage client directly** outside `storage/`. Use `@inject_database` or `DatabaseCache.get_database()`.
- Redis operations must use `redis.asyncio` (the canonical async client; `aioredis` is deprecated and merged into it).

### 5.8 Naming Conventions (already in §3.8)

---

## 6. Agent Workflow Integration

This project uses an **Artifact-Driven PM Workflow** (PM → Executor → Reviewer). Artifacts live in `.project/` (not packaged). When operating as an Executor or Reviewer:

- **Mandatory initial step:** When working with code, you MUST always look for and read `AGENTS.md` (or `CLAUDE.md` if absent) and study it before starting work. This rule applies to BOTH the master model and all sub-agents.
- Read `executor_workflow_skill.md` or `reviewer_workflow_skill.md` first.
- Task files, reports, and reviews follow the naming convention: `{task_name}_task.md`, `_report.md`, `_review.md`.
- NEVER modify task files — those belong to the PM.
- ALL code must comply with this AGENTS.md. **Reviewers enforce every rule from sections 3–5.**

### 6.1 Commit & Changelog Conventions

- **Commit messages:** imperative mood, short summary line, optional body with details.
  ```
  feat(topic): add /clone_thread command with message cloning
  fix(bot): resolve TopicRegistry race condition in post_init
  ```
- **Changelog:** Update `changelog.md` for user-visible changes. Follow the existing format (version → date → change list).
