"""Agent-facing scheduler tools (design §5.1).

Expose the internal ``ChibiScheduler`` to the LLM so it can plan periodic and
one-time tasks on behalf of the user. All jobs are owned by the creating user
under the ``agent:{user_id}:*`` namespace; the chat/thread context of the
current request is fixed in the job payload at creation time.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from typing import Any, Unpack

from apscheduler.jobstores.base import ConflictingIdError, JobLookupError
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import ValidationError

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import SchedulerJobError
from chibi.schemas.scheduler import AgentJobPayload, CommandActionPayload
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, resolve_session_context
from chibi.services.scheduler import AGENT_JOB_PREFIX, ChibiScheduler
from chibi.services.user import get_moderation_provider

_JOB_SUFFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CRON_FIELD_ORDER = ("minute", "hour", "day", "month", "day_of_week")

_ACTION_TYPE_DESCRIPTIONS = (
    "self — the agent wakes up by itself in the fixed thread: it sees the thread context plus the "
    "trigger_text and does the work with a live answer (a true autonomous heartbeat). "
    "notify — sends a dumb static message to the user without waking the agent (zero LLM usage; "
    "may be disabled by server settings). "
    "command — runs a pre-moderated shell command on schedule and reports its output."
)


def _slugify(value: str) -> str:
    """Convert a free-form title or suffix into a valid job_id suffix.

    Args:
        value: Human-readable title or explicit suffix provided by the model.

    Returns:
        A suffix matching ``^[a-z0-9][a-z0-9_-]{0,63}$``.

    Raises:
        ToolException: If no valid suffix can be derived (e.g. a title made of
            non-latin characters only).
    """
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower())
    slug = slug.strip("-_")[:64].rstrip("-_")
    if not slug or not _JOB_SUFFIX_PATTERN.fullmatch(slug):
        raise ToolException(
            f"Could not derive a valid job id suffix from {value!r} "
            "(latin letters, digits, '-' and '_' only). Provide an explicit job_id."
        )
    return slug


class ScheduleTaskTool(ChibiTool):
    register = bool(sys.modules.get("chibi.runners.telegram")) and application_settings.scheduler_tool_enabled
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="schedule_task",
            description=(
                "Create a scheduled task (periodic or one-time) owned by the current user. The task is pinned "
                "to the current chat/thread: that context is fixed at creation time. Schedule variants: "
                "kind='interval' runs every `every_seconds` seconds; kind='cron' runs on a crontab-like "
                "calendar built from the minute/hour/day/month/day_of_week fields (each defaults to '*'); "
                "kind='once' runs a single time at the ISO8601 `at` datetime. Action variants: "
                "type='self' — the agent wakes up by itself in the fixed thread, sees the thread context "
                "plus trigger_text and does the work with a live answer (a true autonomous heartbeat); "
                "type='notify' — sends a dumb static message to the user without waking the agent (zero LLM "
                "usage, may be disabled by server settings); type='command' — runs a pre-moderated shell "
                "command and reports its output. Use list_scheduled_tasks to inspect existing tasks and "
                "delete_scheduled_task to remove one."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Human-readable task name, e.g. 'Daily report'.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": (
                            "Optional task identifier suffix (latin letters, digits, '-' and '_'). "
                            "If omitted, it is generated from the title. A full id under your own "
                            "'agent:<user_id>:' prefix is also accepted."
                        ),
                    },
                    "schedule": {
                        "type": "object",
                        "description": "When the task runs. Exactly one kind per task.",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["interval", "cron", "once"],
                                "description": "Schedule kind: 'interval', 'cron' or 'once'.",
                            },
                            "every_seconds": {
                                "type": "integer",
                                "description": "Interval between runs in seconds, must be > 0 (kind='interval').",
                            },
                            "minute": {
                                "type": "string",
                                "description": (
                                    "Cron minute field (kind='cron'), default '*'. Examples: '0', '*/15', '0,30'."
                                ),
                            },
                            "hour": {
                                "type": "string",
                                "description": ("Cron hour field (kind='cron'), default '*'. Examples: '9', '8-18'."),
                            },
                            "day": {
                                "type": "string",
                                "description": "Cron day-of-month field (kind='cron'), default '*'.",
                            },
                            "month": {
                                "type": "string",
                                "description": "Cron month field (kind='cron'), default '*'.",
                            },
                            "day_of_week": {
                                "type": "string",
                                "description": (
                                    "Cron day-of-week field (kind='cron'), default '*'. Examples: 'mon-fri', '1,3,5'."
                                ),
                            },
                            "at": {
                                "type": "string",
                                "description": (
                                    "ISO8601 datetime of the single run, must be in the future (kind='once'), "
                                    "e.g. '2026-10-01T09:00:00+02:00'."
                                ),
                            },
                        },
                        "required": ["kind"],
                    },
                    "action": {
                        "type": "object",
                        "description": f"What happens on every run. {(_ACTION_TYPE_DESCRIPTIONS)}",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["self", "notify", "command"],
                                "description": (
                                    "Action discriminator: 'self' wakes the agent in the fixed thread, "
                                    "'notify' sends a static message to the user, "
                                    "'command' runs a pre-moderated shell command."
                                ),
                            },
                            "trigger_text": {
                                "type": "string",
                                "description": (
                                    "Required for type='self': the trigger message injected into the thread "
                                    "when the job fires."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": (
                                    "Required for type='notify': the static message text delivered to the chat."
                                ),
                            },
                            "command": {
                                "type": "string",
                                "description": "Required for type='command': the shell command to execute.",
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Working directory for the command (type='command'), optional.",
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": (
                                    "Per-run command timeout in seconds (type='command'), default 60, "
                                    "clamped from above by the server configuration."
                                ),
                            },
                            "notify_on_nonempty_output": {
                                "type": "boolean",
                                "description": (
                                    "type='command': when true (default) the user is notified only if the "
                                    "command produced non-empty stdout."
                                ),
                            },
                        },
                        "required": ["type"],
                    },
                    "replace": {
                        "type": "boolean",
                        "description": (
                            "Replace an existing task with the same id. Use ONLY when the user explicitly "
                            "asked to overwrite an existing scheduled task. Default false."
                        ),
                    },
                },
                "required": ["title", "schedule", "action"],
            },
        ),
    )
    name = "schedule_task"
    allow_model_to_change_background_mode = False

    @classmethod
    def _check_job_limit(cls, user_id: int, existing_job_ids: list[str]) -> None:
        """Extension point for a per-user job limit (design §6.1).

        Intentionally not implemented in v1: the owner decided that no job
        limit is needed yet. Future implementations should raise a
        ``ToolException`` here when the user exceeds their quota.

        Args:
            user_id: Owner of the job being created.
            existing_job_ids: Job ids the user already owns.
        """
        return None

    @classmethod
    def _build_job_id(cls, user_id: int, job_id: str | None, title: str) -> str:
        """Build the fully qualified job id under the user's namespace.

        Args:
            user_id: Owner of the job.
            job_id: Optional explicit suffix (or a full id under the user's own prefix).
            title: Human-readable title used to derive a slug when job_id is omitted.

        Returns:
            The fully qualified job id ``agent:{user_id}:{suffix}``.

        Raises:
            ToolException: If the requested id points into another user's or
                the system namespace, or no valid suffix can be derived.
        """
        prefix = f"{AGENT_JOB_PREFIX}{user_id}:"
        if job_id and job_id.strip():
            candidate = job_id.strip()
            if ":" in candidate:
                if not candidate.startswith(prefix):
                    raise ToolException(
                        f"Access denied: you can only create jobs under your own prefix '{prefix}'. "
                        "'system:*' jobs are managed by Chibi itself."
                    )
                candidate = candidate.removeprefix(prefix)
            suffix = _slugify(candidate)
        else:
            suffix = _slugify(title)
        return f"{prefix}{suffix}"

    @classmethod
    def _resolve_job_context(cls, **kwargs: Unpack[AdditionalOptions]) -> tuple[int, int, int, int]:
        """Fix the current session context for the job payload.

        Args:
            kwargs: Additional options of the tool invocation.

        Returns:
            A ``(user_id, storage_id, thread_id, chat_id)`` tuple. When no
            interface travels with the call (e.g. sub-agent requests), chat_id
            falls back to storage_id — in Telegram chats both always refer to
            the same chat.

        Raises:
            ToolException: If user_id or the session context is unavailable.
        """
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")
        try:
            storage_id, thread_id = resolve_session_context(**kwargs)
        except ValueError as e:
            raise ToolException(str(e)) from e
        interface = kwargs.get("interface")
        chat_id = int(interface.chat_id) if interface is not None else storage_id
        return user_id, storage_id, thread_id, chat_id

    @classmethod
    def _parse_once_datetime(cls, raw: Any) -> datetime:
        """Parse and sanity-check the ``at`` value of a 'once' schedule.

        Args:
            raw: ISO8601 datetime string provided by the model.

        Returns:
            The parsed datetime.

        Raises:
            ToolException: If the value is not a valid ISO8601 datetime or is
                not in the future.
        """
        try:
            run_at = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError) as e:
            raise ToolException(f"Invalid ISO8601 datetime for schedule.at: {raw!r}.") from e
        now = datetime.now(tz=run_at.tzinfo)
        if run_at <= now:
            raise ToolException(f"The schedule.at datetime must be in the future (got {raw!r}).")
        return run_at

    @classmethod
    async def function(
        cls,
        title: str,
        schedule: dict[str, Any],
        action: dict[str, Any],
        job_id: str | None = None,
        replace: bool = False,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        """Create a scheduled task owned by the current user.

        Validates the action against the configured gates, pre-moderates
        shell commands, fixes the session context in the job payload and
        registers the job with the stable ``run_agent_job`` dispatcher.

        Args:
            title: Human-readable task name.
            schedule: Schedule specification (``interval`` / ``cron`` / ``once``).
            action: Action specification (``self`` / ``notify`` / ``command``).
            job_id: Optional explicit id suffix (or full id under the user's prefix).
            replace: Whether to overwrite an existing job with the same id.
            kwargs: Additional options provided to the tool.

        Returns:
            A dictionary describing the created job.

        Raises:
            ToolException: If the context is unavailable, a gate blocks the
                action, the parameters are invalid, the moderator declines a
                command, or the job id is already taken (with ``replace=false``).
        """
        caller_model = kwargs.get("caller_model", "unknown model")
        if not title or not title.strip():
            raise ToolException("Task title cannot be empty.")

        action_type = action.get("type") if isinstance(action, dict) else None
        cls._validate_action_gates(action_type=action_type)

        user_id, storage_id, thread_id, chat_id = cls._resolve_job_context(**kwargs)
        scheduler = ChibiScheduler()
        prefix = f"{AGENT_JOB_PREFIX}{user_id}:"
        existing_job_ids = [info.job_id for info in scheduler.list_jobs(prefix=prefix)]

        cls._check_job_limit(user_id=user_id, existing_job_ids=existing_job_ids)

        full_job_id = cls._build_job_id(user_id=user_id, job_id=job_id, title=title)
        if not replace and full_job_id in existing_job_ids:
            raise ToolException(
                f"Job id '{full_job_id}' is already taken. Existing scheduled tasks: "
                f"{', '.join(existing_job_ids)}. Pass replace=true ONLY if the user explicitly asked "
                "to overwrite an existing task."
            )

        try:
            payload = AgentJobPayload(
                job_id=full_job_id,
                user_id=user_id,
                storage_id=storage_id,
                thread_id=thread_id,
                chat_id=chat_id,
                title=title.strip(),
                action=action,
            )
        except ValidationError as e:
            raise ToolException(f"Invalid task parameters: {e}") from e

        await cls._moderate_scheduled_command(payload=payload, user_id=user_id, caller_model=caller_model)

        job_kwargs = {"job_id": full_job_id, **payload.model_dump(exclude={"job_id"})}
        cls._register_scheduled_job(
            scheduler=scheduler,
            schedule=schedule,
            full_job_id=full_job_id,
            job_kwargs=job_kwargs,
            replace=replace,
        )

        logger.log(
            "TOOL",
            (
                f"[{caller_model}] Scheduled task '{full_job_id}' ('{payload.title}', "
                f"action: {payload.action.type}) for user #{user_id}"
            ),
        )
        job_infos = scheduler.list_jobs(prefix=full_job_id)
        next_run_time = job_infos[0].next_run_time if job_infos else None
        return {
            "status": "ok",
            "job_id": full_job_id,
            "title": payload.title,
            "action_type": payload.action.type,
            "next_run_time": next_run_time.isoformat() if next_run_time else None,
            "message": f"Scheduled task '{payload.title}' ({full_job_id}) created.",
        }

    @classmethod
    def _validate_action_gates(cls, action_type: str | None) -> None:
        """Reject action types that are disabled by server settings.

        Args:
            action_type: The requested action type (``self`` / ``notify`` / ``command``).

        Raises:
            ToolException: If the requested action type is disabled by configuration.
        """
        if action_type == "notify" and not application_settings.scheduler_notify_enabled:
            raise ToolException(
                "The 'notify' action is disabled by the scheduler_notify_enabled setting. "
                "Use the 'self' action so the agent wakes up and handles the task itself."
            )
        if action_type == "command" and not application_settings.scheduler_agent_commands_enabled:
            raise ToolException(
                "Shell commands in scheduled tasks are disabled by the scheduler_agent_commands_enabled setting. "
                "Ask the administrator to enable them, or use the 'self' action instead."
            )

    @classmethod
    async def _moderate_scheduled_command(cls, payload: AgentJobPayload, user_id: int, caller_model: str) -> None:
        """Pre-moderate shell commands before a scheduled task is created.

        Args:
            payload: The validated job payload.
            user_id: Owner of the task (used to select the moderation provider).
            caller_model: Name of the model that requested the task (for logs).

        Raises:
            ToolException: If the moderator declines the command.
        """
        if not isinstance(payload.action, CommandActionPayload):
            return

        moderation_provider = await get_moderation_provider(user_id=user_id)
        logger.log("MODERATOR", f"[{caller_model}] Pre-moderating scheduled command: '{payload.action.command}'")
        moderator_answer = await moderation_provider.moderate_command(
            cmd=payload.action.command, model=gpt_settings.moderation_model
        )
        if moderator_answer.verdict == "declined":
            raise ToolException(
                f"Moderator declined the scheduled command '{payload.action.command}'. "
                f"Reason: {moderator_answer.reason}"
            )

    @classmethod
    def _register_scheduled_job(
        cls,
        scheduler: ChibiScheduler,
        schedule: dict[str, Any],
        full_job_id: str,
        job_kwargs: dict[str, Any],
        replace: bool,
    ) -> None:
        """Register the agent job on the scheduler according to the schedule kind.

        Args:
            scheduler: The scheduler singleton to register the job on.
            schedule: Schedule specification (``interval`` / ``cron`` / ``once``).
            full_job_id: Fully qualified job id (``agent:<user_id>:<suffix>``).
            job_kwargs: Keyword arguments passed to ``run_agent_job`` when the job fires.
            replace: Whether to overwrite an existing job with the same id.

        Raises:
            ToolException: If the schedule kind or its parameters are invalid, the
                scheduler rejects the job, or the job id was concurrently taken.
        """
        from chibi.services.jobs.agent_task import (
            run_agent_job,
        )  # Circular import avoidance: agent_task imports chibi.services.bot, which imports this tools package.

        schedule_kind = schedule.get("kind") if isinstance(schedule, dict) else None
        try:
            if schedule_kind == "interval":
                every_seconds = schedule.get("every_seconds")
                if isinstance(every_seconds, bool) or not isinstance(every_seconds, (int, float)) or every_seconds <= 0:
                    raise ToolException("kind='interval' requires a positive integer 'every_seconds'.")
                scheduler.schedule_interval_job(
                    job_id=full_job_id,
                    func=run_agent_job,
                    interval_seconds=int(every_seconds),
                    kwargs=job_kwargs,
                    replace=replace,
                )
            elif schedule_kind == "cron":
                cron = " ".join(str(schedule.get(field, "*")) for field in _CRON_FIELD_ORDER)
                scheduler.schedule_cron_job(
                    job_id=full_job_id, func=run_agent_job, cron=cron, kwargs=job_kwargs, replace=replace
                )
            elif schedule_kind == "once":
                scheduler.schedule_once_job(
                    job_id=full_job_id,
                    func=run_agent_job,
                    run_at=cls._parse_once_datetime(raw=schedule.get("at")),
                    kwargs=job_kwargs,
                )
            else:
                raise ToolException("schedule.kind must be one of: 'interval', 'cron', 'once'.")
        except SchedulerJobError as e:
            raise ToolException(str(e)) from e
        except ConflictingIdError as e:
            raise ToolException(
                f"Job id '{full_job_id}' was taken while creating the task. "
                "List the existing tasks and retry with replace=true if the user wants to overwrite."
            ) from e


class ListScheduledTasksTool(ChibiTool):
    register = bool(sys.modules.get("chibi.runners.telegram")) and application_settings.scheduler_tool_enabled
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="list_scheduled_tasks",
            description=(
                "List the scheduled tasks of the current user (title, id, schedule, action type and next run). "
                "Only tasks created under this user's own 'agent:<user_id>:' namespace are visible."
            ),
            parameters={"type": "object", "properties": {}},
        ),
    )
    name = "list_scheduled_tasks"
    allow_model_to_change_background_mode = False

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        """List the scheduled tasks of the current user.

        Args:
            kwargs: Additional options provided to the tool.

        Returns:
            A dictionary with the list of the user's scheduled tasks.

        Raises:
            ToolException: If user_id is missing.
        """
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        scheduler = ChibiScheduler()
        prefix = f"{AGENT_JOB_PREFIX}{user_id}:"
        details: dict[str, dict[str, Any]] = {}
        for job in scheduler.get_jobs():
            job_id = str(job.id)
            if not job_id.startswith(prefix):
                continue
            job_kwargs = job.kwargs or {}
            action = job_kwargs.get("action")
            details[job_id] = {
                "title": job_kwargs.get("title"),
                "action_type": action.get("type") if isinstance(action, dict) else None,
            }

        jobs = [
            {
                "job_id": info.job_id,
                "title": details.get(info.job_id, {}).get("title"),
                "schedule": info.trigger,
                "action_type": details.get(info.job_id, {}).get("action_type"),
                "next_run_time": info.next_run_time.isoformat() if info.next_run_time else None,
            }
            for info in scheduler.list_jobs(prefix=prefix)
        ]
        result: dict[str, Any] = {"status": "ok", "count": len(jobs), "jobs": jobs}
        if not jobs:
            result["message"] = "No scheduled tasks found for this user."
        return result


class DeleteScheduledTaskTool(ChibiTool):
    register = bool(sys.modules.get("chibi.runners.telegram")) and application_settings.scheduler_tool_enabled
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="delete_scheduled_task",
            description=(
                "Delete a scheduled task by its full job id. Only tasks under the current user's own "
                "'agent:<user_id>:' namespace can be deleted; 'system:*' tasks are managed by Chibi itself."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Full id of the task to delete, e.g. 'agent:134604548:daily-report'.",
                    },
                },
                "required": ["job_id"],
            },
        ),
    )
    name = "delete_scheduled_task"
    allow_model_to_change_background_mode = False

    @classmethod
    async def function(cls, job_id: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        """Delete a scheduled task of the current user by its id.

        Args:
            job_id: Full identifier of the job to delete.
            kwargs: Additional options provided to the tool.

        Returns:
            A dictionary with the operation status.

        Raises:
            ToolException: If user_id is missing, the id belongs to another
                namespace (including 'system:*'), or no such job exists.
        """
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        prefix = f"{AGENT_JOB_PREFIX}{user_id}:"
        if not job_id.startswith(prefix):
            raise ToolException(
                f"Access denied: you can only delete jobs under your own prefix '{prefix}'. "
                "'system:*' jobs are managed by Chibi itself."
            )

        try:
            ChibiScheduler().remove_scheduled_job(job_id)
        except JobLookupError as e:
            raise ToolException(f"No scheduled task with id '{job_id}'. Use list_scheduled_tasks to check ids.") from e

        logger.bind(user_id=user_id).info(f"Scheduler: agent job '{job_id}' deleted via the delete tool.")
        return {"status": "ok", "message": f"Scheduled task '{job_id}' deleted."}
