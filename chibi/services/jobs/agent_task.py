"""Stable dispatcher for all agent-created scheduler jobs (design §3.3).

This module is contractually frozen: persistent jobs survive restarts because
APScheduler unpickles a reference to :func:`run_agent_job` from the job store,
so the module path and the function name must never change. The actual
behavior of every job lives in its Pydantic payload stored in the job kwargs
(JSON-serializable).
"""

import asyncio
import os
import signal
import time
from typing import TYPE_CHECKING, Any

from apscheduler.jobstores.base import JobLookupError
from loguru import logger
from pydantic import ValidationError

from chibi.config import application_settings, gpt_settings
from chibi.schemas.scheduler import (
    AgentJobPayload,
    CommandActionPayload,
    NotifyActionPayload,
    SelfActionPayload,
)
from chibi.services.bot import handle_scheduler_trigger
from chibi.services.interface import UserInterface
from chibi.services.providers.tools.cmd import _decode_output
from chibi.services.providers.tools.constants import CMD_STDOUT_LIMIT
from chibi.services.scheduler import AGENT_JOB_PREFIX, ChibiScheduler
from chibi.services.scheduler_interface import SchedulerInterface, StdioSchedulerInterface
from chibi.services.user import get_moderation_provider
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database

if TYPE_CHECKING:
    from apscheduler.job import Job

FAILURE_NOTIFY_COOLDOWN_SECONDS = 3600

_failure_notify_timestamps: dict[str, float] = {}


def _build_interface(payload: AgentJobPayload) -> UserInterface:
    """Build the runner-appropriate delivery interface for a fired job.

    The switch is keyed on ``application_settings.client``: the Telegram
    process keeps delivering through the shared Bot of
    :class:`SchedulerInterface`, while stdio/IDE clients deliver through
    :class:`StdioSchedulerInterface`, which hands messages to the session-level
    emitter registered by the stdio runner at handshake.

    Args:
        payload: Validated payload of the fired job.

    Returns:
        Interface delivering messages to the chat/thread fixed in the payload.
    """
    if application_settings.client == "telegram":
        return SchedulerInterface(
            user_id=payload.user_id,
            storage_id=payload.storage_id,
            chat_id=payload.chat_id,
            thread_id=payload.thread_id,
        )
    return StdioSchedulerInterface(
        user_id=payload.user_id,
        storage_id=payload.storage_id,
        chat_id=payload.chat_id,
        thread_id=payload.thread_id,
    )


def _should_notify_failure(job_id: str) -> bool:
    """Check the per-job anti-flood cooldown and reserve a notification slot.

    Args:
        job_id: Identifier of the failed job.

    Returns:
        True if the failure notification should be sent now, False when a
        notification for this job was already sent within the cooldown window.
    """
    now = time.monotonic()
    last_notification = _failure_notify_timestamps.get(job_id)
    if last_notification is not None and now - last_notification < FAILURE_NOTIFY_COOLDOWN_SECONDS:
        return False
    _failure_notify_timestamps[job_id] = now
    return True


async def _notify_failure(payload: AgentJobPayload, reason: str) -> None:
    """Deliver a single anti-flooded notification about a failed job.

    This is the direct-user-delivery fallback of the dispatcher: it is called
    only when the agent itself cannot learn about what happened — a wake-up
    (a `self` trigger or a command result delivery) failed with a provider/LLM
    error, or a non-command action raised. Failures of the command itself are
    not reported here: the agent is woken with the failure context instead.

    Args:
        payload: Validated payload of the failed job.
        reason: Short human-readable failure reason.
    """
    if not application_settings.scheduler_failure_notify:
        return
    if not _should_notify_failure(payload.job_id):
        logger.bind(user_id=payload.user_id).debug(
            f"Scheduler: failure notification for job '{payload.job_id}' suppressed by the anti-flood cooldown."
        )
        return
    message = f"⚠️ Scheduled job '{payload.title}' ({payload.job_id}) failed: {reason}"
    try:
        await _build_interface(payload).send_message(message=message)
    except Exception as e:
        logger.bind(user_id=payload.user_id).error(
            f"Scheduler: failed to deliver the failure notification for job '{payload.job_id}': {e!r}"
        )


async def _run_self_action(payload: AgentJobPayload, action: SelfActionPayload) -> None:
    """Wake the agent with the trigger text from the job payload.

    Args:
        payload: Validated payload with a `self` action.
        action: The narrowed self action from the payload.
    """
    await handle_scheduler_trigger(
        trigger_text=action.trigger_text,
        job_id=payload.job_id,
        interface=_build_interface(payload),
    )


async def _run_notify_action(payload: AgentJobPayload, action: NotifyActionPayload) -> None:
    """Deliver the static notification message fixed in the job payload.

    Args:
        payload: Validated payload with a `notify` action.
        action: The narrowed notify action from the payload.
    """
    if not application_settings.scheduler_notify_enabled:
        logger.bind(user_id=payload.user_id).warning(
            f"Scheduler: notify action of job '{payload.job_id}' skipped — "
            f"scheduler_notify_enabled is disabled. Job is kept."
        )
        return
    await _build_interface(payload).send_message(message=action.message)


def _truncate_output(text: str, limit: int = CMD_STDOUT_LIMIT) -> str:
    """Truncate command output for delivery.

    Args:
        text: Raw decoded output.
        limit: Maximum number of characters to keep.

    Returns:
        The original text when short enough, otherwise its head with a
        truncation marker.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...truncated ({len(text)} characters total)..."


def _build_command_result_trigger(
    payload: AgentJobPayload,
    action: CommandActionPayload,
    *,
    exit_code: int | None = None,
    stdout_text: str = "",
    stderr_text: str = "",
    error_note: str | None = None,
) -> str:
    """Compose the structured trigger text carrying a command result to the agent.

    Args:
        payload: Validated payload of the fired job.
        action: The narrowed command action from the payload.
        exit_code: Process exit code, or None when the command never ran.
        stdout_text: Decoded stdout (truncated for delivery).
        stderr_text: Decoded stderr (truncated for delivery).
        error_note: Extra failure context (timeout or spawn error description).

    Returns:
        Structured trigger text for the scheduler wake-up.
    """
    lines = [
        f"Scheduled command job '{payload.title}' ({payload.job_id}) result.",
        f"Command: `{action.command}`",
    ]
    if error_note is not None:
        lines.append(error_note)
    if exit_code is not None:
        lines.append(f"Exit code: {exit_code}")
    lines.append(f"Output:\n{_truncate_output(stdout_text) if stdout_text.strip() else '(empty)'}")
    if stderr_text.strip():
        lines.append(f"Stderr:\n{_truncate_output(stderr_text)}")
    return "\n".join(lines)


async def _wake_agent_with_command_result(
    payload: AgentJobPayload, trigger_text: str, interface: UserInterface
) -> None:
    """Wake the agent with a command result, falling back to a direct user notification.

    The wake goes through the normal self-wake pipeline
    (:func:`handle_scheduler_trigger`): the agent sees the result context and
    decides itself what — if anything — to tell the user. A failure of the wake
    step itself (LLM/provider error) means the agent never learned about the
    result, so the user is notified directly through the interface, guarded by
    the anti-flood cooldown. Failures of the command itself never reach this
    fallback: they arrive as part of the trigger text.

    Args:
        payload: Validated payload of the fired job.
        trigger_text: Structured command-result context for the agent.
        interface: Interface bound to the job's chat/thread context.
    """
    try:
        await handle_scheduler_trigger(trigger_text=trigger_text, job_id=payload.job_id, interface=interface)
    except Exception as e:
        logger.bind(user_id=payload.user_id).exception(
            f"Scheduler: agent wake for command job '{payload.job_id}' failed: {type(e).__name__}: {e}"
        )
        await _notify_failure(
            payload, reason=f"agent wake failed ({type(e).__name__}: {e}); command result not delivered"
        )


async def _run_command_action(payload: AgentJobPayload, action: CommandActionPayload) -> None:
    """Run the pre-moderated shell command of a `command` action and wake the agent.

    The command is re-moderated before every run (design §6.4); on a declined
    verdict the run is skipped and the job is kept. Execution follows the
    ``cmd.py`` pattern: ``create_subprocess_shell`` with a fresh process group
    and ``killpg`` on timeout. The agent is ALWAYS woken with the result —
    success or failure, empty or non-empty output — through the self-wake
    pipeline; it owns communication with the user. Only a failed wake (LLM
    error) falls back to a direct, anti-flooded user notification.

    Args:
        payload: Validated payload with a `command` action.
        action: The narrowed command action from the payload.
    """
    if not application_settings.scheduler_agent_commands_enabled:
        logger.bind(user_id=payload.user_id).warning(
            f"Scheduler: command action of job '{payload.job_id}' skipped — "
            f"scheduler_agent_commands_enabled is disabled. Job is kept."
        )
        return

    moderation_provider = await get_moderation_provider(user_id=payload.user_id)
    moderator_answer = await moderation_provider.moderate_command(
        cmd=action.command, model=gpt_settings.moderation_model
    )
    if moderator_answer.verdict == "declined":
        logger.bind(user_id=payload.user_id).warning(
            f"Scheduler: moderator declined the command of job '{payload.job_id}' "
            f"({moderator_answer.reason}). Run skipped, job is kept."
        )
        return

    timeout = min(action.timeout_seconds, application_settings.scheduler_command_timeout_max)
    interface = _build_interface(payload)
    try:
        process = await asyncio.create_subprocess_shell(
            cmd=action.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=action.cwd,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=float(timeout))
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
            logger.bind(user_id=payload.user_id).warning(
                f"Scheduler: command of job '{payload.job_id}' timed out after {timeout}s, process group killed."
            )
            await _wake_agent_with_command_result(
                payload,
                _build_command_result_trigger(
                    payload,
                    action,
                    error_note=f"The command was killed after {timeout}s (the process group was killed).",
                ),
                interface,
            )
            return
    except OSError as e:
        await _wake_agent_with_command_result(
            payload,
            _build_command_result_trigger(
                payload, action, error_note=f"The command could not be started: {type(e).__name__}: {e}"
            ),
            interface,
        )
        return

    stdout_text = _decode_output(stdout)
    stderr_text = _decode_output(stderr)
    logger.bind(user_id=payload.user_id).info(
        f"Scheduler: command of job '{payload.job_id}' finished with exit code {process.returncode} "
        f"— waking the agent with the result."
    )
    await _wake_agent_with_command_result(
        payload,
        _build_command_result_trigger(
            payload,
            action,
            exit_code=process.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        ),
        interface,
    )


async def _dispatch(payload: AgentJobPayload) -> None:
    """Route a validated payload to the handler of its action type.

    Args:
        payload: Validated payload of the fired job.

    Raises:
        ValueError: If the action type is unknown (defensive, the Pydantic
            discriminated union makes it unreachable).
    """
    if isinstance(payload.action, SelfActionPayload):
        await _run_self_action(payload, payload.action)
    elif isinstance(payload.action, NotifyActionPayload):
        await _run_notify_action(payload, payload.action)
    elif isinstance(payload.action, CommandActionPayload):
        await _run_command_action(payload, payload.action)
    else:
        raise ValueError(f"Unknown agent job action type: {payload.action!r}")


async def run_agent_job(job_id: str, **kwargs: Any) -> None:
    """Single stable job function for all agent-created scheduler jobs.

    Scheduled through ``ChibiScheduler.schedule_*_job`` with the payload fields
    passed as flat JSON-serializable kwargs (see :class:`AgentJobPayload`).
    The work is awaited directly — ``task_manager.run_task`` is intentionally
    not used, so APScheduler's ``max_instances``/coalesce guarantees hold.

    Args:
        job_id: Fully qualified job identifier.
        **kwargs: Payload fields (``user_id``, ``storage_id``, ``chat_id``,
            ``thread_id``, ``title``, ``action``).
    """
    try:
        payload = AgentJobPayload.model_validate({"job_id": job_id, **kwargs})
    except ValidationError as e:
        logger.error(f"Scheduler: agent job '{job_id}' has an invalid payload, run skipped: {e}")
        return None

    log = logger.bind(user_id=payload.user_id)
    log.info(f"Scheduler: agent job '{payload.job_id}' ('{payload.title}') fired (action: {payload.action.type})")
    try:
        await _dispatch(payload)
    except Exception as e:
        log.exception(f"Scheduler: agent job '{payload.job_id}' failed: {type(e).__name__}: {e}")
        await _notify_failure(payload, reason=f"{type(e).__name__}: {e}")
    return None


def _remove_job(scheduler: ChibiScheduler, job_id: str) -> None:
    """Remove a job ignoring the case when it is already gone.

    Args:
        scheduler: Scheduler instance to remove the job from.
        job_id: Identifier of the job to remove.
    """
    try:
        scheduler.remove_scheduled_job(job_id)
    except JobLookupError:
        logger.debug(f"Scheduler: job '{job_id}' already removed.")


async def recover_agent_jobs(scheduler: ChibiScheduler) -> None:
    """Startup recovery pass over the persistent job store (design §3.3/§3.4).

    Removes broken agent jobs (invalid payload or unexpected job function) and
    orphans (agent jobs of users that no longer exist in the storage). System
    jobs are left untouched. Never raises: a broken job store must not prevent
    Chibi from starting.

    Args:
        scheduler: Started scheduler instance to inspect and clean up.
    """
    try:
        jobs = scheduler.get_jobs()
    except Exception as e:
        logger.exception(f"Scheduler: failed to load jobs from the job store during recovery: {e!r}")
        return None

    broken_job_ids: list[str] = []
    agent_user_ids: set[int] = set()
    for job in jobs:
        job_id = str(job.id)
        if not job_id.startswith(AGENT_JOB_PREFIX):
            continue
        if job.func is not run_agent_job:
            logger.error(f"Scheduler: removing unknown agent job '{job_id}' (unexpected job function {job.func!r}).")
            broken_job_ids.append(job_id)
            continue
        try:
            payload = AgentJobPayload.model_validate({"job_id": job_id, **(job.kwargs or {})})
        except ValidationError as e:
            logger.error(f"Scheduler: removing broken agent job '{job_id}' (invalid payload): {e}")
            broken_job_ids.append(job_id)
            continue
        agent_user_ids.add(payload.user_id)

    for job_id in broken_job_ids:
        _remove_job(scheduler, job_id)
    if broken_job_ids:
        logger.info(f"Scheduler: recovery removed {len(broken_job_ids)} broken agent job(s).")

    try:
        await _remove_orphan_agent_jobs(scheduler=scheduler, jobs=jobs, agent_user_ids=agent_user_ids)
    except Exception as e:
        logger.exception(f"Scheduler: orphan cleanup failed, agent jobs left as is: {e!r}")
    return None


@inject_database
async def _remove_orphan_agent_jobs(
    db: Database,
    scheduler: ChibiScheduler,
    jobs: list["Job"],
    agent_user_ids: set[int],
) -> None:
    """Remove agent jobs belonging to users missing from the storage.

    The storage backend is the source of truth for existing users:
    ``db.get_user(user_id)`` returning None means the user (and all their
    `agent:{user_id}:*` jobs) is gone.

    Args:
        db: Database instance (injected).
        scheduler: Scheduler instance to remove orphaned jobs from.
        jobs: Jobs loaded from the job store.
        agent_user_ids: User ids referenced by the surviving agent jobs.
    """
    existing_user_ids: set[int] = set()
    for user_id in agent_user_ids:
        if await db.get_user(user_id=user_id) is not None:
            existing_user_ids.add(user_id)

    orphan_prefixes = {f"{AGENT_JOB_PREFIX}{user_id}:" for user_id in agent_user_ids - existing_user_ids}
    removed = 0
    for job in jobs:
        job_id = str(job.id)
        if any(job_id.startswith(prefix) for prefix in orphan_prefixes):
            logger.warning(f"Scheduler: removing orphaned agent job '{job_id}' (user no longer exists).")
            _remove_job(scheduler, job_id)
            removed += 1
    if removed:
        logger.info(f"Scheduler: orphan cleanup removed {removed} agent job(s) of deleted users.")
