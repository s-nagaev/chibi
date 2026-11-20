"""Standardized handling of command-moderator unavailability.

Any failure of the moderation provider (HTTP errors, connection problems,
empty or unparsable responses) must surface to the calling agent as one
standardized error instead of an uncaught provider exception or a misleading
"declined" verdict. Genuine moderator declines keep their existing behavior:
this module maps failures only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from loguru import logger
from openai import APIStatusError

from chibi.exceptions import ServiceRateLimitError
from chibi.schemas.app import ModeratorsAnswer
from chibi.services.providers.tools.exceptions import ToolException

if TYPE_CHECKING:
    from chibi.services.providers.provider import Provider

MODERATOR_UNAVAILABLE_MESSAGE_TEMPLATE = (
    "Команда не может быть выполнена: модератор недоступен (HTTP {code}, {error_type}). "
    "Если данная ошибка повторяется, прекрати выполнение задачи."
)

MODERATOR_UNAVAILABLE_CODE_FALLBACK = "n/a"
SERVICE_RATE_LIMIT_CODE = "429"
INVALID_MODERATOR_RESPONSE_TYPE = "InvalidModeratorResponse"


class ModeratorUnavailableError(Exception):
    """The command moderator could not produce a verdict.

    Attributes:
        code: HTTP status code when available, otherwise ``n/a``.
        error_type: Exception class name or a descriptive API error type.
        message: The standardized agent-facing message with code and type substituted.
    """

    def __init__(self, code: str, error_type: str) -> None:
        self.code = code
        self.error_type = error_type
        self.message = MODERATOR_UNAVAILABLE_MESSAGE_TEMPLATE.format(code=code, error_type=error_type)
        super().__init__(self.message)


def _raise_unavailable(
    code: str,
    error_type: str,
    moderation_provider: Provider,
    moderator_model: str | None,
    caller_model: str,
) -> NoReturn:
    """Log the moderator failure at the MODERATOR level and raise the standardized error.

    Args:
        code: HTTP status code when available, otherwise the fallback.
        error_type: Exception class name or descriptive API error type.
        moderation_provider: The moderation provider that failed.
        moderator_model: The model the moderation request targeted.
        caller_model: Name of the model whose command was being moderated.

    Raises:
        ModeratorUnavailableError: Always, with the given code and error type.
    """
    logger.log(
        "MODERATOR",
        (
            f"[{caller_model}] Moderator '{moderation_provider.name}' (model {moderator_model}) is unavailable: "
            f"HTTP {code}, {error_type}."
        ),
    )
    raise ModeratorUnavailableError(code=code, error_type=error_type)


async def moderate_command_standardized(
    moderation_provider: Provider,
    cmd: str,
    model: str | None,
    caller_model: str = "unknown model",
) -> ModeratorsAnswer:
    """Run the command moderation, mapping every moderator failure to one standardized error.

    Raised provider exceptions (``APIStatusError``, connection errors,
    ``ServiceRateLimitError``, any other provider exception) and answers with
    ``status == "error"`` (empty or unparsable response) are converted into
    :class:`ModeratorUnavailableError`. Genuine declines pass through unchanged.

    Args:
        moderation_provider: The provider whose ``moderate_command`` is called.
        cmd: The command to moderate.
        model: The moderation model override, forwarded to the provider.
        caller_model: Name of the model whose command is being moderated (for logs).

    Returns:
        The moderator answer when a verdict was produced.

    Raises:
        ModeratorUnavailableError: On any moderator failure, carrying the HTTP
            code (or ``n/a``) and the error type.
    """
    moderator_model = (
        model
        or getattr(moderation_provider, "default_moderation_model", None)
        or getattr(moderation_provider, "default_model", None)
    )
    try:
        answer: ModeratorsAnswer = await moderation_provider.moderate_command(cmd=cmd, model=model)
    except APIStatusError as e:
        _raise_unavailable(
            code=str(e.status_code),
            error_type=type(e).__name__,
            moderation_provider=moderation_provider,
            moderator_model=moderator_model,
            caller_model=caller_model,
        )
    except ServiceRateLimitError as e:
        _raise_unavailable(
            code=SERVICE_RATE_LIMIT_CODE,
            error_type=type(e).__name__,
            moderation_provider=moderation_provider,
            moderator_model=moderator_model,
            caller_model=caller_model,
        )
    except Exception as e:
        _raise_unavailable(
            code=MODERATOR_UNAVAILABLE_CODE_FALLBACK,
            error_type=type(e).__name__,
            moderation_provider=moderation_provider,
            moderator_model=moderator_model,
            caller_model=caller_model,
        )

    if answer.status == "error":
        _raise_unavailable(
            code=MODERATOR_UNAVAILABLE_CODE_FALLBACK,
            error_type=INVALID_MODERATOR_RESPONSE_TYPE,
            moderation_provider=moderation_provider,
            moderator_model=moderator_model,
            caller_model=caller_model,
        )
    return answer


async def moderate_command_for_tool(
    moderation_provider: Provider,
    cmd: str,
    model: str | None,
    caller_model: str = "unknown model",
) -> ModeratorsAnswer:
    """Run the command moderation for a tool call site.

    Same failure mapping as :func:`moderate_command_standardized`, but the
    standardized error is raised as a :class:`ToolException` so the tool
    machinery turns it into an error tool response the agent sees.

    Args:
        moderation_provider: The provider whose ``moderate_command`` is called.
        cmd: The command to moderate.
        model: The moderation model override, forwarded to the provider.
        caller_model: Name of the model whose command is being moderated (for logs).

    Returns:
        The moderator answer when a verdict was produced.

    Raises:
        ToolException: On any moderator failure, with the standardized message.
    """
    try:
        return await moderate_command_standardized(moderation_provider, cmd=cmd, model=model, caller_model=caller_model)
    except ModeratorUnavailableError as e:
        raise ToolException(e.message) from e
