from enum import Enum
from typing import TYPE_CHECKING

from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import BaseModel, Field, model_validator

from chibi.config import telegram_settings

if TYPE_CHECKING:
    pass


class UsageSchema(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class MetricTagsSchema(UsageSchema):
    user_id: int
    user_name: str | None = None
    provider: str
    model: str
    bot: str = telegram_settings.bot_name


class ChatResponseSchema(BaseModel):
    answer: str
    provider: str
    model: str
    usage: UsageSchema | CompletionUsage | None


class ModelChangeSchema(BaseModel):
    provider: str
    name: str
    display_name: str = ""
    image_generation: bool

    @model_validator(mode="after")
    def set_display_name_if_none(self) -> "ModelChangeSchema":
        if not self.display_name:
            self.display_name = self.name
        return self


class ModeratorsAnswer(BaseModel):
    status: str | None = None
    verdict: str
    reason: str | None = None


class SupervisorVerdict(str, Enum):
    """Enumerates possible verdicts returned by the Supervisor.

    The Supervisor decides whether a model action (tool call or final
    response) conforms to its assigned role and established workflow.
    """

    OK = "ok"
    INTERVENE = "intervene"


class SupervisorCategory(str, Enum):
    """Enumerates the categories of role/workflow violations the Supervisor can flag.

    Used to classify the reason for an ``INTERVENE`` verdict.
    """

    ROLE_VIOLATION = "role_violation"
    SCOPE_CREEP = "scope_creep"
    PROTOCOL_SKIP = "protocol_skip"
    CONTEXT_POLLUTION = "context_pollution"
    OUT_OF_BOUNDS_TOOL = "out_of_bounds_tool"


class SupervisorAnswer(BaseModel):
    """Structured response produced by the Supervisor classifier.

    Encodes a single verdict (``OK`` or ``INTERVENE``) with optional
    context. When the verdict is ``INTERVENE``, both ``category`` and
    ``reason`` must be provided; otherwise, validation fails.

    Attributes:
        verdict: The Supervisor's decision on whether to allow the action.
        category: Violation category when ``verdict`` is ``INTERVENE``; ``None`` otherwise.
        reason: Human-readable explanation when ``verdict`` is ``INTERVENE``; ``None`` otherwise.
        status: Free-form status tag (e.g. ``"ok"``, ``"error"``); mirrors ``ModeratorsAnswer`` and
            surfaces fail-open conditions when the Supervisor itself fails.

    Raises:
        ValueError: If ``verdict`` is ``INTERVENE`` while ``category`` is
            ``None``, ``reason`` is ``None``, or ``reason`` is an empty
            string. Per the supervisor concept ("always explains the reason
            for intervention"), an empty string is not a valid explanation.
    """

    verdict: SupervisorVerdict = Field(description="The Supervisor's verdict on the checked action.")
    category: SupervisorCategory | None = Field(
        default=None,
        description="Violation category; required when verdict is 'intervene'.",
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable explanation of the verdict; required when verdict is 'intervene'.",
    )
    status: str | None = Field(
        default=None,
        description='Free-form status tag (e.g. "ok", "error"); used to signal fail-open conditions.',
    )

    @model_validator(mode="after")
    def category_and_reason_required_on_intervene(self) -> "SupervisorAnswer":
        """Validate that ``category`` and ``reason`` are present on ``INTERVENE``.

        When the verdict is ``INTERVENE``, the Supervisor MUST supply both a
        violation ``category`` and a non-empty human-readable ``reason`` —
        per the supervisor concept ("always explains the reason for
        intervention"). An empty string is not a valid explanation. For
        ``OK`` verdicts, the validator is a no-op.

        Returns:
            The validated ``SupervisorAnswer`` instance.

        Raises:
            ValueError: If ``verdict`` is ``INTERVENE`` while ``category`` is
                ``None``, ``reason`` is ``None``, or ``reason`` is an empty
                string.
        """
        if self.verdict == SupervisorVerdict.INTERVENE and (self.category is None or not self.reason):
            raise ValueError("category and reason are required when verdict is 'intervene'")
        return self


class VisionResultSchema(BaseModel):
    short_description: str = Field(description="Image short description, up to 100 characters")
    full_description: str = Field(description="Image full description")
    text: str | None = Field(
        default=None,
        description=(
            "The text extracted from the image, must be filled if image is document and can be "
            "omitted if image does not contain any text"
        ),
    )
