from uuid import UUID

from loguru import logger
from telegram import (
    CallbackQuery,
)

from chibi.config import application_settings, gpt_settings
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.interface import UserInterface
from chibi.services.providers import RegisteredProviders
from chibi.services.providers.tools import ToolResponse
from chibi.services.providers.utils import get_usage_msg
from chibi.services.user import (
    check_history_and_summarize,
    generate_image,
    get_llm_chat_completion_answer,
    get_models_available,
    reset_chat_history,
    send_scheduled_message_to_llm,
    set_active_model,
    set_api_key,
    user_has_reached_images_generation_limit,
)
from chibi.utils.app import handle_gpt_exceptions
from chibi.utils.bot import indicator


@handle_gpt_exceptions
async def handle_model_selection(
    interface: UserInterface,
    model: ModelChangeSchema,
    query: CallbackQuery,
) -> None:
    await set_active_model(user_id=interface.user_id, model=model)
    logger.info(f"{interface.user_data} switched to model '{model.name} ({model.provider})'")
    await query.edit_message_text(text=f"Selected model: '{model.name} ({model.provider})'")


async def handle_tool_response(tool_response: ToolResponse, interface: UserInterface) -> None:
    chat_response: ChatResponseSchema = await get_llm_chat_completion_answer(
        user_id=interface.user_id, tool_message=tool_response, interface=interface
    )
    usage_message = get_usage_msg(chat_response.usage)

    if "<chibi>ack</chibi>" in chat_response.answer.lower():
        logger.info(
            f"[{interface.user_data}-{interface.chat_data}] LLM silently received tool result "
            f"(answer: {chat_response.answer}). No user notification required. {usage_message}"
        )
        return None

    if application_settings.log_prompt_data:
        answer_to_log = chat_response.answer.replace("\r", " ").replace("\n", " ")
        logged_answer = f"Answer: {answer_to_log}"
    else:
        logged_answer = ""

    logger.info(
        f"{interface.user_data} got {chat_response.provider} ({chat_response.model}) answer in "
        f"the {interface.chat_data}. {logged_answer} {usage_message}"
    )
    await interface.send_message(message=chat_response.answer)


async def handle_scheduled_event(
    message: str,
    event_id: UUID,
    interface: UserInterface,
) -> None:
    chat_response: ChatResponseSchema = await send_scheduled_message_to_llm(
        user_id=interface.user_id, event_id=event_id, message=message, interface=interface
    )
    usage_message = get_usage_msg(chat_response.usage)

    if "<chibi>ack</chibi>" in chat_response.answer.lower():
        logger.info(
            f"[{interface.user_data}-{interface.chat_data}] LLM silently received scheduled message "
            f"(answer: {chat_response.answer}). No user notification required. {usage_message}"
        )
        return None

    if application_settings.log_prompt_data:
        answer_to_log = chat_response.answer.replace("\r", " ").replace("\n", " ")
        logged_answer = f"Answer: {answer_to_log}"
    else:
        logged_answer = ""

    logger.info(
        f"{interface.user_data} got {chat_response.provider} ({chat_response.model}) answer in "
        f"the {interface.chat_data}. {logged_answer} {usage_message}"
    )
    await interface.send_message(message=chat_response.answer)


@handle_gpt_exceptions
async def handle_user_prompt(interface: UserInterface) -> None:
    text_prompt = await interface.get_text_prompt()
    voice_prompt = await interface.get_voice_prompt()

    if text_prompt:
        if text_prompt.startswith("/ask"):
            text_prompt = text_prompt.replace("/ask", "", 1).strip()

    prompt_to_log = text_prompt.replace("\r", " ").replace("\n", " ") if text_prompt else "voice message"

    logger.info(
        f"{interface.user_data} sent a new message in the {interface.user_data}"
        f"{': ' + prompt_to_log if application_settings.log_prompt_data else ''}"
    )

    async with indicator(coro=interface.send_action_typing()):
        chat_response: ChatResponseSchema = await get_llm_chat_completion_answer(
            user_id=interface.user_id,
            user_text_message=text_prompt,
            user_voice_message=voice_prompt,
            interface=interface,
        )

    usage = chat_response.usage
    usage_message = get_usage_msg(usage)

    if application_settings.log_prompt_data:
        answer_to_log = chat_response.answer.replace("\r", " ").replace("\n", " ")
        logged_answer = f"Answer: {answer_to_log}"
    else:
        logged_answer = ""

    if "<chibi>ack</chibi>" in chat_response.answer.lower():
        logger.info(
            f"[{interface.user_data}-{interface.chat_data}] LLM silently received user request "
            f"(answer: {chat_response.answer}). No user notification required. {usage_message}"
        )
        try:
            await interface.send_reaction(reaction="👌")
        except Exception as e:
            logger.error(f"{interface.user_data}: Couldn't set message reaction due to exception: {e}")
        return None

    logger.info(
        f"{interface.user_data} got {chat_response.provider} ({chat_response.model}) answer in "
        f"the {interface.chat_data}. {logged_answer} {usage_message}"
    )
    await interface.send_message(message=chat_response.answer)
    history_is_summarized = await check_history_and_summarize(user_id=interface.user_id)
    if history_is_summarized:
        logger.info(f"{interface.user_data}: history successfully summarized.")


async def handle_reset(interface: UserInterface) -> None:
    logger.info(f"{interface.user_data}: conversation history reset.")

    await reset_chat_history(user_id=interface.user_id)
    await interface.send_message(message="Done!", reply=False)


@handle_gpt_exceptions
async def handle_image_generation(prompt: str, interface: UserInterface) -> None:
    if await user_has_reached_images_generation_limit(user_id=interface.user_id):
        await interface.send_message(
            message=(
                f"Sorry, you have reached your monthly images generation limit "
                f"({gpt_settings.image_generations_monthly_limit}). Please, try again later."
            )
        )
        return None

    logger.info(
        f"{interface.user_data} sent image generation request in the {interface.chat_data}"
        f"{': ' + prompt if application_settings.log_prompt_data else ''}"
    )

    # The user finds it psychologically easier to wait for a response from the chatbot when they see its activity
    # during the entire waiting time.
    async with indicator(coro=interface.send_action_uploading_photo()):
        image_data = await generate_image(user_id=interface.user_id, prompt=prompt)
        await interface.send_images(images=image_data)

    log_message = f"{interface.user_data} got a successfully generated image(s)"
    if application_settings.log_prompt_data and isinstance(image_data[0], str):
        log_message += f": {image_data}"
    logger.info(log_message)


async def handle_provider_api_key_set(provider_name: str, interface: UserInterface) -> None:
    logger.info(f"{interface.user_data} provides API Key for provider '{provider_name}'.")

    api_key = await interface.get_text_prompt()
    if not api_key:
        return None

    api_key = api_key.strip()

    provider = RegisteredProviders.get_class(provider_name)
    if not provider:
        return None

    if not await provider(token=api_key).api_key_is_valid():
        await interface.send_message(message="Sorry, but API key you have provided does not seem correct.")
        logger.warning(f"{interface.user_data} provided invalid key.")
        return None

    await set_api_key(user_id=interface.user_id, api_key=api_key, provider_name=provider_name)
    RegisteredProviders.register_as_available(provider=provider)

    await interface.send_message(
        message=(
            f"Your {provider_name} API Key successfully set! 🦾\n\nNow you may check available models in /gpt_models."
        )
    )
    await interface.delete_last_user_message()
    logger.info(f"{interface.user_data} successfully set up {provider_name} Key.")


@handle_gpt_exceptions
async def handle_available_model_options(
    user_id: int,
    interface: UserInterface,
    image_generation: bool = False,
) -> list[ModelChangeSchema]:
    return await get_models_available(user_id=user_id, image_generation=image_generation)
