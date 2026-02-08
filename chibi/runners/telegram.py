import sys
from contextvars import Context
from typing import Any, Coroutine, TypeVar

import click
from loguru import logger
from telegram import (
    BotCommand,
    CallbackQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
    constants,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

from chibi.config import application_settings, gpt_settings, telegram_settings
from chibi.constants import GROUP_CHAT_TYPES, UserAction, UserContext
from chibi.schemas.app import ModelChangeSchema
from chibi.services.bot import (
    handle_available_model_options,
    handle_available_provider_options,
    handle_image_generation,
    handle_model_selection,
    handle_provider_api_key_set,
    handle_reset,
    handle_user_prompt,
)
from chibi.services.providers import RegisteredProviders
from chibi.services.task_manager import task_manager
from chibi.utils.app import log_application_settings, run_heartbeat
from chibi.utils.telegram import (
    check_user_allowance,
    current_user_action,
    get_telegram_chat,
    get_telegram_message,
    get_user_context,
    set_user_action,
    set_user_context,
    user_interacts_with_bot,
)

_T = TypeVar("_T")


class ChibiBot:
    def __init__(self, telegram_token: str) -> None:
        self.telegram_token = telegram_token
        self.commands = [
            BotCommand(command="help", description="Show this help message"),
            BotCommand(
                command="ask",
                description=(
                    "Ask me any question (in group chat, for example) (e.g. /ask which program language is the best?)"
                ),
            ),
            BotCommand(
                command="reset",
                description="Reset your conversation history (will reduce prompt and save some tokens)",
            ),
        ]
        if not application_settings.hide_imagine:
            self.commands.append(
                BotCommand(command="imagine", description="Generate image from prompt"),
            )
            self.commands.append(BotCommand(command="image_models", description="Select image generation model"))
        if not application_settings.hide_models:
            self.commands.append(BotCommand(command="gpt_models", description="Select GPT model"))

        if gpt_settings.public_mode:
            self.commands.append(
                BotCommand(
                    command="set_api_key",
                    description="Set an API key (token) for any of supported providers",
                )
            )

    def run_task(
        self,
        coro: Coroutine[Any, Any, _T],
        name: str | None = None,
        context: Context | None = None,
    ) -> None:
        task_manager.run_task(coro)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        commands = [f"/{command.command} - {command.description}" for command in self.commands]
        commands_desc = "\n".join(commands)
        help_text = (
            f"Hey! My name is {telegram_settings.bot_name}, and I'm your ChatGPT experience provider!\n\n"
            f"{commands_desc}"
        )
        await telegram_message.reply_text(help_text, disable_web_page_preview=True)

    @check_user_allowance
    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.run_task(handle_reset(update=update, context=context))

    async def _handle_message_with_api_key(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        provider_name = get_user_context(context=context, key=UserContext.SELECTED_PROVIDER, expected_type=str)
        if not provider_name:
            return None
        self.run_task(handle_provider_api_key_set(update=update, context=context, provider_name=provider_name))
        return None

    @check_user_allowance
    async def imagine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        assert telegram_message.text
        prompt = telegram_message.text.replace("/imagine", "", 1).strip()
        if prompt:
            self.run_task(handle_image_generation(update=update, context=context, prompt=prompt))
            return None
        set_user_action(context=context, action=UserAction.IMAGINE)
        await telegram_message.reply_text("Ok, now give me an image prompt.")

    @check_user_allowance
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_chat = get_telegram_chat(update=update)
        telegram_message = get_telegram_message(update=update)
        if telegram_message.voice:
            self.run_task(handle_user_prompt(update=update, context=context))
            return None

        prompt = telegram_message.text

        if not prompt:
            return None

        if current_user_action(context=context) == UserAction.SET_API_KEY:
            set_user_action(context=context, action=UserAction.NONE)
            return await self._handle_message_with_api_key(update=update, context=context)

        if current_user_action(context=context) == UserAction.IMAGINE:
            self.run_task(handle_image_generation(update=update, context=context, prompt=prompt))
            return None

        if (
            telegram_chat.type in GROUP_CHAT_TYPES
            and telegram_settings.answer_direct_messages_only
            and "/ask" not in prompt
            and not user_interacts_with_bot(update=update, context=context)
        ):
            return None
        self.run_task(handle_user_prompt(update=update, context=context))

    @check_user_allowance
    async def ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.run_task(handle_user_prompt(update=update, context=context))

    @check_user_allowance
    async def show_gpt_models_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        reply_markup = await handle_available_model_options(update=update, context=context)

        if active_model := get_user_context(context=context, key=UserContext.ACTIVE_MODEL, expected_type=str):
            message = f"Active model: {active_model}. You  may select another one from the list below:"
        else:
            message = "Please, select model:"
        set_user_action(context=context, action=UserAction.SELECT_MODEL)
        await telegram_message.reply_text(text=message, reply_markup=reply_markup)

    @check_user_allowance
    async def show_image_models_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        reply_markup = await handle_available_model_options(update=update, context=context, image_generation=True)

        if active_model := get_user_context(context=context, key=UserContext.ACTIVE_IMAGE_MODEL, expected_type=str):
            message = f"Active model: {active_model}. You  may select another one from the list below:"
        else:
            message = "Please, select model:"
        set_user_action(context=context, action=UserAction.SELECT_MODEL)
        await telegram_message.reply_text(text=message, reply_markup=reply_markup)

    async def show_api_key_set_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        reply_markup = await handle_available_provider_options()

        message = "Please, select a provider:"
        await telegram_message.reply_text(text=message, reply_markup=reply_markup)
        set_user_action(context=context, action=UserAction.SELECT_PROVIDER)

    async def _compute_model_selection_action(
        self, query: CallbackQuery, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        mapped_models = get_user_context(
            context=context,
            key=UserContext.MAPPED_MODELS,
            expected_type=dict[str, ModelChangeSchema],
        )
        await query.answer()

        if not mapped_models or not query.data:
            await query.delete_message()
            return None

        if query.data == "-1":
            await query.delete_message()
            return None

        model = mapped_models.get(query.data)
        if not model:
            await query.delete_message()
            return None

        if model.image_generation:
            set_user_context(context=context, key=UserContext.ACTIVE_IMAGE_MODEL, value=model.name)
        else:
            set_user_context(context=context, key=UserContext.ACTIVE_MODEL, value=model.name)
        self.run_task(
            handle_model_selection(
                update=update,
                context=context,
                model=model,
                query=query,
            )
        )
        set_user_action(context=context, action=UserAction.NONE)

    async def _compute_provider_selection_action(
        self, query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await query.answer()
        provider_name = query.data
        if not provider_name or provider_name not in RegisteredProviders.all.keys():
            await query.delete_message()
            return
        set_user_context(context=context, key=UserContext.SELECTED_PROVIDER, value=provider_name)
        await query.edit_message_text(
            text=f"{provider_name} selected.\nNow please send me an API key",
        )
        set_user_action(context=context, action=UserAction.SET_API_KEY)

    async def handle_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        action = current_user_action(context=context)
        if not action or action == UserAction.NONE:
            return None

        query = update.callback_query
        if not query:
            return None

        if action == UserAction.SELECT_MODEL:
            return await self._compute_model_selection_action(query=query, update=update, context=context)

        if action == UserAction.SELECT_PROVIDER:
            return await self._compute_provider_selection_action(query=query, context=context)

    @check_user_allowance
    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        inline_query = update.inline_query
        if not inline_query:
            return
        query = inline_query.query
        results = [
            InlineQueryResultArticle(
                id=query,
                title="Ask ChatGPT",
                input_message_content=InputTextMessageContent(query),
                description=query,
                thumbnail_url="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png",
            )
        ]

        await inline_query.answer(results)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f"Error occurred while handling an update: {context.error}")

    async def post_init(self, application: Application) -> None:
        await application.bot.set_my_commands(self.commands)

    def run(self) -> None:
        builder = (
            ApplicationBuilder()
            .base_url(telegram_settings.telegram_base_url)
            .base_file_url(telegram_settings.telegram_base_file_url)
            .token(self.telegram_token)
            .post_init(self.post_init)
            .post_shutdown(task_manager.shutdown)
        )

        if telegram_settings.proxy:
            builder = builder.proxy(telegram_settings.proxy).get_updates_proxy(telegram_settings.proxy)
        app = builder.build()

        if not application_settings.hide_imagine:
            app.add_handler(CommandHandler(command="imagine", callback=self.imagine))

        if not application_settings.hide_models:
            app.add_handler(CommandHandler("gpt_models", self.show_gpt_models_menu))
            app.add_handler(CommandHandler("image_models", self.show_image_models_menu))
        app.add_handler(CallbackQueryHandler(self.handle_selection))

        if gpt_settings.public_mode:
            app.add_handler(CommandHandler("set_api_key", self.show_api_key_set_menu))

        app.add_handler(CommandHandler("ask", self.ask))
        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("reset", self.reset))
        app.add_handler(CommandHandler("start", self.help))

        app.add_handler(MessageHandler(filters.TEXT | filters.VOICE | filters.AUDIO & (~filters.COMMAND), self.prompt))

        app.add_handler(
            InlineQueryHandler(
                self.inline_query,
                chat_types=[
                    constants.ChatType.PRIVATE,
                    constants.ChatType.GROUP,
                    constants.ChatType.SUPERGROUP,
                ],
            )
        )
        # app.add_error_handler(self.error_handler)
        if application_settings.heartbeat_url:
            if not app.job_queue:
                logger.error("Could not launch heartbeat beacon: application job queue was shut down or never started.")
            else:
                url = application_settings.heartbeat_url
                logger.info(
                    f"Launching heartbeat beacon: calling {url[:30]}..{url[-3:]} "
                    f"every {application_settings.heartbeat_frequency_call} seconds."
                )
                app.job_queue.run_repeating(
                    callback=run_heartbeat,
                    interval=application_settings.heartbeat_frequency_call,
                    first=0.0,
                )
        app.run_polling()


def run_chibi():
    if not telegram_settings.token:
        click.echo()
        click.secho(" CONFIGURATION ERROR ".center(60, "="), fg="red", bold=True)
        click.echo("Telegram token not set.\nIf you're using Chibi installed via pip, please set it using")
        click.secho("$ chibi config", fg="green", bold=True)
        click.echo()

        click.echo(
            "Otherwise,  please check the config file manually  or ensure\n"
            f"that you've exported {click.style('TELEGRAM_BOT_TOKEN', bold=True, fg='yellow')} environment variable",
        )
        click.secho("=" * 60, fg="red", bold=True)
        click.echo()
        sys.exit(1)

    log_application_settings()
    telegram_bot = ChibiBot(telegram_settings.token)
    telegram_bot.run()
