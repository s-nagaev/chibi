import json
from typing import TypeVar

from loguru import logger
from telegram import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    PhotoSize,
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
    handle_image_generation,
    handle_image_understanding,
    handle_model_selection,
    handle_provider_api_key_set,
    handle_reset,
    handle_stop,
    handle_user_prompt,
)
from chibi.services.interface import TelegramInterface
from chibi.services.providers import RegisteredProviders
from chibi.services.task_manager import task_manager
from chibi.storage.files.telegram_storage import TelegramFileStorage
from chibi.utils.app import log_application_settings, run_heartbeat
from chibi.utils.telegram import (
    check_user_allowance,
    current_user_action,
    get_telegram_chat,
    get_telegram_message,
    get_telegram_user,
    get_user_context,
    set_user_action,
    set_user_context,
    telegram_security_pre_start_check,
    telegram_setting_pre_start_check,
    user_interacts_with_bot,
)

_T = TypeVar("_T")


class ChibiBot:
    def __init__(self, telegram_token: str | None) -> None:
        if not telegram_token:
            raise RuntimeError("No telegram token provider")
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
                description="Stop LLM and reset your conversation history (will reduce prompt and save some tokens)",
            ),
            BotCommand(
                command="stop",
                description="Stop LLM and all the processes it runs.",
            ),
        ]
        if not application_settings.hide_imagine:
            self.commands.append(
                BotCommand(command="imagine", description="Generate image from prompt"),
            )
            self.commands.append(BotCommand(command="image_models", description="Select image generation model"))
        if not application_settings.hide_models:
            self.commands.append(BotCommand(command="llm_models", description="Select LLM"))

        if gpt_settings.public_mode:
            self.commands.append(
                BotCommand(
                    command="set_api_key",
                    description="Set an API key (token) for any of supported providers",
                )
            )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        commands = [f"/{command.command} - {command.description}" for command in self.commands]
        commands_desc = "\n".join(commands)
        help_text = f"Hey! My name is {telegram_settings.bot_name}, and I am your digital partner!\n\n{commands_desc}"
        await telegram_message.reply_text(help_text, disable_web_page_preview=True)

    @check_user_allowance
    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        interface = TelegramInterface(update=update, context=context)
        task_manager.run_task(
            coro=handle_reset(interface=interface),
            user_id=-1,
        )
        return None

    @check_user_allowance
    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        interface = TelegramInterface(update=update, context=context)
        task_manager.run_task(
            coro=handle_stop(interface=interface),
            user_id=-1,
        )
        return None

    async def _handle_message_with_api_key(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        provider_name = get_user_context(context=context, key=UserContext.SELECTED_PROVIDER, expected_type=str)
        if not provider_name:
            return None
        interface = TelegramInterface(update=update, context=context)
        task_manager.run_task(
            coro=handle_provider_api_key_set(provider_name=provider_name, interface=interface),
            user_id=interface.user_id,
        )
        return None

    @check_user_allowance
    async def imagine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        assert telegram_message.text
        prompt = telegram_message.text.replace("/imagine", "", 1).strip()
        if prompt:
            set_user_action(context=context, action=UserAction.NONE)
            interface = TelegramInterface(update=update, context=context)
            task_manager.run_task(
                coro=handle_image_generation(prompt=prompt, interface=interface),
                user_id=interface.user_id,
            )
            return None

        set_user_action(context=context, action=UserAction.IMAGINE)
        await telegram_message.reply_text("Ok, now give me an image prompt.")
        return None

    @check_user_allowance
    async def file_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message:
            return None

        interface = TelegramInterface(update=update, context=context)
        storage = TelegramFileStorage(interface=interface)

        if document_meta := message.document:
            file_id = await storage.save(file_metadata=document_meta.to_dict())
            logger.info(
                f"{interface.user_data}-{interface.chat_data}: File '{document_meta.file_name}' successfully uploaded."
            )
            caption = {
                "user_caption": message.caption or "no data",
                "file_id": file_id,
            }
            interface.set_caption(json.dumps(caption))

        if photo_variants := message.photo:
            photo_meta: PhotoSize = photo_variants[-1]
            photo_meta_dict = photo_meta.to_dict()
            file_name = f"{photo_meta.file_unique_id}.jpeg"
            photo_meta_dict["file_name"] = file_name
            photo_meta_dict["mime_type"] = "image/jpeg"
            file_id = await storage.save(file_metadata=photo_meta_dict)
            if vision_result := await handle_image_understanding(
                interface=interface,
                storage=storage,
                file_id=file_id,
                mime_type="image/jpeg",
            ):
                photo_meta_dict["full_description"] = vision_result.full_description
                photo_meta_dict["short_description"] = vision_result.short_description
                photo_meta_dict["text"] = vision_result.text
                await storage.save(file_metadata=photo_meta_dict)

                caption = {
                    "user_caption": message.caption or "no data",
                    "photo_short_desc": vision_result.short_description,
                    "file_id": file_id,
                }
                interface.set_caption(json.dumps(caption))

            logger.info(f"{interface.user_data}-{interface.chat_data}: Photo '{file_name}' successfully uploaded.")

        if await interface.get_caption():
            task_manager.run_task(
                coro=handle_user_prompt(interface=interface),
                user_id=interface.user_id,
            )
        return None

    @check_user_allowance
    async def document_uploaded(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: ...

    @check_user_allowance
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_chat = get_telegram_chat(update=update)
        telegram_message = get_telegram_message(update=update)
        interface = TelegramInterface(update=update, context=context)

        if telegram_message.voice:
            task_manager.run_task(
                coro=handle_user_prompt(interface=interface),
                user_id=interface.user_id,
            )
            return None

        prompt = telegram_message.text

        if not prompt:
            return None

        if current_user_action(context=context) == UserAction.SET_API_KEY:
            set_user_action(context=context, action=UserAction.NONE)
            return await self._handle_message_with_api_key(update=update, context=context)

        if current_user_action(context=context) == UserAction.IMAGINE:
            set_user_action(context=context, action=UserAction.NONE)
            task_manager.run_task(
                coro=handle_image_generation(prompt=prompt, interface=interface),
                user_id=interface.user_id,
            )
            return None

        if (
            telegram_chat.type in GROUP_CHAT_TYPES
            and telegram_settings.answer_direct_messages_only
            and "/ask" not in prompt
            and not user_interacts_with_bot(update=update, context=context)
        ):
            return None

        task_manager.run_task(
            coro=handle_user_prompt(interface=interface),
            user_id=interface.user_id,
        )
        return None

    @check_user_allowance
    async def ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        interface = TelegramInterface(update=update, context=context)
        task_manager.run_task(
            coro=handle_user_prompt(interface=interface),
            user_id=interface.user_id,
        )
        return None

    _MODELS_PER_PAGE = 12

    @staticmethod
    def _create_model_selection_keyboad(
        models: list[ModelChangeSchema],
        context: ContextTypes.DEFAULT_TYPE,
        *,
        add_back_button: bool = False,
        page: int = 0,
        per_page: int = _MODELS_PER_PAGE,
    ) -> InlineKeyboardMarkup:
        total = len(models)
        total_pages = max((total - 1) // per_page + 1, 1) if total > 0 else 1
        page = max(0, min(page, total_pages - 1))
        start = page * per_page
        page_models = models[start : start + per_page]

        mapped_models: dict[str, ModelChangeSchema] = {str(k): model for k, model in enumerate(page_models)}
        set_user_context(context=context, key=UserContext.MAPPED_MODELS, value=mapped_models)

        keyboard = [
            [InlineKeyboardButton(f"{model.display_name}", callback_data=key)] for key, model in mapped_models.items()
        ]
        for model in page_models:
            logger.debug(f"{model.provider}: {model.name}")

        if total_pages > 1:
            nav_buttons = []
            if page > 0:
                nav_buttons.append(InlineKeyboardButton(text="\u25c0 Back", callback_data=f"__page_{page - 1}__"))
            nav_buttons.append(
                InlineKeyboardButton(text=f"\U0001f4c4 {page + 1}/{total_pages}", callback_data="__noop__")
            )
            if page < total_pages - 1:
                nav_buttons.append(InlineKeyboardButton(text="More \u25b6", callback_data=f"__page_{page + 1}__"))
            keyboard.append(nav_buttons)

        if add_back_button:
            keyboard.append(
                [InlineKeyboardButton(text="\u2190 Back to providers", callback_data="__back_to_providers__")]
            )
        keyboard.append([InlineKeyboardButton(text="CLOSE (SELECT NOTHING)", callback_data="-1")])
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def _create_provider_selection_keyboad(
        models: list[ModelChangeSchema],
        context: ContextTypes.DEFAULT_TYPE,
        *,
        active_provider: str | None = None,
    ) -> InlineKeyboardMarkup:
        grouped: dict[str, list[ModelChangeSchema]] = {}
        for model in models:
            if model.provider not in grouped:
                grouped[model.provider] = []
            grouped[model.provider].append(model)
        return ChibiBot._create_provider_selection_keyboad_from_grouped(
            grouped=grouped, context=context, active_provider=active_provider
        )

    @staticmethod
    def _create_provider_selection_keyboad_from_grouped(
        grouped: dict[str, list[ModelChangeSchema]],
        context: ContextTypes.DEFAULT_TYPE,
        *,
        active_provider: str | None = None,
    ) -> InlineKeyboardMarkup:
        set_user_context(context=context, key=UserContext.MAPPED_MODELS_GROUPED, value=grouped)
        keyboard = [
            [
                InlineKeyboardButton(
                    f"\u2705 {provider}" if provider == active_provider else provider,
                    callback_data=provider,
                )
            ]
            for provider in grouped
        ]
        keyboard.append([InlineKeyboardButton(text="CLOSE (SELECT NOTHING)", callback_data="-1")])
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def _find_active_provider_in_models(models: list[ModelChangeSchema]) -> str | None:
        """Find the provider of the active model by looking for the 🟢-marked model.

        `get_models_available` marks the active model's display_name with '🟢 ' prefix.
        This is more reliable than checking Telegram context which may not be set.
        """
        for m in models:
            if m.display_name.startswith("\U0001f7e2"):
                return m.provider
        return None

    @staticmethod
    def _find_active_provider_in_grouped(grouped: dict[str, list[ModelChangeSchema]]) -> str | None:
        for provider, models in grouped.items():
            for m in models:
                if m.display_name.startswith("\U0001f7e2"):
                    return provider
        return None

    @check_user_allowance
    async def show_llm_models_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)

        available_models = await handle_available_model_options(
            user_id=get_telegram_user(update=update).id,
            image_generation=False,
            interface=TelegramInterface(update=update, context=context),
        )

        active_model = get_user_context(context=context, key=UserContext.ACTIVE_MODEL, expected_type=str)
        prefix = f"Active model: {active_model}. " if active_model else ""

        if len(available_models) <= 12:
            reply_markup = self._create_model_selection_keyboad(models=available_models, context=context)
            message = f"{prefix}You may select another one from the list below:" if prefix else "Please, select model:"
            set_user_action(context=context, action=UserAction.SELECT_MODEL)
        else:
            active_provider = self._find_active_provider_in_models(available_models)
            reply_markup = self._create_provider_selection_keyboad(
                models=available_models, context=context, active_provider=active_provider
            )
            message = f"{prefix}Select a provider:" if prefix else "Select a provider:"
            set_user_action(context=context, action=UserAction.SELECT_MODEL_PROVIDER)

        await telegram_message.reply_text(text=message, reply_markup=reply_markup)

    @check_user_allowance
    async def show_image_models_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        available_models = await handle_available_model_options(
            user_id=get_telegram_user(update=update).id,
            image_generation=True,
            interface=TelegramInterface(update=update, context=context),
        )

        active_model = get_user_context(context=context, key=UserContext.ACTIVE_IMAGE_MODEL, expected_type=str)
        prefix = f"Active model: {active_model}. " if active_model else ""

        if len(available_models) <= 12:
            reply_markup = self._create_model_selection_keyboad(models=available_models, context=context)
            message = f"{prefix}You may select another one from the list below:" if prefix else "Please, select model:"
            set_user_action(context=context, action=UserAction.SELECT_MODEL)
        else:
            active_provider = self._find_active_provider_in_models(available_models)
            reply_markup = self._create_provider_selection_keyboad(
                models=available_models, context=context, active_provider=active_provider
            )
            message = f"{prefix}Select a provider:" if prefix else "Select a provider:"
            set_user_action(context=context, action=UserAction.SELECT_MODEL_PROVIDER)

        await telegram_message.reply_text(text=message, reply_markup=reply_markup)

    async def show_api_key_set_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        keyboard = [
            [InlineKeyboardButton(name, callback_data=name)]
            for name, klass in RegisteredProviders.all.items()
            if name != "Cloudflare"
            # Temporary removing the Cloudflare provider from the "public mode"
            # because we need to handle account id setting first. Will provide
            # such a support in one of the following releases.
        ]
        keyboard.append([InlineKeyboardButton(text="CLOSE (SELECT NOTHING)", callback_data="-1")])

        reply_markup = InlineKeyboardMarkup(keyboard)

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

        if query.data == "__noop__":
            return None

        if query.data and query.data.startswith("__page_"):
            full_models = get_user_context(
                context=context,
                key=UserContext.MAPPED_MODELS_FULL,
                expected_type=list[ModelChangeSchema],
            )
            if not full_models:
                await query.delete_message()
                return None
            try:
                target_page = int(query.data.removeprefix("__page_").removesuffix("__"))
            except ValueError:
                return None
            reply_markup = self._create_model_selection_keyboad(
                models=full_models, context=context, add_back_button=True, page=target_page
            )
            # Preserve existing keyboard text (provider name + "— select a model:")
            await query.edit_message_reply_markup(reply_markup=reply_markup)
            return None

        if query.data == "__back_to_providers__":
            grouped = get_user_context(
                context=context,
                key=UserContext.MAPPED_MODELS_GROUPED,
                expected_type=dict[str, list[ModelChangeSchema]],
            )
            if not grouped:
                await query.delete_message()
                return None
            active_provider = self._find_active_provider_in_grouped(grouped)
            reply_markup = self._create_provider_selection_keyboad_from_grouped(
                grouped=grouped, context=context, active_provider=active_provider
            )
            set_user_action(context=context, action=UserAction.SELECT_MODEL_PROVIDER)
            await query.edit_message_text(text="Select a provider:", reply_markup=reply_markup)
            return None

        model = mapped_models.get(query.data)
        if not model:
            await query.delete_message()
            return None

        if model.image_generation:
            set_user_context(context=context, key=UserContext.ACTIVE_IMAGE_MODEL, value=model.name)
        else:
            set_user_context(context=context, key=UserContext.ACTIVE_MODEL, value=model.name)
        telegram_interface = TelegramInterface(update=update, context=context)

        task_manager.run_task(
            coro=handle_model_selection(
                interface=telegram_interface,
                model=model,
                query=query,
            ),
            user_id=telegram_interface.user_id,
        )

        set_user_action(context=context, action=UserAction.NONE)

    async def _compute_model_provider_selection_action(
        self, query: CallbackQuery, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        grouped = get_user_context(
            context=context,
            key=UserContext.MAPPED_MODELS_GROUPED,
            expected_type=dict[str, list[ModelChangeSchema]],
        )
        await query.answer()

        if not grouped or not query.data:
            await query.delete_message()
            return None

        if query.data == "-1":
            await query.delete_message()
            return None

        provider_models = grouped.get(query.data)
        if not provider_models:
            await query.delete_message()
            return None

        set_user_context(context=context, key=UserContext.MAPPED_MODELS_FULL, value=provider_models)
        reply_markup = self._create_model_selection_keyboad(
            models=provider_models, context=context, add_back_button=True, page=0
        )
        set_user_action(context=context, action=UserAction.SELECT_MODEL)
        await query.edit_message_text(text=f"{query.data} \u2014 select a model:", reply_markup=reply_markup)

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

        if action == UserAction.SELECT_MODEL_PROVIDER:
            return await self._compute_model_provider_selection_action(query=query, update=update, context=context)

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
                title=f"Ask {telegram_settings.bot_name}",
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
            app.add_handler(CommandHandler("llm_models", self.show_llm_models_menu))
            app.add_handler(CommandHandler("image_models", self.show_image_models_menu))
        app.add_handler(CallbackQueryHandler(self.handle_selection))

        if gpt_settings.public_mode:
            app.add_handler(CommandHandler("set_api_key", self.show_api_key_set_menu))

        app.add_handler(CommandHandler("ask", self.ask))
        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("reset", self.reset))
        app.add_handler(CommandHandler("stop", self.stop))
        app.add_handler(CommandHandler("start", self.help))

        app.add_handler(MessageHandler(filters.TEXT | filters.VOICE | filters.AUDIO & (~filters.COMMAND), self.prompt))
        app.add_handler(
            MessageHandler(
                filters.ATTACHMENT | filters.PHOTO | filters.Document.ALL & (~filters.COMMAND), self.file_upload
            )
        )

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
    telegram_setting_pre_start_check()
    log_application_settings()
    telegram_security_pre_start_check()
    telegram_bot = ChibiBot(telegram_settings.token)
    telegram_bot.run()
