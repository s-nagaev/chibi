import asyncio
from asyncio import Task

from loguru import logger
from telegram import (
    BotCommand,
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

from chibi.config import gpt_settings, telegram_settings
from chibi.services.bot import (
    handle_available_model_options,
    handle_image_generation,
    handle_model_selection,
    handle_openai_key_set,
    handle_prompt,
    handle_reset,
)
from chibi.utils import (
    GROUP_CHAT_TYPES,
    check_openai_api_key,
    check_user_allowance,
    get_telegram_chat,
    get_telegram_message,
    log_application_settings,
    user_interacts_with_bot,
)


class ChibiBot:
    def __init__(self) -> None:
        self.commands = [
            BotCommand(command="help", description="Show this help message"),
            BotCommand(command="imagine", description="Generate image from prompt with DALL-E (e.g. /imagine cat)"),
            BotCommand(
                command="ask",
                description=(
                    "Ask me any question (in group chat, for example) (e.g. /ask which program language is the best?)"
                ),
            ),
            BotCommand(
                command="reset", description="Reset your conversation history (will reduce prompt and save some tokens)"
            ),
            BotCommand(command="menu", description="Select GPT model"),
        ]
        if gpt_settings.public_mode:
            self.commands.append(
                BotCommand(command="set_openai_key", description="Set your own OpenAI key (e.g. /set_api_key sk-XXXXX)")
            )
        self.background_tasks: set[Task] = set()

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        commands = [f"/{command.command} - {command.description}" for command in self.commands]
        commands_desc = "\n".join(commands)
        help_text = (
            f"Hey! My name is {telegram_settings.bot_name}, and I'm your ChatGPT experience provider!\n\n"
            f"{commands_desc}"
        )
        await telegram_message.reply_text(help_text, disable_web_page_preview=True)

    @check_openai_api_key
    @check_user_allowance
    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        task = asyncio.create_task(handle_reset(update=update, context=context))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    @check_openai_api_key
    @check_user_allowance
    async def imagine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        task = asyncio.create_task(handle_image_generation(update=update, context=context))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    @check_openai_api_key
    @check_user_allowance
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_chat = get_telegram_chat(update=update)
        telegram_message = get_telegram_message(update=update)
        prompt = telegram_message.text

        if not prompt:
            return None

        if (
            telegram_chat.type in GROUP_CHAT_TYPES
            and telegram_settings.answer_direct_messages_only
            and "/ask" not in prompt
            and not user_interacts_with_bot(update=update, context=context)
        ):
            return None
        task = asyncio.create_task(handle_prompt(update=update, context=context))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    @check_openai_api_key
    @check_user_allowance
    async def ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        task = asyncio.create_task(handle_prompt(update=update, context=context))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    @check_user_allowance
    async def set_openai_key(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        task = asyncio.create_task(handle_openai_key_set(update=update, context=context))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    @check_user_allowance
    async def show_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_message = get_telegram_message(update=update)
        reply_markup = await handle_available_model_options(update=update, context=context)

        await telegram_message.reply_text("Please, select GPT model:", reply_markup=reply_markup)

    async def select_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        task = asyncio.create_task(handle_model_selection(update=update, context=context))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    @check_openai_api_key
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
        if telegram_settings.proxy:
            app = (
                ApplicationBuilder()
                .token(telegram_settings.token)
                .proxy_url(telegram_settings.proxy)
                .get_updates_proxy_url(telegram_settings.proxy)
                .post_init(self.post_init)
                .build()
            )
        else:
            app = ApplicationBuilder().token(telegram_settings.token).post_init(self.post_init).build()

        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("reset", self.reset))
        app.add_handler(CommandHandler("imagine", self.imagine))
        app.add_handler(CommandHandler("start", self.help))
        app.add_handler(CommandHandler("ask", self.ask))
        app.add_handler(CommandHandler("set_openai_key", self.set_openai_key))
        app.add_handler(CommandHandler("menu", self.show_menu))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        app.add_handler(CallbackQueryHandler(self.select_model))
        app.add_handler(
            InlineQueryHandler(
                self.inline_query,
                chat_types=[constants.ChatType.GROUP, constants.ChatType.SUPERGROUP],
            )
        )
        app.add_error_handler(self.error_handler)
        app.run_polling()


if __name__ == "__main__":
    log_application_settings()
    telegram_bot = ChibiBot()
    telegram_bot.run()
