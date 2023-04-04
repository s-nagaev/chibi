import asyncio

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
from chibi.utils import check_user_allowance


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
        if not gpt_settings.api_key:
            self.commands.append(
                BotCommand(
                    command="set_openai_key", description="Set your own OpenAI key (e.g. /set_openai_key sk-XXXXX)"
                )
            )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        commands = [f"/{command.command} - {command.description}" for command in self.commands]
        commands_desc = "\n".join(commands)
        help_text = (
            f"Hey! My name is {telegram_settings.bot_name}, and I'm your ChatGPT experience provider!\n\n"
            f"{commands_desc}"
        )
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    @check_user_allowance
    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        asyncio.create_task(handle_reset(update=update, context=context))

    @check_user_allowance
    async def imagine(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        asyncio.create_task(handle_image_generation(update=update, context=context))

    @check_user_allowance
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        asyncio.create_task(handle_prompt(update=update, context=context))

    @check_user_allowance
    async def ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        asyncio.create_task(handle_prompt(update=update, context=context))

    @check_user_allowance
    async def set_openai_key(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        asyncio.create_task(handle_openai_key_set(update=update, context=context))

    @check_user_allowance
    async def show_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        reply_markup = await handle_available_model_options(update=update, context=context)

        await update.message.reply_text("Please, select GPT model:", reply_markup=reply_markup)

    # @check_user_allowance
    async def select_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        asyncio.create_task(handle_model_selection(update=update, context=context))

    @check_user_allowance
    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.inline_query.query
        if not query:
            return
        results = [
            InlineQueryResultArticle(
                id=query,
                title="Ask ChatGPT",
                input_message_content=InputTextMessageContent(query),
                description=query,
                thumb_url="https://seeklogo.com/images/C/chatgpt-logo-02AFA704B5-seeklogo.com.png",
            )
        ]

        await update.inline_query.answer(results)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f"Error occurred while handling an update: {context.error}")

    async def post_init(self, application: Application) -> None:
        await application.bot.set_my_commands(self.commands)

    def run(self) -> None:
        app = (
            ApplicationBuilder()
            .token(telegram_settings.token)
            .proxy_url(telegram_settings.proxy)
            .get_updates_proxy_url(telegram_settings.proxy)
            .post_init(self.post_init)
            .build()
        )
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
    telegram_bot = ChibiBot()
    telegram_bot.run()
