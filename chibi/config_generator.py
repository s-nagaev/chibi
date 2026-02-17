import os
import shutil
from pathlib import Path

CHIBI_BOT_DIR = Path.home() / "chibi-bot"
DATA_DIR = CHIBI_BOT_DIR / "data"
SKILLS_DIR = CHIBI_BOT_DIR / "skills"
HOME_DIR = CHIBI_BOT_DIR / "home"
CONFIG_PATH = CHIBI_BOT_DIR / "settings"

DEFAULT_CONFIG = f"""# Chibi Agent Configuration File
# Chibi comes fully preconfigured with a set of optimal parameters.
# However, for security reasons, its access to the file system is DISABLED by default.

# If you only need a chatbot without access to your PC’s file system, just set the following values:
# • TELEGRAM_BOT_TOKEN
# • USERS_WHITELIST - enter your nickname or Telegram user ID here; you can also specify multiple
#                     nicknames or IDs separated by commas for everyone who should have access to your bot
# • at least one LLM provider key, for example, OPENAI_API_KEY
# That’s it, you’re awesome! :) The chatbot is ready to launch!

# For advanced usage (agent mode), you additionally need to set:
# • FILESYSTEM_ACCESS=true
# • ENABLE_MCP_STDIO=true (optional but recommended)

# Official documentation: https://chibi.bot/configuration

# ============================================================================
#                                PLEASE NOTE
#                    If you change the value of a setting,
#              please make sure that this setting is uncommented
#                   (does not contain a leading `#` symbol).
# ============================================================================

# ============================================================================
# 1. GETTING STARTED (Required)
# ============================================================================

# Telegram bot token (REQUIRED) - Get one from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=

# Comma-separated list of Telegram user IDs or nicnames allowed to interact with the bot
# Example: "123456789,987654321,@FooBar"
USERS_WHITELIST=

# Comma-separated list of Telegram group IDs where the bot can operate
# Example: "-1001234567890,-1009876543210"
GROUPS_WHITELIST=


# ============================================================================
# 2. LLM PROVIDERS (At least one required for private mode)
# ============================================================================

# GPT-5.2, Whisper, Dall-E, etc. https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Claude models. https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=

# Gemini, Nano Banana, Imagen models. https://makersuite.google.com/app/apikey
GEMINI_API_KEY=

# Grok from xAI. https://console.x.ai/
GROK_API_KEY=

# Mistral, Magistral, Codestral, etc. https://console.mistral.ai/
MISTRALAI_API_KEY=

# Qwen, Wan models. https://modelstudio.console.alibabacloud.com?tab=playground#/api-key
ALIBABA_API_KEY=

# Deepseek-chat, deepseek-reasoner. https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY=

# MiniMax models. https://www.minimax.io
MINIMAX_API_KEY=

# Kimi models. https://platform.moonshot.cn/
MOONSHOTAI_API_KEY=

# Zhipu AI, GLM models. https://z.ai/manage-apikey/apikey-list
ZHIPUAI_API_KEY=

# Open-source models from Cloudflare. Chat-only. https://dash.cloudflare.com/profile/api-tokens
CLOUDFLARE_API_KEY=
# Cloudflare account ID. Required if CLOUDFLARE_API_KEY set.
CLOUDFLARE_ACCOUNT_ID=

# Custom OpenAI API key (for local/self-hosted models)
#CUSTOMOPENAI_API_KEY=""
# Custom OpenAI URL (default: http://localhost:1234/v1)
CUSTOMOPENAI_URL=http://localhost:1234/v1


# ============================================================================
# 3. MEDIA PROVIDERS
# ============================================================================

# API key for unofficial Suno API music generation service. https://sunoapi.org
SUNO_API_ORG_API_KEY=

# ElevenLabs API key for text-to-speech. https://elevenlabs.io/app/settings/api-keys
ELEVEN_LABS_API_KEY=


# ============================================================================
# 4. STORAGE CONFIGURATION
# ============================================================================

# LOCAL STORAGE (DEFAULT)
LOCAL_DATA_PATH={DATA_DIR.absolute()}

# REDIS STORAGE (if set, the local storage setting will be ignored)
# Format: redis://[:password@]host[:port][/db][?option=value]
# REDIS=

# AWS DYNAMODB STORAGE (if set, the local storage and redis setting will be ignored)
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# DynamoDB table name for users
DDB_USERS_TABLE=
# DynamoDB table name for messages
DDB_MESSAGES_TABLE=


# ============================================================================
# 5. AGENT CAPABILITIES
# ============================================================================

# Allow AI agent to access the filesystem (default: false)
FILESYSTEM_ACCESS=false

# Allow AI agent to delegate tasks to sub-agents (default: true)
ALLOW_DELEGATION=true

# Timeout in seconds for delegated sub-agent tasks (default: None — no timeout)
#DELEGATE_TASK_TIMEOUT=600

# AI agent home directory (default: ~/chibi-bot/home)
HOME_DIR={HOME_DIR.absolute()}

# AI agent working directory (default: ~/chibi-bot/home)
WORKING_DIR={HOME_DIR.absolute()}

# Absolute path to directory with skills
SKILLS_DIR={SKILLS_DIR.absolute()}


# ============================================================================
# 6. MCP CONFIGURATION (Model Context Protocol)
# ============================================================================

# Enable MCP SSE (Server-Sent Events) connections (default: true)
ENABLE_MCP_SSE=true

# Enable MCP stdio (standard input/output) connections (default: false)
ENABLE_MCP_STDIO=false


# ============================================================================
# 7. WHITELISTS AND RESTRICTIONS
# ============================================================================

# Comma-separated list of allowed models
# Example: "gpt-5.2,claude-sonnet-4-5-20250929,models/gemini-2.5-pro"
MODELS_WHITELIST=

# Comma-separated list of allowed tools
# Example: "run_command_in_terminal,delegate_task,create_file"
TOOLS_WHITELIST=

# Comma-separated list of users allowed to generate images
# Example: "123456789,987654321"
IMAGE_GENERATIONS_WHITELIST=

# Hide model options in UI (default: false)
HIDE_MODELS=false

# Hide imagine commands (default: false)
HIDE_IMAGINE=false


# ============================================================================
# 8. BEHAVIOR SETTINGS
# ============================================================================

# Default LLM model to use when not specified
# Example: "gpt-5.2,claude-sonnet-4-5-20250929,models/gemini-2.5-pro"
DEFAULT_MODEL=

# Default LLM provider to use when not specified
# Example: "DeepSeek"
DEFAULT_PROVIDER=

# Sampling temperature for LLM responses (0.0-2.0, default: 1.0)
# Higher values increase randomness, lower values make responses more deterministic
TEMPERATURE=1.0

# Maximum number of tokens in LLM responses (default: 32000)
MAX_TOKENS=32000

# Penalty for token frequency in LLM responses (0.0-2.0, default: 0.0)
# Higher values reduce repetition
FREQUENCY_PENALTY=0.0

# Penalty for token presence in LLM responses (0.0-2.0, default: 0.0)
# Higher values encourage new topics
PRESENCE_PENALTY=0.0

# Backoff factor for API retry exponential backoff (default: 0.5)
BACKOFF_FACTOR=0.5

# Number of retry attempts for failed API calls (default: 3)
RETRIES=3

# Timeout in seconds for API calls (default: 600)
TIMEOUT=600


# ============================================================================
# 9. IMAGE GENERATION SETTINGS
# ============================================================================

# Number of image choices to generate (1-4, default: 1)
IMAGE_N_CHOICES=1

# Image quality: standard or hd (default: standard)
IMAGE_QUALITY=standard

# Image size for general image generation (default: 1024x1024)
IMAGE_SIZE=1024x1024

# Image aspect ratio for generation (default: 16:9)
IMAGE_ASPECT_RATIO=16:9

# Image size for Nano Banana provider: 1K, 2K, or 4K (default: 2K)
IMAGE_SIZE_NANO_BANANA=2K

# Image size for Imagen provider: 1K or 2K (default: 2K)
IMAGE_SIZE_IMAGEN=2K

# Image size for Alibaba provider (default: 1664*928)
IMAGE_SIZE_ALIBABA=1664*928


# ============================================================================
# 10. PROVIDER SELECTION
# ============================================================================

# Default speech-to-text provider. Please ensure that the provider you set is available
# Example: "OpenAI"
# STT_PROVIDER=

# Default speech-to-text model
# Example: whisper-1
# STT_MODEL=

# Default text-to-speech provider
# Example: "elevenlabs"
# TTS_PROVIDER=

# Default text-to-speech model
# Example: "eleven_multilingual_v2"
# TTS_MODEL=

# Default content moderation provider
# Example: "Gemini"
# MODERATION_PROVIDER=

# Default content moderation model
# Example: "models/gemini-2.5-flash"
# MODERATION_MODEL=


# ============================================================================
# 11. CONVERSATION SETTINGS
# ============================================================================

# Maximum age of conversation history in minutes before cleanup (default: 360)
MAX_CONVERSATION_AGE_MINUTES=360

# Maximum number of tokens to keep in conversation history (default: 64000)
MAX_HISTORY_TOKENS=64000


# ============================================================================
# 12. SEARCH CAPABILITIES
# ============================================================================

# Google Custom Search API key
# Get from: https://developers.google.com/custom-search/v1/overview
# GOOGLE_SEARCH_API_KEY=

# Google Custom Search Engine ID
# Get from: https://programmablesearchengine.google.com/
# GOOGLE_SEARCH_CX=


# ============================================================================
# 13. TELEGRAM ADVANCED
# ============================================================================

# Base URL for Telegram Bot API (default: https://api.telegram.org/bot)
TELEGRAM_BASE_URL=https://api.telegram.org/bot

# Base URL for Telegram file downloads (default: https://api.telegram.org/file/bot)
TELEGRAM_BASE_FILE_URL=https://api.telegram.org/file/bot

# Allow other bots to interact with this bot (default: false)
ALLOW_BOTS=false

# Only answer direct messages (not group messages) (default: true)
ANSWER_DIRECT_MESSAGES_ONLY=true

# Display name for the bot (default: Chibi)
BOT_NAME=Chibi

# Message shown to users not in whitelist (default: shown below)
MESSAGE_FOR_DISALLOWED_USERS=You're not allowed to interact with me, sorry. Contact my owner first, please.


# ============================================================================
# 14. HEARTBEAT MONITORING
# ============================================================================

# Heartbeat check URL (for monitoring bot availability)
# HEARTBEAT_URL=

# Heartbeat frequency in seconds (minimum: 30, default: 60)
HEARTBEAT_FREQUENCY_CALL=60

# Heartbeat retry count (default: 3)
HEARTBEAT_RETRY_CALLS=3

# Heartbeat proxy URL
# HEARTBEAT_PROXY=


# ============================================================================
# 15. INFLUXDB METRICS
# ============================================================================

# InfluxDB URL for metrics collection
# INFLUXDB_URL=

# InfluxDB token for authentication
# INFLUXDB_TOKEN=

# InfluxDB organization
# INFLUXDB_ORG=

# InfluxDB bucket for metrics
# INFLUXDB_BUCKET=


# ============================================================================
# 16. LOGGING AND DEBUGGING
# ============================================================================

# Log prompt data (for debugging, default: false)
LOG_PROMPT_DATA=false

# Show LLM reasoning/thought process (default: false)
SHOW_LLM_THOUGHTS=false


# ============================================================================
# 17. GENERAL SETTINGS
# ============================================================================

# General proxy URL for all API calls
# PROXY=

# Enable public mode (default: false)
PUBLIC_MODE=false
"""


def generate_default_config() -> None:
    """Generate default configuration file if it doesn't exist."""
    config_dir = os.path.dirname(CONFIG_PATH)
    Path(config_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "w") as config_file:
                config_file.write(DEFAULT_CONFIG)
            print(f"Created default configuration at {CONFIG_PATH}")
        except IOError as e:
            print(f"Error creating configuration file: {e}")

    # Create directory structure
    try:
        for directory in [CHIBI_BOT_DIR, DATA_DIR, SKILLS_DIR, HOME_DIR]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {directory}")
    except IOError as e:
        print(f"Error creating directories: {e}")

    # Sync Skills
    package_skills_dir = Path(__file__).parent.parent / "skills"
    if package_skills_dir.exists():
        try:
            synced_count = 0
            for skill_file in package_skills_dir.iterdir():
                if skill_file.is_file():
                    target_path = SKILLS_DIR / skill_file.name
                    if not target_path.exists():
                        shutil.copy2(skill_file, target_path)
                        synced_count += 1
            if synced_count > 0:
                print(f"Synced {synced_count} skills to {SKILLS_DIR}")
        except IOError as e:
            print(f"Error syncing skills: {e}")
