<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="logo"></h1>

[![Build](https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg)](https://github.com/s-nagaev/chibi/actions/workflows/build.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/s-nagaev/chibi/badge)](https://www.codefactor.io/repository/github/s-nagaev/chibi)
[![docker hub](https://img.shields.io/docker/pulls/pysergio/chibi)](https://hub.docker.com/r/pysergio/chibi)
[![docker image arch](https://img.shields.io/badge/docker%20image%20arch-arm64%20%7C%20amd64-informational)](https://hub.docker.com/r/pysergio/chibi/tags)
[![docker image size](https://img.shields.io/docker/image-size/pysergio/chibi/latest)](https://hub.docker.com/r/pysergio/chibi/tags)
![license](https://img.shields.io/github/license/s-nagaev/chibi)


Chibi is a Python-based Telegram chatbot that allows users to interact with a variety of powerful large language models (LLMs) and image generation models. The bot is asynchronous, providing fast response times and serving multiple users simultaneously without blocking each other's requests. Chibi supports session management, enabling models to remember the user's conversation history. The conversation history is stored for a configurable duration and is preserved even if the bot is restarted (when using persistent storage).

[Docker Hub](https://hub.docker.com/r/pysergio/chibi)

## Supported platforms

Currently supported architectures:
*   `linux/amd64`
*   `linux/arm64`

_**Note on armv7:** Support for the `linux/armv7` architecture was discontinued after version `0.8.2`. However, if there is sufficient user interest, resuming `armv7` builds can be investigated._

## Supported providers

Chibi currently supports models from the following providers:
*   Alibaba (Qwen models)
*   Anthropic (Claude models)
*   Cloudflare (44+ open-source models)
*   DeepSeek
*   Google (Gemini models, including image generation)
*   MistralAI
*   MoonshotAI
*   OpenAI (including DALL-E for images)
*   xAI (Grok models, including image generation)
*   ElevenLabs (ElevenLabs models, including text to speech, speech to text, music generation)
*   MiniMax (Minimax models, text to speech)

## Features

*   **Expanded AI Provider Support:** Interact with a wider range of leading LLM and image generation models from various providers, including OpenAI, Google (Gemini), Anthropic, Alibaba, Deepseek, Grok, MoonshotAI, Cloudflare, and custom OpenAI-compatible endpoints.
*   **Seamless Provider Switching:** Change the underlying LLM provider anytime without losing the current conversation context. The history is automatically adapted for the new model.
*   **Advanced Tooling & Recursive Delegation:** The bot leverages an enhanced suite of built-in tools for tasks like file system operations (with `FILESYSTEM_ACCESS` setting), terminal command execution (with LLM-moderated access), web searching, and more. Recursive delegation allows sub-agents to handle complex, multi-step tasks efficiently, significantly reducing token usage.
*   **Initial Voice Interaction:** Engage with the bot using voice messages and receive audio responses. This is an initial implementation with ongoing development planned for richer voice capabilities.
*   **Configurable Image Generation:** Request images with `Nano Banana`, `Imagen`, `Wan/Qwen`, `DALL-E` or `Grok`, with enhanced control over quality, size, aspect ratio, and quantity per request.
*   **Music Generation:** Generate high-quality music via Suno AI by describing the style and lyrics. Use the `/image_model` command to see if Suno is available or simply ask the bot to "compose a song".
*   **Context Management:**
    *   Automatic conversation summarization (optional) to save tokens on long conversations by replacing older parts of the history with a summary.
    *   Manual context reset (`/reset` command) to start fresh and save tokens.
*   **Flexible Session Storage:** Store conversation history locally (requires mounting a volume for persistence), in Redis, or DynamoDB, or simply keep it in memory (lost on restart).
*   **Optional "Public Mode":** Run the bot without master API keys. Each user will be prompted to provide their own key via private message to the bot.
*   **Granular Access Control:** Restrict bot access to specific users or chat groups, whitelist allowed models, and control image generation limits for certain users.
*   **Dynamic Proxy Support:** Configure HTTP/SOCKS proxies for AI provider API requests and heartbeat pings, separate from the Telegram proxy.
*   **Health Monitoring (Heartbeat):** The bot can periodically ping a configured URL (e.g., a healthchecks.io endpoint) to signal its operational status, allowing external systems to monitor its health and availability.
*   **Easy Deployment with Docker:** Pre-configured Docker images (including experimental ARMv7 support) ready to run with minimal setup. Public mode works out-of-the-box without needing API keys in the environment variables.
*   **Low Resource Usage:** Runs efficiently even on low-spec hardware like a Raspberry Pi 4.
*   **Asynchronous:** Fast, non-blocking performance.
*   **Highly Configurable:** Extensive options via environment variables for fine-grained control over bot behavior.
*   **MIT Licensed:** Open source and free to use.

## System Requirements

Chibi is designed to be very resource-efficient. It runs smoothly on hardware like a Raspberry Pi 4 (or higher) or even minimal cloud instances such as an AWS EC2 `t4g.nano` (2 arm64 vCPUs, 512MB RAM), capable of serving multiple users concurrently.

Essentially, the main requirements are:
*   A machine running on a supported architecture (`linux/amd64` or `linux/arm64`).
*   Docker installed and running on that machine.

## Prerequisites

- Docker
- Docker Compose (optional)

## Getting Started

### Quick Install: Dobby Agent Mode (Experimental)

⚠️ **WARNING**: This installation mode enables **full filesystem access**. Only use on trusted systems.

**Dobby** is an experimental agent mode that gives the bot access to your filesystem and shell commands. Perfect for development assistance, automation, and advanced AI interactions.

```bash
curl -fsSL https://raw.githubusercontent.com/s-nagaev/chibi/main/scripts/install.sh | bash
```

**What you get:**
- Autonomous AI agent with filesystem access
- Pre-configured for agent workflows
- Simple command-line interface: `dobby run`, `dobby attach`, `dobby free`
- 200K token context window (extended conversations)
- Whitelisted access only (security first)

**Requirements:**
- Telegram Bot Token (required)
- At least one AI provider API key
- User whitelist configuration

See [scripts/README.md](scripts/README.md) for full installation guide and security considerations.

### Standard Installation (Docker)

### Using Docker Run

1. Pull the Chibi Docker image:

    ```shell
    docker pull pysergio/chibi:latest
    ```

2. Run the Docker container with the necessary environment variables:
This command runs the bot in private mode using a Google Gemini key. It requires your Telegram token and at least one AI provider API key. Data is *not* persisted between restarts.
    ```shell
   docker run -d --name chibi \
     -e TELEGRAM_BOT_TOKEN='YOUR_TELEGRAM_TOKEN' \
     -e GEMINI_API_KEY='AIzaSyYourGeminiKey...' \
     # Add other keys like -e OPENAI_API_KEY='sk-...' if needed
     pysergio/chibi:latest
   ```
Replace placeholders with your actual values.
**Hint:** talk to [BotFather](https://t.me/BotFather) on Telegram and use the `/newbot` command to create a bot and get your token.

## Docker Compose Examples

Here are a few examples to get you started with docker-compose. Remember to create a .env file in the same directory as your docker-compose.yml to store your secrets like TELEGRAM_BOT_TOKEN and API keys.

### Minimal Private Mode Example

This is the bare minimum setup to run Chibi in private mode. It uses the OpenAI API key provided in your .env file.

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      # Required: Get from BotFather
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}

      # Required for Private Mode: Add at least one provider API key
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      # Add other keys (GEMINI_API_KEY, ANTHROPIC_API_KEY, etc.) here if needed

# --- .env file example ---
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234...
# OPENAI_API_KEY=sk-YourOpenAIKey...
# -------------------------
```
---
### Advanced Private Mode Example

This example shows a more customized setup for private mode:
*   Uses multiple AI providers.
*   Stores conversation history locally and persistently using a Docker volume.
*   Sets user/group whitelists.
*   Defines a custom assistant prompt and default model.
*   Sets a monthly limit for image generations.
```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      # --- Required ---
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}

      # --- API Keys (add whichever you use) ---
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      # ... add other provider keys as needed

      # --- Whitelists ---
      USERS_WHITELIST: "@your_telegram_username,123456789" # Usernames and User IDs
      GROUPS_WHITELIST: "-100987654321" # Group Chat IDs (usually negative)

      # --- Customization ---
      BOT_NAME: "MyChibiBot"
      ASSISTANT_PROMPT: "You are a specialized assistant for code reviews."
      MODEL_DEFAULT: "gpt-4o" # Set your preferred default model

      # --- Limits & History ---
      MAX_HISTORY_TOKENS: 8000 # Adjust context window size
      IMAGE_GENERATIONS_LIMIT: 20 # Monthly limit per user (requires volume)

      # --- Storage ---
      # LOCAL_DATA_PATH: /app/data # Default path

    volumes:
      # Mounts the 'chibi_data' volume for persistence
      - chibi_data:/app/data

volumes:
  # Defines the Docker volume
  chibi_data: {}

# --- .env file example ---
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234...
# OPENAI_API_KEY=sk-YourOpenAIKey...
# GEMINI_API_KEY=AIzaSy...
# ANTHROPIC_API_KEY=sk-ant-...
# -------------------------
```

### Public Mode Example

This setup runs the bot in public mode. No master API keys are needed here; the bot will ask each user to provide their own key via private message.
```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      # Required: Get from BotFather
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}

      # --- Enable Public Mode ---
      PUBLIC_MODE: "True"

      # --- Optional Customization ---
      BOT_NAME: "ChibiPublic"
      # HIDE_MODELS: "True"
      # HIDE_IMAGINE: "True"

      # --- Whitelists (Optional for Public Mode) ---
      # You can still restrict *which* users/groups can use the bot
      # USERS_WHITELIST: "allowed_user1,123456789"
      # GROUPS_WHITELIST: "-1001122334455"

    # Volumes are optional unless needed for IMAGE_GENERATIONS_LIMIT
    # or specific local storage needs.
    # volumes:
    #   - chibi_public_data:/app/data

# volumes:
#   chibi_public_data: {}

# --- .env file example ---
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234...
# (No API keys needed here for public mode)
# -------------------------
```

Please, visit the [examples](examples) directory of the current repository for more examples.

### Telegram Bot Settings
| Variable                       | Description                                                                                             | Default Value                                                                      |
|:-------------------------------|:--------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| `TELEGRAM_BOT_TOKEN`           | **Required.** Your Telegram Bot API token. Get one from [@BotFather](https://t.me/BotFather).          |                                                                                    |
| `TELEGRAM_BASE_URL`            | Base URL for Telegram Bot API requests. Useful for local Bot API server.                                | `https://api.telegram.org/bot`                                                     |
| `TELEGRAM_BASE_FILE_URL`       | Base URL for Telegram file downloads. Useful for local Bot API server.                                  | `https://api.telegram.org/file/bot`                                                |
| `BOT_NAME`                     | The name the bot uses for itself (e.g., in the default prompt).                                         | `Chibi`                                                                            |
| `ANSWER_DIRECT_MESSAGES_ONLY`  | If `True`, the bot only responds to direct messages, ignoring group messages (unless whitelisted and mentioned). | `True`                                                                             |
| `ALLOW_BOTS`                   | If `True`, allows the bot to respond to messages sent by other bots.                                    | `False`                                                                            |
| `MESSAGE_FOR_DISALLOWED_USERS` | The message sent to users or groups not in the whitelists when they try to interact with the bot.       | `"You\'re not allowed to interact with me, sorry. Contact my owner first, please."` |
| `PROXY`                        | Optional HTTP/SOCKS proxy specifically for connecting to the Telegram Bot API (e.g., `socks5://user:pass@host:port`). | `None`                                                                             |
| `GROUPS_WHITELIST`             | Comma-separated list of Telegram group chat IDs where the bot is allowed to operate. If empty or unset, allow all groups. | `None`                                                                             |
| `USERS_WHITELIST`              | Comma-separated list of Telegram usernames (with or without `@`) or user IDs allowed to interact with the bot. If empty or unset, allow all users. | `None`                                                                             |
### General Application Settings
| Variable          | Description                         | Default Value |
|:------------------|:------------------------------------|:--------------|
| `LOG_PROMPT_DATA` | Whether to log prompt data.         | `False`       |
| `HIDE_MODELS`     | Hide model options in UI.           | `False`       |
| `HIDE_IMAGINE`    | Hide imagine commands.              | `False`       |
| `WORKING_DIR`     | The default working directory for the AI agent's filesystem tools. | `~/chibi` || `HOME_DIR`        | The directory considered as the "home" for the AI agent. | `~/chibi`     |
| `SKILLS_DIR`      | Absolute path to the directory containing LLM skills/prompts. | `./skills`    |
### Storage Settings
| Variable              | Description                                                                                             | Default Value |
|:----------------------|:--------------------------------------------------------------------------------------------------------|:--------------|
| `REDIS`               | Redis connection URL.                                                                                   | `None`        |
| `REDIS_PASSWORD`      | Password for Redis.                                                                                     | `None`        |
| `AWS_REGION`          | AWS region for DynamoDB.                                                                                | `None`        |
| `AWS_ACCESS_KEY_ID`   | AWS access key ID.                                                                                      | `None`        |
| `AWS_SECRET_ACCESS_KEY`| AWS secret access key.                                                                                  | `None`        |
| `DDB_USERS_TABLE`     | DynamoDB table name for users.                                                                          | `None`        |
| `DDB_MESSAGES_TABLE`  | DynamoDB table name for messages.                                                                       | `None`        |
| `LOCAL_DATA_PATH`     | Filesystem path for local storage.                                                                      | `/app/data`   |
### API Keys (Master Keys)
These keys are used when `PUBLIC_MODE` is `False`. If `PUBLIC_MODE` is `True`, these are ignored (users provide their own keys).

| Variable                | Description                                       | Default Value              |
|:------------------------|:--------------------------------------------------|:---------------------------|
| `ALIBABA_API_KEY`       | API key for Alibaba (Qwen) models.                | `None`                     |
| `ANTHROPIC_API_KEY`     | API key for Anthropic (Claude) models.            | `None`                     |
| `CLOUDFLARE_API_KEY`    | API key for Cloudflare (44+ open-source models).  | `None`                     |
| `CLOUDFLARE_ACCOUNT_ID` | Account ID in the Cloudflare platform.            | `None`                     |
| `CUSTOMOPENAI_API_KEY`  | API key for custom OpenAI-compatible endpoints.   | `None`                     |
| `CUSTOMOPENAI_URL`      | URL for custom OpenAI-compatible endpoints.       | `http://localhost:1234/v1` |
| `DEEPSEEK_API_KEY`      | API key for DeepSeek models.                      | `None`                     |
| `GEMINI_API_KEY`        | API key for Google (Gemini & Imagen) models.      | `None`                     |
| `GOOGLE_SEARCH_API_KEY` | API key for Google Custom Search.                 | `None`                     |
| `GOOGLE_SEARCH_CX`      | Custom Search Engine ID for Google Custom Search. | `None`                     |
| `GROK_API_KEY`          | API key for xAI (Grok) models.                    | `None`                     |
| `MISTRALAI_API_KEY`     | API key for MistralAI models.                     | `None`                     |
| `MOONSHOTAI_API_KEY`    | API key for MoonshotAI (Kimi) models.             | `None`                     |
| `OPENAI_API_KEY`        | API key for OpenAI (GPT & DALL-E) models.         | `None`                     |
| `SUNO_API_ORG_API_KEY`  | API key for Suno music generation (via sunoapi.org).       | `None`                     |
| `ELEVEN_LABS_API_KEY`   | API key for ElevenLabs models.                    | `None`                     |
| `MINIMAX_API_KEY`       | API key for MiniMax models.                       | `None`                     |

### Model & Conversation Settings### MCP Settings
| Variable           | Description                                      | Default Value |
|:-------------------|:-------------------------------------------------|:--------------|
| `ENABLE_MCP_SSE`   | Enable Model Context Protocol via SSE transport. | `True`        |
| `ENABLE_MCP_STDIO` | Enable Model Context Protocol via Stdio transport. | `False`       |

| Variable                       | Description                                                                                                                                     | Default Value                                                                                     |
|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
| `PUBLIC_MODE`                  | If `True`, the bot doesn't require master API keys. Users will be asked to provide their own keys via PM.                                       | `False`                                                                                           |
| `PROXY`                        | Optional HTTP/SOCKS proxy for **outgoing AI provider API requests** (e.g., `socks5://user:pass@host:port`). Distinct from Telegram proxy.     | `None`                                                                                            |
| `RETRIES`                      | Number of times to retry a failed AI provider API request.                                                                                      | `3`                                                                                               |
| `TIMEOUT`                      | Timeout in seconds for waiting for a response from the AI provider API.                                                                         | `600`                                                                                             |
| `BACKOFF_FACTOR`               | A backoff factor to apply between attempts after the first failed try.                                                                          | `0.5`                                                                                             |
| `DEFAULT_MODEL`                | Default model ID used for new conversations (e.g., `gpt-5.2`, `claude-sonnet-4-5-20250929`). If unset, the provider\'s default is used.               | `None`                                                                                            |
| `DEFAULT_PROVIDER`             | Default provider for models if `DEFAULT_MODEL` is not fully qualified.                                                                          | `None`                                                                                            |
| `MAX_CONVERSATION_AGE_MINUTES` | Maximum age (in minutes) of messages kept in the active history. Older messages might be summarized or dropped.                                 | `360`                                                                                             |
| `MAX_HISTORY_TOKENS`           | Maximum number of tokens to retain in the conversation history sent to the model. Helps manage context window size and cost.                    | `64000`                                                                                           |
| `MAX_TOKENS`                   | Maximum number of tokens the model is allowed to generate in a single response.                                                                 | `32000`                                                                                           |
| `TEMPERATURE`                  | Controls randomness (0.0 to 2.0). Lower values are more deterministic, higher values are more creative/random.                                  | `1.0`                                                                                             |
| `FREQUENCY_PENALTY`            | Penalty applied to tokens based on their frequency in the text so far (positive values decrease repetition). Range: -2.0 to 2.0.                | `0.0`                                                                                             |
| `PRESENCE_PENALTY`             | Penalty applied to tokens based on whether they appear in the text so far (positive values encourage exploring new topics). Range: -2.0 to 2.0. | `0.0`                                                                                             |
| `MODELS_WHITELIST`             | Comma-separated list of specific model IDs users are allowed to switch to. If empty or unset, all available models are allowed.                  | `None`                                                                                            |
| `FILESYSTEM_ACCESS`            | If `True`, enables file system access for the bot.                                                                                              | `False`                                                                                           || `ALLOW_DELEGATION`           | If `True`, the agent can spawn sub-agents to handle complex tasks.                                                                              | `True`                                                                                            |
| `TOOLS_WHITELIST`              | Comma-separated list of specific tools the agent is allowed to use. If not set, all available tools are enabled.                                | `None`                                                                                            |
| `SHOW_LLM_THOUGHTS`            | If `True`, shows the bot\'s internal thoughts in the chat.                                                                                       | `False`                                                                                           |
### Image Generation Settings
| Variable                      | Description                                                                                                            | Default Value |
|:------------------------------|:-----------------------------------------------------------------------------------------------------------------------|:--------------|
| `IMAGE_GENERATIONS_LIMIT`     | Monthly limit on the number of `/image` commands per user (0 means unlimited). Requires persistent storage.            | `0`           |
| `IMAGE_N_CHOICES`             | Default number of images to generate per request (currently supported mainly by DALL-E).                               | `1`           |
| `IMAGE_QUALITY`               | Default image quality for providers that support it (e.g., DALL-E: `standard` or `hd`).                                | `standard`    |
| `IMAGE_SIZE`                  | Default image size (e.g., `1024x1024`, `1792x1024`). Check provider documentation for supported values.                | `1024x1024`   |
| `IMAGE_ASPECT_RATIO`          | Default image aspect ratio for providers that support it (e.g.: `1:1`, `16:9`, `9:16`).                                | `16:9`        |
| `IMAGE_GENERATIONS_WHITELIST` | Comma-separated list of Telegram usernames (with or without `@`) or user IDs excluded from the image generation limit. | `None`        |
| `IMAGE_SIZE_NANO_BANANA`      | An image size for the Nano Banana Pro model only. Can be `1K`, `2K`, `4K`                                              | `2K`          |
### Whitelists
Whitelist variables have been moved to their respective sections (e.g., `USERS_WHITELIST` and `GROUPS_WHITELIST` are now under Telegram Bot Settings, `MODELS_WHITELIST` under Model & Conversation Settings, and `IMAGE_GENERATIONS_WHITELIST` under Image Generation Settings).
### Heartbeat
| Variable                   | Description                                                                                                                                                                                                                                        | Default Value |
|:---------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|
| `HEARTBEAT_URL`            | The target URL endpoint for sending periodic heartbeat GET requests. _Setting this variable enables the heartbeat feature._ Used to signal to an external monitoring system (like healthchecks.io, uptime kuma, etc.) that the bot is operational. | `None`        |
| `HEARTBEAT_FREQUENCY_CALL` | The interval (in seconds) between sending heartbeat pings to the `HEARTBEAT_URL`. This only applies if `HEARTBEAT_URL` is set.                                                                                                                     | `30`          |
| `HEARTBEAT_RETRY_CALLS`    | Number of times the HTTP client (`httpx`) will automatically retry sending the heartbeat request upon transient failures (e.g., network errors, specific server responses) before logging an error.                                                | `3`           |
| `HEARTBEAT_PROXY`          | Optional proxy URL (e.g., `http://user:pass@host:port` or `socks5://host:port`) to use for sending the heartbeat requests.                                                                                                                         | `None`        |
### Metrics & Monitoring
| Variable          | Description                                           | Default Value |
|:------------------|:------------------------------------------------------|:--------------|
| `INFLUXDB_URL`    | The URL of your InfluxDB instance.                    | `None`        |
| `INFLUXDB_TOKEN`  | The authentication token for your InfluxDB instance.  | `None`        |
| `INFLUXDB_ORG`    | The organization to be used in InfluxDB.              | `None`        |
| `INFLUXDB_BUCKET` | The data bucket where the metrics will be stored.     | `None`        |

## Getting API Keys

To use Chibi in private mode, or for users interacting with the bot in public mode, you'll need API keys from the desired AI providers. Here's where you can typically find information or generate keys:

*   Alibaba (Qwen via DashScope): https://dashscope.console.aliyun.com/apiKey
*   Anthropic (Claude): https://console.anthropic.com/ (Sign up and navigate to API Keys)
*   Cloudflare: https://dash.cloudflare.com/profile/api-tokens (Need API Key with `Account.Workers AI` permissions)
*   DeepSeek: https://platform.deepseek.com/ (Sign up and navigate to API Keys)
*   Google (Gemini): https://aistudio.google.com/app/apikey
*   MistralAI: https://console.mistral.ai/ (Sign up and navigate to API Keys)
*   MoonshotAI (Kimi): https://platform.moonshot.cn (Sign up and navigate to API Keys)
*   OpenAI (GPT & DALL-E): https://platform.openai.com/api-keys
*   xAI (Grok): https://docs.x.ai/ (Check documentation for API access details)
*   ElevenLabs (Music, STT, TTS): https://elevenlabs.io/ (Sign up and navigate to API Keys)
*   Suno (Music, Unofficial API): https://sunoapi.org (Sign up and navigate to API Keys)
*   MiniMax (TTS): https://api.minimax.io/ (Sign up and navigate to API Keys)

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/s-nagaev/chibi/tags).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
 