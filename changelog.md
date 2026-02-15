# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

## [1.6.3] - 2026-02-15

### Changed
- **ZhipuAI Provider:** Renamed env variable containing API Key.
- **Config Generator:** Added `ZHIPUAI_API_KEY` variable with short description.


## [1.6.2] - 2026-02-15

### Added
- **ZhipuAI Provider:** Added support for ZhipuAI (GLM models) as a new LLM provider.
- **MiniMax Image Generation:** Enabled image generation capabilities for the MiniMax provider.
- **Loop Detection:** Implemented `LoopDetectedException` to prevent infinite loops in tool calls.

### Changed
- **Documentation:** Updated README with new providers and API key links.



## [1.6.1] - 2026-02-11
### Added
- **Pre-start security checks** to prevent dangerous bot configurations that could lead to serious security issues.

## [1.6.0] - 2026-02-10
### Added
- **CLI Interface & Pip Installation:** New `chibi` command for easy bot management (`start`, `stop`, `restart`, `config`, `logs`)
- **MiniMax Provider:** Added chat completion support via Anthropic-compatible API
- **Chinese Localization:** Complete README translations for Simplified (zh-CN) and Traditional (zh-TW) Chinese

### Changed
- **Command Moderation System:** Upgraded feature allowing terminal commands to be validated by LLM providers before execution
- **Dependencies Updated:** anthropic 0.75.0 → 0.79.0, google-genai 1.53.0 → 1.62.0, mcp 1.23.1 → 1.26.0, redis 5.2.1 → 7.1.0, and more
- **Thinking Models:** Improved `reasoning_content` preservation for DeepSeek-Reasoner and Moonshot KIMI models
- **MoonshotAI:** Default temperature changed from 0.3 to 1.0 for better reasoning
- **Provider Defaults:** Gemini → `gemini-2.5-pro`, Grok → `grok-4-1-fast-reasoning`, OpenAI → `gpt-5.2`

### Fixed
- **Anthropic Messages:** Fixed loading messages with tools usage when switching providers
- **Telegram Reaction:** Bot now sets `OK` reaction to user message when bot doesn't answer immediately

## [1.5.2] - 2026-01-18
### Added
- **Telegram message reaction:** Bot now will set `OK` reaction to user message bot doesn't answer immediately.

### Changed
- **Base Prompt updated:** Multi-task aggregation protocol described.

### Changed
- **Base Prompt updated:** Multi-task aggregation protocol described.

## [1.5.1] - 2026-01-17

### Added
- **Background Tasks:** Long-running tools (Image/Music generation, Delegation) now execute in the background, freeing up the chat interface.
- **Global Tool Parameter:** Added a global `run_in_background` parameter to all compatible tools, allowing the LLM to choose execution mode.
- **Silent ACK:** Implemented a protocol for the LLM to silently acknowledge tool completion without notifying the user.
- **System Prompt:** Updated system prompt with instructions for the "ACK" rule and async tool handling.
- **STT/TTS Configuration:** New settings in `GPTSettings` for explicit STT/TTS provider selection (`stt_provider`, `tts_provider`).
- **Provider Helpers:** Added `first_stt_ready` and `first_tts_ready` helpers to `RegisteredProviders` for automatic provider selection.

### Changed
- **Tool Execution:** `GenerateImageTool`, `GenerateMusicViaSunoTool`, `TextToSpeechTool`, and `DelegateTool` default to background execution.
- **Provider Logic:** Refactored how STT and TTS providers are selected; added fallback to the first available provider if not explicitly configured.
- **Provider Interface:** Standardized `transcribe` and `speech` methods in the `Provider` base class.
- **Anthropic:** Enabled ephemeral caching for prompts to optimize costs/performance.
- **Internal:** Renamed core message handling functions (`handle_user_prompt`, `get_llm_chat_completion_answer`) for clarity.
- **Internal:** `GenerateAdvancedMusicViaSunoTool` migrated to the new global background execution mechanism.

## [1.5.0] - 2026-01-06
### Added
- **Suno AI Integration:**
    - Support for high-quality music generation via Suno AI.
    - New tools: `generate_music_via_suno` and `generate_music_via_suno_custom_mode`.
    - Asynchronous polling for generation results with robust retry logic.
- **Enhanced Media Tools:**
    - New tools for direct media interaction: `send_audio`, `send_image`, `send_video`, and `send_media_group`.
- **AI Skills System:**
    - Added specialized prompting guides (Skills) for `Imagen`, `Jina Reader`, `Nano Banana`, `Suno`, and `Wan` models to improve output quality and consistency.
- **Custom Telegram API Support:**
    - Added `TELEGRAM_BASE_URL` and `TELEGRAM_BASE_FILE_URL` settings for compatibility with local Telegram Bot API servers.
- **Agent Environment:**
    - Added `WORKING_DIR` setting to define the default path for agent operations.

### Changed
- **Architecture:** Refactored `BackgroundTaskManager` to use a Singleton pattern for better task lifecycle management.
- **MCP Improvements:** Increased `call_tool` timeout to 600s and improved error reporting for SSE connections.
- **Dependencies:** Added `tenacity` for resilient API polling and updated `mistralai` and `mcp` versions.
- **Logging:** Standardized tool call logging to include the specific model name requesting the action.

### Fixed
- Improved cleanup and error handling in background tasks to prevent "zombie" tasks on failure.

## [1.4.1] - 2025-12-27
### Added
- **Model Context Protocol (MCP)** integration:
    - Support for **Stdio** transport (local tools like SQLite, Filesystem).
    - Support for **SSE** transport (remote tools).
    - Lifecycle management for MCP servers (connect/disconnect/list tools).
    - New settings `ENABLE_MCP_SSE` (default `True`) and `ENABLE_MCP_STDIO` (default `False`).

### Changed
- Improved `BackgroundTaskManager` to return `asyncio.Task` objects for better control.
- Added an option to set the size of the image generated by `Imagen`.
- Implemented retry for Gemini with proper handling of the 429 response.

## [1.4.0] - 2025-12-09
### Added
- **InfluxDB Integration:** Added integration with InfluxDB for storing and visualizing metrics (token usage, cost, latency, etc.). Metrics can now be pushed to InfluxDB for monitoring via Grafana or other tools.
- **Mistral Native Client:** Reimplemented the Mistral AI client using the native `mistralai` library (v1.0.0+) for better stability and feature support, replacing the generic requests-based implementation.
- **Metrics Service:** Introduced a dedicated `MetricsService` (`chibi/services/metrics.py`) to handle data collection and reporting to InfluxDB.
- **Dobby Installation Scripts:** Added a comprehensive suite of scripts (`scripts/`) for installing, configuring, and controlling the "Dobby" local agent mode, including `install.sh`, `dobby-control.sh`, and documentation.

### Changed
- **Task Manager Refactoring:** Refactored `TaskManager` to better handle recursive delegation and metrics tracking.
- **Provider Interface Updates:** Updated the base `Provider` class and implementations (`Anthropic`, `Gemini`, `DeepSeek`) to support the new metrics collection system.
- **Dependency Updates:** Updated `poetry.lock` and requirements files to include `influxdb-client` and updated `mistralai` SDK.
### Changed
- image generation functionality for Gemini models reimplemented using the official `google.genai` library
- upload image to Telegram chat logic updated

## [1.3.0] - 2025-11-23

### Added
- **Enhanced Tooling with Recursive Delegation:** Implemented recursive delegation for sub-agents, enabling the bot to tackle complex, multi-step tasks efficiently. This approach drastically reduces token consumption (up to 76% in various tests) by offloading processing to sub-agents.
- **Initial Voice Interaction Support:** Introduced preliminary support for voice messages, allowing the bot to receive voice input and respond with generated speech. This feature is an initial implementation and is planned for significant future development.
- **Moderated Terminal Access:** Integrated a moderated terminal access tool. All terminal commands are pre-moderated by a specialized LLM using strict rules to ensure safety and prevent unauthorized operations. This is an initial implementation with extensive future enhancements planned. (Refer to `chibi/constants.py` for moderation prompt details).
- **Google Web Search Integration:** Added a dedicated tool for web searching using the official Google Search API for more accurate and comprehensive search results.
- **Enhanced Built-in Toolset:** Expanded the bot's capabilities with a new suite of built-in tools, including command execution (`cmd.py`), general utilities (`common.py`), robust file editing (`file_editor.py`), memory management (`memory.py`), and a structured framework for tool development (`tool.py`, `schemas.py`, `utils.py`, `exceptions.py`).
    - *Note:* Access to the filesystem via tools is governed by the `FILESYSTEM_ACCESS` setting and is strongly discouraged for publicly accessible bots due to potential security implications.
- **DynamoDB Support:** Implemented initial support for DynamoDB as a persistent storage backend.

### Changed
- **Major Codebase Refactoring:** Performed extensive refactoring across the core codebase, models, and provider logic to significantly improve maintainability, scalability, and overall code quality.
- **Rewritten Gemini Provider:** The Gemini provider has been completely rewritten to leverage the native `genai` library, moving away from the previously used OpenAI-compatible interface.
- **Rewritten Anthropic Integration:** The integration with Anthropic has been entirely re-engineered to utilize the official Python module. This change enables Claude models to fully access the bot's comprehensive toolset and benefits from native caching mechanisms.
- **Reworked Base Chatbot Prompt:** The fundamental system prompt used to initialize the chatbot's personality and guidelines has undergone a significant overhaul for improved performance and clarity.
- **Updated Existing AI Provider Integrations:** Updated integrations for Alibaba, Grok, and OpenAI to align with their latest APIs and introduce new features.
- **Revised Tool Initialization and Web Search:** The process for initializing tools has been overhauled, and the web search tool (`web_search.py`) specifically refactored for improved performance and reliability.
- **Project Configuration and Dependencies Refresh:** Updated `pyproject.toml`, `Taskfile.yml`, and managed dependencies (`poetry.lock`, `requirements-dev.txt`, `requirements.txt`).

### Removed
- **Unnecessary Dependencies and Configurations:** Cleaned up unneeded dependencies and configurations as part of the overall project restructuring.

### Chore
- **GitHub Actions Workflows Refinements:** Added new steps to GitHub Actions workflows (`.github/workflows/main.yml`) and updated the GitHub setup action (`.github/actions/setup/action.yml`) for optimized CI/CD processes.
- **ARMv7 Dockerfile Experimentation:** Included `armv7.Dockerfile` as part of ongoing experimental efforts to restore ARMv7 platform support. This work is still in progress.
- **Linter and Test Configuration Fixes:** Addressed issues within linter configurations and made improvements to test setup for greater accuracy and coverage.
- **New Comprehensive Test Suites:** Introduced new test files covering database interactions (`test_database.py`), file editor functionalities (`test_file_editor.py`), and model integrity (`test_models.py`).


## [1.2.1] - 2025-04-21

### Added
- Optional heartbeat functionality. The bot can now periodically fetch a specified URL (e.g., a healthchecks.io endpoint or a custom monitoring system endpoint) to signal that it is operational.


## [1.2.0] - 2025-04-20

### Added
- Implemented **initial** support for LLM function calling (tool use).
- Added initial tools available for LLM invocation:
  - `web_search`
  - `search_news`
  - `read_web_page`
  - `get_current_datetime`
- Integrated Vulture for dead code detection in the development workflow.

### Changed
- Updated core project dependencies to their latest compatible versions.
- Replaced Flake8, isort, and Black with Ruff for linting and formatting, streamlining the development toolchain.
- Updated GitHub Actions CI workflow (`Quality Gate`) to use Ruff and Vulture, and optimized setup steps.

### Fixed
- Addressed issues with Telegram Markdown rendering when LLM responses were split into multiple messages.

## [1.1.0] - 2025-04-13

### Added
- Integrated support for new LLM providers:
  - `MoonshotAI (Kimi)`
  - `Cloudflare` (available in the private mode only: temporary unavailable in the public mode)

### Changed
- Project dependencies updated.

## [1.0.0] - 2025-04-01

### Added
- Integrated support for new LLM providers:
  - `Alibaba (Qwen)`
  - `Deepseek`
  - `xAI (Grok)`
  - `Google (Gemini)`
- Enabled image generation using models from various supported providers, extending beyond the default `DALL-E`.
- Implemented the `/image_model` command, allowing users to select their preferred model for image generation tasks.

### Changed
- Redesigned and improved the user interface and workflow for setting up provider tokens in the public mode.
- Extensively refactored the core application codebase to significantly simplify the future integration of new LLM providers.
- Updated project dependencies to their latest compatible versions.
- Thoroughly rewrote the `README.md` file to accurately reflect the current features and structure.
- Updated provided examples to align them with the latest codebase and functionalities.

### Fixed
- Addressed issues related to inconsistent or incorrect Markdown rendering in responses from Language Models, significantly reducing the occurrence of these problems.


## [0.10.0] - 2025-02-06

### Changed
- Dependencies updated: `openai` module updated to `1.61.1`.


## [0.9.0] - 2025-02-06

### Added
- OpenAI O1/O3 models support.

### Changed
- Mistral AI models selection updated.
- Anthropic available models list updated. 



## [0.8.2] - 2024-08-01

### Fixed
- Data validation error after Mistral AI updated their `GET /models` API signature.

### Changed
- Project dependencies updated.


## [0.8.1] - 2024-07-11

### Added
- `Claude 3.5 Sonnet` model support.

## [0.8.0] - 2024-06-05

### Fixed
- A bug, when after the first image generation, all subsequent messages were treated as image generation prompts.

## [0.7.0] - 2024-05-24

### Added
- Anthropic provider support.
- MistralAI provider support.
- Optional `HIDE_MODELS` setting to hide `models` option from the bot menu.
- Optional `HIDE_IMAGINE` setting to hide `imagine` option from the bot menu.

### Changed
- Project structure significantly refactored.
- README updated: environment variables description significantly improved.


## [0.6.2] - 2024-02-18

### Added
- Optional `MODELS_WHITELIST` setting to limit the number of available GPT models.

## [0.6.1] - 2024-01-12

### Added
- Optional `IMAGE_GENERATIONS_LIMIT` setting to limit the number of images a user can generate within a month using 
DALL-E (to avoid excessive spending). The default value is 0, which means the feature is turned off, and no limits are 
applied.
- Optional `IMAGE_GENERATIONS_WHITELIST` setting for a list of user IDs exempt from image generation limits.


## [0.6.0] - 2023-11-28

### Fixed
- Fixed an issue preventing the bots' use with local data storage
- Resolved a problem where the bot would lose asynchronous tasks during long server response times from OpenAI
- Fixed a bug that sometimes prevented sending images to users

### Changed
- Updated key project dependencies and adapted code for integration with the latest version of the OpenAI library
- Significantly improved the informativeness of logs
- Implemented the ability to send large responses across multiple messages
- Added the option to select the model for generating images (currently only through bot setting using environment variables)
- Refactored the code
- Base Docker image updated to `python:3.11-alpine`
- `REDIS_PASSWORD` environment variable is deprecated.


## [0.5.3] - 2023-06-20

### Changed

- Project dependencies updated.  

## [0.5.2] - 2023-04-15

### Fixed

- A bug when users whitelist couldn't work with the telegram user ID.  


## [0.5.1] - 2023-04-15

### Added

- `LOG_PROMPT_DATA` setting. If true, the application will log user's prompts and GPT answers. The default value is `False`.

### Changed

- The log of disallowed user actions now includes the user's ID.

## [0.5.0] - 2023-04-15

### Fixed

- A bug when the bot didn't answer messages quoting the bot's ones.
- A bug when the bot checked the user whitelist using the user's username, not the user's name.
- A bug when setting the application via `.env` was impossible.

### Added

- Now it's possible to connect to Redis via password (using the `REDIS_PASSWORD` setting).

### Changed
- Settings refactored.

## [0.4.0] - 2023-04-10

### Added

- "Public mode" - now the bot can run without a "master" OpenAI API Key, and every user will have to provide their own.

### Changed

- Codebase has been refactored.

## [0.3.4] - 2023-04-08

### Added

- `ANSWER_DIRECT_MESSAGES_ONLY` setting: if it True the bot in group chats will respond only to messages, containing its name (see the `BOT_NAME` setting) 

## [0.3.3] - 2023-04-07

### Changed

- Dependencies updated, redundant modules removed.
- Now, if the telegram bot raises the markdown parse error and the GPT answer contains some code, such an answer will be additionally provided to the user in the MD file as the attachment.
- Readme updated.

## [0.3.2] - 2023-04-04

### Fixed

- An error when user without Telegram username could not interact with the bot.

### Changed

- Now the `USERS_WHITELIST` setting can also contain telegram user IDs.
- Logging additionally set up.

## [0.3.1] - 2023-04-03

### Added

- Setting `ALLOW_BOTS` with `False` default value. If `ALLOW_BOTS` is set to `False`, it means that other bot requests will be ignored, and only non-bot requests will be processed. 

## [0.3.0] - 2023-04-02

### Fixed

- A bug in local data storage management when the conversation summarizing function didn't clean the summarized history.

### Changed

- Now the `/menu` option displays all the available GPT-based models.
- A `GPT4_ENABLED` setting (default: `True`) allows administrators to exclude GPT-4 models from the available choices in /menu option. The reason for this is that the GPT-4 token is 15 times more expensive than the GPT-3 one.
- A `GPT4_WHITELIST` setting, that allow andinisstrator to specify users for whom the restriction on using the GPT-4 model does not apply.
- Now, during the entire waiting time for a response, the user sees the bot's activity such as "typing..." or "uploading photo...".
- Code slightly refactored.
- README updated.

## [0.2.1] - 2023-03-31

### Added

- Added the ability to switch between GPT-3 and GPT-4 models.

## [0.2.0] - 2023-03-30

### Added

- Added the ability to configure many application properties through environment variables.
- Redis support implemented.
- Added README and this changelog.
- Added examples of running the application.

### Changed

- Now each request is processed in a separate thread, greatly speeding up the bot's performance with multiple users simultaneously.
- Saving session management between application restarts (by using Redis or local storage).
- Project dependencies updated.
- Code refactored.

## [0.1.0] - 2023-03-05

### Added

- Basic functionality.
- Session management.
- Dockerfile.
- Flake8 and Mypy setups.
- GitHub Action for linters.

[Unreleased]: https://github.com/s-nagaev/chibi/compare/v1.6.1...HEAD
[1.6.1]: https://github.com/s-nagaev/chibi/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/s-nagaev/chibi/compare/v1.5.2...v1.6.0
[1.5.2]: https://github.com/s-nagaev/chibi/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/s-nagaev/chibi/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/s-nagaev/chibi/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/s-nagaev/chibi/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/s-nagaev/chibi/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/s-nagaev/chibi/compare/v1.2.0...v1.3.1
[1.3.0]: https://github.com/s-nagaev/chibi/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/s-nagaev/chibi/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/s-nagaev/chibi/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/s-nagaev/chibi/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/s-nagaev/chibi/compare/v0.10.0...v1.0.0
[0.10.0]: https://github.com/s-nagaev/chibi/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/s-nagaev/chibi/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/s-nagaev/chibi/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/s-nagaev/chibi/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/s-nagaev/chibi/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/s-nagaev/chibi/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/s-nagaev/chibi/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/s-nagaev/chibi/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/s-nagaev/chibi/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/s-nagaev/chibi/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/s-nagaev/chibi/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/s-nagaev/chibi/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/s-nagaev/chibi/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/s-nagaev/chibi/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/s-nagaev/chibi/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/s-nagaev/chibi/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/s-nagaev/chibi/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/s-nagaev/chibi/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/s-nagaev/chibi/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/s-nagaev/chibi/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/s-nagaev/chibi/tree/v0.2.0