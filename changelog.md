# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/s-nagaev/chibi/compare/v0.8.0...HEAD

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