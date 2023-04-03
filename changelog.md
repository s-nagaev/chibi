# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2023-04-03

### Added

- Setting `ALLOW_BOTS` with `True` default value. If `ALLOW_BOTS` is set to `True`, it means that other bot requests will be ignored, and only non-bot requests will be processed. 

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
