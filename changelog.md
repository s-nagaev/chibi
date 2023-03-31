# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2023-03-31

### Added

- Added the ability to switch between GPT-3 and GPT-4 models.
- Redis support implemented.
- Added README and this changelog.


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
