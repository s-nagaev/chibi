# Chibi

Chibi is a Python-based Telegram chatbot that allows users to interact with the powerful ChatGPT 3.5 and DALL-E neural networks. The bot is asynchronous, providing fast response times and serving multiple users simultaneously without blocking each other's requests. Chibi supports session management, enabling ChatGPT to remember the user's conversation history. The conversation history is stored for a configurable duration and is preserved even if the bot is restarted. The bot offers a wide range of settings through environment variables and can operate in both private and public modes. It is distributed as a Docker image, with support for amd64, arm64, and armv7 platforms (yes, you can run it on a Raspberry Pi!).

[Docker Hub]()

## Features

- Asynchronous code for fast and non-blocking performance
- Session management for ChatGPT
- Wide range of configuration options through environment variables
- Private and public mode support
- Docker image for easy deployment
- Cross-platform support (amd64, arm64, armv7)
- MIT License

## Prerequisites

- Docker
- Docker Compose (optional)
