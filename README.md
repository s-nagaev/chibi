<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="logo"></h1>

[![Build](https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg)](https://github.com/s-nagaev/chibi/actions/workflows/build.yml)
[![docker hub](https://img.shields.io/docker/pulls/pysergio/chibi)](https://hub.docker.com/r/pysergio/chibi)
[![docker image arch](https://img.shields.io/badge/docker%20image%20arch-amd64%20%7C%20arm64%20%7C%20armv7-informational)](https://hub.docker.com/r/pysergio/chibi/tags)
[![docker image size](https://img.shields.io/docker/image-size/pysergio/chibi/latest)](https://hub.docker.com/r/pysergio/chibi/tags)
![license](https://img.shields.io/github/license/s-nagaev/chibi)


Chibi is a Python-based Telegram chatbot that allows users to interact with the powerful ChatGPT and DALL-E neural networks. The bot is asynchronous, providing fast response times and serving multiple users simultaneously without blocking each other's requests. Chibi supports session management, enabling ChatGPT to remember the user's conversation history. The conversation history is stored for a configurable duration and is preserved even if the bot is restarted.

[Docker Hub](https://hub.docker.com/r/pysergio/chibi)

## Supported platforms

- linux/amd64
- linux/arm64
- linux/arm/v7 *(Yeah! Your Raspberry Pi is supported!)*

## Features

- Optional "Public Mode", when no master `OPENAI_API_KEY` is required, but every user will be asked to provide one own while interacting with the bot.
- Switch between GPT models (including GPT-4) at any time.
- Session management for ChatGPT (by storing data locally or using Redis).
- Request DALL-E to create an image from the same chat.
- User and group whitelists.
- Asynchronous code for quick and non-blocking performance.
- Extensive configuration options through environment variables.
- Docker image for easy deployment.
- Cross-platform support (amd64, arm64, armv7).
- MIT License.

## Try it!

- You can try a DEMO version of the Chibi bot by using [@ChibiDemoBot](https://t.me/ChibiDemoBot). An OpenAI API Key is not required, but there are only 4 free requests available.
- You can also use a public version of the Chibi bot by using [@ChibiPublicBot](https://t.me/ChibiPublicBot). An OpenAI Key is required.


## System Requirements

The application is not resource-demanding at all. It works perfectly on both Raspberry Pi 3A with 512MB RAM and the cheapest AWS EC2 Instance `t4g.nano` (2 arm64 cores, 512MB RAM), while being able to serve many people simultaneously. I would say that if your machine belongs to a supported architecture and can run Docker, the application will work.

## Prerequisites

- Docker
- Docker Compose (optional)

## Getting Started

### Using Docker Run

1. Pull the Chibi Docker image:

    ```shell
    docker pull pysergio/chibi:latest
    ```

2. Run the Docker container with the necessary environment variables:

    ```shell
    docker run -d \
      -e TELEGRAM_BOT_TOKEN=<your_telegram_token> \
      -e OPENAI_API_KEY=<your_chatgpt_api_key> \
      -v <path_to_local_data_directory>:/app/data \
      --name chibi \
      pysergio/chibi:latest
    ```

   Replace `<your_telegram_token>`, `<your_chatgpt_api_key>`, and `<path_to_local_data_directory>` with appropriate values.

### Using Docker Compose

1. Create a `docker-compose.yml` file with the following contents:

   ```yaml
      version: '3'

      services:
        chibi:
         restart: unless-stopped
         image: pysergio/chibi:latest
         environment:
           OPENAI_API_KEY: <your_chatgpt_api_key>
           TELEGRAM_BOT_TOKEN: <your_telegram_token>
         volumes:
           - chibi_data:/app/data
      
      volumes:
        chibi_data:
   ```

   Replace `<your_telegram_token>` and `<your_chatgpt_api_key>` with appropriate values.

2. Run the Docker container:

   ```shell
   docker-compose up -d
   ```

Please, visit the [examples](examples) directory of the current repository for more examples.

## Configuration

You can configure Chibi using the following environment variables:

| Variable                      | Description                                                                                                                                                       | Required | Default Value                                                                    |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|----------------------------------------------------------------------------------|
| TELEGRAM_BOT_TOKEN            | Your Telegram bot token                                                                                                                                           | Yes      |                                                                                  |
| OPENAI_API_KEY                | OpenAI API key. If not provided, the user will be asked to provide it while interacting with bot.                                                                 | No       |                                                                                  |
| ALLOW_BOTS                    | Allow other bots to interact with Chibi                                                                                                                           | No       | `false`                                                                          |
| ANSWER_DIRECT_MESSAGES_ONLY   | If True the bot in group chats will respond only to messages, containing its name (see the `BOT_NAME` setting)                                                    | NO       | `true`                                                                           |
| ASSISTANT_PROMPT              | Initial assistant prompt for OpenAI Client                                                                                                                        | No       | `You're helpful and friendly assistant. Your name is Chibi`                      |
| BOT_NAME                      | Name of the bot                                                                                                                                                   | No       | `Chibi`                                                                          |
| DALL_E_MODEL                  | DALL-E model. Available values: `dall-e-2`, `dall-e-3`                                                                                                            | No       | `dall-e-3`                                                                       |
| GPT4_ENABLED                  | Allow users (every one) to use GPT-4 model if it is available                                                                                                     | No       | `false`                                                                          |
| GPT4_WHITELIST                | Comma-separated list of user ids and/or usernames for whom the restriction on using the GPT-4 model does not apply, i.e. "@YourName,@YourFriendName,@YourCatName" | No       |                                                                                  |
| GROUPS_WHITELIST              | Comma-separated list of whitelisted group IDs, i.e `"-799999999,-788888888"`                                                                                      | No       |                                                                                  |
| IMAGE_GENERATIONS_LIMIT       | The number of images that a typical non-whitelist user can generate per month.`0` - no limits.                                                                    | No       | `0`                                                                              |
| IMAGE_GENERATIONS_WHITELIST   | Comma-separated list of user ids (not usernames) for whom the images generation limit does not apply                                                              | No       |                                                                                  |
| IMAGE_QUALITY                 | Quality of the image generated by DALL-E. Available values: `standard`, `hd`                                                                                      | No       | `standard`                                                                       |
| IMAGE_SIZE                    | Size of the image generated by DALL-E. Available values: `256x256`, `512x512`, `1024x1024`, `1792x1024`, `1024x1792`                                              | No       | `1024x1024`                                                                      |
| LOG_PROMPT_DATA               | Log user's prompts and GPT answers                                                                                                                                | No       | `false`                                                                          |
| MAX_CONVERSATION_AGE_MINUTES  | Maximum age of conversations (in minutes)                                                                                                                         | No       | `60`                                                                             |
| MAX_HISTORY_TOKENS            | Maximum number of tokens in conversation history                                                                                                                  | No       | `1800`                                                                           |
| MAX_TOKENS                    | Maximum tokens for the OpenAI Client                                                                                                                              | No       | `1000`                                                                           |
| MESSAGE_FOR_DISALLOWED_USERS  | Message to show disallowed users                                                                                                                                  | No       | `You're not allowed to interact with me, sorry. Contact my owner first, please.` |
| MODEL_DEFAULT                 | Default OpenAI model to use                                                                                                                                       | No       | `gpt-3.5-turbo`                                                                  |
| MODELS_WHITELIST              | Comma-separated list of allowed models, i.e.: "gpt-4,gpt-3.5-turbo"                                                                                               | No       |                                                                                  |
| OPENAI_FREQUENCY_PENALTY      | OpenAI frequency penalty                                                                                                                                          | No       | `0`                                                                              |
| OPENAI_IMAGE_N_CHOICES        | Number of choices for image generation in DALL-E                                                                                                                  | No       | `4`                                                                              |
| OPENAI_PRESENCE_PENALTY       | OpenAI presence penalty                                                                                                                                           | No       | `0`                                                                              |
| OPENAI_TEMPERATURE            | OpenAI temperature for response generation                                                                                                                        | No       | `0.5`                                                                            |
| PROXY                         | Proxy settings for your application                                                                                                                               | No       |                                                                                  |
| REDIS                         | Redis connection string, i.e. `redis://localhost` or `redis://:my-secret-password@127.0.0.1:6379/1`                                                               | No       |                                                                                  |
| REDIS_PASSWORD                | **DEPRECATED! Please, use `REDIS` instead.** Redis password (optional)                                                                                            | No       |                                                                                  |
| TIMEOUT                       | Timeout (in seconds) for processing requests                                                                                                                      | No       | `120`                                                                            |
| USERS_WHITELIST               | Comma-separated list of whitelisted usernames, i.e. `"@YourName,@YourFriendName,@YourCatName"`                                                                    | No       |                                                                                  |


Please, visit the [examples](examples) directory for the example of `.env`-file.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/s-nagaev/chibi/tags).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
