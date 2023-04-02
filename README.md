<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="logo"></h1>

Chibi is a Python-based Telegram chatbot that allows users to interact with the powerful ChatGPT and DALL-E neural networks. The bot is asynchronous, providing fast response times and serving multiple users simultaneously without blocking each other's requests. Chibi supports session management, enabling ChatGPT to remember the user's conversation history. The conversation history is stored for a configurable duration and is preserved even if the bot is restarted.

[Docker Hub](https://hub.docker.com/r/pysergio/chibi)

## Supported platforms

- linux/amd64
- linux/arm64
- linux/arm/v7 *(Yeah! Your Raspberry Pi is supported!)*

## Features
- Switch between GPT-3 & GPT-4 models any time.
- Asynchronous code for fast and non-blocking performance
- Session management for ChatGPT  (by storing data locally or using Redis)
- Wide range of configuration options through environment variables
- Docker image for easy deployment
- Cross-platform support (amd64, arm64, armv7)
- MIT License

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

| Variable                     | Description                                                                                                                                       | Required | Default Value                                                                    |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------|----------------------------------------------------------------------------------|
| OPENAI_API_KEY               | Your OpenAI API key                                                                                                                               | Yes      |                                                                                  |
| TELEGRAM_BOT_TOKEN           | Your Telegram bot token                                                                                                                           | Yes      |                                                                                  |
| PROXY                        | Proxy settings for your application                                                                                                               | No       |                                                                                  |
| BOT_NAME                     | Name of the bot                                                                                                                                   | No       | "Chibi"                                                                          |
| MAX_CONVERSATION_AGE_MINUTES | Maximum age of conversations (in minutes)                                                                                                         | No       | 60                                                                               |
| MAX_HISTORY_TOKENS           | Maximum number of tokens in conversation history                                                                                                  | No       | 1800                                                                             |
| MESSAGE_FOR_DISALLOWED_USERS | Message to show disallowed users                                                                                                                  | No       | "You're not allowed to interact with me, sorry. Contact my owner first, please." |
| REDIS                        | Redis connection string, i.e. "redis://localhost"                                                                                                 | No       |                                                                                  |
| TIMEOUT                      | Timeout (in seconds) for processing requests                                                                                                      | No       | 15                                                                               |
| ASSISTANT_PROMPT             | Initial assistant prompt for OpenAI Client                                                                                                        | No       | "You're helpful and friendly assistant. Your name is Chibi"                      |
| IMAGE_SIZE                   | Size of the image generated by DALL-E                                                                                                             | No       | "512x512"                                                                        |
| MAX_TOKENS                   | Maximum tokens for the OpenAI Client                                                                                                              | No       | 1000                                                                             |
| MODEL_DEFAULT                | Default OpenAI model to use                                                                                                                       | No       | "gpt-3.5-turbo"                                                                  |
| GPT4_ENABLED                 | Allow users (every one) to use GPT-4 model if it is available                                                                                     | No       | false                                                                            |
| GPT4_WHITELIST               | Comma-separated list of usernames for whom the restriction on using the GPT-4 model does not apply, i.e. "@YourName,@YourFriendName,@YourCatName" | No       |                                                                                  |
| OPENAI_FREQUENCY_PENALTY     | OpenAI frequency penalty                                                                                                                          | No       | 0                                                                                |
| OPENAI_IMAGE_N_CHOICES       | Number of choices for image generation in DALL-E                                                                                                  | No       | 4                                                                                |
| OPENAI_PRESENCE_PENALTY      | OpenAI presence penalty                                                                                                                           | No       | 0                                                                                |
| OPENAI_TEMPERATURE           | OpenAI temperature for response generation                                                                                                        | No       | 0.5                                                                              |
| GROUPS_WHITELIST             | Comma-separated list of whitelisted group IDs, i.e `"-799999999,-788888888"`                                                                      | No       |                                                                                  |
| USERS_WHITELIST              | Comma-separated list of whitelisted usernames, i.e. `"@YourName,@YourFriendName,@YourCatName"`                                                    | No       |                                                                                  |

Please, visit the [examples](examples) directory for the example of `.env`-file.

## The Plan for future versions

- [ ] implement the "Public Mode", when no general `OPENAI_API_KEY` is required, but every user can provide one own;
- [ ] add MongoDB support;
- [ ] add the option to upload answers containing code to the Pastebin-like service.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
