# Chibi

Self-hosted, asynchronous Telegram bot that orchestrates multiple AI providers, tools, and sub-agents to get real work done.

---

**Maintained by:** [Sergio](https://github.com/s-nagaev)

[![Docker Pulls](https://img.shields.io/docker/pulls/pysergio/chibi)](https://hub.docker.com/r/pysergio/chibi)
[![Image Size](https://img.shields.io/docker/image-size/pysergio/chibi/latest)](https://hub.docker.com/r/pysergio/chibi/tags)
[![License](https://img.shields.io/github/license/s-nagaev/chibi)](https://github.com/s-nagaev/chibi/blob/main/LICENSE)
[![Version](https://img.shields.io/docker/v/pysergio/chibi/latest?sort=semver)](https://hub.docker.com/r/pysergio/chibi/tags)

---

## Quick Start

Get Chibi running in under 2 minutes.

**Prerequisites:** Docker, a [Telegram Bot Token](https://t.me/BotFather), and at least one AI provider API key.

```bash
docker run -d \
  --name chibi \
  -v chibi_data:/app/data \
  -e TELEGRAM_BOT_TOKEN=your_bot_token_here \
  -e USERS_WHITELIST=your_telegram_id \
  -e OPENAI_API_KEY=sk-... \
  pysergio/chibi:latest
```

**Verify it works:** Open Telegram, find your bot, and send `/start`.

> **Tip:** Get your Telegram user ID from [@userinfobot](https://t.me/userinfobot).

| I want to…                    | Jump to                                       |
|-------------------------------|-----------------------------------------------|
| Deploy with Docker Compose    | [Docker Compose Example](#docker-compose-example) |
| See all configuration options | [Environment Variables](#environment-variables)   |
| Understand what Chibi can do  | [What is Chibi?](#what-is-chibi)                  |

---

## What is Chibi?

Chibi is a **Telegram-based digital companion** that orchestrates multiple AI providers and tools to deliver outcomes — code changes, research syntheses, media generation, and operational tasks.

Built for the moment you need more than "an AI tool": a **partner** that coordinates models, runs work in the background, and integrates with your systems.

**Key Features:**
- **One interface (Telegram).** Mobile, desktop, web — always with you.
- **Provider-agnostic.** Use the best model for each task, no vendor lock-in.
- **Autonomous execution.** Sub-agents work in parallel; long tasks run asynchronously.
- **Tool-connected.** Filesystem, terminal, and MCP integrations (GitHub, browser, DBs, etc.).
- **Self-hosted.** Your data, your keys, your rules.

**Supported Providers:**
**LLMs:** OpenAI · Anthropic · Google Gemini · DeepSeek · xAI · Mistral AI · Alibaba (Qwen) · Moonshot AI · MiniMax · ZhipuAI · Cloudflare Workers AI
**OpenAI-compatible:** Ollama · vLLM · LM Studio · any OpenAI-compatible API
**Image Generation:** Google (Imagen, Nano Banana) · OpenAI (DALL·E) · Alibaba · xAI · Wan · ZhipuAI · MiniMax
**Music:** Suno · **Voice:** ElevenLabs · MiniMax · OpenAI (Whisper)

---

## Supported Tags and Architectures

### Tags

- [`latest`](https://github.com/s-nagaev/chibi/blob/main/Dockerfile) — most recent stable release
- `vX.Y.Z` (e.g., `v2.5.0`) — specific version, pinned
- `vX.Y` (e.g., `v2.5`) — latest patch within a minor version

### Architectures

| Architecture | Platform           | Examples                              |
|--------------|--------------------|---------------------------------------|
| `amd64`      | x86_64             | Intel/AMD servers, most cloud VMs     |
| `arm64/v8`   | ARM 64-bit         | Apple Silicon, AWS Graviton, RPi 4+   |

All tags are multi-arch manifests — Docker automatically pulls the correct image for your platform.

---

## Docker Compose Example

Recommended for production deployments. Redis enables persistent conversation history across container restarts.

Create a `compose.yaml`:

```yaml
services:
  chibi:
    image: pysergio/chibi:latest
    container_name: chibi
    restart: unless-stopped
    depends_on:
      - redis
    volumes:
      - chibi_data:/app/data
      - ./skills:/app/skills:ro    # Custom skills (optional, read-only)
    environment:
      # Required
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}
      USERS_WHITELIST: ${USERS_WHITELIST}
      REDIS_URL: redis://redis:6379/0

      # At least one AI provider
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      # ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      # GEMINI_API_KEY: ${GEMINI_API_KEY}

      # Optional: Agent mode
      # FILESYSTEM_ACCESS: "true"
      # ENABLE_MCP_STDIO: "true"
    env_file:
      - .env

  redis:
    image: redis:alpine
    container_name: chibi-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data

volumes:
  chibi_data:
  redis_data:
```

Create a `.env` file alongside it:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
USERS_WHITELIST=your_telegram_id
OPENAI_API_KEY=sk-...
```

Then start:

```bash
docker compose up -d
```

> **Important:** Never hardcode API keys in `compose.yaml`. Use the `.env` file and keep it out of version control.

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Telegram Bot Token from [@BotFather](https://t.me/BotFather) | `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz` |
| `USERS_WHITELIST` | Comma-separated Telegram user IDs or usernames | `123456789,@username` |

### AI Providers

At least one provider API key is required.

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `DEEPSEEK_API_KEY` | DeepSeek |

For the full list of environment variables (50+), see the [Configuration Reference](https://chibi.bot/configuration).

---

## Volumes and Persistence

| Mount Point | Purpose | Required? |
|-------------|---------|-----------|
| `/app/data` | Conversations, user data, settings | **Yes** — use a named volume |
| `/app/skills` | Custom skills directory | No — mount read-only if used |

```bash
# Backup data volume
docker run --rm -v chibi_data:/data -v $(pwd):/backup alpine tar czf /backup/chibi-backup.tar.gz /data
```

---

## Security

Chibi ships with secure defaults:

- Runs as **non-root** user (`chibi`)
- **Filesystem access disabled** by default
- Only **whitelisted users** can interact with the bot
- Delegation enabled but restrictable via `TOOLS_WHITELIST`

**Best Practices:**

1. Always set `USERS_WHITELIST` — never run without it
2. Keep `FILESYSTEM_ACCESS=false` unless you need agent mode
3. Store API keys in `.env` files or Docker secrets, never in compose files
4. Use `TOOLS_WHITELIST` to restrict available tools in production

**Read-only filesystem** (enhanced security):

```bash
docker run --read-only \
  -v chibi_data:/app/data \
  -v /tmp:/tmp \
  -v /var/run:/var/run \
  pysergio/chibi:latest
```

> The `/app/data` volume must remain read-write for conversation history and user data.

---

## Troubleshooting

**Bot not starting?**

1. Check logs: `docker logs chibi`
2. Verify `TELEGRAM_BOT_TOKEN` is correct
3. Ensure `USERS_WHITELIST` contains your Telegram ID

**Cannot connect to a provider?**

1. Verify the API key is set correctly
2. Check network connectivity from the container
3. Review API key permissions in the provider's dashboard

**Data not persisting?**

1. Always use named volumes: `-v chibi_data:/app/data`
2. Check volume mount permissions

**Enable debug logging:**

```yaml
environment:
  LOG_PROMPT_DATA: "true"
```

---

## Advanced Usage

### Custom Dockerfile

```dockerfile
FROM pysergio/chibi:latest

# Install additional dependencies
RUN pip install some-package

# Add custom skills
COPY skills/ /app/skills/
```

### Model Context Protocol (MCP)

Chibi connects to MCP servers **dynamically at runtime** — there is no static configuration file to mount. The AI agent uses built-in tools to establish and manage MCP connections on demand:

- **`initialize_stdio_mcp_server`** — connect to an MCP server via stdio (e.g., a local CLI tool).
- **`initialize_sse_mcp_server`** — connect to an MCP server via SSE (e.g., a remote HTTP endpoint).
- **`deinitialize_mcp_server`** — disconnect from a previously connected server.

Once connected, the MCP server's tools are automatically registered and become available to the agent for the duration of the session.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MCP_SSE` | `true` | Allow the agent to connect to MCP servers via SSE |
| `ENABLE_MCP_STDIO` | `false` | Allow the agent to connect to MCP servers via stdio |

> **Note:** `ENABLE_MCP_STDIO` is `false` by default because stdio-based servers execute local processes inside the container. Enable it only when you understand the security implications.

#### Secure Secret Handling

When connecting to MCP servers that require sensitive credentials (API tokens, passwords, etc.), **never pass secrets directly in the `env` parameter** — doing so sends them to your AI provider's API.

Instead, use the **`secret_envs`** parameter:

1. Define sensitive variables in your `.env` file or `compose.yaml`:
   ```bash
   GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_token_here
   ```

2. When the AI connects to an MCP server, it provides only the **variable names** via `secret_envs`:
   ```json
   {
     "server_name": "github",
     "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-github"],
     "secret_envs": ["GITHUB_PERSONAL_ACCESS_TOKEN"]
   }
   ```

3. Chibi automatically pulls the actual values from its own environment and passes them to the MCP server — **the AI provider never sees the token values**.

**Comparison:**

| Method | Security | Use Case |
|--------|----------|----------|
| `env: {"TOKEN": "ghp_..."}` | ❌ Insecure — token sent to AI provider | Never use for secrets |
| `secret_envs: ["TOKEN"]` | ✅ Secure — token stays in Chibi's environment | Always use for sensitive data |

**Example usage (from Telegram):**

Simply ask the bot to connect to an MCP server using `secret_envs`:

```
Connect to the GitHub MCP server:
- server name: "github"
- command: npx
- args: ["-y", "@modelcontextprotocol/server-github"]
- secret_envs: ["GITHUB_PERSONAL_ACCESS_TOKEN"]
```

No volume mounts or config files needed — MCP is fully managed through the conversation, with secrets kept secure.

---

## Updating

```bash
docker pull pysergio/chibi:latest
docker compose up -d
```

> **Always back up your data volume before updating** (see [Volumes and Persistence](#volumes-and-persistence)).

---

## License

MIT License — see [LICENSE](https://github.com/s-nagaev/chibi/blob/main/LICENSE).

---

## Resources

- **Documentation:** https://chibi.bot
- **GitHub:** https://github.com/s-nagaev/chibi
- **Configuration Reference:** https://chibi.bot/configuration
- **Installation Guide:** https://chibi.bot/installation
- **Issues & Support:** https://github.com/s-nagaev/chibi/issues
