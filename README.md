<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi Logo"></h1>

<p align="center">
  <strong>Your Digital Companion. Not a Tool. A Partner.</strong><br/>
  <span>Self-hosted, asynchronous Telegram bot that orchestrates multiple AI providers, tools, and sub-agents to get real work done.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/docker%20image%20arch-arm64%20%7C%20amd64-informational" alt="Architectures"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="License"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="Documentation"></a>

<p align="center">
  <strong>🌍 Read this in other languages:</strong><br/>
  <a href="docs/README.es.md">Español</a> •
  <a href="docs/README.pt-BR.md">Português (Brasil)</a> •
  <a href="docs/README.uk.md">Українська</a> •
  <a href="docs/README.id.md">Bahasa Indonesia</a> •
  <a href="docs/README.tr.md">Türkçe</a> •
  <a href="docs/README.ru.md">Русский</a> •
  <a href="docs/README.ja.md">日本語</a> •
  <a href="docs/README.zh-TW.md">繁體中文</a> •
  <a href="docs/README.zh-CN.md">简体中文</a>
</p>

---

Chibi is built for the moment you realize you need more than “an AI tool.” You need a **partner** that can coordinate models, run work in the background, and integrate with your systems - without you babysitting prompts.

**Chibi** is an asynchronous, self-hosted **Telegram-based digital companion** that orchestrates multiple AI providers and tools to deliver outcomes: code changes, research syntheses, media generation, and operational tasks.

---

## Why Chibi

- **One interface (Telegram).** Mobile/desktop/web, always with you.
- **Provider-agnostic.** Use the best model for each task - without vendor lock-in.
- **Autonomous execution.** Sub-agents work in parallel; long tasks run asynchronously.
- **Tool-connected.** Filesystem + terminal + MCP integrations (GitHub, browser, DBs, etc.).
- **Self-hosted.** Your data, your keys, your rules.

---

## Supported AI providers (and endpoints)

Chibi supports multiple providers behind a single conversation. Add one key or many - Chibi can route per task.

### LLM providers

- **OpenAI** (GPT models)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **DeepSeek**
- **Alibaba Cloud** (Qwen)
- **xAI** (Grok)
- **Mistral AI**
- **Moonshot AI**
- **MiniMax**
- **Cloudflare Workers AI** (many open-source models)

### OpenAI-compatible endpoints (self-host / local)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Any** OpenAI-compatible API

### Multimodal providers (optional)

- **Images:** Google (Imagen, Nano Banana), OpenAI (DALL·E), Alibaba (Qwen Image), xAI (Grok Image), Wan
- **Music:** Suno
- **Voice:** ElevenLabs, MiniMax, OpenAI (Whisper)

> Exact model availability depends on your configured provider keys and enabled features.

---

## 🚀 Quick start (Docker)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # Required
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # Or any other provider
      # Add more API keys as needed
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) Get a bot token from [@BotFather](https://t.me/BotFather)

2) Put secrets into `.env`

3) Run:

```bash
docker-compose up -d
```

Next:
- **Installation guide:** https://chibi.bot/installation
- **Configuration reference:** https://chibi.bot/configuration

---

## 🔑 Getting API Keys

Each provider requires its own API key. Here are the direct links:

**Major Providers:**
- **OpenAI** (GPT, DALL·E): [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic** (Claude): [console.anthropic.com](https://console.anthropic.com/)
- **Google** (Gemini, Nano Banana, Imagen): [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/)
- **xAI** (Grok): [console.x.ai](https://console.x.ai/)
- **Alibaba** (Qwen, Wan): [modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com?tab=playground#/api-key)
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/)
- **Moonshot** (Kimi): [platform.moonshot.cn](https://platform.moonshot.cn/)
- **MiniMax** (Voice, MiniMax-M2.x): [minimax.io](https://www.minimax.io)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**Creative Tools:**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **Full guide with setup instructions:** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## Try this in the first 5 minutes

Paste these into Telegram after you deploy.

1) **Planning + execution**
> Ask me 3 questions to clarify my goal, then propose a plan and execute step 1.

2) **Parallel work (sub-agents)**
> Spawn 3 sub-agents: one to research options, one to draft a recommendation, one to list risks. Return a single decision.

3) **Agent mode (tools)**
> Inspect the project files and summarize what this repo does. Then propose 5 improvements and open a checklist.

4) **Background task**
> Start a background task: gather sources on X and deliver a synthesis in 30 minutes. Keep me updated.

---

## What makes Chibi different

### 🎭 Multi-provider orchestration
Chibi can keep context while switching providers mid-thread, or choose the best model per step - balancing **cost**, **capability**, and **speed**.

### 🤖 Autonomous agent capabilities
- **Recursive delegation:** spawn sub-agents that can spawn their own sub-agents
- **Background processing:** long-running tasks execute asynchronously
- **Filesystem access:** read/write/search/organize files
- **Terminal execution:** run commands with LLM-moderated security
- **Persistent memory:** conversation history survives restarts with context management/summarization

### 🔌 Extensible via MCP (Model Context Protocol)
Connect Chibi to external tools and services (or build your own):

- GitHub (PRs, issues, code review)
- Browser automation
- Docker / cloud services
- Databases
- Creative tools (Blender, Figma)

If a tool can be exposed via MCP, Chibi can learn to use it.

### 🎨 Rich content generation
- **Images:** Nano Banana, Imagen, Qwen, Wan, DALL·E, Grok
- **Music:** Suno (including custom mode: style/lyrics/vocals)
- **Voice:** transcription + text-to-speech (ElevenLabs, MiniMax, OpenAI)

---

## Use cases

**Developers**
```
You: “Run the tests and fix what’s broken. I’ll work on the frontend.”
Chibi: *spawns sub-agent, executes tests, analyzes failures, proposes fixes*
```

**Researchers**
```
You: “Research the latest developments in quantum computing. I need a synthesis by tomorrow.”
Chibi: *spawns multiple research agents, aggregates sources, delivers a report*
```

**Creators**
```
You: “Generate a cyberpunk cityscape and compose a synthwave track to match.”
Chibi: *generates an image, creates music, delivers both*
```

**Teams**
```
You: “Review this PR and update the documentation accordingly.”
Chibi: *analyzes changes, suggests improvements, updates docs via MCP*
```

---

## Privacy, control, and safety

- **Self-hosted:** your data stays on your infrastructure
- **Public Mode:** users can bring their own API keys (no shared master key required)
- **Access control:** whitelist users/groups/models
- **Storage options:** local volumes, Redis, or DynamoDB
- **Tool safety:** agent tools are configurable; terminal execution is moderated and can be restricted

---

## Documentation

- **Start here:** https://chibi.bot
- Introduction & philosophy: https://chibi.bot/introduction
- Installation: https://chibi.bot/installation
- Configuration: https://chibi.bot/configuration
- Agent mode: https://chibi.bot/agent-mode
- MCP guide: https://chibi.bot/guides/mcp
- Support / troubleshooting: https://chibi.bot/support

---

## System requirements

- **Minimum:** Raspberry Pi 4 / AWS EC2 t4g.nano (2 vCPU, 512MB RAM)
- **Architectures:** `linux/amd64`, `linux/arm64`
- **Dependencies:** Docker (and optionally Docker Compose)

---

## Contributing

- Issues: https://github.com/s-nagaev/chibi/issues
- PRs: https://github.com/s-nagaev/chibi/pulls
- Discussions: https://github.com/s-nagaev/chibi/discussions

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting.

---

## License

MIT  -  see [LICENSE](LICENSE).

---

<p align="center">
  <strong>Ready to meet your digital companion?</strong><br/>
  <a href="https://chibi.bot/start"><strong>Get Started →</strong></a>
</p>
