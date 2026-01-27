<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi Logo"></h1>

<p align="center">
  <strong>您的数字伙伴。不仅是工具，更是合作伙伴。</strong><br/>
  <span>自托管的异步 Telegram 机器人，可协调多个 AI 提供商、工具 and 子代理，完成实际工作。</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/docker%20image%20arch-arm64%20%7C%20amd64-informational" alt="架构"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="授权"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="文档"></a>
</p>

<p align="center">
  <strong>🌍 Read this in other languages:</strong><br/>
  <a href="../README.md">English</a> •
  <a href="README.es.md">Español</a> •
  <a href="README.pt-BR.md">Português (Brasil)</a> •
  <a href="README.uk.md">Українська</a> •
  <a href="README.id.md">Bahasa Indonesia</a> •
  <a href="README.tr.md">Türkçe</a> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <strong>简体中文</strong>
</p>
---

Chibi 专为那些意识到自己需要的不仅仅是“一个 AI 工具”的时刻而打造。您需要一个能够协调模型、在后台运行工作并集成到您的系统中的**合作伙伴**——无需您时刻监督提示词。

**Chibi** 是一个异步、自托管的**基于 Telegram 的数字伙伴**，可协调多个 AI 提供商和工具，交付实际成果：代码变更、研究综述、媒体生成和运营任务。

---

## 为什么选择 Chibi

- **单一界面（Telegram）**。移动端/桌面端/网页端，始终伴您左右。
- **提供商无关**。为每个任务使用最佳模型——无需被供应商锁定。
- **自主执行**。子代理并行工作；长时间任务异步运行。
- **工具连接**。文件系统 + 终端 + MCP 集成（GitHub、浏览器、数据库等）。
- **自托管**。您的数据，您的密钥，您的规则。

---

## 支持的 AI 提供商（和端点）

Chibi 在单一对话中支持多个提供商。添加一个或多个密钥——Chibi 可以按任务路由。

### LLM 提供商

- **OpenAI**（GPT 系列模型）
- **Anthropic**（Claude）
- **Google**（Gemini）
- **DeepSeek**
- **阿里云**（通义千问/Qwen）
- **xAI**（Grok）
- **Mistral AI**
- **月之暗面**（Moonshot AI）
- **MiniMax**
- **Cloudflare Workers AI**（众多开源模型）

### OpenAI 兼容端点（自托管/本地）

- **Ollama**
- **vLLM**
- **LM Studio**
- **任意** OpenAI 兼容 API

### 多模态提供商（可选）

- **图像**：Google（Imagen、Nano Banana）、OpenAI（DALL·E）、阿里云（通义万相）、xAI（Grok Image）、Wan
- **音乐**：Suno
- **语音**：ElevenLabs、MiniMax、OpenAI（Whisper）

> 具体模型可用性取决于您配置的提供商密钥和启用的功能。

---

## 🚀 快速开始（Docker）

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # 必需
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # 或其他提供商
      # 根据需要添加更多 API 密钥
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) 从 [@BotFather](https://t.me/BotFather) 获取机器人令牌

2) 将密钥放入 `.env` 文件

3) 运行：

```bash
docker-compose up -d
```

下一步：
- **安装指南**：https://chibi.bot/installation
- **配置参考**：https://chibi.bot/configuration

---

## 部署后前 5 分钟尝试

将以下内容粘贴到 Telegram 中。

1) **规划 + 执行**
> 请向我提出 3 个问题以明确我的目标，然后提出计划并执行第一步。

2) **并行工作（子代理）**
> 启动 3 个子代理：一个用于研究选项，一个用于起草建议，一个用于列出风险。返回单一决策。

3) **代理模式（工具）**
> 检查项目文件并总结此仓库的功能。然后提出 5 项改进建议并创建检查清单。

4) **后台任务**
> 启动后台任务：收集关于 X 的资料，并在 30 分钟内提供综述。保持更新进度。

---

## Chibi 的独特之处

### 🎭 多提供商编排
Chibi 可以在对话中途切换提供商的同时保持上下文，或为每个步骤选择最佳模型——平衡**成本**、**能力**和**速度**。

### 🤖 自主代理能力
- **递归委派**：启动可自行启动子代理的子代理
- **后台处理**：长时间运行任务异步执行
- **文件系统访问**：读取/写入/搜索/组织文件
- **终端执行**：运行经 LLM 审核的安全命令
- **持久化记忆**：对话历史在重启后仍保留，具备上下文管理和摘要功能

### 🔌 通过 MCP（模型上下文协议）扩展
将 Chibi 连接到外部工具和服务（或构建您自己的）：

- GitHub（拉取请求、问题、代码审查）
- 浏览器自动化
- Docker / 云服务
- 数据库
- 创意工具（Blender、Figma）

如果某个工具可以通过 MCP 暴露，Chibi 就能学会使用它。

### 🎨 丰富的内容生成
- **图像**：Nano Banana、Imagen、通义万相、Wan、DALL·E、Grok
- **音乐**：Suno（包括自定义模式：风格/歌词/人声）
- **语音**：转录 + 文本转语音（ElevenLabs、MiniMax、OpenAI）

---

## 使用场景

**开发者**
```
您：“运行测试并修复问题。我来处理前端。”
Chibi：*启动子代理，执行测试，分析失败原因，提出修复方案*
```

**研究人员**
```
您：“研究量子计算的最新进展。我明天需要一份综述。”
Chibi：*启动多个研究代理，聚合资料来源，交付报告*
```

**创作者**
```
您：“生成一幅赛博朋克城市景观，并创作一首匹配的合成器浪潮音乐。”
Chibi：*生成图像，创作音乐，同时交付*
```

**团队**
```
您：“审查此拉取请求并相应更新文档。”
Chibi：*分析变更，提出改进建议，通过 MCP 更新文档*
```

---

## 隐私、控制和安全

- **自托管**：您的数据保留在您的基础设施上
- **公共模式**：用户可自带 API 密钥（无需共享主密钥）
- **访问控制**：白名单用户/群组/模型
- **存储选项**：本地卷、Redis 或 DynamoDB
- **工具安全**：代理工具可配置；终端执行经过审核且可限制

---

## 文档

- **从这里开始**：https://chibi.bot
- 介绍与理念：https://chibi.bot/introduction
- 安装：https://chibi.bot/installation
- 配置：https://chibi.bot/configuration
- 代理模式：https://chibi.bot/agent-mode
- MCP 指南：https://chibi.bot/guides/mcp
- 支持/故障排除：https://chibi.bot/support

---

## 系统要求

- **最低配置**：树莓派 4 / AWS EC2 t4g.nano（2 vCPU，512MB RAM）
- **架构**：`linux/amd64`，`linux/arm64`
- **依赖**：Docker（可选 Docker Compose）

---

## 贡献

- 问题：https://github.com/s-nagaev/chibi/issues
- 拉取请求：https://github.com/s-nagaev/chibi/pulls
- 讨论：https://github.com/s-nagaev/chibi/discussions

提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)。

---

<p align="center">
  <strong>准备好迎接您的数字伙伴了吗？</strong><br/>
  <a href="https://chibi.bot/start"><strong>开始使用 →</strong></a>
</p>