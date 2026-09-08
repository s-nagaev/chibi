<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi Logo"></h1>

<p align="center">
  <strong>您的数字伙伴。不仅是工具，更是合作伙伴。</strong><br/>
  <span>自托管的异步 Telegram 机器人，可协调多个 AI 提供商、工具和子代理，完成实际工作。</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="PyPI 下载"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="架构"></a>
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
- **智谱AI**（GLM 系列模型）
- **Novita AI**
- **小米**（MiMo 系列模型）
- **Melious**
- **Cheaper Inference**
- **DeepInfra**
- **Together AI**
- **Fireworks AI**
- **Nebius**（Token Factory）
- **Baseten**
- **SambaNova**
- **SiliconFlow**
- **Parasail**
- **Featherless**
- **OpenRouter**（统一访问多种模型）
- **Cloudflare Workers AI**（众多开源模型）

### OpenAI 兼容端点（自托管/本地）

- **Ollama**
- **vLLM**
- **LM Studio**
- **任意** OpenAI 兼容 API

### 多模态提供商（可选）

- **图像**：Google（Imagen、Nano Banana）、OpenAI（GPT Image）、阿里云（Qwen Image、Wan）、xAI（Grok Imagine）、智谱AI（GLM Image）、MiniMax、Cheaper Inference（Nano Banana Pro）
- **音乐**：Suno
- **语音**：ElevenLabs、MiniMax、OpenAI（GPT-4o Transcribe / TTS）

> 具体模型可用性取决于您配置的提供商密钥和启用的功能。

---

## 🚀 快速开始 (pip)

通过 pip 安装 Chibi 并作为命令行应用程序运行：

```bash
# 安装软件包
pip install chibi-bot

# 设置代理（添加 API 密钥、更新设置等）
chibi config

# 启动机器人
chibi start
```

机器人将以后台服务运行。使用 CLI 命令进行管理。

### CLI 命令

| 命令 | 描述 |
|---------|-------------|
| `chibi start` | 启动机器人作为后台服务 |
| `chibi stop` | 停止运行的机器人 |
| `chibi restart` | 重启机器人 |
| `chibi config` | 生成或编辑配置 |
| `chibi logs` | 查看机器人日志 |

---

## 本地客户端前端

Chibi 可通过版本化的本地 JSONL 协议为本地客户端提供服务：

```bash
chibi stdio --tui
chibi stdio --vscode
chibi stdio --pycharm
chibi stdio --neovim
```

该命令由客户端前端启动，不用于交互式使用。Chibi 负责维护 `v1` 协议并支持协商相同版本的兼容客户端。stdio 进程仅向 stdout 写入协议帧，向 stderr 写入诊断信息。现有的 Chibi 命令、工具、权限和审核行为仍具有权威性；客户端不添加额外的工具策略层。

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

## 🗝️ 获取 API 密钥

每个提供商都需要自己的 API 密钥。以下是直接链接：

**主要提供商：**
- **OpenAI** (GPT, DALL·E): [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic** (Claude): [console.anthropic.com](https://console.anthropic.com/)
- **Google** (Gemini, Nano Banana, Imagen, Voice): [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/)
- **xAI** (Grok): [console.x.ai](https://console.x.ai/)
- **Alibaba** (Qwen, Wan): [modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com?tab=playground#/api-key)
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/)
- **Moonshot** (Kimi): [platform.moonshot.cn](https://platform.moonshot.cn/)
- **MiniMax** (Voice, MiniMax-M2.x): [minimax.io](https://www.minimax.io)
- **智谱AI** (GLM, CogView): [z.ai/manage-apikey/apikey-list](https://z.ai/manage-apikey/apikey-list)
- **OpenRouter** (统一访问多种模型): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**创意工具：**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📚 **完整指南及设置说明：** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

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
- **技能**：可复用的指令模块，代理可按需加载到系统提示中（`load_builtin_skill`）

### 🔌 通过 MCP（模型上下文协议）扩展
将 Chibi 连接到外部工具和服务（或构建您自己的）：

- GitHub（拉取请求、问题、代码审查）
- 浏览器自动化
- Docker / 云服务
- 数据库
- 创意工具（Blender、Figma）

如果某个工具可以通过 MCP 暴露，Chibi 就能学会使用它。

### 🎨 丰富的内容生成
- **图像**：Nano Banana、Imagen、Qwen、Wan、GPT Image、Grok Imagine、GLM Image
- **音乐**：Suno（包括自定义模式：风格/歌词/人声）
- **语音**：转录 + 文本转语音（ElevenLabs、OpenAI、MiniMax）

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

### `MAX_HISTORY_TOKENS` — 上下文摘要阈值（默认值变更）

`MAX_HISTORY_TOKENS` 是 Chibi 自动摘要对话以保持上下文可控的阈值。其**语义已变更**：现在与**真实的、提供商报告的提示词 token 数量**进行比较（整个发出的请求：系统提示 + 已激活的技能 + 工具模式 + 工具调用参数 + 每条消息的结构开销 + 对话内容），而非旧的仅衡量对话 `content` + `role` 的启发式方法。对于相同对话，真实数字约为旧估计的 **4.8 倍**。

- **默认值已重设**：`64000` → `100000`。新值保护最小常用上下文窗口（128k tokens）：`100000` 约为 128k 窗口的 78%（因此摘要会在 128k 模型溢出前触发），约为 200k 窗口的 50%（留有充裕空间）。旧的 `64000` 仅是历史记录的估计，从未在真正溢出前被触发，因为该估计对西里尔文的计算少约 2 倍，且排除了固定的每轮开销（系统提示约 3.7k、工具模式约 6.8k、已激活技能约 5.9k、`user_info` 约 0.75–3k）——因此 128k 模型在约 128k 真实 token 时溢出，而启发式仍远低于 64k。
- **迁移**：如果您在 `.env` 中显式设置了 `MAX_HISTORY_TOKENS`，旧值是针对旧的历史估计调整的，现在与一个约为相同对话 4.8 倍的真实数字进行比较。请将其重新调整到 **~100k 量级**（例如 `64000` → `100000`），以便在最小模型的上下文窗口溢出前触发摘要。如果从未设置，新默认值将自动生效。
- 冷启动回退（进程重启后的第一轮，尚无提供商数据缓存时）仍使用旧的启发式，因此摘要功能保持可用。

### `REACTIVE_CONTEXT_RECOVERY` — 上下文溢出后的一次性重试

`REACTIVE_CONTEXT_RECOVERY`（默认：`true`）是补充主动 `MAX_HISTORY_TOKENS` 阈值的反应式安全网。当提供商因类型化 `context_length_exceeded` 错误拒绝请求时，Chibi 会自动摘要对话历史并**恰好重试一次**该轮。如果重试成功，用户将收到正常回复（附带一条简短说明上下文已压缩的注释）。如果重试再次溢出或因任何原因恢复失败，则该轮回退到现有的道歉路径——摘要+重试循环在每个原始轮次中最多只能运行一次。设为 `false` 可禁用反应式恢复并保留旧行为（记录日志 + 道歉，不重试）。

### 工作目录按线程隔离（`WORKING_DIR`）

代理的工作目录——用于终端命令并作为当前工作目录报告给模型——是**按线程隔离的**，与所选 LLM 模型按线程绑定的方式一致。

- **默认值**：来自 `WORKING_DIR` 设置（默认 `~/chibi`），通过遗留用户级值获取；全新部署直接继承该设置。
- **每线程覆盖**：任何线程/对话均可**独立地**通过代理的 `set_working_dir` 工具覆盖其工作目录（仅限 LLM 驱动——没有斜杠命令，需启用 `FILESYSTEM_ACCESS`）。这允许不同线程中的两个代理同时在不同项目上工作而互不干扰。
- **解析顺序**：线程覆盖 → 遗留用户级目录 → `WORKING_DIR` 设置。
- **路径规范化**：设置的值在保存时会被扩展为绝对路径（`~/x` 变为 `/abs/x`）；未修改的默认值保持原始形式。
- **子代理**在某线程中生成时共享该线程的工作目录——相同的有效路径会被注入其系统提示和工具调用中。
- 覆盖**在线程克隆后保留**：`/new_thread_with_current_context` 会将工作目录与消息和模型偏好一起携带。

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