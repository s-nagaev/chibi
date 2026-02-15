<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi Logo"></h1>

<p align="center">
  <strong>您的數位夥伴。不僅是工具，更是夥伴。</strong><br/>
  <span>自託管、非同步的 Telegram 機器人，可協調多個 AI 提供商、工具和子代理，完成實際工作。</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/docker%20image%20arch-arm64%20%7C%20amd64-informational" alt="架構"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="授權"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="文件"></a>
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
  <strong>繁體中文</strong> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

當您意識到需要的不僅僅是「一個 AI 工具」時，Chibi 應運而生。您需要的是一位能夠協調模型、在背景執行工作並與您的系統整合的**夥伴**——無需您時時監督提示詞。

**Chibi** 是一個非同步、自託管的**基於 Telegram 的數位夥伴**，可協調多個 AI 提供商和工具，交付實際成果：程式碼變更、研究彙整、媒體生成和營運任務。

---

## 為什麼選擇 Chibi

- **單一介面（Telegram）**。支援行動裝置/桌面/網頁，隨時隨地陪伴您。
- **提供者無關**。為每項任務選擇最佳模型，無需擔心廠商綁定。
- **自主執行**。子代理平行工作；長時間任務非同步執行。
- **工具整合**。檔案系統 + 終端機 + MCP 整合（GitHub、瀏覽器、資料庫等）。
- **自託管**。您的資料、您的金鑰、您的規則。

---

## 支援的 AI 提供商（及端點）

Chibi 在單一對話中支援多個提供商。可新增單一金鑰或多個金鑰——Chibi 能依任務路由。

### LLM 提供商

- **OpenAI**（GPT 模型）
- **Anthropic**（Claude）
- **Google**（Gemini）
- **DeepSeek**
- **阿里雲**（Qwen）
- **xAI**（Grok）
- **Mistral AI**
- **月之暗面**（Moonshot AI）
- **MiniMax**
- **智譜AI**（GLM 系列模型）
- **Cloudflare Workers AI**（多個開源模型）

### OpenAI 相容端點（自託管 / 本機）

- **Ollama**
- **vLLM**
- **LM Studio**
- **任何** OpenAI 相容 API

### 多模態提供商（選用）

- **影像**：Google（Imagen、Nano Banana）、OpenAI（DALL·E）、阿里雲（Qwen Image）、xAI（Grok Image）、Wan、**智譜AI（CogView）、MiniMax**
- **音樂**：Suno
- **語音**：ElevenLabs、MiniMax、OpenAI（Whisper）

> 實際可用的模型取決於您設定的提供者金鑰和啟用的功能。

---

## 🚀 快速入門 (pip)

透過 pip 安裝 Chibi 並作為命令列應用程式執行：

```bash
# 安裝軟體套件
pip install chibi-bot

# 設定代理（新增 API 金鑰、更新設定等）
chibi config

# 啟動機器人
chibi start
```

機器人將以背景服務運行。使用 CLI 命令進行管理。

### CLI 命令

| 命令 | 描述 |
|---------|-------------|
| `chibi start` | 啟動機器人作為背景服務 |
| `chibi stop` | 停止執行中的機器人 |
| `chibi restart` | 重新啟動機器人 |
| `chibi config` | 產生或編輯組態 |
| `chibi logs` | 查看機器人日誌 |

---

## 🚀 快速入門（Docker）

建立 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # 必填
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # 或任何其他提供者
      # 依需要新增更多 API 金鑰
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) 從 [@BotFather](https://t.me/BotFather) 取得機器人權杖

2) 將機密資訊放入 `.env`

3) 執行：

```bash
docker-compose up -d
```

接下來：
- **安裝指南**：https://chibi.bot/installation
- **設定參考**：https://chibi.bot/configuration

---

## 🗝️ 取得 API 金鑰

每個提供商都需要自己的 API 金鑰。以下是直接連結：

**主要提供商：**
- **OpenAI** (GPT, DALL·E): [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic** (Claude): [console.anthropic.com](https://console.anthropic.com/)
- **Google** (Gemini, Nano Banana, Imagen): [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/)
- **xAI** (Grok): [console.x.ai](https://console.x.ai/)
- **Alibaba** (Qwen, Wan): [modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com?tab=playground#/api-key)
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/)
- **Moonshot** (Kimi): [platform.moonshot.cn](https://platform.moonshot.cn/)
- **MiniMax** (Voice, MiniMax-M2.x): [minimax.io](https://www.minimax.io)
- **智譜AI** (GLM, CogView): [z.ai/manage-apikey/apikey-list](https://z.ai/manage-apikey/apikey-list)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**創意工具：**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📚 **完整指南及設置說明：** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## 部署後前 5 分鐘試試這些

將以下內容貼到 Telegram 中。

1) **規劃 + 執行**
> 請問我 3 個問題以釐清我的目標，然後提出計畫並執行第一步。

2) **平行工作（子代理）**
> 啟動 3 個子代理：一個研究選項，一個起草建議，一個列出風險。返回單一決策。

3) **代理模式（工具）**
> 檢查專案檔案並摘要此儲存庫的功能。然後提出 5 項改進建議並建立檢查清單。

4) **背景任務**
> 啟動背景任務：蒐集關於 X 的資料來源，並在 30 分鐘內提供彙整報告。隨時向我更新進度。

---

## Chibi 的獨特之處

### 🎭 多提供商協調
Chibi 能在對話中途切換提供商時保持上下文，或為每個步驟選擇最佳模型——平衡**成本**、**能力**和**速度**。

### 🤖 自主代理功能
- **遞迴委派**：啟動可再啟動自身子代理的子代理
- **背景處理**：長時間執行的任務非同步執行
- **檔案系統存取**：讀取/寫入/搜尋/整理檔案
- **終端機執行**：執行指令並具備 LLM 調節的安全機制
- **持久記憶**：對話歷史在重啟後仍保留，並具備上下文管理/摘要功能

### 🔌 透過 MCP（模型上下文協定）擴充
將 Chibi 連接到外部工具和服務（或自行建構）：

- GitHub（PR、議題、程式碼審查）
- 瀏覽器自動化
- Docker / 雲端服務
- 資料庫
- 創意工具（Blender、Figma）

只要工具能透過 MCP 公開，Chibi 就能學會使用它。

### 🎨 豐富的內容生成
- **影像**：Nano Banana、Imagen、Qwen、Wan、DALL·E、Grok
- **音樂**：Suno（包含自訂模式：風格/歌詞/人聲）
- **語音**：轉錄 + 文字轉語音（ElevenLabs、MiniMax、OpenAI）

---

## 使用案例

**開發者**
```
您：「執行測試並修復損壞的部分。我會處理前端。」
Chibi：*啟動子代理，執行測試，分析失敗原因，提出修復方案*
```

**研究人員**
```
您：「研究量子運算的最新發展。我明天需要一份彙整報告。」
Chibi：*啟動多個研究代理，彙整資料來源，交付報告*
```

**創作者**
```
您：「生成一幅賽博龐克城市景觀，並創作一首匹配的合成器浪潮曲目。」
Chibi：*生成影像，創作音樂，同時交付兩者*
```

**團隊**
```
您：「審查此 PR 並相應更新文件。」
Chibi：*分析變更，建議改進，透過 MCP 更新文件*
```

---

## 隱私、控制與安全

- **自託管**：您的資料保留在您的基礎設施上
- **公開模式**：使用者可使用自己的 API 金鑰（無需共用主金鑰）
- **存取控制**：白名單使用者/群組/模型
- **儲存選項**：本機磁碟區、Redis 或 DynamoDB
- **工具安全**：代理工具可設定；終端機執行經過調節且可限制

---

## 文件

- **從這裡開始**：https://chibi.bot
- 介紹與理念：https://chibi.bot/introduction
- 安裝：https://chibi.bot/installation
- 設定：https://chibi.bot/configuration
- 代理模式：https://chibi.bot/agent-mode
- MCP 指南：https://chibi.bot/guides/mcp
- 支援 / 疑難排解：https://chibi.bot/support

---

## 系統需求

- **最低需求**：Raspberry Pi 4 / AWS EC2 t4g.nano（2 vCPU，512MB RAM）
- **架構**：`linux/amd64`、`linux/arm64`
- **相依性**：Docker（及選用的 Docker Compose）

---

## 貢獻

- 議題：https://github.com/s-nagaev/chibi/issues
- PR：https://github.com/s-nagaev/chibi/pulls
- 討論：https://github.com/s-nagaev/chibi/discussions

提交前請先閱讀 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 授權

MIT — 請參閱 [LICENSE](LICENSE)。

---

<p align="center">
  <strong>準備好見見您的數位夥伴了嗎？</strong><br/>
  <a href="https://chibi.bot/start"><strong>立即開始 →</strong></a>
</p>