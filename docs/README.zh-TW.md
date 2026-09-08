<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi Logo"></h1>

<p align="center">
  <strong>您的數位夥伴。不僅是工具，更是夥伴。</strong><br/>
  <span>自託管、非同步的 Telegram 機器人，可協調多個 AI 提供商、工具和子代理，完成實際工作。</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="PyPI 下載"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="架構"></a>
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
- **OpenRouter**（統一存取多種模型）
- **Cloudflare Workers AI**（多個開源模型）

### OpenAI 相容端點（自託管 / 本機）

- **Ollama**
- **vLLM**
- **LM Studio**
- **任何** OpenAI 相容 API

### 多模態提供商（選用）

- **影像**：Google（Imagen、Nano Banana）、OpenAI（GPT Image）、阿里雲（Qwen Image、Wan）、xAI（Grok Imagine）、智譜AI（GLM Image）、MiniMax、Cheaper Inference（Nano Banana Pro）
- **音樂**：Suno
- **語音**：ElevenLabs、MiniMax、OpenAI（GPT-4o Transcribe / TTS）

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

## 本地客戶端前端

Chibi 可透過版本化的本機 JSONL 協定為本地客戶端提供服務：

```bash
chibi stdio --tui
chibi stdio --vscode
chibi stdio --pycharm
chibi stdio --neovim
```

此命令由客戶端前端啟動，不供互動式使用。Chibi 負責維護 `v1` 協定，並支援協商相同版本的相容客戶端。stdio 程序僅將協定幀寫入 stdout，將診斷資訊寫入 stderr。現有的 Chibi 指令、工具、權限和審核行為仍具權威性；客戶端不會新增額外的工具策略層。

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
- **Google** (Gemini, Nano Banana, Imagen, Voice): [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/)
- **xAI** (Grok): [console.x.ai](https://console.x.ai/)
- **Alibaba** (Qwen, Wan): [modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com?tab=playground#/api-key)
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/)
- **Moonshot** (Kimi): [platform.moonshot.cn](https://platform.moonshot.cn/)
- **MiniMax** (Voice, MiniMax-M2.x): [minimax.io](https://www.minimax.io)
- **智譜AI** (GLM, CogView): [z.ai/manage-apikey/apikey-list](https://z.ai/manage-apikey/apikey-list)
- **OpenRouter** (統一存取多種模型): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
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
- **技能**：可複用的指令模組，代理可按需載入到系統提示中（`load_builtin_skill`）

### 🔌 透過 MCP（模型上下文協定）擴充
將 Chibi 連接到外部工具和服務（或自行建構）：

- GitHub（PR、議題、程式碼審查）
- 瀏覽器自動化
- Docker / 雲端服務
- 資料庫
- 創意工具（Blender、Figma）

只要工具能透過 MCP 公開，Chibi 就能學會使用它。

### 🎨 豐富的內容生成
- **影像**：Nano Banana、Imagen、Qwen、Wan、GPT Image、Grok Imagine、GLM Image
- **音樂**：Suno（包含自訂模式：風格/歌詞/人聲）
- **語音**：轉錄 + 文字轉語音（ElevenLabs、OpenAI、MiniMax）

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

### `MAX_HISTORY_TOKENS` — 上下文摘要閾值（預設值變更）

`MAX_HISTORY_TOKENS` 是 Chibi 自動摘要對話以維持上下文可控的閾值。其**語義已變更**：現在與**真實的、提供者回報的提示詞 token 數量**進行比較（整個發出的請求：系統提示 + 已啟用的技能 + 工具架構 + 工具呼叫引數 + 每則訊息的結構開銷 + 對話內容），而非舊的僅衡量對話 `content` + `role` 的啟發式方法。對於相同對話，真實數字約為舊估計的 **4.8 倍**（詳見 `fix_context_size/context_size_accounting_analysis.md` 的實測分析）。

- **預設值已重設**：`64000` → `100000`。新值保護最小常用上下文視窗（128k tokens）：`100000` 約為 128k 視窗的 78%（因此摘要會在 128k 模型溢位前觸發），約為 200k 視窗的 50%（留有充裕空間）。舊的 `64000` 僅是歷史記錄的估計，從未在真正溢位前被觸發，因為該估計對西里爾文的計算少約 2 倍，且排除了固定的每輪開銷（系統提示約 3.7k、工具架構約 6.8k、已啟用技能約 5.9k、`user_info` 約 0.75–3k）——因此 128k 模型在約 128k 真實 token 時溢位，而啟發式仍遠低於 64k。
- **遷移**：如果您在 `.env` 中明確設定了 `MAX_HISTORY_TOKENS`，舊值是針對舊的歷史估計調整的，現在與一個約為相同對話 4.8 倍的真實數字進行比較。請將其重新調整到 **~100k 量級**（例如 `64000` → `100000`），以便在最小模型的上下文視窗溢位前觸發摘要。如果從未設定，新預設值將自動生效。
- 冷啟動回退（程序重啟後的第一輪，尚無提供者資料快取時）仍使用舊的啟發式，因此摘要功能維持可用。

### `REACTIVE_CONTEXT_RECOVERY` — 上下文溢位後的一次性重試

`REACTIVE_CONTEXT_RECOVERY`（預設：`true`）是補充主動 `MAX_HISTORY_TOKENS` 閾值的反應式安全網。當提供者因具型別的 `context_length_exceeded` 錯誤拒絕請求時，Chibi 會自動摘要對話歷史並**恰好重試一次**該輪。如果重試成功，使用者將收到正常回應（附帶一則簡短說明上下文已壓縮的註記）。如果重試再次溢位或因任何原因恢復失敗，則該輪回退到現有的道歉路徑——摘要+重試迴圈在每個原始輪次中最多只能執行一次。設為 `false` 可停用反應式恢復並保留舊行為（記錄日誌 + 道歉，不重試）。

### 工作目錄按執行緒隔離（`WORKING_DIR`）

代理的工作目錄——用於終端機指令並作為目前工作目錄回報給模型——是**按執行緒隔離的**，與所選 LLM 模型按執行緒綁定的方式一致。

- **預設值**：來自 `WORKING_DIR` 設定（預設 `~/chibi`），透過遺留使用者層級值取得；全新部署直接繼承該設定。
- **每執行緒覆寫**：任何執行緒/對話均可**獨立地**透過代理的 `set_working_dir` 工具覆寫其工作目錄（僅限 LLM 驅動——沒有斜線命令，需啟用 `FILESYSTEM_ACCESS`）。這允許不同執行緒中的兩個代理同時在不同專案上工作而互不干擾。
- **解析順序**：執行緒覆寫 → 遺留使用者層級目錄 → `WORKING_DIR` 設定。
- **路徑正規化**：設定的值在儲存時會被展開為絕對路徑（`~/x` 變為 `/abs/x`）；未修改的預設值保持原始形式。
- **子代理**在某執行緒中產生時共用該執行緒的工作目錄——相同的有效路徑會被注入其系統提示和工具呼叫中。
- 覆寫**在執行緒複製後保留**：`/new_thread_with_current_context` 會將工作目錄與訊息和模型偏好一起攜帶。

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