<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi ロゴ"></h1>

<p align="center">
  <strong>あなたのデジタルコンパニオン。ツールではなく、パートナー。</strong><br/>
  <span>複数のAIプロバイダー、ツール、サブエージェントをオーケストレーションし、実務を遂行する、セルフホスト可能な非同期Telegramボットです。</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="PyPI ダウンロード"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="アーキテクチャ"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="ライセンス"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="ドキュメント"></a>
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
  <strong>日本語</strong> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

Chibi は、「AIツール」だけでは足りないと気づいた瞬間のために作られています。必要なのは、モデルを連携させ、バックグラウンドで作業を進め、あなたのシステムと統合できる **パートナー** です。プロンプトの調整に追われる必要はありません。

**Chibi** は、複数のAIプロバイダーとツールをオーケストレーションし、成果（コード変更、調査結果の集約、メディア生成、運用タスクなど）を届ける、非同期・セルフホスト可能な **Telegramベースのデジタルコンパニオン** です。

---

## Chibi を選ぶ理由

- **インターフェースは1つ（Telegram）。** モバイル/デスクトップ/ウェブで、いつでもどこでも。
- **プロバイダー非依存。** タスクごとに最適なモデルを選べます（ベンダーロックインなし）。
- **自律的な実行。** サブエージェントが並列に動作し、長いタスクは非同期で進行します。
- **ツール連携。** ファイルシステム + ターミナル + MCP連携（GitHub、ブラウザ、DB など）。
- **セルフホスト。** データもキーもルールも、あなたの管理下に。

---

## 対応AIプロバイダー（およびエンドポイント）

Chibi は、単一の会話の中で複数プロバイダーを扱えます。キーは1つでも複数でも構いません。タスクに応じてルーティングできます。

### LLMプロバイダー

- **OpenAI**（GPT 系）
- **Anthropic**（Claude）
- **Google**（Gemini）
- **DeepSeek**
- **Alibaba Cloud**（Qwen）
- **xAI**（Grok）
- **Mistral AI**
- **Moonshot AI**
- **MiniMax**
- **ZhipuAI**（GLMモデル）
- **Cloudflare Workers AI**（多数のオープンソースモデル）

### OpenAI互換エンドポイント（セルフホスト / ローカル）

- **Ollama**
- **vLLM**
- **LM Studio**
- **任意の** OpenAI互換API

### マルチモーダルプロバイダー（任意）

- **画像:** Google（Imagen, Nano Banana）、OpenAI（DALL·E）、Alibaba（Qwen Image）、xAI（Grok Image）、Wan、ZhipuAI（CogView）、MiniMax
- **音楽:** Suno
- **音声:** ElevenLabs、MiniMax、OpenAI（Whisper）

> 利用可能なモデルは、設定したプロバイダーキーと有効化した機能により異なります。

---

## 🚀 クイックスタート (pip)

pip 経由で Chibi をインストールし、コマンドラインアプリケーションとして実行します：

```bash
# パッケージのインストール
pip install chibi-bot

# エージェントの設定（API キーの追加、設定の更新など）
chibi config

# ボットの起動
chibi start
```

ボットはバックグラウンドサービスとして実行されます。CLI コマンドを使用して管理します。

### CLIコマンド

| コマンド | 説明 |
|---------|-------------|
| `chibi start` | ボットをバックグラウンドサービスとして起動 |
| `chibi stop` | 実行中のボットを停止 |
| `chibi restart` | ボットを再起動 |
| `chibi config` | 設定の生成または編集 |
| `chibi logs` | ボットのログを表示 |

---

## 🚀 クイックスタート（Docker）

`docker-compose.yml` を作成します:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # 必須
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # または他のプロバイダー
      # 必要に応じてAPIキーを追加
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) [@BotFather](https://t.me/BotFather) からボットトークンを取得します

2) `.env` にシークレットを設定します

3) 実行:

```bash
docker-compose up -d
```

次はこちら:
- **インストールガイド:** https://chibi.bot/installation
- **設定リファレンス:** https://chibi.bot/configuration

---

## 🔑 APIキーの取得

各プロバイダーには独自のAPIキーが必要です。直接リンクは以下の通りです：

**主要プロバイダー：**
- **OpenAI** (GPT, DALL·E): [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic** (Claude): [console.anthropic.com](https://console.anthropic.com/)
- **Google** (Gemini, Nano Banana, Imagen): [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/)
- **xAI** (Grok): [console.x.ai](https://console.x.ai/)
- **Alibaba** (Qwen, Wan): [modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com?tab=playground#/api-key)
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/)
- **Moonshot** (Kimi): [platform.moonshot.cn](https://platform.moonshot.cn/)
- **MiniMax** (Voice, MiniMax-M2.x): [minimax.io](https://www.minimax.io)
- **ZhipuAI** (GLM, CogView): [z.ai/manage-apikey/apikey-list](https://z.ai/manage-apikey/apikey-list)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**クリエイティブツール：**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **セットアップ手順を含む完全ガイド：** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## 最初の5分で試せること

デプロイ後、Telegram に以下を貼り付けて試してください。

1) **計画 + 実行**
> 目標を明確にするために質問を3つしてください。その後、計画を提案し、ステップ1を実行してください。

2) **並列作業（サブエージェント）**
> サブエージェントを3つ起動してください。1つは選択肢の調査、1つは推奨案のドラフト、1つはリスクの列挙を担当。最終的に1つの意思決定としてまとめてください。

3) **エージェントモード（ツール）**
> プロジェクトファイルを確認し、このリポジトリが何をするものか要約してください。その後、改善案を5つ提案し、チェックリストを作成してください。

4) **バックグラウンドタスク**
> バックグラウンドタスクを開始してください: X に関する情報源を集め、30分後に要約を提出。進捗も随時共有してください。

---

## Chibi の違い

### 🎭 マルチプロバイダー・オーケストレーション
Chibi は、スレッドの途中でプロバイダーを切り替えてもコンテキストを維持できます。あるいは、ステップごとに最適なモデルを選択し、**コスト**・**性能**・**速度** のバランスを取れます。

### 🤖 自律エージェント機能
- **再帰的な委任:** サブエージェントを起動し、そのサブエージェントがさらにサブエージェントを起動できます
- **バックグラウンド処理:** 長時間タスクを非同期で実行します
- **ファイルシステムアクセス:** ファイルの読み書き/検索/整理
- **ターミナル実行:** LLM によるセキュリティモデレーション付きでコマンドを実行
- **永続メモリ:** 会話履歴は再起動後も保持され、コンテキスト管理/要約に対応

### 🔌 MCP（Model Context Protocol）による拡張
Chibi を外部ツールやサービスに接続できます（または自作も可能です）。

- GitHub（PR、Issue、コードレビュー）
- ブラウザ自動化
- Docker / クラウドサービス
- データベース
- クリエイティブツール（Blender、Figma）

MCP 経由で公開できるツールであれば、Chibi は使い方を学習して活用できます。

### 🎨 リッチなコンテンツ生成
- **画像:** Nano Banana、Imagen、Qwen、Wan、DALL·E、Grok
- **音楽:** Suno（カスタムモード: スタイル/歌詞/ボーカル指定を含む）
- **音声:** 文字起こし + テキスト読み上げ（ElevenLabs、MiniMax、OpenAI）

---

## ユースケース

**開発者**
```
あなた: 「テストを実行して、壊れているところを直しておいて。フロントエンドは私がやる。」
Chibi: *サブエージェントを起動し、テストを実行、失敗を分析し、修正案を提示*
```

**研究者**
```
あなた: 「量子コンピューティングの最新動向を調べて。明日までに要約が必要。」
Chibi: *複数の調査エージェントを起動し、情報源を集約してレポートを作成*
```

**クリエイター**
```
あなた: 「サイバーパンクな都市景観を生成して、それに合うシンセウェーブ曲も作って。」
Chibi: *画像を生成し、音楽を作成し、両方を提供*
```

**チーム**
```
あなた: 「このPRをレビューして、ドキュメントもそれに合わせて更新して。」
Chibi: *変更を分析し、改善提案を行い、MCP 経由でドキュメントを更新*
```

---

## プライバシー、制御、安全性

- **セルフホスト:** データはあなたのインフラ内に留まります
- **パブリックモード (Public Mode):** ユーザーが自分のAPIキーを持ち込めます（共有のマスターキーは不要）
- **アクセス制御:** ユーザー/グループ/モデルのホワイトリスト
- **ストレージの選択肢:** ローカルボリューム、Redis、DynamoDB
- **ツールの安全性:** エージェントツールは設定可能です。ターミナル実行はモデレートされ、制限できます

---

## ドキュメント

- **まずはこちら:** https://chibi.bot
- Introduction & philosophy: https://chibi.bot/introduction
- Installation: https://chibi.bot/installation
- Configuration: https://chibi.bot/configuration
- Agent mode: https://chibi.bot/agent-mode
- MCP guide: https://chibi.bot/guides/mcp
- Support / troubleshooting: https://chibi.bot/support

---

## システム要件

- **最小:** Raspberry Pi 4 / AWS EC2 t4g.nano（2 vCPU、512MB RAM）
- **アーキテクチャ:** `linux/amd64`、`linux/arm64`
- **依存関係:** Docker（および任意で Docker Compose）

---

## コントリビュート

- Issues: https://github.com/s-nagaev/chibi/issues
- PRs: https://github.com/s-nagaev/chibi/pulls
- Discussions: https://github.com/s-nagaev/chibi/discussions

投稿前に [CONTRIBUTING.md](CONTRIBUTING.md) をお読みください。

---

## ライセンス

MIT — 詳細は [LICENSE](LICENSE) を参照してください。

---

<p align="center">
  <strong>デジタルコンパニオンに会う準備はできましたか？</strong><br/>
  <a href="https://chibi.bot/start"><strong>はじめる →</strong></a>
</p>
