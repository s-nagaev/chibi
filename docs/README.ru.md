<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Логотип Chibi"></h1>

<p align="center">
  <strong>Ваш цифровой компаньон. Не инструмент. Партнёр.</strong><br/>
  <span>Self-hosted, асинхронный Telegram-бот, который координирует работу AI‑моделей, инструментов и субагентов для решения реальных задач.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="PyPI Загрузки"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="Архитектуры"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="Лицензия"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="Документация"></a>
</p>

<p align="center">
  <strong>🌍 Read this in other languages:</strong><br/>
  <a href="../README.md">English</a> •
  <a href="README.es.md">Español</a> •
  <a href="README.pt-BR.md">Português (Brasil)</a> •
  <a href="README.uk.md">Українська</a> •
  <a href="README.id.md">Bahasa Indonesia</a> •
  <a href="README.tr.md">Türkçe</a> •
  <strong>Русский</strong> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

Chibi создан для тех моментов, когда вы понимаете: вам нужен не просто очередной «AI‑инструмент». Вам нужен **партнёр**, способный координировать модели, выполнять работу в фоне и интегрироваться с вашими системами — без необходимости постоянно «нянчиться» с промптами.

**Chibi** — это асинхронный, self‑hosted **цифровой компаньон в Telegram**, который оркестрирует работу множества AI‑провайдеров и инструментов для получения конкретных результатов: изменений в коде, аналитических сводок, генерации медиа и выполнения операционных задач.

---

## Почему Chibi

- **Один интерфейс (Telegram).** Мобильный, десктоп, веб — всегда с вами.
- **Агностик к провайдерам.** Используйте лучшую модель для каждой задачи — без привязки к вендору.
- **Автономность.** Субагенты работают параллельно, длительные задачи выполняются в фоне.
- **Инструментарий.** Файловая система + терминал + интеграции MCP (GitHub, браузер, БД и т.д.).
- **Self‑hosted.** Ваши данные, ваши ключи, ваши правила.

---

## Поддерживаемые AI‑провайдеры (и endpoints)

Chibi поддерживает множество провайдеров в рамках одного диалога. Добавьте один ключ или несколько — Chibi умеет маршрутизировать запросы в зависимости от задачи.

### LLM‑провайдеры

- **OpenAI** (модели GPT)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **DeepSeek**
- **Alibaba Cloud** (Qwen)
- **xAI** (Grok)
- **Mistral AI**
- **Moonshot AI**
- **MiniMax**
- **ZhipuAI** (модели GLM)
- **OpenRouter** (унифицированный доступ ко многим моделям)
- **Cloudflare Workers AI** (множество open‑source моделей)

### OpenAI‑совместимые endpoints (self‑host / local)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Любой** OpenAI‑совместимый API

### Мультимодальные провайдеры (опционально)

- **Изображения:** Google (Imagen, Nano Banana), OpenAI (DALL·E), Alibaba (Qwen Image), xAI (Grok Image), Wan, ZhipuAI (CogView), MiniMax
- **Музыка:** Suno
- **Голос:** ElevenLabs, MiniMax, OpenAI (Whisper)

> Доступность конкретных моделей зависит от настроенных ключей провайдеров и включённых возможностей.

---

## 🚀 Быстрый старт (pip)

Установите Chibi через pip и запустите как приложение командной строки:

```bash
# Установка пакета
pip install chibi-bot

# Настройка агента (добавьте API-ключи, измените настройки и т.д.)
chibi config

# Запуск бота
chibi start
```

Бот будет работать как фоновый сервис. Используйте CLI-команды для управления.



| Команда | Описание |
|---------|-------------|
| `chibi start` | Запуск бота в фоновом режиме |
| `chibi stop` | Остановка работающего бота |
| `chibi restart` | Перезапуск бота |
| `chibi config` | Создание или редактирование конфигурации |
| `chibi logs` | Просмотр логов бота |

---

## 🚀 Быстрый старт (Docker)

Создайте `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # Обязательно
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # Или любой другой провайдер
      # Добавьте другие API-ключи при необходимости
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) Получите токен бота у [@BotFather](https://t.me/BotFather)

2) Сохраните секреты в `.env`

3) Запустите:

```bash
docker-compose up -d
```

Дальше:
- **Инструкция по установке:** https://chibi.bot/installation
- **Справочник по конфигурации:** https://chibi.bot/configuration

---

## 🔑 Получение API-ключей

Каждый провайдер требует свой API-ключ. Вот прямые ссылки:

**Основные провайдеры:**
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
- **OpenRouter** (унифицированный доступ ко многим моделям): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**Креативные инструменты:**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **Полное руководство с инструкциями по настройке:** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## Что попробовать в первые 5 минут

Скопируйте эти сообщения в Telegram после деплоя.

1) **Планирование + выполнение**
> Задай 3 вопроса для уточнения цели, затем предложи план и выполни первый шаг.

2) **Параллельная работа (субагенты)**
> Создай 3 субагентов: один исследует варианты, второй подготовит рекомендации, третий оценит риски. Сведи всё в одно итоговое решение.

3) **Режим агента (инструменты)**
> Изучи файлы проекта и кратко опиши суть репозитория. Предложи 5 улучшений и оформи их как чек‑лист.

4) **Фоновая задача**
> Запусти фоновую задачу: собери источники по теме X и подготовь сводку через 30 минут. Держи в курсе.

---

## Чем Chibi отличается

### 🎭 Оркестрация нескольких провайдеров
Chibi сохраняет контекст при переключении между провайдерами и выбирает оптимальную модель для каждого шага, балансируя между **стоимостью**, **качеством** и **скоростью**.

### 🤖 Автономные агентные возможности
- **Рекурсивное делегирование:** субагенты могут создавать своих субагентов
- **Фоновая обработка:** длительные задачи выполняются асинхронно
- **Доступ к файловой системе:** чтение, запись, поиск и организация файлов
- **Терминал:** выполнение команд с модерацией безопасности через LLM
- **Постоянная память:** история диалога сохраняется после перезапуска благодаря управлению контекстом и суммаризации

### 🔌 Расширяемость через MCP (Model Context Protocol)
Подключайте Chibi к внешним инструментам и сервисам (или создавайте свои):

- GitHub (PR, issues, code review)
- Автоматизация браузера
- Docker / облачные сервисы
- Базы данных
- Креативные инструменты (Blender, Figma)

Если инструмент можно подключить через MCP, Chibi сможет научиться им пользоваться.

### 🎨 Генерация контента
- **Изображения:** Nano Banana, Imagen, Qwen, Wan, DALL·E, Grok
- **Музыка:** Suno (включая custom mode: стиль/текст/вокал)
- **Голос:** транскрибация + синтез речи (ElevenLabs, MiniMax, OpenAI)

---

## Сценарии использования

**Разработчики**
```
Вы: «Запусти тесты и исправь ошибки. Я займусь фронтендом».
Chibi: *создаёт субагента, запускает тесты, анализирует ошибки, предлагает исправления*
```

**Исследователи**
```
Вы: «Изучи последние достижения в квантовых вычислениях. Нужна сводка к завтрашнему дню».
Chibi: *создаёт несколько исследовательских агентов, собирает источники и выдаёт отчёт*
```

**Создатели контента**
```
Вы: «Сгенерируй город в стиле киберпанк и напиши к нему synthwave‑трек».
Chibi: *генерирует изображение, создаёт музыку и отдаёт оба результата*
```

**Команды**
```
Вы: «Проверь этот PR и обнови документацию».
Chibi: *анализирует изменения, предлагает улучшения, обновляет доки через MCP*
```

---

## Приватность, контроль и безопасность

- **Self‑hosted:** ваши данные остаются на вашей инфраструктуре
- **Публичный режим:** пользователи могут использовать свои API‑ключи (общий мастер‑ключ не требуется)
- **Контроль доступа:** белые списки пользователей/групп/моделей
- **Варианты хранения:** локальные тома, Redis или DynamoDB
- **Безопасность:** инструменты настраиваются, выполнение команд в терминале модерируется и может быть ограничено

---

## Документация

- **Начните здесь:** https://chibi.bot
- Введение и философия: https://chibi.bot/introduction
- Установка: https://chibi.bot/installation
- Конфигурация: https://chibi.bot/configuration
- Режим агента: https://chibi.bot/agent-mode
- Руководство по MCP: https://chibi.bot/guides/mcp
- Поддержка / troubleshooting: https://chibi.bot/support

---

## Системные требования

- **Минимум:** Raspberry Pi 4 / AWS EC2 t4g.nano (2 vCPU, 512MB RAM)
- **Архитектуры:** `linux/amd64`, `linux/arm64`
- **Зависимости:** Docker (и при желании Docker Compose)

---

## Участие в разработке

- Issues: https://github.com/s-nagaev/chibi/issues
- PR: https://github.com/s-nagaev/chibi/pulls
- Обсуждения: https://github.com/s-nagaev/chibi/discussions

Перед отправкой, пожалуйста, прочитайте [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

---

<p align="center">
  <strong>Готовы познакомиться с вашим цифровым компаньоном?</strong><br/>
  <a href="https://chibi.bot/start"><strong>Начать →</strong></a>
</p>
