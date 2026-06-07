<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi Logo"></h1>

<p align="center">
  <strong>Ваш цифровий компаньйон. Не інструмент. Партнер.</strong><br/>
  <span>Self-hosted, асинхронний Telegram-бот, який оркеструє кількох AI-провайдерів, інструменти та субагентів для виконання реальної роботи.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Збірка"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Завантажень Docker"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="Завантажень PyPI"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="Архітектури"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="Ліцензія"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="Документація"></a>

<p align="center">
  <strong>🌍 Читайте іншими мовами:</strong><br/>
  <a href="../README.md">English</a> •
  <a href="README.es.md">Español</a> •
  <a href="README.pt-BR.md">Português (Brasil)</a> •
  <strong>Українська</strong> •
  <a href="README.id.md">Bahasa Indonesia</a> •
  <a href="README.tr.md">Türkçe</a> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

## Підтримувані AI-провайдери (та ендпоінти)

Chibi підтримує кількох провайдерів у межах однієї розмови. Додайте один ключ або багато - Chibi може маршрутизувати запити під конкретну задачу.

### LLM-провайдери

- **OpenAI** (моделі GPT)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **DeepSeek**
- **Alibaba Cloud** (Qwen)
- **xAI** (Grok)
- **Mistral AI**
- **Moonshot AI**
- **MiniMax**
- **ZhipuAI** (моделі GLM)
- **OpenRouter** (уніфікований доступ до багатьох моделей)
- **Cloudflare Workers AI** (багато open-source моделей)

### OpenAI-сумісні ендпоінти (self-host / локально)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Будь-який** OpenAI-сумісний API

### Мультимодальні провайдери (опціонально)

- **Зображення:** Google (Imagen, Nano Banana), OpenAI (DALL·E), Alibaba (Qwen Image), xAI (Grok Image), Wan, ZhipuAI (CogView), MiniMax
- **Музика:** Suno
- **Голос:** ElevenLabs, MiniMax, OpenAI (Whisper)

> Точна доступність моделей залежить від налаштованих ключів провайдерів і ввімкнених можливостей.

---

## 🚀 Швидкий старт (pip)

Встановіть Chibi через pip та запустіть як додаток командного рядка:

```bash
# Встановлення пакета
pip install chibi-bot

# Налаштування агента (додайте API-ключі, змініть налаштування тощо)
chibi config

# Запуск бота
chibi start
```

Бот працюватиме як фонова служба. Використовуйте CLI-команди для керування.



| Команда | Опис |
|---------|-------------|
| `chibi start` | Запуск бота у фоні |
| `chibi stop` | Зупинка працюючого бота |
| `chibi restart` | Перезапуск бота |
| `chibi config` | Створення або редагування конфігурації |
| `chibi logs` | Перегляд логів бота |

---

## 🚀 Швидкий старт (Docker)

Створіть `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # Обов’язково
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # Або будь-який інший провайдер
      # Додайте інші API-ключі за потреби
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) Отримайте токен бота у [@BotFather](https://t.me/BotFather)

2) Додайте секрети в `.env`

3) Запустіть:

```bash
docker-compose up -d
```

Далі:
- **Гайд з інсталяції:** https://chibi.bot/installation
- **Довідник з конфігурації:** https://chibi.bot/configuration

---

## 🔑 Отримання API-ключів

Кожен провайдер вимагає свій API-ключ. Ось прямі посилання:

**Основні провайдери:**
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
- **OpenRouter** (уніфікований доступ до багатьох моделей): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**Креативні інструменти:**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **Повний посібник з інструкціями з налаштування:** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## Спробуйте це в перші 5 хвилин

Скопіюйте та вставте ці запити в Telegram після деплою.

1) **Планування + виконання**
> Постав мені 3 запитання, щоб уточнити мою ціль, потім запропонуй план і виконай крок 1.

2) **Паралельна робота (субагенти)**
> Створи 3 субагенти: один - щоб дослідити варіанти, другий - щоб підготувати рекомендацію, третій - щоб перелічити ризики. Поверни одне узгоджене рішення.

3) **Режим агента (інструменти)**
> Переглянь файли проєкту та підсумуй, що робить цей репозиторій. Потім запропонуй 5 покращень і відкрий чекліст.

4) **Фонова задача**
> Запусти фонову задачу: збери джерела про X і підготуй синтез за 30 хвилин. Тримай мене в курсі.

---

## Чим Chibi відрізняється від інших

### 🎭 Оркестрація кількох провайдерів
Chibi може зберігати контекст, перемикаючись між провайдерами в межах одного діалогу, або обирати найкращу модель для кожного кроку - балансуючи **вартість**, **можливості** та **швидкість**.

### 🤖 Автономні можливості агента
- **Рекурсивне делегування:** створюйте субагентів, які можуть створювати власних субагентів
- **Фонова обробка:** довгі задачі виконуються асинхронно
- **Доступ до файлової системи:** читання/запис/пошук/організація файлів
- **Виконання команд у терміналі:** запуск команд із LLM-модерованою безпекою
- **Постійна пам’ять:** історія розмов зберігається між перезапусками з керуванням контекстом/підсумовуванням

### 🔌 Розширюваність через MCP (Model Context Protocol)
Підключайте Chibi до зовнішніх інструментів і сервісів (або створюйте власні):

- GitHub (PR, issues, code review)
- Автоматизація браузера
- Docker / хмарні сервіси
- Бази даних
- Креативні інструменти (Blender, Figma)

Якщо інструмент можна реалізувати через MCP, Chibi зможе навчитися ним користуватися.

### 🎨 Генерація контенту
- **Зображення:** Nano Banana, Imagen, Qwen, Wan, DALL·E, Grok
- **Музика:** Suno (включно з custom mode: стиль/лірика/вокал)
- **Голос:** транскрипція + text-to-speech (ElevenLabs, MiniMax, OpenAI)

---

## Сценарії використання

**Розробники**
```
Ви: «Запусти тести й виправ те, що зламалося. Я займуся фронтендом».
Chibi: *створює субагента, запускає тести, аналізує помилки, пропонує виправлення*
```

**Дослідники**
```
Ви: «Досліди останні розробки у квантових обчисленнях. Мені потрібен синтез до завтра».
Chibi: *створює кількох дослідницьких агентів, агрегує джерела, готує звіт*
```

**Креатори**
```
Ви: «Згенеруй кіберпанковий міський пейзаж і створи під нього синтвейв-трек».
Chibi: *генерує зображення, створює музику, надсилає обидва результати*
```

**Команди**
```
Ви: «Переглянь цей PR і відповідно онови документацію».
Chibi: *аналізує зміни, пропонує покращення, оновлює документацію через MCP*
```

---

## Приватність, контроль і безпека

- **Self-hosted:** ваші дані залишаються у вашій інфраструктурі
- **Public Mode:** користувачі можуть підключати власні API-ключі (спільний майстер-ключ не потрібен)
- **Контроль доступу:** білий список (whitelist) користувачів/груп/моделей
- **Варіанти зберігання:** локальні томи, Redis або DynamoDB
- **Безпека інструментів:** інструменти агента налаштовуються; виконання в терміналі модеруються та можуть бути обмежені

---

## Документація

- **Почніть тут:** https://chibi.bot
- Вступ і філософія: https://chibi.bot/introduction
- Інсталяція: https://chibi.bot/installation
- Конфігурація: https://chibi.bot/configuration
- Режим агента: https://chibi.bot/agent-mode
- Гайд з MCP: https://chibi.bot/guides/mcp
- Підтримка / усунення проблем: https://chibi.bot/support

---

## Системні вимоги

- **Мінімум:** Raspberry Pi 4 / AWS EC2 t4g.nano (2 vCPU, 512MB RAM)
- **Архітектури:** `linux/amd64`, `linux/arm64`
- **Залежності:** Docker (і, за бажанням, Docker Compose)

---

## Участь у розробці

- Issues: https://github.com/s-nagaev/chibi/issues
- PR: https://github.com/s-nagaev/chibi/pulls
- Discussions: https://github.com/s-nagaev/chibi/discussions

Будь ласка, прочитайте [CONTRIBUTING.md](CONTRIBUTING.md) перед тим, як надсилати зміни.

---

## Ліцензія

MIT - див. [LICENSE](LICENSE).

---

<p align="center">
  <strong>Готові познайомитися зі своїм цифровим компаньйоном?</strong><br/>
  <a href="https://chibi.bot/start"><strong>Почати →</strong></a>
</p>
