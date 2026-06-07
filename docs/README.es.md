<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Logo de Chibi"></h1>

<p align="center">
  <strong>Tu compañero digital. No una herramienta. Un socio.</strong><br/>
  <span>Bot de Telegram autoalojado y asíncrono que orquesta múltiples proveedores de IA, herramientas y subagentes para hacer trabajo real.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="Descargas PyPI"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="Arquitecturas"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="Licencia"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="Documentación"></a>
</p>

<p align="center">
  <strong>🌍 Read this in other languages:</strong><br/>
  <a href="../README.md">English</a> •
  <strong>Español</strong> •
  <a href="README.pt-BR.md">Português (Brasil)</a> •
  <a href="README.uk.md">Українська</a> •
  <a href="README.id.md">Bahasa Indonesia</a> •
  <a href="README.tr.md">Türkçe</a> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

Chibi está hecho para ese momento en el que te das cuenta de que necesitas algo más que “una herramienta de IA”. Necesitas un **socio** que pueda coordinar modelos, ejecutar trabajo en segundo plano e integrarse con tus sistemas—sin que tengas que estar cuidando prompts.

**Chibi** es un **compañero digital basado en Telegram**, asíncrono y autoalojado, que orquesta múltiples proveedores de IA y herramientas para entregar resultados: cambios de código, síntesis de investigación, generación de medios y tareas operativas.

---

## Por qué Chibi

- **Una sola interfaz (Telegram).** Móvil/escritorio/web, siempre contigo.
- **Agnóstico al proveedor.** Usa el mejor modelo para cada tarea—sin dependencia de un único proveedor.
- **Ejecución autónoma.** Los subagentes trabajan en paralelo; las tareas largas se ejecutan de forma asíncrona.
- **Conectado a herramientas.** Sistema de archivos + terminal + integraciones MCP (GitHub, navegador, BD, etc.).
- **Autoalojado.** Tus datos, tus claves, tus reglas.

---

## Proveedores de IA compatibles (y endpoints)

Chibi admite múltiples proveedores detrás de una sola conversación. Añade una clave o varias—Chibi puede enrutar por tarea.

### Proveedores LLM

- **OpenAI** (modelos GPT)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **DeepSeek**
- **Alibaba Cloud** (Qwen)
- **xAI** (Grok)
- **Mistral AI**
- **Moonshot AI**
- **MiniMax**
- **ZhipuAI** (modelos GLM)
- **OpenRouter** (acceso unificado a muchos modelos)
- **Cloudflare Workers AI** (muchos modelos open-source)

### Endpoints compatibles con OpenAI (autoalojado / local)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Cualquier** API compatible con OpenAI

### Proveedores multimodales (opcional)

- **Imágenes:** Google (Imagen, Nano Banana), OpenAI (DALL·E), Alibaba (Qwen Image), xAI (Grok Image), Wan, ZhipuAI (CogView), MiniMax
- **Música:** Suno
- **Voz:** ElevenLabs, MiniMax, OpenAI (Whisper)

> La disponibilidad exacta de modelos depende de tus claves de proveedor configuradas y de las funciones habilitadas.

---

## 🚀 Inicio rápido (pip)

Instala Chibi a través de pip y ejecútalo como una aplicación de línea de comandos:

```bash
# Instalar el paquete
pip install chibi-bot

# Configurar el agente (añadir claves API, actualizar ajustes, etc.)
chibi config

# Iniciar el bot
chibi start
```

El bot se ejecutará como un servicio en segundo plano. Utiliza los comandos de CLI para gestionarlo.
## 🚀 Inicio rápido (Docker)

Crea `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # Obligatorio
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # O cualquier otro proveedor
      # Añade más claves de API según sea necesario
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) Obtén un token de bot en [@BotFather](https://t.me/BotFather)

2) Pon los secretos en `.env`

3) Ejecuta:

```bash
docker-compose up -d
```

Siguiente:
- **Guía de instalación:** https://chibi.bot/installation
- **Referencia de configuración:** https://chibi.bot/configuration

---

## 🔑 Obtener claves API

Cada proveedor requiere su propia clave API. Aquí están los enlaces directos:

**Proveedores principales:**
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
- **OpenRouter** (acceso unificado a muchos modelos): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**Herramientas creativas:**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **Guía completa con instrucciones de configuración:** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## Pruébalo en los primeros 5 minutos

Pega esto en Telegram después de desplegar.

1) **Planificación + ejecución**
> Hazme 3 preguntas para aclarar mi objetivo, luego propone un plan y ejecuta el paso 1.

2) **Trabajo en paralelo (subagentes)**
> Crea 3 subagentes: uno para investigar opciones, otro para redactar una recomendación y otro para listar riesgos. Devuelve una única decisión.

3) **Modo agente (herramientas)**
> Inspecciona los archivos del proyecto y resume qué hace este repo. Luego propone 5 mejoras y abre una checklist.

4) **Tarea en segundo plano**
> Inicia una tarea en segundo plano: reúne fuentes sobre X y entrega una síntesis en 30 minutos. Manténme al tanto.

---

## Qué hace a Chibi diferente

### 🎭 Orquestación multi-proveedor
Chibi puede mantener el contexto mientras cambia de proveedor a mitad de hilo, o elegir el mejor modelo por paso—equilibrando **coste**, **capacidad** y **velocidad**.

### 🤖 Capacidades de agente autónomo
- **Delegación recursiva:** crea subagentes que pueden crear sus propios subagentes
- **Procesamiento en segundo plano:** las tareas de larga duración se ejecutan de forma asíncrona
- **Acceso al sistema de archivos:** leer/escribir/buscar/organizar archivos
- **Ejecución en terminal:** ejecutar comandos con seguridad moderada por LLM
- **Memoria persistente:** el historial de conversación sobrevive reinicios con gestión de contexto/resumen

### 🔌 Extensible vía MCP (Model Context Protocol)
Conecta Chibi a herramientas y servicios externos (o crea los tuyos):

- GitHub (PRs, issues, revisión de código)
- Automatización del navegador
- Docker / servicios cloud
- Bases de datos
- Herramientas creativas (Blender, Figma)

Si una herramienta puede exponerse vía MCP, Chibi puede aprender a usarla.

### 🎨 Generación de contenido enriquecido
- **Imágenes:** Nano Banana, Imagen, Qwen, Wan, DALL·E, Grok
- **Música:** Suno (incluye modo personalizado: estilo/letra/voces)
- **Voz:** transcripción + texto a voz (ElevenLabs, MiniMax, OpenAI)

---

## Casos de uso

**Desarrolladores**
```
Tú: “Ejecuta los tests y arregla lo que esté roto. Yo me encargo del frontend.”
Chibi: *crea un subagente, ejecuta tests, analiza fallos, propone arreglos*
```

**Investigadores**
```
Tú: “Investiga los últimos avances en computación cuántica. Necesito una síntesis para mañana.”
Chibi: *crea múltiples agentes de investigación, agrega fuentes, entrega un informe*
```

**Creadores**
```
Tú: “Genera una ciudad cyberpunk y compón un tema synthwave que encaje.”
Chibi: *genera una imagen, crea música, entrega ambos*
```

**Equipos**
```
Tú: “Revisa este PR y actualiza la documentación en consecuencia.”
Chibi: *analiza cambios, sugiere mejoras, actualiza docs vía MCP*
```

---

## Privacidad, control y seguridad

- **Autoalojado:** tus datos se quedan en tu infraestructura
- **Modo público:** los usuarios pueden traer sus propias claves de API (no se requiere una clave maestra compartida)
- **Control de acceso:** lista blanca de usuarios/grupos/modelos
- **Opciones de almacenamiento:** volúmenes locales, Redis o DynamoDB
- **Seguridad de herramientas:** las herramientas del agente son configurables; la ejecución en terminal está moderada y puede restringirse

---

## Documentación

- **Empieza aquí:** https://chibi.bot
- Introducción y filosofía: https://chibi.bot/introduction
- Instalación: https://chibi.bot/installation
- Configuración: https://chibi.bot/configuration
- Modo agente: https://chibi.bot/agent-mode
- Guía MCP: https://chibi.bot/guides/mcp
- Soporte / solución de problemas: https://chibi.bot/support

---

## Requisitos del sistema

- **Mínimo:** Raspberry Pi 4 / AWS EC2 t4g.nano (2 vCPU, 512MB RAM)
- **Arquitecturas:** `linux/amd64`, `linux/arm64`
- **Dependencias:** Docker (y opcionalmente Docker Compose)

---

## Contribuir

- Issues: https://github.com/s-nagaev/chibi/issues
- PRs: https://github.com/s-nagaev/chibi/pulls
- Debates: https://github.com/s-nagaev/chibi/discussions

Por favor, lee [CONTRIBUTING.md](CONTRIBUTING.md) antes de enviar.

---

## Licencia

MIT — ver [LICENSE](LICENSE).

---

<p align="center">
  <strong>¿Listo para conocer a tu compañero digital?</strong><br/>
  <a href="https://chibi.bot/start"><strong>Empezar →</strong></a>
</p>
