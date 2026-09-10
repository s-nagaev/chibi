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
- **Novita AI**
- **Xiaomi** (modelos MiMo)
- **Melious**
- **Cheaper Inference**
- **DeepInfra**
- **Together AI**
- **Fireworks AI**
- **Nebius** (Token Factory)
- **Baseten**
- **SambaNova**
- **SiliconFlow**
- **Parasail**
- **Featherless**
- **OpenRouter** (acceso unificado a muchos modelos)
- **Cloudflare Workers AI** (muchos modelos de código abierto)

### Endpoints compatibles con OpenAI (autoalojado / local)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Cualquier** API compatible con OpenAI

### Proveedores multimodales (opcional)

- **Imágenes:** Google (Imagen, Nano Banana), OpenAI (GPT Image), Alibaba (Qwen Image, Wan), xAI (Grok Imagine), ZhipuAI (GLM Image), MiniMax, Cheaper Inference (Nano Banana Pro)
- **Música:** Suno
- **Voz:** ElevenLabs, OpenAI (GPT-4o Transcribe / TTS), MiniMax (TTS)

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

### Comandos de CLI

| Comando         | Descripción                              |
|-----------------|------------------------------------------|
| `chibi start`   | Inicia el bot como servicio en segundo plano |
| `chibi stop`    | Detiene el bot en ejecución              |
| `chibi restart` | Reinicia el bot                          |
| `chibi config`  | Genera o edita la configuración          |
| `chibi logs`    | Muestra los registros del bot            |

---

## Clientes locales

Chibi puede atender a sus clientes locales mediante el protocolo JSONL local versionado:

```bash
chibi stdio --tui
chibi stdio --vscode
chibi stdio --pycharm
chibi stdio --neovim
```

Este comando está pensado para que lo inicie el cliente, no para usarse de forma interactiva. Chibi es propietario del protocolo `v1` y admite clientes compatibles que negocien la misma versión. El proceso stdio escribe únicamente tramas de protocolo en stdout y diagnósticos en stderr. El comportamiento existente de Chibi para comandos, herramientas, permisos y moderación sigue siendo autoritativo; el cliente no añade una capa de políticas para herramientas.

---
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
- **Skills:** módulos de instrucciones reutilizables que el agente puede cargar en su prompt del sistema bajo demanda (`load_builtin_skill`)

### 🔌 Extensible vía MCP (Model Context Protocol)
Conecta Chibi a herramientas y servicios externos (o crea los tuyos):

- GitHub (PRs, issues, revisión de código)
- Automatización del navegador
- Docker / servicios cloud
- Bases de datos
- Herramientas creativas (Blender, Figma)

Si una herramienta puede exponerse vía MCP, Chibi puede aprender a usarla.

### 🎨 Generación de contenido enriquecido
- **Imágenes:** Nano Banana, Imagen, Qwen, Wan, GPT Image, Grok Imagine, GLM Image
- **Música:** Suno (incluye modo personalizado: estilo/letra/voces)
- **Voz:** transcripción + texto a voz (ElevenLabs, OpenAI, MiniMax)

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

### `MAX_HISTORY_TOKENS` — umbral de resumen de contexto (cambio incompatible del valor predeterminado)

`MAX_HISTORY_TOKENS` es el umbral a partir del cual Chibi resume automáticamente una conversación para mantener el contexto manejable. Su **semántica cambió**: ahora se compara con el **recuento real de tokens del prompt informado por el proveedor** (toda la solicitud saliente: prompt del sistema + Skills activadas + esquemas de herramientas + argumentos de llamadas a herramientas + sobrecarga estructural por mensaje + contenido de la conversación), en vez de la heurística anterior que medía solo `content` + `role` de la conversación. La cifra real es aproximadamente **4,8 veces mayor** que la estimación anterior para una conversación idéntica (consulta `fix_context_size/context_size_accounting_analysis.md` para el desglose medido).

- **Predeterminado reajustado:** `64000` → `100000`. El nuevo valor protege la ventana de contexto común más pequeña (128k tokens): `100000` es aproximadamente el 78 % de una ventana de 128k (el resumen se activa *antes* de que un modelo de 128k se desborde) y aproximadamente el 50 % de una ventana de 200k (dejando margen suficiente). El anterior `64000` era una estimación solo del historial que nunca se alcanzaba antes de un desbordamiento real, pues subestima el cirílico unas 2 veces y excluye la sobrecarga fija por turno (prompt del sistema ~3,7k, esquemas de herramientas ~6,8k, Skills activadas ~5,9k, `user_info` ~0,75–3k).
- **Migración:** si configuraste `MAX_HISTORY_TOKENS` explícitamente en `.env`, su valor anterior se ajustó a la antigua estimación basada solo en el historial y ahora se compara con una cifra veraz unas 4,8 veces mayor para la misma conversación. Reajústalo a la escala de **~100k** (por ejemplo, `64000` → `100000`) para que el resumen se active antes de que se desborde la ventana de contexto de tu modelo más pequeño. Si nunca lo configuraste, el nuevo valor predeterminado se aplica automáticamente.
- La alternativa de arranque en frío (el primer turno tras reiniciar el proceso, cuando aún no hay datos del proveedor en caché) sigue usando la heurística anterior para que el resumen continúe funcionando.

### `REACTIVE_CONTEXT_RECOVERY` — reintento único tras un desbordamiento de contexto

`REACTIVE_CONTEXT_RECOVERY` (predeterminado: `true`) es la red de seguridad reactiva que complementa el umbral proactivo `MAX_HISTORY_TOKENS`. Cuando un proveedor rechaza una solicitud con un error tipado `context_length_exceeded`, Chibi resume automáticamente el historial y reintenta el turno **exactamente una vez**. Si tiene éxito, recibes una respuesta normal (con una nota breve indicando que se comprimió el contexto). Si vuelve a desbordarse o la recuperación falla, el turno recurre a la vía de disculpa existente: el bucle de resumen y reintento nunca se ejecuta más de una vez por turno original. Establécelo en `false` para desactivarlo y conservar el comportamiento previo (registro + disculpa, sin reintento).

### El directorio de trabajo se limita al hilo (`WORKING_DIR`)

El directorio de trabajo del agente —usado para comandos de terminal e informado al modelo como su CWD actual— tiene alcance **por hilo**, igual que el modelo LLM seleccionado se asocia a cada hilo.

- **Predeterminado:** proviene del ajuste `WORKING_DIR` (predeterminado `~/chibi`) mediante el valor heredado de usuario; los despliegues nuevos heredan el ajuste directamente.
- **Anulación por hilo:** cualquier hilo/conversación puede anular su directorio de trabajo **de forma aislada** mediante la herramienta `set_working_dir` del agente (solo controlada por LLM; no hay comando slash, disponible si `FILESYSTEM_ACCESS` está activado). Esto permite que dos agentes en hilos diferentes trabajen simultáneamente en proyectos distintos sin interferir.
- **Orden de resolución:** anulación por hilo → directorio heredado de usuario → ajuste `WORKING_DIR`.
- **Normalización de rutas:** los valores establecidos se expanden a rutas absolutas al guardarse (`~/x` pasa a ser `/abs/x`); los valores predeterminados sin modificar mantienen su forma original.
- Los **subagentes** generados en un hilo comparten su directorio de trabajo: la misma ruta efectiva se inyecta en sus prompts del sistema y llamadas a herramientas.
- Las anulaciones **sobreviven a la clonación de hilos**: `/new_thread_with_current_context` conserva el directorio de trabajo junto con los mensajes y las preferencias de modelo.

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
