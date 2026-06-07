<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Logo do Chibi"></h1>

<p align="center">
  <strong>Seu companheiro digital. Não uma ferramenta. Um parceiro.</strong><br/>
  <span>Bot do Telegram auto-hospedado e assíncrono que orquesta múltiplos provedores de IA, ferramentas e subagentes para fazer trabalho de verdade.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="Downloads PyPI"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="Arquiteturas"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="Licença"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="Documentação"></a>
</p>

<p align="center">
  <strong>🌍 Read this in other languages:</strong><br/>
  <a href="../README.md">English</a> •
  <a href="README.es.md">Español</a> •
  <strong>Português (Brasil)</strong> •
  <a href="README.uk.md">Українська</a> •
  <a href="README.id.md">Bahasa Indonesia</a> •
  <a href="README.tr.md">Türkçe</a> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

O Chibi foi feito para aquele momento em que você percebe que precisa de mais do que “uma ferramenta de IA”. Você precisa de um **parceiro** que coordene modelos, execute trabalho em segundo plano e se integre aos seus sistemas — sem você ficar “cuidando” de prompts.

**Chibi** é um **companheiro digital baseado no Telegram**, assíncrono e auto-hospedado, que orquestra múltiplos provedores de IA e ferramentas para entregar resultados: mudanças de código, sínteses de pesquisa, geração de mídia e tarefas operacionais.

---

## Por que Chibi

- **Uma interface (Telegram).** Mobile/desktop/web, sempre com você.
- **Agnóstico a provedores.** Use o melhor modelo para cada tarefa — sem vendor lock-in.
- **Execução autônoma.** Subagentes trabalham em paralelo; tarefas longas rodam de forma assíncrona.
- **Conectado a ferramentas.** Sistema de arquivos + terminal + integrações MCP (GitHub, navegador, bancos de dados etc.).
- **Auto-hospedado.** Seus dados, suas chaves, suas regras.

---

## Provedores de IA suportados (e endpoints)

O Chibi suporta múltiplos provedores por trás de uma única conversa. Adicione uma chave ou várias — o Chibi pode rotear por tarefa.

### Provedores de LLM

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
- **OpenRouter** (acesso unificado a muitos modelos)
- **Cloudflare Workers AI** (muitos modelos open-source)

### Endpoints compatíveis com OpenAI (auto-hospedado / local)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Qualquer** API compatível com OpenAI

### Provedores multimodais (opcional)

- **Imagens:** Google (Imagen, Nano Banana), OpenAI (DALL·E), Alibaba (Qwen Image), xAI (Grok Image), Wan, ZhipuAI (CogView), MiniMax
- **Música:** Suno
- **Voz:** ElevenLabs, MiniMax, OpenAI (Whisper)

> A disponibilidade exata de modelos depende das suas chaves configuradas e dos recursos habilitados.

---

## 🚀 Começo rápido (pip)

Instale o Chibi via pip e execute-o como um aplicativo de linha de comando:

```bash
# Instalar o pacote
pip install chibi-bot

# Configurar o agente (adicionar chaves de API, atualizar configurações, etc.)
chibi config

# Iniciar o bot
chibi start
```

O bot será executado como um serviço em segundo plano. Use comandos de CLI para gerenciá-lo.
## 🚀 Começo rápido (Docker)

Crie `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # Obrigatório
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # Ou qualquer outro provedor
      # Adicione mais chaves de API conforme necessário
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) Pegue um token de bot com o [@BotFather](https://t.me/BotFather)

2) Coloque os segredos no `.env`

3) Rode:

```bash
docker-compose up -d
```

Próximos passos:
- **Guia de instalação:** https://chibi.bot/installation
- **Referência de configuração:** https://chibi.bot/configuration

---

## 🔑 Obter chaves de API

Cada provedor requer sua própria chave de API. Aqui estão os links diretos:

**Provedores principais:**
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
- **OpenRouter** (acesso unificado a muitos modelos): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)

**Ferramentas criativas:**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **Guia completo com instruções de configuração:** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## Experimente nos primeiros 5 minutos

Cole isto no Telegram depois de fazer o deploy.

1) **Planejamento + execução**
> Faça 3 perguntas para esclarecer meu objetivo, depois proponha um plano e execute o passo 1.

2) **Trabalho em paralelo (subagentes)**
> Crie 3 subagentes: um para pesquisar opções, um para rascunhar uma recomendação e um para listar riscos. Retorne uma única decisão.

3) **Modo agente (ferramentas)**
> Inspecione os arquivos do projeto e resuma o que este repositório faz. Depois proponha 5 melhorias e abra uma checklist.

4) **Tarefa em segundo plano**
> Inicie uma tarefa em segundo plano: reúna fontes sobre X e entregue uma síntese em 30 minutos. Mantenha-me atualizado.

---

## O que torna o Chibi diferente

### 🎭 Orquestração multi-provedor
O Chibi consegue manter o contexto enquanto troca de provedor no meio da conversa, ou escolher o melhor modelo por etapa — equilibrando **custo**, **capacidade** e **velocidade**.

### 🤖 Capacidades de agente autônomo
- **Delegação recursiva:** subagentes podem criar seus próprios subagentes
- **Processamento em segundo plano:** tarefas longas executam de forma assíncrona
- **Acesso ao sistema de arquivos:** ler/escrever/pesquisar/organizar arquivos
- **Execução no terminal:** rodar comandos com segurança moderada por LLM
- **Memória persistente:** histórico de conversa sobrevive a reinícios com gestão de contexto/sumarização

### 🔌 Extensível via MCP (Model Context Protocol)
Conecte o Chibi a ferramentas e serviços externos (ou crie os seus):

- GitHub (PRs, issues, code review)
- Automação de navegador
- Docker / serviços de nuvem
- Bancos de dados
- Ferramentas criativas (Blender, Figma)

Se uma ferramenta puder ser exposta via MCP, o Chibi pode aprender a usá-la.

### 🎨 Geração de conteúdo rica
- **Imagens:** Nano Banana, Imagen, Qwen, Wan, DALL·E, Grok
- **Música:** Suno (inclui modo custom: estilo/letra/voz)
- **Voz:** transcrição + texto-para-fala (ElevenLabs, MiniMax, OpenAI)

---

## Casos de uso

**Desenvolvedores**
```
Você: “Rode os testes e conserte o que estiver quebrado. Eu vou trabalhar no frontend.”
Chibi: *cria um subagente, executa testes, analisa falhas, propõe correções*
```

**Pesquisadores**
```
Você: “Pesquise os últimos avanços em computação quântica. Preciso de uma síntese até amanhã.”
Chibi: *cria múltiplos agentes de pesquisa, agrega fontes, entrega um relatório*
```

**Criadores**
```
Você: “Gere uma cidade cyberpunk e componha uma faixa synthwave para combinar.”
Chibi: *gera uma imagem, cria música, entrega ambos*
```

**Times**
```
Você: “Revise este PR e atualize a documentação de acordo.”
Chibi: *analisa mudanças, sugere melhorias, atualiza docs via MCP*
```

---

## Privacidade, controle e segurança

- **Auto-hospedado:** seus dados ficam na sua infraestrutura
- **Modo público:** usuários podem trazer suas próprias chaves de API (não é necessária uma chave mestra compartilhada)
- **Controle de acesso:** whitelist de usuários/grupos/modelos
- **Opções de armazenamento:** volumes locais, Redis ou DynamoDB
- **Segurança de ferramentas:** ferramentas do agente são configuráveis; execução no terminal é moderada e pode ser restrita

---

## Documentação

- **Comece aqui:** https://chibi.bot
- Introdução e filosofia: https://chibi.bot/introduction
- Instalação: https://chibi.bot/installation
- Configuração: https://chibi.bot/configuration
- Modo agente: https://chibi.bot/agent-mode
- Guia de MCP: https://chibi.bot/guides/mcp
- Suporte / troubleshooting: https://chibi.bot/support

---

## Requisitos do sistema

- **Mínimo:** Raspberry Pi 4 / AWS EC2 t4g.nano (2 vCPU, 512MB RAM)
- **Arquiteturas:** `linux/amd64`, `linux/arm64`
- **Dependências:** Docker (e opcionalmente Docker Compose)

---

## Contribuindo

- Issues: https://github.com/s-nagaev/chibi/issues
- PRs: https://github.com/s-nagaev/chibi/pulls
- Discussões: https://github.com/s-nagaev/chibi/discussions

Por favor, leia [CONTRIBUTING.md](CONTRIBUTING.md) antes de enviar.

---

## Licença

MIT — veja [LICENSE](LICENSE).

---

<p align="center">
  <strong>Pronto para conhecer seu companheiro digital?</strong><br/>
  <a href="https://chibi.bot/start"><strong>Começar →</strong></a>
</p>
