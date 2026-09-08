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
- **OpenRouter** (acesso unificado a muitos modelos)
- **Cloudflare Workers AI** (muitos modelos de código aberto)

### Endpoints compatíveis com OpenAI (auto-hospedado / local)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Qualquer** API compatível com OpenAI

### Provedores multimodais (opcional)

- **Imagens:** Google (Imagen, Nano Banana), OpenAI (GPT Image), Alibaba (Qwen Image, Wan), xAI (Grok Imagine), ZhipuAI (GLM Image), MiniMax, Cheaper Inference (Nano Banana Pro)
- **Música:** Suno
- **Voz:** ElevenLabs, OpenAI (GPT-4o Transcribe / TTS), MiniMax (TTS)

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

### Comandos da CLI

| Comando         | Descrição                                  |
|-----------------|--------------------------------------------|
| `chibi start`   | Inicia o bot como serviço em segundo plano |
| `chibi stop`    | Para o bot em execução                     |
| `chibi restart` | Reinicia o bot                             |
| `chibi config`  | Gera ou edita a configuração               |
| `chibi logs`    | Exibe os logs do bot                       |

---

## Clientes locais

O Chibi pode atender seus clientes locais pelo protocolo JSONL local versionado:

```bash
chibi stdio --tui
chibi stdio --vscode
chibi stdio --pycharm
chibi stdio --neovim
```

O comando deve ser iniciado pelo cliente, não usado interativamente. O Chibi é proprietário do protocolo `v1` e oferece suporte a clientes compatíveis que negociam a mesma versão. O processo stdio grava somente frames do protocolo em stdout e diagnósticos em stderr. O comportamento existente do Chibi para comandos, ferramentas, permissões e moderação permanece autoritativo; o cliente não adiciona uma camada de políticas de ferramentas.

---
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
- **Skills:** módulos de instruções reutilizáveis que o agente pode carregar sob demanda em seu prompt de sistema (`load_builtin_skill`)

### 🔌 Extensível via MCP (Model Context Protocol)
Conecte o Chibi a ferramentas e serviços externos (ou crie os seus):

- GitHub (PRs, issues, code review)
- Automação de navegador
- Docker / serviços de nuvem
- Bancos de dados
- Ferramentas criativas (Blender, Figma)

Se uma ferramenta puder ser exposta via MCP, o Chibi pode aprender a usá-la.

### 🎨 Geração de conteúdo rica
- **Imagens:** Nano Banana, Imagen, Qwen, Wan, GPT Image, Grok Imagine, GLM Image
- **Música:** Suno (inclui modo custom: estilo/letra/voz)
- **Voz:** transcrição + texto-para-fala (ElevenLabs, OpenAI, MiniMax)

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

### `MAX_HISTORY_TOKENS` — limite de sumarização do contexto (alteração incompatível no padrão)

`MAX_HISTORY_TOKENS` é o limite no qual o Chibi resume automaticamente uma conversa para manter o contexto administrável. Sua **semântica mudou**: agora ele compara a **contagem real de tokens do prompt informada pelo provedor** (toda a solicitação enviada: prompt de sistema + Skills ativadas + esquemas de ferramentas + argumentos de chamadas de ferramentas + sobrecarga estrutural por mensagem + conteúdo da conversa), em vez da heurística anterior que media apenas `content` + `role` da conversa. O valor real é cerca de **4,8 vezes maior** que a estimativa antiga para uma conversa idêntica (consulte `fix_context_size/context_size_accounting_analysis.md` para a divisão medida).

- **Padrão recalibrado:** `64000` → `100000`. O novo valor protege a menor janela de contexto com suporte comum (128k tokens): `100000` é ~78% de uma janela de 128k (logo a sumarização dispara *antes* de um modelo de 128k estourar) e ~50% de uma janela de 200k (deixando boa margem). O antigo `64000` era uma estimativa apenas do histórico que nunca era atingida antes de um estouro real, pois subconta cirílico em ~2x e exclui a sobrecarga fixa por turno (prompt de sistema ~3,7k, esquemas de ferramentas ~6,8k, Skills ativadas ~5,9k, `user_info` ~0,75–3k).
- **Migração:** se você definiu `MAX_HISTORY_TOKENS` explicitamente no `.env`, o valor antigo foi ajustado para a estimativa anterior, baseada apenas no histórico, e agora é comparado a uma medida verdadeira ~4,8 vezes maior para a mesma conversa. Recalibre-o para a escala de **~100k** (por exemplo, `64000` → `100000`) para que a sumarização ocorra antes de a menor janela de contexto dos seus modelos estourar, e não depois. Se nunca o definiu, o novo padrão é aplicado automaticamente.
- O fallback de inicialização a frio (primeiro turno após reiniciar o processo, quando ainda não há dados do provedor em cache) continua usando a heurística antiga para que a sumarização permaneça funcional.

### `REACTIVE_CONTEXT_RECOVERY` — nova tentativa única após estouro de contexto

`REACTIVE_CONTEXT_RECOVERY` (padrão: `true`) é a rede de segurança reativa que complementa o limite proativo `MAX_HISTORY_TOKENS`. Quando um provedor rejeita uma solicitação com o erro tipado `context_length_exceeded`, o Chibi resume automaticamente o histórico e tenta novamente o turno **exatamente uma vez**. Se funcionar, você recebe uma resposta normal (com uma observação curta de que o contexto foi comprimido). Se a nova tentativa também estourar ou a recuperação falhar por qualquer motivo, o turno usa o caminho de desculpa existente; o ciclo resumir+tentar novamente nunca pode ocorrer mais de uma vez por turno original. Defina como `false` para desativá-lo e manter o comportamento anterior (log + desculpa, sem nova tentativa).

### O diretório de trabalho tem escopo de thread (`WORKING_DIR`)

O diretório de trabalho do agente — usado para comandos de terminal e informado ao modelo como seu CWD atual — tem escopo **por thread**, assim como o modelo LLM selecionado é vinculado por thread.

- **Padrão:** vem da configuração `WORKING_DIR` (padrão `~/chibi`) pelo valor legado de nível de usuário; novas instalações herdam a configuração diretamente.
- **Substituição por thread:** qualquer thread/conversa pode substituir seu diretório de trabalho **isoladamente** pela ferramenta `set_working_dir` do agente (somente controlada por LLM; não há slash command, disponível quando `FILESYSTEM_ACCESS` está ativado). Isso permite que dois agentes em threads diferentes trabalhem simultaneamente em projetos distintos sem interferência.
- **Ordem de resolução:** substituição da thread → diretório legado do usuário → configuração `WORKING_DIR`.
- **Normalização de caminho:** valores definidos são expandidos para caminhos absolutos ao salvar (`~/x` torna-se `/abs/x`); padrões inalterados mantêm a forma original.
- **Subagentes** criados em uma thread compartilham seu diretório de trabalho — o mesmo caminho efetivo é injetado em seus prompts de sistema e chamadas de ferramentas.
- As substituições **sobrevivem à clonagem de thread**: `/new_thread_with_current_context` leva o diretório de trabalho junto com mensagens e preferências de modelo.

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
