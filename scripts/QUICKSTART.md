# Dobby Quick Start

## TL;DR

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/s-nagaev/chibi/main/scripts/install.sh | bash

# Start
dobby run

# View logs
dobby attach

# Stop
dobby free
```

## What You Need

1. **Telegram Bot Token** (get from [@BotFather](https://t.me/BotFather))
2. **At least one API key**:
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/
   - Gemini: https://aistudio.google.com/app/apikey
   - DeepSeek: https://platform.deepseek.com/
3. **Your Telegram username or ID**

## Installation Flow

The installer will:
1. ✓ Check requirements (git, Python 3.11+, Poetry)
2. ✓ Clone repo to `~/dobby`
3. ✓ Install dependencies
4. ✓ Ask for API keys (optional, press Enter to skip)
5. ✓ Ask for whitelist user (REQUIRED)
6. ✓ Ask for Telegram token (REQUIRED)
7. ✓ Create `.env` file
8. ✓ Setup command aliases

## Commands

| Command           | Description                |
|-------------------|----------------------------|
| `dobby run`       | Start the agent            |
| `dobby stop`      | Stop the agent             |
| `dobby attach`    | Watch live logs            |
| `dobby logs [n]`  | Show last n lines          |
| `dobby status`    | Check if running           |
| `dobby update`    | Update to latest version   |
| `dobby uninstall` | Remove completely          |

## Configuration

Edit `~/dobby/.env` to customize:

```env
# Required
TELEGRAM_BOT_TOKEN=your_token
USERS_WHITELIST=your_username

# Recommended (at least one)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...

# Pre-configured
BOT_NAME=Dobby
FILESYSTEM_ACCESS=True
MAX_HISTORY_TOKENS=200000
MAX_TOKENS=32000
MAX_CONVERSATION_AGE_MINUTES=2400
```

## Security

⚠️ **Dobby has full filesystem access**

Only whitelist users you **absolutely trust**.

What Dobby can do:
- ✓ Read any file
- ✓ Write/modify files
- ✓ Execute shell commands
- ✓ Install packages
- ✓ Delete files

Built-in protections:
- ✓ Command moderation (blocks dangerous operations)
- ✓ Secret protection (won't read .env, credentials)
- ✓ Whitelist enforcement
- ✓ Operation logging

**Recommendations**:
1. Use dedicated user account
2. Run in VM or container (optional)
3. Monitor logs regularly
4. Keep whitelist minimal

## First Run

After installation:

```bash
# Option 1: Use alias (after restarting terminal)
dobby run

# Option 2: Direct run
cd ~/dobby && poetry run python main.py
```

In Telegram:
```
You: /start
Dobby: Hello! I'm Dobby, your autonomous AI agent...

You: Create a hello.py file
Dobby: [Creates file]
       ✓ Created hello.py successfully

You: Show me the code
Dobby: [Shows file content]
```

## Troubleshooting

### Command not found

```bash
# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Make permanent
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Bot not responding

```bash
# Check logs
dobby logs

# Common issues:
# - Invalid Telegram token → check .env
# - Wrong whitelist → check .env
# - No API keys → add at least one

# Restart after fixing
dobby restart
```

### Permission errors

```bash
chmod +x ~/dobby/scripts/dobby-control.sh
chmod 600 ~/dobby/.env
```

## Getting Help

- **Full documentation**: [scripts/README.md](README.md)
- **Demo & examples**: [scripts/DEMO.md](DEMO.md)
- **Configuration template**: [scripts/dobby.env.example](dobby.env.example)
- **Project docs**: [Main README](../README.md)
- **Issues**: https://github.com/s-nagaev/chibi/issues

## Next Steps

1. **Configure providers**: Add more API keys to `.env`
2. **Customize prompt**: Edit `ASSISTANT_PROMPT` in `.env`
3. **Enable web search**: Add `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_CX`
3. **Enable web search**: Add `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_CX`
4. **Setup metrics**: Configure InfluxDB (optional)
## Example Workflows

### Development Assistant

```
You: Analyze this Python file and suggest improvements
Dobby: [Reads file, analyzes, suggests changes]

You: Apply the changes
Dobby: [Modifies file]
       ✓ Changes applied
```

### System Administration

```
You: Check disk space
Dobby: [Runs df -h]
       / - 45% used
       /home - 78% used

You: Find large files in /home
Dobby: [Runs find command]
       Top 5 largest files:
       1. video.mp4 (2.3GB)
       2. backup.tar.gz (1.8GB)
       ...
```

### Code Generation

```
You: Create a FastAPI app with user authentication
Dobby: [Creates multiple files]
       ✓ main.py
       ✓ models.py
       ✓ auth.py
       ✓ requirements.txt
       
       Ready to run with: uvicorn main:app --reload
```

## Uninstall

```bash
dobby uninstall
```

Then manually remove aliases from `~/.zshrc` or `~/.bashrc`:
```bash
# Look for and delete:
# DOBBY ALIASES (Auto-generated)
```

---

**Ready to start?**

```bash
curl -fsSL https://raw.githubusercontent.com/s-nagaev/chibi/main/scripts/install.sh | bash
```
