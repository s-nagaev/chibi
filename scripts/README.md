# Dobby Installation Scripts

## 📁 Files in This Directory

| File                    | Description                                          |
|-------------------------|------------------------------------------------------|
| **install.sh**          | Main installation script (cyberpunk-styled)          |
| **dobby-control.sh**    | Control script for managing Dobby lifecycle          |
| **dobby.env.example**   | Example .env file with all configuration options     |
| **QUICKSTART.md**       | TL;DR quick start guide                              |
| **README.md**           | This file - full documentation                       |

---

# Dobby Installation Scripts

## Quick Install (Agent Mode with Filesystem Access)

⚠️ **WARNING**: This installation mode enables **full filesystem access**. Only use on trusted systems with proper security measures.

### One-Line Installation

```bash
curl -fsSL https://raw.githubusercontent.com/s-nagaev/chibi/main/scripts/install.sh | bash
```

Or download and inspect first:

```bash
curl -fsSL https://raw.githubusercontent.com/s-nagaev/chibi/main/scripts/install.sh -o install-dobby.sh
chmod +x install-dobby.sh
./install-dobby.sh
```

### What It Does

1. **Pre-flight checks**: Verifies git, Python 3.11+, and Poetry
2. **Repository cloning**: Clones Chibi to `~/dobby`
3. **Dependency installation**: Installs packages via Poetry
4. **API key configuration**: Interactive prompts for provider keys
5. **Security setup**: Configures user whitelist (REQUIRED)
6. **Telegram token**: Sets up bot token (REQUIRED)
7. **Alias creation**: Creates convenient shell commands

### Configuration

The installer will prompt for:

**Required**:
- Telegram Bot Token
- User whitelist (username or ID)

**Optional** (press Enter to skip):
- OpenAI API Key
- Anthropic API Key
- Google Gemini API Key
- DeepSeek API Key
- MistralAI API Key
- Alibaba DashScope API Key
- xAI Grok API Key
- MoonshotAI API Key

### Hardcoded Settings

The following settings are pre-configured:

```env
BOT_NAME=Dobby
FILESYSTEM_ACCESS=True
MAX_HISTORY_TOKENS=200000
MAX_TOKENS=32000
MAX_CONVERSATION_AGE_MINUTES=2400
SHOW_USAGE=True
```

## Available Commands

After installation, these commands will be available:

| Command             | Description                          |
|---------------------|--------------------------------------|
| `dobby run`         | Start Dobby agent                    |
| `dobby attach`      | View Dobby logs (tail -f)            |
| `dobby free`        | Stop Dobby agent                     |
| `dobby update`      | Update Dobby (git pull + poetry install) |
| `dobby uninstall`   | Remove Dobby completely              |

### Starting Dobby

After installation, restart your terminal (or run `source ~/.zshrc` / `source ~/.bashrc`), then:

```bash
dobby run
```

To view logs:

```bash
dobby attach
```

To stop:

```bash
dobby free
```

## Manual Installation

If you prefer manual setup:

```bash
# Clone repository
git clone https://github.com/s-nagaev/chibi.git ~/dobby
cd ~/dobby

# Install dependencies
poetry install

# Create .env file
cp .env.example .env
# Edit .env with your configuration

# Run
poetry run python main.py
```

## Security Considerations

### Filesystem Access

When `FILESYSTEM_ACCESS=True`, the agent can:
- Read any file accessible to the user
- Write/modify files
- Execute shell commands
- Create and delete files/directories

### Recommendations

1. **Use dedicated user account**: Run Dobby under a restricted user
2. **Whitelist only trusted users**: Be extremely selective
3. **Monitor logs**: Regularly check for suspicious activity
4. **Use read-only mode**: Consider disabling write operations if not needed
5. **Isolate environment**: Use containers or VMs for additional isolation

### Command Moderation

The bot includes command moderation to prevent:
- Access to sensitive files (`.env`, credentials)
- Dangerous operations (rm -rf /, etc.)
- Secret exposure

However, a determined malicious user can still cause harm. **Only whitelist users you absolutely trust.**

## Uninstallation

To completely remove Dobby:

```bash
dobby uninstall
```

This will:
1. Stop the running agent
2. Remove Poetry virtual environment
3. Delete `~/dobby` directory
4. Prompt for manual alias removal

You'll need to manually remove aliases from your shell config (`~/.zshrc` or `~/.bashrc`).

## Troubleshooting

### Poetry Not Found After Installation

If Poetry is installed but not found, add it to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add this line to your `~/.zshrc` or `~/.bashrc` to make it permanent.

### Python Version Issues

Ensure Python 3.11+ is installed:

```bash
python3 --version
```

If needed, install via:
- **macOS**: `brew install python@3.11`
- **Ubuntu**: `sudo apt install python3.11`
- **Fedora**: `sudo dnf install python3.11`

### Permission Denied Errors

If you encounter permission errors:

```bash
chmod +x ~/dobby/scripts/dobby-control.sh
chmod 600 ~/dobby/.env
```

## Development Mode

To contribute or develop:

```bash
cd ~/dobby
poetry install --with dev
task test  # Run tests
task lint  # Run linters
```

See [`.project/agents.md`](../.project/agents.md) for development workflow.

## Support

- **Issues**: https://github.com/s-nagaev/chibi/issues
- **Documentation**: See main [README.md](../README.md)
- **Project docs**: `.project/` directory

---

**Remember**: With great power comes great responsibility. Use Dobby wisely.
