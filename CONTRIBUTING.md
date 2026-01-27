# Contributing to Chibi

Thanks for your interest in contributing to Chibi! Whether you're fixing a bug, adding a feature, or improving documentation, your help is welcome.

## 🤝 How to Contribute

### Reporting Issues

Found a bug or have a feature request?

1. **Search existing issues** first to avoid duplicates
2. **Open a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (OS, Python version, Docker version)
   - Relevant logs (redact any API keys!)

### Suggesting Features

Have an idea for a new capability?

1. **Check the roadmap** in issues/discussions first
2. **Open a feature request** describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches you considered
   - How it fits Chibi's "Digital Companion" philosophy

### Contributing Code

#### Getting Started

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/chibi.git
   cd chibi
   ```

3. **Set up development environment:**
   ```bash
   # Install Poetry if you haven't
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install

   # Activate virtual environment
   poetry shell
   ```

4. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

#### Development Workflow

1. **Make your changes**
   - Follow existing code style (we use `ruff` for linting)
   - Add type hints where appropriate
   - Keep commits atomic and well-described

2. **Run tests:**
   ```bash
   pytest
   ```

3. **Run linters:**
   ```bash
   ruff check .
   ruff format .
   ```

4. **Test manually** (if applicable):
   ```bash
   # Set up test environment variables
   cp .env.example .env
   # Edit .env with test API keys

   # Run the bot
   python main.py
   ```

#### Code Style

- **Python 3.11+** features are welcome
- **Type hints** for function signatures
- **Docstrings** Google style preferred
- **Async/await** always :)
- **Descriptive variable names** over comments

**Example:**
```python
async def fetch_user_history(
    user_id: int,
    limit: int = 100
) -> list[Message]:
    """Fetch conversation history for a user.
    
    Args:
        user_id: Telegram user ID
        limit: Maximum number of messages to retrieve
        
    Returns:
        List of Message objects, newest first
    """
    # Implementation
```

#### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add support for new AI provider`
- `fix: resolve memory leak in task manager`
- `docs: update MCP integration guide`
- `refactor: simplify provider registry logic`
- `test: add tests for file editor tools`

#### Pull Requests

1. **Push your branch** to your fork
2. **Open a Pull Request** against `main`
3. **Fill out the PR template** with:
   - What changed and why
   - How to test it
   - Any breaking changes
   - Related issues (use `Fixes #123` to auto-close)

4. **Respond to review feedback**
   - Be open to suggestions
   - Keep discussions focused and respectful
   - Update your PR based on feedback

### Contributing Documentation

Documentation improvements are highly valued!

- **README.md** — Main project overview
- **chibi-doc/** — Full documentation (separate repo)
- **Code comments** — Explain "why", not "what"
- **Docstrings** — Public API documentation

### Adding AI Providers

Want to add support for a new AI provider?

1. **Create provider class** in `chibi/services/providers/`
   - Extend `RestApiFriendlyProvider`, `OpenAIFriendlyProvider` or `AnthropicFriendlyProvider` base class
   - Handle tool calls
   - Add error handling with retries if necessary

2.**Add configuration** in `chibi/config/gpt.py`

3.**Update documentation** with setup instructions

4.**Add tests** for the new provider

**Example structure:**
```python
from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class VeryNewProvider(OpenAIFriendlyProvider):
    api_key = gpt_settings.very_new_provider_key
    chat_ready = True

    name = "VeryNewProvider"
    model_name_keywords = ["model_substring"]
    base_url = "https://api.verynewprovider.com"
    default_model = "verynewprovider-chat"
```

### Adding Tools

Want to add a new tool for the agent?

1. **Create tool class** in `chibi/services/tools/`
   - Extend `ChibiTool` base class
   - Define `name`, `description`, `parameters`
   - Implement `function()` method

2. **Register tool** (automatic via `ChibiTool` metaclass)

3. **Add tests** for the tool

4. **Update documentation** if it's a major feature

**Example:**
```python
from chibi.services.tools.tool import ChibiTool

class MyNewTool(ChibiTool):
    register = True
    name = "my_new_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input parameter"}
        },
        "required": ["input"]
    }
    
    @classmethod
    async def function(cls, input: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        ...
```

## 🧪 Testing

- **Unit tests** for business logic
- **Integration tests** for provider interactions (use mocks)
- **Manual testing** for UI/UX changes

Run tests with:
```bash
pytest                    # All tests
pytest tests/test_foo.py  # Specific file
pytest -v                 # Verbose output
pytest -k "test_name"     # Specific test
```

## 📋 Code Review Process

1. **Automated checks** must pass (linting, tests)
2. **Maintainer review** — usually within a few days
3. **Feedback iteration** — address comments
4. **Approval & merge** — squash or rebase as appropriate

## 🎯 Project Philosophy

When contributing, keep Chibi's core principles in mind:

- **Companion, not tool** — Design for partnership, not utility
- **Autonomous by default** — Minimize user intervention
- **Provider-agnostic** — No vendor lock-in
- **Privacy-first** — User data stays with the user
- **Extensible** — MCP and modular architecture
- **Production-ready** — Reliability over features

## 💬 Community

- **GitHub Issues** — Bug reports and feature requests
- **GitHub Discussions** — Questions and ideas
- **Telegram Group** — Real-time chat (link in README)

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

**Thank you for making Chibi better!** 🎉

Every contribution, no matter how small, helps build a more capable digital companion for everyone.
