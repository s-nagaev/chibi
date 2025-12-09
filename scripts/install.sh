#!/usr/bin/env bash

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    DOBBY INSTALLATION PROTOCOL                        ║
# ║                   Autonomous AI Agent Deployment                      ║
# ║                         Version 1.0.0                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

set -e

# Colors & Styles
NEON_CYAN='\033[96m'
NEON_PINK='\033[95m'
NEON_GREEN='\033[92m'
NEON_YELLOW='\033[93m'
NEON_RED='\033[91m'
NEON_BLUE='\033[94m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'
BLINK='\033[5m'

# Installation paths
INSTALL_DIR="$HOME/dobby"
REPO_URL="https://github.com/s-nagaev/chibi.git"

# ═══════════════════════════════════════════════════════════════════════
# ASCII Art Banner
# ═══════════════════════════════════════════════════════════════════════

print_banner() {
    clear
    echo -e "${NEON_CYAN}"
    cat << "EOF"
    ██████╗  ██████╗ ██████╗ ██████╗ ██╗   ██╗
    ██╔══██╗██╔═══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
    ██║  ██║██║   ██║██████╔╝██████╔╝ ╚████╔╝
    ██║  ██║██║   ██║██╔══██╗██╔══██╗  ╚██╔╝  
    ██████╔╝╚██████╔╝██████╔╝██████╔╝   ██║   
    ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝    ╚═╝   
EOF
    echo -e "${RESET}"
    echo -e "${NEON_PINK}╔═══════════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${BOLD}Autonomous AI Agent with Filesystem Access${RESET}                           ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${DIM}Powered by Chibi Framework${RESET}                                           ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}║${RESET}                                                                       ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${NEON_YELLOW}⚠️ EXPERIMENTAL VERSION${RESET}                                              ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${NEON_CYAN}🔗 Repository: ${RESET}${DIM}https://github.com/s-nagaev/chibi${RESET}                     ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}╚═══════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "${NEON_CYAN}╔═══════════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${NEON_CYAN}║${RESET}  ${BOLD}📋 PREREQUISITES${RESET}                                                     ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}╠═══════════════════════════════════════════════════════════════════════╣${RESET}"
    echo -e "${NEON_CYAN}║${RESET}                                                                       ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}  ${NEON_GREEN}1.${RESET} ${BOLD}Telegram Bot Token${RESET}                                                ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}     ${DIM}Get it from @BotFather: ${RESET}${NEON_BLUE}https://t.me/BotFather${RESET}                    ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}                                                                       ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}  ${NEON_GREEN}2.${RESET} ${BOLD}At least one API key from supported providers:${RESET}                    ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}     ${DIM}• OpenAI, Anthropic, Google, DeepSeek, xAI, Alibaba,${RESET}              ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}     ${DIM}  MistralAI, Cloudflare${RESET}                                           ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}     ${DIM}(Full provider links will be shown during installation)${RESET}           ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}                                                                       ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}  ${NEON_GREEN}3.${RESET} ${BOLD}Your Telegram username or user ID${RESET}                                 ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}║${RESET}                                                                       ${NEON_CYAN}║${RESET}"
    echo -e "${NEON_CYAN}╚═══════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

log_info() {
    echo -e "${NEON_CYAN}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${NEON_GREEN}[✓]${RESET} $1"
}

log_warning() {
    echo -e "${NEON_YELLOW}[!]${RESET} $1"
}

log_error() {
    echo -e "${NEON_RED}[✗]${RESET} $1"
}

log_step() {
    echo -e "\n${NEON_PINK}▶${RESET} ${BOLD}$1${RESET}"
}

prompt_input() {
    local prompt_text="$1"
    local var_name="$2"
    local required="${3:-false}"
    
    while true; do
        echo -ne "${NEON_BLUE}${prompt_text}${RESET} "
        read -r input
        
        if [[ -n "$input" ]]; then
            eval "$var_name='$input'"
            break
        elif [[ "$required" == "false" ]]; then
            eval "$var_name=''"
            break
        else
            log_warning "This field is required. Please provide a value."
        fi
    done
}

prompt_secret() {
    local prompt_text="$1"
    local var_name="$2"
    local required="${3:-false}"
    
    while true; do
        echo -ne "${NEON_BLUE}${prompt_text}${RESET} "
        read -rs input
        echo ""
        
        if [[ -n "$input" ]]; then
            eval "$var_name='$input'"
            break
        elif [[ "$required" == "false" ]]; then
            eval "$var_name=''"
            break
        else
            log_warning "This field is required. Please provide a value."
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════════
# Pre-flight Checks
# ═══════════════════════════════════════════════════════════════════════

check_requirements() {
    log_step "Running pre-flight checks..."
    
    # Check for git
    if ! command -v git &> /dev/null; then
        log_error "git is not installed. Please install git first."
        exit 1
    fi
    log_success "git found"
    
    # Check for Python 3.11+
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
        
        if [[ "$PYTHON_MAJOR" -ge 3 ]] && [[ "$PYTHON_MINOR" -ge 11 ]]; then
            log_success "Python $PYTHON_VERSION found"
        else
            log_error "Python 3.11+ is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check for Poetry or offer to install
    if ! command -v poetry &> /dev/null; then
        log_warning "Poetry is not installed"
        echo -ne "${NEON_BLUE}Install Poetry now? [Y/n]${RESET} "
        read -r install_poetry
        
        if [[ "$install_poetry" =~ ^[Nn]$ ]]; then
            log_error "Poetry is required for installation. Aborting."
            exit 1
        fi
        
        log_info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH for this session
        export PATH="$HOME/.local/bin:$PATH"
        
        if ! command -v poetry &> /dev/null; then
            log_error "Poetry installation failed"
            exit 1
        fi
        log_success "Poetry installed successfully"
    else
        log_success "Poetry found"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# Clone Repository
# ═══════════════════════════════════════════════════════════════════════

clone_repository() {
    log_step "Cloning Dobby repository..."
    
    if [[ -d "$INSTALL_DIR" ]]; then
        log_warning "Directory $INSTALL_DIR already exists"
        echo -ne "${NEON_BLUE}Remove existing directory and continue? [y/N]${RESET} "
        read -r remove_dir
        
        if [[ "$remove_dir" =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
            log_success "Removed existing directory"
        else
            log_error "Installation aborted"
            exit 1
        fi
    fi
    
    git clone "$REPO_URL" "$INSTALL_DIR"
    log_success "Repository cloned to $INSTALL_DIR"
}

# ═══════════════════════════════════════════════════════════════════════
# Install Dependencies
# ═══════════════════════════════════════════════════════════════════════

install_dependencies() {
    log_step "Installing dependencies..."
    
    cd "$INSTALL_DIR"
    poetry install --no-interaction --no-ansi
    
    log_success "Dependencies installed"
}

# ═══════════════════════════════════════════════════════════════════════
# Configure Environment
# ═══════════════════════════════════════════════════════════════════════

configure_environment() {
    log_step "Configuring environment..."
    
    local CONFIG_SKIPPED=false
    
    echo ""
    echo -e "${NEON_PINK}╔═══════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${BOLD}API Keys Configuration${RESET}                                   ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${DIM}Press Enter to skip optional providers${RESET}                   ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}╚═══════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    
    # Collect API keys for each provider
    prompt_secret "OpenAI API Key (optional):" OPENAI_API_KEY false
    if [[ -z "$OPENAI_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    prompt_secret "Anthropic API Key (optional):" ANTHROPIC_API_KEY false
    if [[ -z "$ANTHROPIC_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    prompt_secret "Google Gemini API Key (optional):" GEMINI_API_KEY false
    if [[ -z "$GEMINI_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    prompt_secret "DeepSeek API Key (optional):" DEEPSEEK_API_KEY false
    if [[ -z "$DEEPSEEK_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    prompt_secret "MistralAI API Key (optional):" MISTRALAI_API_KEY false
    if [[ -z "$MISTRALAI_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    prompt_secret "Alibaba DashScope API Key (optional):" ALIBABA_API_KEY false
    if [[ -z "$ALIBABA_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    prompt_secret "xAI Grok API Key (optional):" GROK_API_KEY false
    if [[ -z "$GROK_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    prompt_secret "MoonshotAI API Key (optional):" MOONSHOTAI_API_KEY false
    if [[ -z "$MOONSHOTAI_API_KEY" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    echo ""
    echo -e "${NEON_PINK}╔═══════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${BOLD}Security Configuration${RESET}                                   ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${NEON_RED}⚠  CRITICAL: Filesystem access is DANGEROUS${RESET}              ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${DIM}Only whitelist users you absolutely trust${RESET}                ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}╚═══════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    
    # User whitelist (changed to optional)
    prompt_input "Enter your Telegram username or ID (or press Enter to skip): " "USERS_WHITELIST" "false"
    if [[ -z "$USERS_WHITELIST" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    echo ""
    echo -e "${NEON_PINK}╔═══════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${NEON_PINK}║${RESET}  ${BOLD}Telegram Bot Configuration${RESET}                               ${NEON_PINK}║${RESET}"
    echo -e "${NEON_PINK}╚═══════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    
    # Telegram token (changed to optional)
    prompt_secret "Enter your Telegram Bot Token (or press Enter to skip): " "TELEGRAM_BOT_TOKEN" "false"
    if [[ -z "$TELEGRAM_BOT_TOKEN" ]]; then
        CONFIG_SKIPPED=true
    fi
    
    # Create .env file
    log_info "Generating .env file..."
    
    ENV_FILE="$INSTALL_DIR/.env"
    
    cat > "$ENV_FILE" << EOF
# Dobby Configuration
# Generated on $(date)

# Telegram Bot
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
BOT_NAME=Dobby

# Security
USERS_WHITELIST=$USERS_WHITELIST
FILESYSTEM_ACCESS=True

# Model Configuration
MAX_HISTORY_TOKENS=200000
MAX_TOKENS=32000
MAX_CONVERSATION_AGE_MINUTES=2400
SHOW_USAGE=True

# API Keys
EOF
    
    [[ -n "$OPENAI_API_KEY" ]] && echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> "$ENV_FILE"
    [[ -n "$ANTHROPIC_API_KEY" ]] && echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> "$ENV_FILE"
    [[ -n "$GEMINI_API_KEY" ]] && echo "GEMINI_API_KEY=$GEMINI_API_KEY" >> "$ENV_FILE"
    [[ -n "$DEEPSEEK_API_KEY" ]] && echo "DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY" >> "$ENV_FILE"
    [[ -n "$MISTRALAI_API_KEY" ]] && echo "MISTRALAI_API_KEY=$MISTRALAI_API_KEY" >> "$ENV_FILE"
    [[ -n "$ALIBABA_API_KEY" ]] && echo "ALIBABA_API_KEY=$ALIBABA_API_KEY" >> "$ENV_FILE"
    [[ -n "$GROK_API_KEY" ]] && echo "GROK_API_KEY=$GROK_API_KEY" >> "$ENV_FILE"
    [[ -n "$MOONSHOTAI_API_KEY" ]] && echo "MOONSHOTAI_API_KEY=$MOONSHOTAI_API_KEY" >> "$ENV_FILE"
    
    chmod 600 "$ENV_FILE"
    log_success "Configuration saved to .env"
    
    # If any configuration was skipped, open .env file for manual editing
    if [[ "$CONFIG_SKIPPED" == "true" ]]; then
        log_warning "Some configuration was skipped"
        log_info "Opening .env file for manual configuration..."
        echo ""
        
        # Detect editor
        if command -v nano >/dev/null 2>&1; then
            nano "$ENV_FILE"
        elif command -v vim >/dev/null 2>&1; then
            vim "$ENV_FILE"
        elif command -v vi >/dev/null 2>&1; then
            vi "$ENV_FILE"
        else
            log_warning "No terminal editor found. Please edit manually:"
            log_info "File: $ENV_FILE"
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# Setup Aliases
# ═══════════════════════════════════════════════════════════════════════

setup_aliases() {
    log_step "Setting up command aliases..."
    
    if [[ -n "$ZSH_VERSION" ]] || [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    elif [[ -n "$BASH_VERSION" ]] || [[ "$SHELL" == *"bash"* ]]; then
        SHELL_RC="$HOME/.bashrc"
    else
        log_warning "Could not detect shell. Please add aliases manually."
        log_info "Alias file location: $INSTALL_DIR/scripts/dobby-aliases.sh"
        return
    fi
    
    # dobby-control.sh is already in the repo, just make it executable
    chmod +x "$INSTALL_DIR/scripts/dobby-control.sh"
    
    # Check if aliases already exist
    if grep -q "# DOBBY ALIASES" "$SHELL_RC" 2>/dev/null; then
        log_warning "Aliases already exist in $SHELL_RC"
        return
    fi
    
    # Create alias file with actual install directory
    sed "s|__INSTALL_DIR__|$INSTALL_DIR|g" "$INSTALL_DIR/scripts/dobby-aliases.sh" > "$INSTALL_DIR/scripts/dobby-aliases.generated.sh"
    
    # Add source line to shell config
    cat >> "$SHELL_RC" << EOF

# DOBBY ALIASES (Auto-generated - do not edit manually)
if [[ -f "$INSTALL_DIR/scripts/dobby-aliases.generated.sh" ]]; then
    source "$INSTALL_DIR/scripts/dobby-aliases.generated.sh"
fi
EOF
    
    log_success "Aliases added to $SHELL_RC"
}

# ═══════════════════════════════════════════════════════════════════════
# Installation Complete
# ═══════════════════════════════════════════════════════════════════════

print_completion() {
    echo ""
    echo -e "${NEON_GREEN}"
    cat << "EOF"
    ╔═══════════════════════════════════════════════════╗
    ║              INSTALLATION COMPLETE                ║
    ║             Dobby is ready to serve               ║
    ╚═══════════════════════════════════════════════════╝
EOF
    echo -e "${RESET}"
    
    echo -e "${NEON_CYAN}Available commands:${RESET}"
    echo -e "  ${BOLD}dobby run${RESET}        - Start Dobby agent"
    echo -e "  ${BOLD}dobby attach${RESET}     - View Dobby logs"
    echo -e "  ${BOLD}dobby free${RESET}       - Stop Dobby agent"
    echo -e "  ${BOLD}dobby update${RESET}     - Update Dobby"
    echo -e "  ${BOLD}dobby uninstall${RESET}  - Remove Dobby"
    echo ""
    echo -e "${NEON_YELLOW}⚠️  Important:${RESET}"
    echo -e "  • Restart your terminal or run: ${DIM}source $SHELL_RC${RESET}"
    echo -e "  • To start Dobby now: ${DIM}cd $INSTALL_DIR && poetry run python main.py${RESET}"
    echo ""
    echo -e "${NEON_PINK}═══════════════════════════════════════════════════${RESET}"
}

# ═══════════════════════════════════════════════════════════════════════
# Main Installation Flow
# ═══════════════════════════════════════════════════════════════════════

main() {
    print_banner
    
    echo -e "${NEON_YELLOW}⚠️  WARNING:${RESET} This installation enables ${BOLD}FILESYSTEM ACCESS${RESET}"
    echo -e "   The agent will have full moderated access to your filesystem."
    echo -e "   Only proceed if you understand the security implications."
    echo ""
    echo -ne "${NEON_BLUE}Continue with installation? [y/N]${RESET} "
    read -r confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled"
        exit 0
    fi
    
    check_requirements
    clone_repository
    install_dependencies
    configure_environment
    setup_aliases
    print_completion
}

# Run installation
main
