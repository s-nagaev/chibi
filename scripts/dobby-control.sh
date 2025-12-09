#!/usr/bin/env bash

# ═══════════════════════════════════════════════════════════════════════
# Dobby Control Script
# Manages Dobby agent lifecycle
# ═══════════════════════════════════════════════════════════════════════

DOBBY_DIR="$HOME/dobby"
PID_FILE="$DOBBY_DIR/.dobby.pid"
LOG_DIR="$DOBBY_DIR/logs"

# Colors
CYAN='\033[96m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
RESET='\033[0m'

# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

log_info() {
    echo -e "${CYAN}[Dobby]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[Dobby]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[Dobby]${RESET} $1"
}

log_error() {
    echo -e "${RED}[Dobby]${RESET} $1"
}

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

get_pid() {
    [[ -f "$PID_FILE" ]] && cat "$PID_FILE"
}

# ═══════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════

cmd_start() {
    if is_running; then
        log_warning "Dobby is already running (PID: $(get_pid))"
        return 1
    fi
    
    log_info "Starting Dobby agent..."
    
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    
    # Start Dobby in background
    cd "$DOBBY_DIR" || exit 1
    nohup poetry run python main.py > "$LOG_DIR/dobby.log" 2>&1 &
    echo $! > "$PID_FILE"
    
    sleep 2
    
    if is_running; then
        log_success "Dobby started (PID: $(get_pid))"
        log_info "Use 'dobby attach' to view logs"
    else
        log_error "Failed to start Dobby"
        rm -f "$PID_FILE"
        return 1
    fi
}

cmd_stop() {
    if ! is_running; then
        log_warning "Dobby is not running"
        rm -f "$PID_FILE"
        return 1
    fi
    
    local pid
    pid=$(get_pid)
    
    log_info "Stopping Dobby (PID: $pid)..."
    
    kill "$pid" 2>/dev/null
    
    # Wait for process to terminate (max 10 seconds)
    for i in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warning "Force stopping Dobby..."
        kill -9 "$pid" 2>/dev/null
    fi
    
    rm -f "$PID_FILE"
    log_success "Dobby stopped"
}

cmd_restart() {
    log_info "Restarting Dobby..."
    cmd_stop
    sleep 2
    cmd_start
}

cmd_status() {
    if is_running; then
        log_success "Dobby is running (PID: $(get_pid))"
        
        # Show uptime if ps is available
        if command -v ps &> /dev/null; then
            local uptime
            uptime=$(ps -p "$(get_pid)" -o etime= 2>/dev/null | xargs)
            [[ -n "$uptime" ]] && log_info "Uptime: $uptime"
        fi
    else
        log_warning "Dobby is not running"
        [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
    fi
}

cmd_attach() {
    local log_file="$LOG_DIR/dobby.log"
    
    if [[ ! -f "$log_file" ]]; then
        log_error "Log file not found: $log_file"
        return 1
    fi
    
    log_info "Attaching to Dobby logs (Ctrl+C to exit)..."
    echo ""
    tail -f "$log_file"
}

cmd_logs() {
    local log_file="$LOG_DIR/dobby.log"
    local lines="${1:-50}"
    
    if [[ ! -f "$log_file" ]]; then
        log_error "Log file not found: $log_file"
        return 1
    fi
    
    tail -n "$lines" "$log_file"
}


cmd_settings() {
    log_info "Opening configuration file..."
    
    ENV_FILE="$DOBBY_DIR/.env"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Configuration file not found: $ENV_FILE"
        return 1
    fi
    
    # Detect and use available editor
    if command -v nano >/dev/null 2>&1; then
        nano "$ENV_FILE"
    elif command -v vim >/dev/null 2>&1; then
        vim "$ENV_FILE"
    elif command -v vi >/dev/null 2>&1; then
        vi "$ENV_FILE"
    else
        log_warning "No terminal editor found. File location:"
        echo "$ENV_FILE"
        return 1
    fi
    
    log_success "Configuration edited"
    log_warning "Restart Dobby for changes to take effect: dobby restart"
}
cmd_update() {
    log_info "Updating Dobby..."
    
    local was_running=false
    if is_running; then
        was_running=true
        cmd_stop
    fi
    
    cd "$DOBBY_DIR" || exit 1
    
    log_info "Pulling latest changes..."
    if ! git pull; then
        log_error "Git pull failed"
        return 1
    fi
    
    log_info "Updating dependencies..."
    if ! poetry install; then
        log_error "Poetry install failed"
        return 1
    fi
    
    log_success "Dobby updated successfully"
    
    if [[ "$was_running" == true ]]; then
        log_info "Restarting Dobby..."
        cmd_start
    fi
}

cmd_uninstall() {
    echo -e "${RED}⚠️  WARNING: This will completely remove Dobby${RESET}"
    echo "   - Stop running process (if active)"
    echo "   - Delete $DOBBY_DIR"
    echo "   - Remove virtual environment"
    echo "   - Remove shell aliases"
    echo ""
    echo -ne "${YELLOW}Are you sure? Type 'yes' to confirm:${RESET} "
    read -r confirm
    
    if [[ "$confirm" != "yes" ]]; then
        log_info "Uninstallation cancelled"
        return 0
    fi
    
    # 1. Stop if running
    if is_running; then
        log_info "Stopping Dobby..."
        cmd_stop
    fi
    
    # 2. Remove virtual environment (before deleting directory)
    log_info "Removing virtual environment..."
    cd "$DOBBY_DIR" && poetry env remove --all 2>/dev/null || true
    
    # 3. Remove installation directory
    log_info "Removing installation directory..."
    rm -rf "$DOBBY_DIR"
    
    # 4. Remove aliases from shell configs
    log_info "Removing shell aliases..."
    for rc_file in "$HOME/.zshrc" "$HOME/.bashrc"; do
        if [[ -f "$rc_file" ]]; then
            # Check if aliases exist
            if grep -q "# DOBBY ALIASES" "$rc_file" 2>/dev/null; then
                # Create backup
                cp "$rc_file" "${rc_file}.bak.$(date +%s)"
                # Remove alias section (from marker until 'fi')
                sed -i.tmp '/# DOBBY ALIASES/,/^fi$/d' "$rc_file" 2>/dev/null || \
                    sed -i '' '/# DOBBY ALIASES/,/^fi$/d' "$rc_file" 2>/dev/null
                rm -f "${rc_file}.tmp"
                log_success "Aliases removed from $rc_file"
            fi
        fi
    done
    
    log_success "Dobby uninstalled successfully"
    log_warning "Please restart your shell or run: source ~/.zshrc (or ~/.bashrc)"
}

# Main
# ═══════════════════════════════════════════════════════════════════════

show_usage() {
    cat << EOF
Dobby Control - Manage your AI agent

Usage: dobby <command> [options]

Commands:
    run, start          Start Dobby agent
    stop, free          Stop Dobby agent
    restart             Restart Dobby agent
    status              Check Dobby status
    attach, show        Attach to Dobby logs (live)
    logs [n]            Show last n lines of logs (default: 50)
    settings            Open .env configuration file for editing
    update              Update Dobby
    uninstall           Remove Dobby completely
    help                Show this help message

Examples:
    dobby start         # Start the agent
    dobby show          # Watch live logs
    dobby logs 100      # Show last 100 log lines
    dobby settings      # Edit configuration
    dobby update        # Update to latest version

EOF
}

main() {
    local command="${1:-}"
    
    case "$command" in
        run|start)
            cmd_start
            ;;
        stop|free)
            cmd_stop
            ;;
        restart)
            cmd_restart
            ;;
        status)
            cmd_status
            ;;
        attach|show)
            cmd_attach
            ;;
        logs)
            cmd_logs "${2:-50}"
            ;;
        update)
            cmd_update
            ;;
        settings)
            cmd_settings
            ;;
        uninstall)
            cmd_uninstall
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
