#!/usr/bin/env bash
# Development helper script for viewTag
# Usage: ./dev.sh <command>

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

show_help() {
    cat <<EOF
viewTag Development Helper

Usage: ./dev.sh <command>

Commands:
  setup         Create venv, install deps, install pre-commit hooks
  test          Run all tests (unit + integration)
  test-unit     Run unit tests only
  test-int      Run integration tests only
  lint          Run ruff linter and formatter
  type-check    Run mypy type checker
  format        Auto-format code with ruff
  check         Run all checks (lint + type-check + tests)
  run           Run API server (uses .env config)
  run-docker    Run API in Docker (builds if needed)
  build-docker  Build Docker image
  clean         Clean cache and build artifacts
  pre-commit    Install/update pre-commit hooks
  help          Show this help

Examples:
  ./dev.sh setup
  ./dev.sh test
  ./dev.sh run
  ./dev.sh run-docker
EOF
}

cmd_setup() {
    log_info "Setting up development environment..."
    python -m venv .venv
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    else
        echo "No se encontró el entorno virtual .venv"

    fi
    python -m pip install --upgrade pip
    python -m pip install -e .[test,dev]
    pre-commit install
    log_success "Setup complete!"
}

cmd_test() {
    log_info "Running all tests..."
    python -m pytest tests -v --tb=short
}

cmd_test_unit() {
    log_info "Running unit tests..."
    python -m pytest tests/unit -v --tb=short
}

cmd_test_int() {
    log_info "Running integration tests..."
    python -m pytest tests/integration -v --tb=short -m "integration"
}

cmd_lint() {
    log_info "Running ruff linter..."
    ruff check src tests
}

cmd_format() {
    log_info "Formatting code with ruff..."
    ruff format src tests
    ruff check --fix src tests
}

cmd_type_check() {
    log_info "Running mypy type checker..."
    mypy src/
}

cmd_check() {
    log_info "Running all checks..."
    cmd_lint
    cmd_type_check
    cmd_test_unit
    log_success "All checks passed!"
}

cmd_run() {
    log_info "Starting API server..."
    if [ -f .env ]; then
        log_info "Using .env configuration"
    fi
    viewtag-api
}

cmd_run_docker() {
    log_info "Starting API with Docker Compose..."
    docker compose up --build
}

cmd_build_docker() {
    log_info "Building Docker image..."
    docker build -t viewtag .
}

cmd_clean() {
    log_info "Cleaning cache and build artifacts..."
    rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__ .coverage htmlcov
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    log_success "Clean complete!"
}

cmd_pre_commit() {
    log_info "Installing/updating pre-commit hooks..."
    pre-commit install
    pre-commit autoupdate
    log_success "Pre-commit hooks ready!"
}

# Main command dispatcher
case "${1:-help}" in
    setup)          cmd_setup ;;
    test)           cmd_test ;;
    test-unit)      cmd_test_unit ;;
    test-int)       cmd_test_int ;;
    lint)           cmd_lint ;;
    format)         cmd_format ;;
    type-check)     cmd_type_check ;;
    check)          cmd_check ;;
    run)            cmd_run ;;
    run-docker)     cmd_run_docker ;;
    build-docker)   cmd_build_docker ;;
    clean)          cmd_clean ;;
    pre-commit)     cmd_pre_commit ;;
    help|*)         show_help ;;
esac
