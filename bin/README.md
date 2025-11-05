# dataknobs Development Scripts

This directory contains helper scripts for developing dataknobs packages.

## Main Development Script

### `dev.sh`
The main development helper script with common commands:

```bash
# Set up development environment
./bin/dev.sh setup

# Build all packages
./bin/dev.sh build

# Install packages in dev mode
./bin/dev.sh install

# Run all tests
./bin/dev.sh test

# Run linting and type checking
./bin/dev.sh lint

# Clean build artifacts
./bin/dev.sh clean

# Prepare for release
./bin/dev.sh release
```

## Individual Scripts

### `build-packages.sh`
Build all dataknobs packages in the correct order:

```bash
./bin/build-packages.sh
```

### `install-packages.sh`
Install dataknobs packages with various options:

```bash
# Install in dev mode (default)
./bin/install-packages.sh

# Install in production mode
./bin/install-packages.sh -m prod

# Install in a new virtual environment
./bin/install-packages.sh -e myenv

# Force reinstall
./bin/install-packages.sh -f
```

### `test-packages.sh`
Run tests for dataknobs packages:

```bash
# Test all packages
./bin/test-packages.sh

# Test specific package
./bin/test-packages.sh structures

# Run with verbose output
./bin/test-packages.sh -v

# Generate coverage report
./bin/test-packages.sh -c
```

### `fix.sh`
Auto-fix code issues using ruff:

```bash
# Fix all packages (linting and formatting)
./bin/fix.sh

# Fix specific package
./bin/fix.sh utils

# Format only (skip linting fixes)
./bin/fix.sh -f
```

### `validate.sh`
Validate code quality and catch common errors:

```bash
# Validate all packages
./bin/validate.sh

# Validate specific package
./bin/validate.sh utils

# Quick validation (skip slow checks)
./bin/validate.sh -q

# Validate and attempt auto-fix
./bin/validate.sh -f
```

This script checks:
- Python syntax errors
- Type annotation issues
- Import problems
- Common code quality issues

## Development Services

### Docker-based Services

Most development services (Postgres, Elasticsearch, S3/LocalStack) run via Docker and are managed automatically:

```bash
# Start all services
./bin/manage-services.sh start

# Stop all services
./bin/manage-services.sh stop

# Check service status
./bin/check-services.py
```

### Ollama (Local Installation Required)

Unlike other services, **Ollama runs locally on your development machine** due to hardware requirements (GPU access). It cannot be easily containerized for local development.

#### Installation

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

#### Starting Ollama

```bash
# Start the Ollama service
ollama serve
```

Ollama will run on `http://localhost:11434` by default.

#### Required Models

Pull the models needed for tests:
```bash
ollama pull <model-name>  # e.g., llama2, mistral, etc.
```

Check which models your tests require in the bots package tests.

#### Verifying Ollama

```bash
# Check if Ollama is running
./bin/check-ollama.sh

# Or manually:
curl http://localhost:11434/api/tags
```

#### Running Tests Without Ollama

If you don't have Ollama installed or don't need to run those tests:

```bash
# Skip Ollama tests explicitly
export TEST_OLLAMA=false
dk test

# Or use quick test mode (skips all integration tests)
dk testquick
```

## Quick Start

For a fresh development setup:

```bash
# 1. Set up development environment
./bin/dev.sh setup

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run tests to verify everything works
./bin/dev.sh test
```