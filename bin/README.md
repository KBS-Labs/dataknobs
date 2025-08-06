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