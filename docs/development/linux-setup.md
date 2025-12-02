# Linux Development Setup

This guide covers Linux-specific setup instructions and troubleshooting for developing with Dataknobs.

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+**: `python3 --version`
- **Docker & Docker Compose**: `docker --version` and `docker compose version`
- **UV Package Manager**: `uv --version` (or install with `pip install uv`)
- **Git**: `git --version`

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kbs-labs/dataknobs.git
cd dataknobs

# Install dependencies
uv sync --all-packages

# Install the dk command
./setup-dk.sh

# Start development services
dk up

# Run quality checks
dk pr
```

## Docker Compose Configuration

### Docker Compose v2 vs v1

Linux systems typically use Docker Compose v2 (plugin) instead of the standalone `docker-compose` command:

- **Docker Compose v2 (plugin)**: `docker compose` (no hyphen)
- **Docker Compose v1 (standalone)**: `docker-compose` (with hyphen)

The `dk` command and `manage-services.sh` script automatically detect which version is installed and use the appropriate command.

If you encounter issues, verify your installation:

```bash
# Check for v2 plugin (preferred)
docker compose version

# Check for v1 standalone
docker-compose --version
```

### Starting Services

Start all development services (PostgreSQL, Elasticsearch, LocalStack):

```bash
dk up
```

This command:
1. Creates necessary data directories in `~/.dataknobs/data/`
2. Starts Docker containers for PostgreSQL, Elasticsearch, and LocalStack
3. Waits for services to become healthy

## Common Issues and Solutions

### Elasticsearch Startup Failure

**Problem**: Elasticsearch fails to start with permission errors.

```
Elasticsearch failed to start after 90 seconds
```

**Cause**: Elasticsearch runs as UID 1000 inside the container, but the data directory may have different ownership.

**Solution**: Change ownership of the Elasticsearch data directory:

```bash
sudo chown -R 1000:1000 ~/.dataknobs/data/elasticsearch
```

After fixing permissions, restart services:

```bash
dk restart
```

### PostgreSQL Startup Failure

**Problem**: PostgreSQL fails to start within the timeout period.

**Solutions**:

1. **Check if port 5432 is in use**:
   ```bash
   sudo lsof -i :5432
   ```

   If another service is using the port, stop it or change the port in `docker-compose.yml`.

2. **Check container logs**:
   ```bash
   dk logs postgres
   ```

3. **Clean and restart**:
   ```bash
   dk down
   docker volume rm dataknobs_postgres_data 2>/dev/null || true
   dk up
   ```

### Service Health Check Issues

If services start but health checks fail:

1. **Increase timeout**: Edit `bin/manage-services.sh` to increase timeout values

2. **Check Docker resources**: Ensure Docker has sufficient memory allocated

3. **Check system resources**:
   ```bash
   free -h          # Check available memory
   df -h ~/.dataknobs  # Check disk space
   ```

## Ollama Setup

Ollama must be installed locally (not in Docker) to access GPU hardware.

### Installation

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### Pull Required Models

For running the integration tests:

```bash
ollama pull gemma3:1b
```

For running examples:

```bash
ollama pull gemma3:1b           # General chat
ollama pull nomic-embed-text    # Embeddings for RAG
ollama pull phi3:mini           # Tool-calling (optional)
```

### Verify Ollama

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Or use the check script
./bin/check-ollama.sh
```

### Running Tests with Ollama

By default, tests that require Ollama are skipped. To run them:

```bash
# Run all tests including Ollama tests
TEST_OLLAMA=true dk test

# Or just the integration tests
TEST_OLLAMA=true dk int
```

## Data Directories

Dataknobs stores data in `~/.dataknobs/`:

```
~/.dataknobs/
├── data/
│   ├── elasticsearch/    # Elasticsearch data (owned by UID 1000)
│   ├── postgres/         # PostgreSQL data
│   └── localstack/       # LocalStack data
└── logs/                 # Service logs
```

### Permissions Summary

| Directory | Required Owner | Notes |
|-----------|---------------|-------|
| `~/.dataknobs/data/elasticsearch/` | 1000:1000 | Elasticsearch container runs as UID 1000 |
| `~/.dataknobs/data/postgres/` | Your user | PostgreSQL container handles this |
| `~/.dataknobs/data/localstack/` | Your user | LocalStack container handles this |

### Resetting Data

To completely reset all service data:

```bash
dk down
rm -rf ~/.dataknobs/data/*
dk up
```

**Note**: Remember to fix Elasticsearch permissions after resetting:

```bash
mkdir -p ~/.dataknobs/data/elasticsearch
sudo chown -R 1000:1000 ~/.dataknobs/data/elasticsearch
```

## SELinux Considerations

On systems with SELinux enabled (Fedora, RHEL, CentOS), you may need to adjust contexts for mounted volumes:

```bash
# Check SELinux status
getenforce

# If Enforcing, you may need to add :z or :Z to volume mounts
# or run:
sudo chcon -Rt svirt_sandbox_file_t ~/.dataknobs/data
```

## Firewall Configuration

If using a firewall (ufw, firewalld), ensure Docker bridge networking works:

### UFW (Ubuntu)

```bash
# Allow Docker traffic
sudo ufw allow from 172.16.0.0/12
```

### Firewalld (Fedora/RHEL)

```bash
# Trust Docker zone
sudo firewall-cmd --zone=trusted --add-interface=docker0 --permanent
sudo firewall-cmd --reload
```

## Differences from macOS

| Feature | macOS | Linux |
|---------|-------|-------|
| Docker Compose | `docker-compose` or `docker compose` | Usually `docker compose` (v2 plugin) |
| Elasticsearch permissions | Works automatically | May need `chown 1000:1000` |
| Ollama installation | `brew install ollama` | `curl -fsSL https://ollama.ai/install.sh \| sh` |
| File ownership in Docker | Mapped to current user | Needs explicit UID/GID handling |

## Troubleshooting Checklist

When services fail to start, check these in order:

1. **Docker is running**: `docker info`
2. **Ports are available**: `sudo lsof -i :5432 -i :9200 -i :4566`
3. **Docker Compose version**: `docker compose version`
4. **Data directory exists**: `ls -la ~/.dataknobs/data/`
5. **Elasticsearch permissions**: `ls -ln ~/.dataknobs/data/elasticsearch/`
6. **Container logs**: `dk logs` or `docker logs <container-name>`
7. **System resources**: `free -h` and `df -h`

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/kbs-labs/dataknobs/issues)
2. Run diagnostics: `dk diagnose`
3. Create a new issue with:
   - Linux distribution and version (`cat /etc/os-release`)
   - Docker version (`docker --version`)
   - Docker Compose version (`docker compose version`)
   - Full error output
   - Steps to reproduce
