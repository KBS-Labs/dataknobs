# Docker Setup for Dataknobs

This directory contains Docker configurations for the Dataknobs project.

## Files

- **Dockerfile.dev**: Development environment with Jupyter and additional tools
- **Dockerfile.prod**: Production environment optimized for running the Flask API

## Usage

### Using Docker Compose (Recommended)

From the project root:

```bash
# Start development server
docker-compose up dataknobs-dev

# Start production server
docker-compose up dataknobs-prod

# Start Jupyter notebook server
docker-compose up jupyter
```

### Building Images Manually

From the project root:

```bash
# Build development image
docker build -f docker/Dockerfile.dev -t dataknobs:dev .

# Build production image
docker build -f docker/Dockerfile.prod -t dataknobs:prod .
```

### Running Containers Manually

```bash
# Run development container
docker run -it -v $(pwd):/workdir -p 5000:5000 dataknobs:dev

# Run production container
docker run -it -p 5000:5000 dataknobs:prod
```

## Key Changes from Poetry to uv

1. **Package Manager**: Now using `uv` instead of `poetry`
2. **Commands**: Use `uv run` instead of `poetry run`
3. **Dependencies**: Managed via `uv sync` with workspace-level lock file
4. **Performance**: Faster dependency resolution and installation

## Environment Variables

- `ENV`: Set to "dev" or "prod" to control server mode
- `FLASK_PORT`: Port for the Flask API (default: 5000)