#!/bin/bash
# Script to run integration tests with Docker services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
COMPOSE_OVERRIDE="docker-compose.override.yml"
TEST_PATH="${1:-packages/data/tests/integration}"
VERBOSE="${2:-}"

# Detect if running inside a Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER}" ]; then
    IN_DOCKER=true
fi

echo -e "${GREEN}DataKnobs Integration Test Runner${NC}"
echo "======================================="
if [ "$IN_DOCKER" = true ]; then
    echo -e "${YELLOW}Running inside Docker container${NC}"
fi

# Function to check if service is healthy
check_service_health() {
    local service=$1
    local max_attempts=$2
    local check_command=$3
    
    echo -n "Waiting for $service to be ready..."
    
    for i in $(seq 1 $max_attempts); do
        if eval $check_command 2>/dev/null; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo -e " ${RED}✗${NC}"
    echo -e "${RED}ERROR: $service failed to start after $max_attempts seconds${NC}"
    return 1
}

# Function to cleanup resources
cleanup() {
    if [ "$IN_DOCKER" = false ]; then
        echo -e "\n${YELLOW}Cleaning up...${NC}"
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE down -v 2>/dev/null || true
    fi
}

# Set trap for cleanup on exit (only if not in Docker)
if [ "$IN_DOCKER" = false ]; then
    trap cleanup EXIT INT TERM
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --path|-p)
            TEST_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose       Run tests in verbose mode"
            echo "  -p, --path PATH     Path to integration tests (default: packages/data/tests/integration)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

if [ "$IN_DOCKER" = false ]; then
    # Ensure network exists
    if [ -f "bin/ensure-network.sh" ]; then
        ./bin/ensure-network.sh
    else
        docker network create devnet 2>/dev/null || true
    fi
    
    # Step 1: Start Docker services
    echo -e "\n${YELLOW}Starting Docker services...${NC}"
    docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE up -d postgres elasticsearch localstack
    
    # Step 2: Wait for services to be healthy
    echo -e "\n${YELLOW}Checking service health...${NC}"
    
    # Check PostgreSQL
    check_service_health "PostgreSQL" 30 \
        "docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres pg_isready -U postgres"
    
    # Check Elasticsearch (may take longer to start)
    check_service_health "Elasticsearch" 90 \
        "curl -s http://localhost:9200/_cluster/health | grep -q '\"status\":\"[green|yellow]\"'"
    
    # Check LocalStack (S3)
    check_service_health "LocalStack" 30 \
        "curl -s http://localhost:4566/_localstack/health | grep -q '\"s3\":\"available\"'"
else
    # When in Docker, use Python-based service checks
    echo -e "\n${YELLOW}Checking service connectivity...${NC}"
    
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Use Python script for checking services (works without additional tools)
    # Check PostgreSQL
    check_service_health "PostgreSQL" 30 \
        "python3 ${SCRIPT_DIR}/check-services.py postgres"
    
    # Check Elasticsearch (may take longer to start)
    check_service_health "Elasticsearch" 90 \
        "python3 ${SCRIPT_DIR}/check-services.py elasticsearch"
    
    # Check LocalStack (S3)
    check_service_health "LocalStack" 30 \
        "python3 ${SCRIPT_DIR}/check-services.py localstack"
fi

# Step 3: Set environment variables for tests
echo -e "\n${YELLOW}Setting up test environment...${NC}"

if [ "$IN_DOCKER" = true ]; then
    # Use Docker network hostnames when inside container
    export POSTGRES_HOST=postgres
    export ELASTICSEARCH_HOST=elasticsearch
    export AWS_ENDPOINT_URL=http://localstack:4566
else
    # Use localhost when running on host
    export POSTGRES_HOST=localhost
    export ELASTICSEARCH_HOST=localhost
    export AWS_ENDPOINT_URL=http://localhost:4566
fi

export POSTGRES_PORT=5432
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_DB=dataknobs_test

export ELASTICSEARCH_PORT=9200

export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

# Step 4: Create test database
echo -e "\n${YELLOW}Creating test database...${NC}"
if [ "$IN_DOCKER" = false ]; then
    docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres \
        psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'dataknobs_test'" | grep -q 1 || \
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres \
        psql -U postgres -c "CREATE DATABASE dataknobs_test;" 2>/dev/null || true
else
    # When inside Docker, try to create database using available tools
    # First check if psql is available
    if command -v psql &> /dev/null; then
        # Check if database exists first, then create if needed
        PGPASSWORD=postgres psql -h postgres -p 5432 -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'dataknobs_test'" | grep -q 1 || \
        PGPASSWORD=postgres psql -h postgres -p 5432 -U postgres -c "CREATE DATABASE dataknobs_test;" 2>/dev/null || true
    else
        # Try using Python with psycopg2 if available
        python3 -c "
try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    conn = psycopg2.connect(
        host='postgres',
        port=5432,
        user='postgres',
        password='postgres',
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    # Check if database exists
    cursor.execute(\"SELECT 1 FROM pg_database WHERE datname = 'dataknobs_test'\")
    if not cursor.fetchone():
        cursor.execute('CREATE DATABASE dataknobs_test')
        print('Database created successfully')
    else:
        print('Database already exists')
    cursor.close()
    conn.close()
except ImportError:
    print('Warning: psycopg2 not available, skipping database creation')
    print('The test database may need to be created manually')
except Exception as e:
    print(f'Warning: Could not create database: {e}')
" 2>&1 || true
    fi
fi

# Step 5: Run integration tests
echo -e "\n${YELLOW}Running integration tests...${NC}"
echo "Test path: $TEST_PATH"
echo ""

# Run tests with pytest
# Check if uv is available
if command -v uv &> /dev/null; then
    # Use uv if available
    if [ -n "$VERBOSE" ]; then
        uv run pytest $TEST_PATH \
            -m integration \
            --tb=short \
            -v \
            --color=yes \
            --junit-xml=.quality-artifacts/integration-junit.xml \
            --html=.quality-artifacts/integration-report.html \
            --self-contained-html
    else
        uv run pytest $TEST_PATH \
            -m integration \
            --tb=short \
            --color=yes \
            --junit-xml=.quality-artifacts/integration-junit.xml
    fi
else
    # Fallback to regular pytest
    if [ -n "$VERBOSE" ]; then
        python3 -m pytest $TEST_PATH \
            -m integration \
            --tb=short \
            -v \
            --color=yes \
            --junit-xml=.quality-artifacts/integration-junit.xml
    else
        python3 -m pytest $TEST_PATH \
            -m integration \
            --tb=short \
            --color=yes \
            --junit-xml=.quality-artifacts/integration-junit.xml
    fi
fi

TEST_EXIT_CODE=$?

# Step 6: Display results
echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Integration tests passed successfully!${NC}"
else
    echo -e "${RED}✗ Integration tests failed with exit code $TEST_EXIT_CODE${NC}"
fi

# Step 7: Optional - Keep services running for debugging
if [ "$IN_DOCKER" = false ] && [ "$KEEP_SERVICES" = "true" ]; then
    echo -e "\n${YELLOW}Services are still running. To stop them, run:${NC}"
    echo "docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE down"
    trap - EXIT INT TERM
fi

exit $TEST_EXIT_CODE
