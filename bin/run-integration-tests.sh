#!/bin/bash
# Integration test runner with package selection and proper test discovery

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
COMPOSE_OVERRIDE="docker-compose.override.yml"
PACKAGE=""
VERBOSE=""
SERVICES_ONLY=false
SKIP_SERVICES=false
KEEP_SERVICES=false

# Detect if running inside a Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER}" ]; then
    IN_DOCKER=true
fi

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [PACKAGE]

Run integration tests for DataKnobs packages with Docker services.

Options:
    -p, --package PACKAGE  Package to test (e.g., data, config, structures)
                          If not specified, tests all packages with integration tests
    -v, --verbose         Run tests in verbose mode
    -s, --services-only   Only start services, don't run tests
    -n, --no-services     Don't start services (assume they're already running)
    -k, --keep-services   Keep services running after tests
    -h, --help           Show this help message

Examples:
    $0                    # Run integration tests for all packages
    $0 data              # Run integration tests for data package
    $0 -v -p config      # Run config package tests verbosely
    $0 -s                # Only start services
    $0 -n data           # Run data tests without starting services

Available packages with integration tests:
EOF
    
    # List packages with integration tests
    for pkg_dir in "$ROOT_DIR"/packages/*/; do
        if [ -d "$pkg_dir/tests/integration" ]; then
            pkg_name=$(basename "$pkg_dir")
            echo "    - $pkg_name"
        fi
    done
    
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--package)
            PACKAGE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -s|--services-only)
            SERVICES_ONLY=true
            shift
            ;;
        -n|--no-services)
            SKIP_SERVICES=true
            shift
            ;;
        -k|--keep-services)
            KEEP_SERVICES=true
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            # Assume it's a package name if no package specified yet
            if [ -z "$PACKAGE" ]; then
                PACKAGE="$1"
            else
                echo -e "${RED}Unknown option: $1${NC}"
                show_usage
            fi
            shift
            ;;
    esac
done

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

# Function to start services
start_services() {
    echo -e "${YELLOW}Starting Docker services...${NC}"
    
    # Ensure network exists
    docker network create devnet 2>/dev/null || true
    
    # Start services
    docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE up -d postgres elasticsearch localstack
    
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
    
    # Create test database
    echo -e "\n${YELLOW}Creating test database...${NC}"
    docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres \
        psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'dataknobs_test'" | grep -q 1 || \
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres \
        psql -U postgres -c "CREATE DATABASE dataknobs_test;" 2>/dev/null || true
    
    echo -e "${GREEN}Services are ready!${NC}"
}

# Function to cleanup resources
cleanup() {
    if [ "$KEEP_SERVICES" = false ] && [ "$SKIP_SERVICES" = false ] && [ "$IN_DOCKER" = false ]; then
        echo -e "\n${YELLOW}Cleaning up services...${NC}"
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE down -v 2>/dev/null || true
    elif [ "$KEEP_SERVICES" = true ]; then
        echo -e "\n${YELLOW}Services are still running. To stop them, run:${NC}"
        echo "docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE down -v"
    fi
}

# Set trap for cleanup on exit (only if not in Docker)
if [ "$IN_DOCKER" = false ]; then
    trap cleanup EXIT INT TERM
fi

# Main execution
echo -e "${GREEN}DataKnobs Integration Test Runner${NC}"
echo "======================================="

# Start services if needed
if [ "$SKIP_SERVICES" = false ] && [ "$IN_DOCKER" = false ]; then
    start_services
fi

# If services-only mode, exit here
if [ "$SERVICES_ONLY" = true ]; then
    echo -e "\n${GREEN}Services started successfully!${NC}"
    trap - EXIT INT TERM  # Remove cleanup trap
    exit 0
fi

# Set environment variables for tests
echo -e "\n${YELLOW}Setting up test environment...${NC}"

if [ "$IN_DOCKER" = true ]; then
    # Use Docker network hostnames when inside container
    export POSTGRES_HOST=postgres
    export ELASTICSEARCH_HOST=elasticsearch
    export AWS_ENDPOINT_URL=http://localstack:4566
    export LOCALSTACK_ENDPOINT=http://localstack:4566
else
    # Use localhost when running on host
    export POSTGRES_HOST=localhost
    export ELASTICSEARCH_HOST=localhost
    export AWS_ENDPOINT_URL=http://localhost:4566
    export LOCALSTACK_ENDPOINT=http://localhost:4566
fi

export POSTGRES_PORT=5432
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_DB=dataknobs_test
export ELASTICSEARCH_PORT=9200
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1

# Determine which packages to test
if [ -n "$PACKAGE" ]; then
    # Test specific package
    if [ ! -d "$ROOT_DIR/packages/$PACKAGE" ]; then
        echo -e "${RED}Package not found: $PACKAGE${NC}"
        exit 1
    fi
    
    if [ ! -d "$ROOT_DIR/packages/$PACKAGE/tests/integration" ]; then
        echo -e "${RED}No integration tests found for package: $PACKAGE${NC}"
        exit 1
    fi
    
    TEST_PACKAGES=("$PACKAGE")
else
    # Find all packages with integration tests
    TEST_PACKAGES=()
    for pkg_dir in "$ROOT_DIR"/packages/*/; do
        if [ -d "$pkg_dir/tests/integration" ]; then
            pkg_name=$(basename "$pkg_dir")
            TEST_PACKAGES+=("$pkg_name")
        fi
    done
    
    if [ ${#TEST_PACKAGES[@]} -eq 0 ]; then
        echo -e "${YELLOW}No packages with integration tests found${NC}"
        exit 0
    fi
fi

echo -e "Packages to test: ${BLUE}${TEST_PACKAGES[*]}${NC}\n"

# Track overall test result
OVERALL_RESULT=0

# Run tests for each package
for pkg in "${TEST_PACKAGES[@]}"; do
    echo -e "\n${YELLOW}Running integration tests for package: $pkg${NC}"
    echo "----------------------------------------"
    
    TEST_PATH="$ROOT_DIR/packages/$pkg/tests/integration"
    
    # Run ALL tests in the integration directory, not just marked ones
    # This ensures test_s3_backend.py is included
    if command -v uv &> /dev/null; then
        uv run pytest "$TEST_PATH" \
            $VERBOSE \
            --tb=short \
            --color=yes \
            --junit-xml=".quality-artifacts/${pkg}-integration-junit.xml" || OVERALL_RESULT=$?
    else
        python3 -m pytest "$TEST_PATH" \
            $VERBOSE \
            --tb=short \
            --color=yes \
            --junit-xml=".quality-artifacts/${pkg}-integration-junit.xml" || OVERALL_RESULT=$?
    fi
done

# Display results
echo ""
echo "======================================="
if [ $OVERALL_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All integration tests passed!${NC}"
else
    echo -e "${RED}✗ Some integration tests failed${NC}"
fi

exit $OVERALL_RESULT