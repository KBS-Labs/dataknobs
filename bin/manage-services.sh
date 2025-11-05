#!/bin/bash
# Service management script for DataKnobs test infrastructure
# This script provides unified service management for integration tests and quality checks

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
SERVICES_STARTED_FLAG="/tmp/.dataknobs_services_started_$$"
VERBOSE=false

# Default action
ACTION=""
SELECTED_SERVICES=()

# Function to auto-detect available services from docker-compose files
detect_services() {
    local services=()
    
    # Parse services from docker-compose.override.yml (excluding main app services)
    if [ -f "$ROOT_DIR/$COMPOSE_OVERRIDE" ]; then
        # Use yq if available, otherwise fall back to basic parsing
        if command -v yq &> /dev/null; then
            services=($(yq eval '.services | keys | .[]' "$ROOT_DIR/$COMPOSE_OVERRIDE" | grep -v '^dataknobs'))
        elif command -v python3 &> /dev/null; then
            # Use Python as fallback for YAML parsing
            services=($(python3 -c "
import yaml
import sys
with open('$ROOT_DIR/$COMPOSE_OVERRIDE', 'r') as f:
    data = yaml.safe_load(f)
    if 'services' in data:
        for service in data['services']:
            if not service.startswith('dataknobs'):
                print(service)
" 2>/dev/null || echo ""))
        else
            # Basic grep fallback - less reliable but works for simple cases
            services=($(grep -E '^[[:space:]]*[a-z_-]+:$' "$ROOT_DIR/$COMPOSE_OVERRIDE" | grep -v 'dataknobs' | sed 's/://g' | sed 's/^[[:space:]]*//' | sort -u))
        fi
    fi
    
    # If no services detected, use sensible defaults
    if [ ${#services[@]} -eq 0 ]; then
        services=("postgres" "elasticsearch" "localstack")
    fi
    
    echo "${services[@]}"
}

# Get available services
AVAILABLE_SERVICES=($(detect_services))

# Function to get health check command for a service
get_health_check() {
    local service=$1
    
    if is_in_docker; then
        # Inside Docker - use network hostnames
        case "$service" in
            postgres)
                echo "pg_isready -h $POSTGRES_HOST -U postgres"
                ;;
            elasticsearch)
                echo "curl -s http://$ELASTICSEARCH_HOST:9200/_cluster/health | grep -qE '\"status\":\"(green|yellow)\"'"
                ;;
            localstack)
                echo "curl -s http://$LOCALSTACK_HOST:4566/_localstack/health | jq -r '.services.s3' 2>/dev/null | grep -q 'available'"
                ;;
            *)
                echo ""
                ;;
        esac
    else
        # On host - use docker-compose exec
        case "$service" in
            postgres)
                echo "docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres pg_isready -U postgres"
                ;;
            elasticsearch)
                echo "curl -s http://$ELASTICSEARCH_HOST:9200/_cluster/health | grep -qE '\"status\":\"(green|yellow)\"'"
                ;;
            localstack)
                echo "curl -s http://$LOCALSTACK_HOST:4566/_localstack/health | jq -r '.services.s3' 2>/dev/null | grep -q 'available'"
                ;;
            *)
                echo ""
                ;;
        esac
    fi
}

# Function to get wait time for a service
get_wait_time() {
    local service=$1
    case "$service" in
        elasticsearch)
            echo 90
            ;;
        postgres|localstack)
            echo 30
            ;;
        *)
            echo 30
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS] [SERVICES...]

Manage Docker services for DataKnobs testing and development.

Commands:
    start       Start test services
    stop        Stop test services
    restart     Restart test services
    status      Check status of test services
    logs        Show logs for services
    cleanup     Stop services and remove volumes
    ensure      Start services if not running (used by other scripts)
    list        List available services

Options:
    -v, --verbose     Show detailed output
    -h, --help        Show this help message

Services:
    If no services are specified, all available services will be affected.
    Available services (auto-detected):
EOF
    
    # List detected services
    for service in "${AVAILABLE_SERVICES[@]}"; do
        echo "        - $service"
    done
    
    cat << EOF

Environment Variables:
    KEEP_SERVICES     If set to 'true', services won't be stopped automatically
    SKIP_SERVICES     If set to 'true', service management will be skipped

Examples:
    $0 start                      # Start all test services
    $0 start postgres             # Start only PostgreSQL
    $0 start postgres elasticsearch  # Start PostgreSQL and Elasticsearch
    $0 stop elasticsearch         # Stop only Elasticsearch
    $0 status                     # Check status of all services
    $0 status postgres            # Check status of PostgreSQL only
    $0 logs elasticsearch -v      # Show verbose logs for Elasticsearch
    $0 list                       # List available services
    $0 ensure                     # Start all services if needed (for automation)

Service Started Flag:
    When services are started by this script, a flag file is created at:
    $SERVICES_STARTED_FLAG
    
    This allows calling scripts to know if they started the services
    and should clean them up when done.
EOF
    exit 0
}

# Function to print status messages
print_status() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to check if we're in Docker
is_in_docker() {
    if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER}" ]; then
        return 0
    fi
    return 1
}

# Set service hostnames based on environment
if is_in_docker; then
    # Inside Docker - use service names as hostnames
    POSTGRES_HOST="postgres"
    ELASTICSEARCH_HOST="elasticsearch"
    LOCALSTACK_HOST="localstack"
else
    # On host machine - use localhost
    POSTGRES_HOST="localhost"
    ELASTICSEARCH_HOST="localhost"
    LOCALSTACK_HOST="localhost"
fi

# Function to check if Docker is running
check_docker() {
    # Skip Docker check if we're inside a container
    if is_in_docker; then
        return 0
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check service health
check_service_health() {
    local service=$1
    local max_attempts=$2
    local check_command=$3
    
    if [ "$VERBOSE" = true ]; then
        echo -n "Waiting for $service to be ready..."
    fi
    
    for i in $(seq 1 $max_attempts); do
        if eval $check_command 2>/dev/null; then
            if [ "$VERBOSE" = true ]; then
                echo -e " ${GREEN}✓${NC}"
            fi
            return 0
        fi
        if [ "$VERBOSE" = true ]; then
            echo -n "."
        fi
        sleep 1
    done
    
    if [ "$VERBOSE" = true ]; then
        echo -e " ${RED}✗${NC}"
    fi
    print_error "$service failed to start after $max_attempts seconds"
    return 1
}

# Function to start services
start_services() {
    # Check if we're in Docker container
    if is_in_docker; then
        print_warning "Cannot start services from within Docker container"
        print_status "Services should be managed from the host machine"
        return 1
    fi
    
    check_docker
    
    # Determine which services to start
    local services_to_start=()
    if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then
        services_to_start=("${AVAILABLE_SERVICES[@]}")
        print_status "Starting all Docker services..."
    else
        services_to_start=("${SELECTED_SERVICES[@]}")
        print_status "Starting selected services: ${services_to_start[*]}"
    fi
    
    # Change to project root for docker-compose
    cd "$ROOT_DIR"
    
    # Ensure network exists
    docker network create devnet 2>/dev/null || true
    
    # Start services
    if docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE up -d "${services_to_start[@]}" 2>/dev/null; then
        print_success "Services started"
    else
        print_warning "Some services may already be running"
    fi
    
    print_status "Checking service health..."
    
    # Check health of each started service
    for service in "${services_to_start[@]}"; do
        local health_check=$(get_health_check "$service")
        if [ -n "$health_check" ]; then
            local wait_time=$(get_wait_time "$service")
            # Capitalize service name for display
            case "$service" in
                postgres) local service_name="PostgreSQL" ;;
                elasticsearch) local service_name="Elasticsearch" ;;
                localstack) local service_name="LocalStack" ;;
                *) local service_name="$service" ;;
            esac
            
            if ! check_service_health "$service_name" "$wait_time" "$health_check"; then
                # Special handling for non-critical services
                if [ "$service" = "localstack" ]; then
                    print_warning "$service_name may not be fully ready"
                else
                    return 1
                fi
            else
                print_success "$service_name is ready"
            fi
        else
            print_warning "No health check defined for $service"
        fi
    done
    
    # Special post-start actions for specific services
    if [[ " ${services_to_start[@]} " =~ " postgres " ]]; then
        # Create test database if PostgreSQL was started
        print_status "Ensuring test database exists..."
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres \
            psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'dataknobs_test'" | grep -q 1 || \
            docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres \
            psql -U postgres -c "CREATE DATABASE dataknobs_test;" 2>/dev/null || true
    fi
    
    # Mark that we started the services
    touch "$SERVICES_STARTED_FLAG"
    
    if [ ${#services_to_start[@]} -eq 1 ]; then
        print_success "Service is ready!"
    else
        print_success "Services are ready!"
    fi
    return 0
}

# Function to stop services
stop_services() {
    # Check if we're in Docker container
    if is_in_docker; then
        print_warning "Cannot stop services from within Docker container"
        print_status "Services should be managed from the host machine"
        return 1
    fi
    
    check_docker
    
    # Determine which services to stop
    local services_to_stop=()
    if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then
        # Stop all with docker-compose down
        print_status "Stopping all Docker services..."
        cd "$ROOT_DIR"
        
        if docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE down 2>/dev/null; then
            print_success "All services stopped"
        else
            print_warning "Services may not have been running"
        fi
    else
        # Stop specific services
        services_to_stop=("${SELECTED_SERVICES[@]}")
        print_status "Stopping selected services: ${services_to_stop[*]}"
        cd "$ROOT_DIR"
        
        if docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE stop "${services_to_stop[@]}" 2>/dev/null; then
            docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE rm -f "${services_to_stop[@]}" 2>/dev/null
            print_success "Selected services stopped"
        else
            print_warning "Some services may not have been running"
        fi
    fi
    
    # Remove the flag file only if we stopped all services
    if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then
        rm -f "$SERVICES_STARTED_FLAG"
    fi
    
    return 0
}

# Function to restart services
restart_services() {
    stop_services
    sleep 2
    start_services
}

# Function to show service status
show_status() {
    check_docker
    
    print_status "Checking service status..."
    
    cd "$ROOT_DIR"
    
    # Determine which services to check
    local services_to_check=()
    if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then
        services_to_check=("${AVAILABLE_SERVICES[@]}")
    else
        services_to_check=("${SELECTED_SERVICES[@]}")
    fi
    
    # Show container status only if on host
    if ! is_in_docker; then
        echo ""
        if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then
            docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE ps
        else
            docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE ps "${services_to_check[@]}"
        fi
        echo ""
    else
        echo ""
        print_status "Running from within Docker container - showing network connectivity"
        echo ""
    fi
    
    # Check individual service health
    echo "Service Health:"
    
    for service in "${services_to_check[@]}"; do
        case "$service" in
            postgres)
                if is_in_docker; then
                    # Inside container - check directly
                    if pg_isready -h $POSTGRES_HOST -U postgres >/dev/null 2>&1; then
                        echo -e "  PostgreSQL:    ${GREEN}✓ Healthy${NC}"
                    else
                        echo -e "  PostgreSQL:    ${RED}✗ Not responding${NC}"
                    fi
                else
                    # On host - use docker-compose
                    if docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
                        echo -e "  PostgreSQL:    ${GREEN}✓ Healthy${NC}"
                    else
                        echo -e "  PostgreSQL:    ${RED}✗ Not responding${NC}"
                    fi
                fi
                ;;
            elasticsearch)
                local es_url="http://$ELASTICSEARCH_HOST:9200/_cluster/health"
                if curl -s "$es_url" >/dev/null 2>&1; then
                    CLUSTER_STATUS=$(curl -s "$es_url" | jq -r '.status' 2>/dev/null || echo "unknown")
                    if [ "$CLUSTER_STATUS" = "green" ]; then
                        echo -e "  Elasticsearch: ${GREEN}✓ Healthy (green)${NC}"
                    elif [ "$CLUSTER_STATUS" = "yellow" ]; then
                        echo -e "  Elasticsearch: ${YELLOW}⚠ Healthy (yellow)${NC}"
                    else
                        echo -e "  Elasticsearch: ${RED}✗ Unhealthy ($CLUSTER_STATUS)${NC}"
                    fi
                else
                    echo -e "  Elasticsearch: ${RED}✗ Not responding${NC}"
                fi
                ;;
            localstack)
                local ls_url="http://$LOCALSTACK_HOST:4566/_localstack/health"
                if curl -s "$ls_url" >/dev/null 2>&1; then
                    local s3_status=$(curl -s "$ls_url" | jq -r '.services.s3' 2>/dev/null || echo "unknown")
                    if [ "$s3_status" = "available" ]; then
                        echo -e "  LocalStack:    ${GREEN}✓ Healthy (S3 available)${NC}"
                    else
                        echo -e "  LocalStack:    ${YELLOW}⚠ Running but S3 not available${NC}"
                    fi
                else
                    echo -e "  LocalStack:    ${RED}✗ Not responding${NC}"
                fi
                ;;
            *)
                # Generic check
                if is_in_docker; then
                    # Inside container - try to ping the service
                    if ping -c 1 -W 1 "$service" >/dev/null 2>&1; then
                        echo -e "  ${service}:    ${GREEN}✓ Reachable${NC}"
                    else
                        echo -e "  ${service}:    ${RED}✗ Not reachable${NC}"
                    fi
                else
                    # On host - check if container is running
                    if docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE ps "$service" 2>/dev/null | grep -q "Up"; then
                        echo -e "  ${service}:    ${GREEN}✓ Running${NC}"
                    else
                        echo -e "  ${service}:    ${RED}✗ Not running${NC}"
                    fi
                fi
                ;;
        esac
    done

    # Check Ollama (non-Docker service)
    echo ""
    echo "Local Services:"
    OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
    if curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        MODEL_COUNT=$(curl -s "${OLLAMA_HOST}/api/tags" 2>/dev/null | grep -o '"name"' | wc -l | tr -d ' ')
        if [ "$MODEL_COUNT" -gt 0 ]; then
            echo -e "  Ollama:        ${GREEN}✓ Running ($MODEL_COUNT models)${NC}"
        else
            echo -e "  Ollama:        ${YELLOW}⚠ Running (no models installed)${NC}"
        fi
    else
        echo -e "  Ollama:        ${YELLOW}⚠ Not running${NC} (install locally or skip with TEST_OLLAMA=false)"
    fi

    return 0
}

# Function to show logs
show_logs() {
    check_docker
    
    cd "$ROOT_DIR"
    
    # Determine which services to show logs for
    local services_to_log=()
    if [ ${#SELECTED_SERVICES[@]} -eq 0 ]; then
        services_to_log=()  # Empty means all
    else
        services_to_log=("${SELECTED_SERVICES[@]}")
    fi
    
    local tail_lines=50
    if [ "$VERBOSE" = true ]; then
        tail_lines=100
    fi
    
    if [ ${#services_to_log[@]} -eq 0 ]; then
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE logs --tail=$tail_lines
    else
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE logs --tail=$tail_lines "${services_to_log[@]}"
    fi
    
    return 0
}

# Function to list available services
list_services() {
    print_status "Available services:"
    echo ""
    
    if is_in_docker; then
        # Inside container - check network connectivity
        for service in "${AVAILABLE_SERVICES[@]}"; do
            case "$service" in
                postgres)
                    if pg_isready -h $POSTGRES_HOST -U postgres >/dev/null 2>&1; then
                        echo -e "  ${GREEN}●${NC} $service (reachable)"
                    else
                        echo -e "  ${RED}○${NC} $service (not reachable)"
                    fi
                    ;;
                elasticsearch)
                    if curl -s "http://$ELASTICSEARCH_HOST:9200" >/dev/null 2>&1; then
                        echo -e "  ${GREEN}●${NC} $service (reachable)"
                    else
                        echo -e "  ${RED}○${NC} $service (not reachable)"
                    fi
                    ;;
                localstack)
                    if curl -s "http://$LOCALSTACK_HOST:4566/_localstack/health" >/dev/null 2>&1; then
                        local s3_status=$(curl -s "http://$LOCALSTACK_HOST:4566/_localstack/health" | jq -r '.services.s3' 2>/dev/null || echo "unknown")
                        if [ "$s3_status" = "available" ]; then
                            echo -e "  ${GREEN}●${NC} $service (reachable, S3 available)"
                        else
                            echo -e "  ${YELLOW}●${NC} $service (reachable, S3: $s3_status)"
                        fi
                    else
                        echo -e "  ${RED}○${NC} $service (not reachable)"
                    fi
                    ;;
                *)
                    if ping -c 1 -W 1 "$service" >/dev/null 2>&1; then
                        echo -e "  ${GREEN}●${NC} $service (reachable)"
                    else
                        echo -e "  ${RED}○${NC} $service (not reachable)"
                    fi
                    ;;
            esac
        done
    else
        # On host - check if containers are running
        for service in "${AVAILABLE_SERVICES[@]}"; do
            if docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE ps "$service" 2>/dev/null | grep -q "Up"; then
                echo -e "  ${GREEN}●${NC} $service (running)"
            else
                echo -e "  ${RED}○${NC} $service (stopped)"
            fi
        done
    fi
    
    echo ""
    return 0
}

# Function to cleanup (stop and remove volumes)
cleanup_services() {
    # Check if we're in Docker container
    if is_in_docker; then
        print_warning "Cannot cleanup services from within Docker container"
        print_status "Services should be managed from the host machine"
        return 1
    fi
    
    check_docker
    
    print_status "Stopping services and removing volumes..."
    
    cd "$ROOT_DIR"
    
    if docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE down -v 2>/dev/null; then
        print_success "Services stopped and volumes removed"
    else
        print_warning "Services may not have been running"
    fi
    
    # Remove the flag file
    rm -f "$SERVICES_STARTED_FLAG"
    
    return 0
}

# Function to ensure services are running (for automation)
ensure_services() {
    # Skip if SKIP_SERVICES is set
    if [ "${SKIP_SERVICES}" = "true" ]; then
        print_status "Skipping service management (SKIP_SERVICES=true)"
        return 0
    fi
    
    # Check if we're in Docker
    if is_in_docker; then
        print_status "Running in Docker container, checking service connectivity..."
        
        # Check if services are reachable from within container
        local all_reachable=true
        
        # Check PostgreSQL
        if ! pg_isready -h $POSTGRES_HOST -U postgres >/dev/null 2>&1; then
            print_warning "PostgreSQL not reachable at $POSTGRES_HOST"
            all_reachable=false
        fi
        
        # Check Elasticsearch
        if ! curl -s "http://$ELASTICSEARCH_HOST:9200/_cluster/health" >/dev/null 2>&1; then
            print_warning "Elasticsearch not reachable at $ELASTICSEARCH_HOST:9200"
            all_reachable=false
        fi
        
        if [ "$all_reachable" = true ]; then
            print_success "All services are reachable from container"
            return 0
        else
            print_error "Some services are not reachable from container"
            print_status "Please ensure services are running on the host machine"
            return 1
        fi
    fi
    
    check_docker
    
    cd "$ROOT_DIR"
    
    # Check if services are already running
    SERVICES_RUNNING=true
    
    # Check PostgreSQL
    if ! docker-compose -f $COMPOSE_FILE -f $COMPOSE_OVERRIDE exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
        SERVICES_RUNNING=false
    fi
    
    # Check Elasticsearch
    if ! curl -s http://localhost:9200/_cluster/health >/dev/null 2>&1; then
        SERVICES_RUNNING=false
    fi
    
    if [ "$SERVICES_RUNNING" = true ]; then
        print_success "Services are already running"
        # Don't create the flag file since we didn't start them
        return 0
    else
        print_status "Services not running, starting them..."
        start_services
        return $?
    fi
}

# Function to check if we started the services
should_cleanup() {
    # Don't cleanup if KEEP_SERVICES is set
    if [ "${KEEP_SERVICES}" = "true" ]; then
        return 1
    fi
    
    # Only cleanup if we started the services (flag file exists)
    if [ -f "$SERVICES_STARTED_FLAG" ]; then
        return 0
    fi
    
    return 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|status|logs|cleanup|ensure|list)
            ACTION="$1"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            # Check if it's a valid service name
            if [[ " ${AVAILABLE_SERVICES[@]} " =~ " $1 " ]]; then
                SELECTED_SERVICES+=("$1")
                shift
            else
                print_error "Unknown option or service: $1"
                echo "Available services: ${AVAILABLE_SERVICES[*]}"
                echo "Use '$0 list' to see all available services"
                exit 1
            fi
            ;;
    esac
done

# Execute the requested action
case $ACTION in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup_services
        ;;
    ensure)
        ensure_services
        ;;
    list)
        list_services
        ;;
    *)
        print_error "No command specified"
        show_usage
        ;;
esac

exit $?