#!/bin/bash
# Diagnostic script to test service connectivity

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Service Connectivity Diagnostic ===${NC}"
echo ""

# Detect if running inside Docker
IN_DOCKER=false
if [ -f /.dockerenv ] || [ -n "${DOCKER_CONTAINER}" ]; then
    IN_DOCKER=true
    echo -e "${YELLOW}Running inside Docker container${NC}"
else
    echo -e "${YELLOW}Running on host machine${NC}"
fi

echo ""
echo -e "${BLUE}Environment Information:${NC}"
echo "Hostname: $(hostname)"
echo "IP Address: $(hostname -I 2>/dev/null || echo 'N/A')"

# Check network interfaces
echo ""
echo -e "${BLUE}Network Interfaces:${NC}"
ip addr show 2>/dev/null | grep -E "^[0-9]+:|inet " | head -10 || echo "ip command not available"

# Check DNS resolution
echo ""
echo -e "${BLUE}DNS Resolution:${NC}"
if [ "$IN_DOCKER" = true ]; then
    for host in postgres elasticsearch localstack; do
        echo -n "$host: "
        getent hosts $host 2>/dev/null || nslookup $host 2>/dev/null | grep Address | tail -1 || echo "Cannot resolve"
    done
else
    echo "Using localhost for all services"
fi

# Test raw connectivity
echo ""
echo -e "${BLUE}Testing Raw Connectivity:${NC}"

test_connection() {
    local service=$1
    local host=$2
    local port=$3
    
    echo -n "  $service ($host:$port): "
    
    # Try netcat first
    if command -v nc &> /dev/null; then
        if nc -zv -w2 $host $port &>/dev/null; then
            echo -e "${GREEN}✓ Connected${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed (nc)${NC}"
            return 1
        fi
    fi
    
    # Try Python as fallback
    if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('$host', $port)); s.close()" 2>/dev/null; then
        echo -e "${GREEN}✓ Connected${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed (python)${NC}"
        return 1
    fi
}

if [ "$IN_DOCKER" = true ]; then
    test_connection "PostgreSQL" "postgres" 5432
    test_connection "Elasticsearch" "elasticsearch" 9200
    test_connection "LocalStack" "localstack" 4566
else
    test_connection "PostgreSQL" "localhost" 5432
    test_connection "Elasticsearch" "localhost" 9200
    test_connection "LocalStack" "localhost" 4566
fi

# Test service health
echo ""
echo -e "${BLUE}Testing Service Health:${NC}"

# PostgreSQL
echo -n "  PostgreSQL: "
if [ "$IN_DOCKER" = true ]; then
    if python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(host='postgres', port=5432, user='postgres', password='postgres', database='postgres', connect_timeout=2)
    conn.close()
    print('Connected')
    exit(0)
except Exception as e:
    print(f'Failed: {e}')
    exit(1)
" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
else
    if PGPASSWORD=postgres psql -h localhost -U postgres -c "SELECT 1" postgres &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
fi

# Elasticsearch
echo -n "  Elasticsearch: "
if [ "$IN_DOCKER" = true ]; then
    RESPONSE=$(curl -s -w "\n%{http_code}" http://elasticsearch:9200/_cluster/health 2>/dev/null | tail -1)
else
    RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:9200/_cluster/health 2>/dev/null | tail -1)
fi

if [ "$RESPONSE" = "200" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ (HTTP $RESPONSE)${NC}"
fi

# LocalStack
echo -n "  LocalStack S3: "
if [ "$IN_DOCKER" = true ]; then
    RESPONSE=$(curl -s http://localstack:4566/_localstack/health 2>/dev/null | grep -o '"s3":"[^"]*"' | cut -d'"' -f4)
else
    RESPONSE=$(curl -s http://localhost:4566/_localstack/health 2>/dev/null | grep -o '"s3":"[^"]*"' | cut -d'"' -f4)
fi

if [ "$RESPONSE" = "available" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ ($RESPONSE)${NC}"
fi

# Check Docker networks
if [ "$IN_DOCKER" = false ]; then
    echo ""
    echo -e "${BLUE}Docker Network Status:${NC}"
    docker network ls | grep devnet || echo "devnet network not found"
    
    echo ""
    echo -e "${BLUE}Container Status:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "postgres|elasticsearch|localstack" || echo "Services not running"
fi

echo ""
echo -e "${BLUE}=== Diagnostic Complete ===${NC}"