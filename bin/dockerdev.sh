#!/bin/bash

BIN_DIR=$(dirname "${BASH_SOURCE[0]}")

PROJ_DIR="${BIN_DIR}/.."

test -e "${PROJ_DIR}/.project_vars" && . "${PROJ_DIR}/.project_vars"
test -e "${PROJ_DIR}/.env" && . "${PROJ_DIR}/.env"

NOTEBOOK_PORT=${NOTEBOOK_PORT:=8888}
DOCUMENTATION_PORT=${DOCUMENTATION_PORT:=8000}

# Environment variables for database and services
# Use service names since we're on the devnet network
DATABASE_URL="postgresql://postgres:postgres@postgres:5432/dataknobs"
ELASTICSEARCH_URL="http://elasticsearch:9200"
AWS_ENDPOINT_URL="http://localstack:4566"
AWS_ACCESS_KEY_ID="test"
AWS_SECRET_ACCESS_KEY="test"
AWS_DEFAULT_REGION="us-east-1"
S3_BUCKET="dataknobs-local"

$BIN_DIR/start_docker_process.sh \
    -p $NOTEBOOK_PORT:$NOTEBOOK_PORT \
    -p $DOCUMENTATION_PORT:$DOCUMENTATION_PORT \
    -e DATABASE_URL:$DATABASE_URL \
    -e ELASTICSEARCH_URL:$ELASTICSEARCH_URL \
    -e AWS_ENDPOINT_URL:$AWS_ENDPOINT_URL \
    -e AWS_ACCESS_KEY_ID:$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY:$AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION:$AWS_DEFAULT_REGION \
    -e S3_BUCKET:$S3_BUCKET \
    --network_name devnet
