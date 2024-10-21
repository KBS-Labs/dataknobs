#!/bin/bash

BIN_DIR=$(dirname "${BASH_SOURCE[0]}")

PROJ_DIR="${BIN_DIR}/.."

test -e "${PROJ_DIR}/.project_vars" && . "${PROJ_DIR}/.project_vars"
test -e "${PROJ_DIR}/.env" && . "${PROJ_DIR}/.env"

NOTEBOOK_PORT=${NOTEBOOK_PORT:=8888}

$BIN_DIR/start_docker_process.sh -p $NOTEBOOK_PORT:$NOTEBOOK_PORT
