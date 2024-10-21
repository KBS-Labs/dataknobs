#!/bin/bash

CUR_FILE=$(realpath "${BASH_SOURCE[0]}")
BIN_DIR=$(dirname "${CUR_FILE}")
PROJ_DIR="$(dirname ${BIN_DIR})"

PROJ_NAME="$(basename ${PROJ_DIR} | sed s/-/_/g)"

echo "CUR_FILE=$CUR_FILE, BIN_DIR=$BIN_DIR, PROJ_DIR=$PROJ_DIR, PROJ_NAME=$PROJ_NAME"

env_file=""

if test -e "${PROJ_DIR}/.project_vars"; then
    . "${PROJ_DIR}/.project_vars"
    env_file="--env-file ${PROJ_DIR}/.project_vars"
fi

if test -e "${PROJ_DIR}/.env"; then
    . "${PROJ_DIR}/.env"
    env_file="--env-file ${PROJ_DIR}/.env"
fi

FLASK_PORT=${FLASK_PORT:=5000}

$BIN_DIR/start_docker_process.sh -e ENV=prod --prod ${env_file} -p $FLASK_PORT:$FLASK_PORT --entrypoint ""
