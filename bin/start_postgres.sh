#!/bin/bash
#
# Start (dockerhub) elasticsearch (within the "devnet" network)
#
# Usage:
#   start_postgres.sh [debug]
#
# Where "debug" is any non-empty text and, when present, starts the postgres
# docker container interactively.
#
# This script will:
#
#   1. Remove *any* existing (exited) container(s)
#   2. Start the postgres container if it exists
#   3. Or run the container, either
#      a. Starting the postgres server or
#      b. Dropping into a bash shell
#

DEBUG="$1"

BIN_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJ_DIR="${BIN_DIR}/.."
test -e "${PROJ_DIR}/.project_vars" && . "${PROJ_DIR}/.project_vars"
test -e "${PROJ_DIR}/.env" && . "${PROJ_DIR}/.env"
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:=pgpass123}

tag=latest
#tag=8.4.2
#tag=8.5.1

DOCKER_CMD="docker"
if test -n "$(uname -a | grep -i linux)"; then
    DOCKER_CMD="sudo docker"
fi

test -z "$($DOCKER_CMD network list | grep devnet)" && $DOCKER_CMD network create --attachable devnet

$DOCKER_CMD container prune -f

PG_CONTAINER_ID=`$DOCKER_CMD container ls -a | grep postgres | cut -d\  -f1`
if test -z "$PG_CONTAINER_ID"; then
    if test -n "$DEBUG"; then
        echo "$DOCKER_CMD run --rm -it --name postgres --net devnet -p 5432:5432 -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} -e PGDATA=/var/lib/postgresql/data/pgdata -v $HOME/data/docker_pg/data:/var/lib/postgresql/data postgres"
        #$DOCKER_CMD run --rm -it --name postgres --net devnet -p 5432:5432 -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} -e PGDATA=/var/lib/postgresql/data/pgdata -v $HOME/data/docker_pg/data:/var/lib/postgresql/data --entrypoint /bin/bash postgres
        $DOCKER_CMD run --rm -it --name postgres --net devnet -p 5432:5432 -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} -e PGDATA=/var/lib/postgresql/data/pgdata -v $HOME/data/docker_pg/data:/var/lib/postgresql/data postgres
    else
        echo "$DOCKER_CMD run -d --name postgres --net devnet -p 5432:5432 -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} -e PGDATA=/var/lib/postgresql/data/pgdata -v $HOME/data/docker_pg/data:/var/lib/postgresql/data postgres"
        $DOCKER_CMD run -d --name postgres --net devnet -p 5432:5432 -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} -e PGDATA=/var/lib/postgresql/data/pgdata -v $HOME/data/docker_pg/data:/var/lib/postgresql/data postgres
    fi;
else
    $DOCKER_CMD start $PG_CONTAINER_ID
fi
