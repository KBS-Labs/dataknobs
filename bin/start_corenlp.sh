#!/bin/bash
#
# Start Stanford CoreNLP (docker, within the "devnet" network)
#
# Usage:
#   start_corenlp.sh [debug]
#
# Where "debug" is any non-empty text and, when present, starts an interactive
# shell in the CoreNLP docker container without starting the server.
#
# This script will:
#
#   1. Build the "corenlp" docker image if it doesn't exist
#   2. Remove *any* existing (exited) container(s)
#   3. Start the corenlp container if it exists
#   4. Or run the container, either
#      a. Starting the CoreNLP server or
#      b. Dropping into a bash shell
#

tag=1.0

DEBUG="$1"

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

test -z "$(docker network list | grep devnet)" && docker network create devnet

gpus=""
NVSMI="$(nvidia-smi > /dev/null; echo $?)"
[ ${NVSMI} -eq 0 ] && gpus="--gpus all"

if test -z "$(docker images | grep corenlp)"; then
    echo "Building corenlp..."
    docker build --compress \
        -t corenlp:${tag} \
        -f "${SCRIPT_DIR}/../docker/stanford/corenlp/Dockerfile" .
    echo "...done building corenlp"
fi

CONTAINER_ID=''

docker container prune -f

DOCKER_CMD="docker"
if test -n "$(uname -a | grep -i linux)"; then
    DOCKER_CMD="sudo docker"
fi

CONTAINER_ID=`docker container ls -a | grep corenlp | cut -d\  -f1`
if test -z "$CONTAINER_ID"; then
    if test -n "$DEBUG"; then
        $DOCKER_CMD run --init --name corenlp --rm -it $gpus --net devnet -v $HOME/data:/data -p 9000:9000 --entrypoint /bin/bash corenlp:${tag};
    else
        $DOCKER_CMD run -d --name corenlp $gpus --net devnet -v $HOME/data:/data -p 9000:9000 corenlp:${tag}
    fi;
else
    $DOCKER_CMD start $ES_CONTAINER_ID;
fi
