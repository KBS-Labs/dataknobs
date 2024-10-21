#!/bin/bash
#
# Start (dockerhub) elasticsearch (within the "devnet" network)
#
# Usage:
#   start_elasticsearch.sh [debug]
#
# Where "debug" is any non-empty text and, when present, starts the elasticsearch
# docker container interactively.
#
# This script will:
#
#   1. Remove *any* existing (exited) container(s)
#   2. Start the elasticsearch container if it exists
#   3. Or run the container, either
#      a. Starting the Elasticsearch server or
#      b. Dropping into a bash shell
#

DEBUG="$1"

tag=8.15.2
#tag=8.4.2
#tag=8.5.1

DOCKER_CMD="docker"
if test -n "$(uname -a | grep -i linux)"; then
    DOCKER_CMD="sudo docker"
fi

test -z "$($DOCKER_CMD network list | grep devnet)" && $DOCKER_CMD network create --attachable devnet

$DOCKER_CMD container prune -f

ES_CONTAINER_ID=`$DOCKER_CMD container ls -a | grep elasticsearch | cut -d\  -f1`
if test -z "$ES_CONTAINER_ID"; then
    if test -n "$DEBUG"; then
        echo "$DOCKER_CMD run --name elasticsearch --rm -it --net devnet -v $HOME/data:/data -v $HOME/data/docker_es/data:/usr/share/elasticsearch/data -v $HOME/data/docker_es/config:/usr/share/elasticsearch/config -p 9200:9200 -p 9300:9300 -e \"discovery.type=single-node\" -e TINI_SUBREAPER=true -e ES_JAVA_OPTS=\"-Xms1g -Xmx2g\"  elasticsearch:$tag"
        #$DOCKER_CMD run --name elasticsearch --rm -it --net devnet -v $HOME/data:/data -v $HOME/data/docker_es/data:/usr/share/elasticsearch/data -v $HOME/data/docker_es/config:/usr/share/elasticsearch/config -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e TINI_SUBREAPER=true -e ES_JAVA_OPTS="-Xms1g -Xmx2g" --entrypoint /bin/bash elasticsearch:$tag
        $DOCKER_CMD run --name elasticsearch --rm -it --net devnet -v $HOME/data:/data -v $HOME/data/docker_es/data:/usr/share/elasticsearch/data -v $HOME/data/docker_es/config:/usr/share/elasticsearch/config -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e TINI_SUBREAPER=true -e ES_JAVA_OPTS="-Xms1g -Xmx2g"  elasticsearch:$tag
    else
        echo "$DOCKER_CMD run -d --name elasticsearch --net devnet -v $HOME/data:/data -v $HOME/data/docker_es/data:/usr/share/elasticsearch/data -v $HOME/data/docker_es/config:/usr/share/elasticsearch/config -p 9200:9200 -p 9300:9300 -e \"discovery.type=single-node\" -e TINI_SUBREAPER=true -e ES_JAVA_OPTS=\"-Xms1g -Xmx2g\"  elasticsearch:$tag"
        $DOCKER_CMD run -d --name elasticsearch --net devnet -v $HOME/data:/data -v $HOME/data/docker_es/data:/usr/share/elasticsearch/data -v $HOME/data/docker_es/config:/usr/share/elasticsearch/config -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e TINI_SUBREAPER=true -e ES_JAVA_OPTS="-Xms1g -Xmx2g"  elasticsearch:$tag
    fi;
else
    $DOCKER_CMD start $ES_CONTAINER_ID
fi

