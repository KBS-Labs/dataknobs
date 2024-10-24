#!/bin/bash
#
# Script to start a development shell
#

test -e .project_vars && . .project_vars
test -e .env && . .env

set -e

poetry install

/bin/bash -i
