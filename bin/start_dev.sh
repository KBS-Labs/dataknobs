#!/bin/bash
#
# Script to start a development shell
#

test -e .project_vars && . .project_vars

set -e

poetry install

/bin/bash -i
