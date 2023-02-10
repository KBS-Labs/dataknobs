#!/bin/bash
#
# Script to start a jupyter notebook server.
#

test -e .project_vars && . .project_vars

set -e

poetry install

poetry run jupyter nbextension enable --py --sys-prefix widgetsnbextension

poetry run jupyter notebook --no-browser --ip 0.0.0.0 --port ${NOTEBOOK_PORT} --allow-root
