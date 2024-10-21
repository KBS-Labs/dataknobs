#/bin/bash

if test -e "${PROJ_DIR}/.project_vars"; then
    . "${PROJ_DIR}/.project_vars";
fi;
if test -e "${PROJ_DIR}/.env"; then
    . "${PROJ_DIR}/.env";
fi;

FLASK_PORT=${FLASK_PORT:=5000};

ENV=${ENV:=dev};

if [ "${ENV}" != "prod" ]; then
    echo "Starting DEV flask server on $(cat /etc/hosts | grep $(hostname))"
    poetry run python -m dataknobs.flask_api
else
    echo "Starting PROD flask server on $(cat /etc/hosts | grep $(hostname))"
    uwsgi --http 127.0.0.1:${FLASK_PORT} --master -p 1 -w dataknobs.flask_api:app
fi;
