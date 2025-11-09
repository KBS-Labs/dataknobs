#!/usr/bin/env python

import argparse
import json
import os
import sys
from pathlib import Path

from flask import request

from dataknobs import create_app
from dataknobs.utils import sys_utils

app = create_app()


@app.route("/echoArgs")
def echo_args():
    indent = request.args.get("indent", type=int, default=2)
    verbose = request.args.get("verbose", type=bool, default=True)
    rv = request.args.to_dict(flat=False)
    if verbose:
        print(json.dumps(rv, indent=indent), file=sys.stdout)
    return rv


@app.route("/printenv")
def printenv():
    rv = os.environ.copy()
    verbose = request.args.get("verbose", type=bool, default=True)
    if verbose:
        print(json.dumps(rv, indent=2), file=sys.stdout)
    return rv


@app.route("/listdir")
def listdir():
    name = request.args.get("name", default=".")
    verbose = request.args.get("verbose", type=bool, default=True)
    rv = [p.name for p in Path(name).iterdir()]
    if verbose:
        print(json.dumps(rv, indent=2), file=sys.stdout)
    return rv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="The server's port")
    parser.add_argument(
        "--pvname", help="The name of the project variables file", default=".project_vars"
    )
    args = parser.parse_args()
    port = args.port
    pvname = args.pvname

    my_subnet = sys_utils.MySubnet()
    print(f"Starting flask app on {my_subnet.my_ip}...")

    if port is None:
        # Initialize from .project_vars
        pvars = sys_utils.load_project_vars(pvname=pvname)
        if pvars is not None:
            port = pvars.get("FLASK_PORT", os.getenv("FLASK_PORT", "5000"))
        else:
            port = os.getenv("FLASK_PORT", "5000")

    app.run(port=port)
