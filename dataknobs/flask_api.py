#!/usr/bin/env python

import argparse
import dataknobs.utils.sys_utils as sys_utils
import json
import os
from dataknobs import create_app
from flask import request


app = create_app()


@app.route('/echoArgs')
def echo_args():
    indent = request.args.get("indent", type=int, default=2)
    verbose = request.args.get("verbose", type=bool, default=True)
    rv = request.args.to_dict(flat=False)
    if verbose:
        print(json.dumps(rv, indent=indent))
    return rv
        

@app.route('/printenv')
def printenv():
    rv = {x: y for x, y in os.environ.items()}
    print(json.dumps(rv, indent=2))
    return rv


@app.route('/listdir')
def listdir():
    name = request.args.get("name", default=None)
    rv = os.listdir(name)
    print(json.dumps(rv, indent=2))
    return rv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', help="The server's port")
    parser.add_argument(
        '--pvname',
        help='The name of the project variables file',
        default=".project_vars"
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
