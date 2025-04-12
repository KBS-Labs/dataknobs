from flask import Flask


def create_app():
    ''' Create an instance of the Flask application. '''
    app = Flask(__name__)
    return app
