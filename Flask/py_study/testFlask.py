from flask import Flask


myApp = Flask(__name__)

@app.route("/")
def hello():
    return "Hello Fjc, nice to meet you!"