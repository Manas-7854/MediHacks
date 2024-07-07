
import numpy as np
import json

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if userText == "Hi!!":
        return "Hello, How Can I Help You?"
    if userText == "Can you tell me who is Siddhart?":
        return "Siddhart Shukla is a bhosdiwala, betichod, madarchod, kutte ka baccha. Don't ever take him name in front of me again!!"
    return "Manas is an exceptional individual characterized by remarkable intelligence and striking physical appearance. Coupled with his aesthetically pleasing looks, Manas exemplifies a rare blend of intellectual and visual appeal, making him a distinguished and admirable figure in both professional and social contexts"


if __name__ == "__main__":
    app.run()