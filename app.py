from flask import Flask,render_template,request, jsonify
import google.generativeai as palm
import os, pickle
import pickle
import numpy as np
from textblob import TextBlob

api = os.getenv("MAKERSUITE_API_TOKEN") 
palm.configure(api_key=api)
model = {"model": "models/chat-bison-001"}

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)


@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/financial_QA",methods=["GET","POST"])
def financial_QA():
    return(render_template("financial_QA.html"))

@app.route("/makersuite",methods=["GET","POST"])
def makersuite():
    q = request.form.get("q")
    r = palm.chat(prompt=q, **model)
    return(render_template("makersuite.html",r=r.last))

@app.route("/prediction",methods=["GET","POST"])
def prediction():

    return(render_template("prediction.html"))

@app.route('/joke', methods=['GET'])
def joke():
    text = "The only thing faster than Singapore's MRT during peak hours is the way we 'chope' seats with a tissue packet."
    return jsonify({'text': text})

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json['text']
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run()
