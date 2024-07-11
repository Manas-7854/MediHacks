import numpy as np
import json

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

count = 0

# ------------------------------------------------------------- Model ------------------------------------------------------------------------------------- #

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder

import torch
import numpy as np

import pandas as pd
from datasets import Dataset

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler

def load_model():
    print("Loading Model.....")
    
    label_encoder = LabelEncoder()

    # Load the model and tokenizer
    model_path = "kaggle/working/saved_model"
    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Model loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    
    # load the model on entry
    if count == 0:
        load_model()
        count+=1
    
    # GET PREDICTION FROM THE MODEL #
    input_text = userText
    print("User Input: "+input_text)
    
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model(**inputs)
    
    logits = outputs.logits
    print(logits)

        # Given tensor
    output = torch.tensor(logits)

    # Find the index with the maximum value
    max_index = torch.argmax(output, dim=1).item()

    # Define the class labels
    class_labels = ["Addiction", "Anxiety", "Depression", "PTSD"]

    # Get the class label based on the index
    predicted_class = class_labels[max_index]

    print(f"The predicted Disorder is: {predicted_class}")
    
    # RETURN THE PREDICTOIN
    return "The predicted Disorder is: " + predicted_class


if __name__ == "__main__":
    app.run()