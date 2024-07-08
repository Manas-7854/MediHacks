from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder

import torch
import numpy as np

import pandas as pd
from datasets import Dataset

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler


label_encoder = LabelEncoder()

# Load the model and tokenizer
model_path = "kaggle/working/saved_model"
print(f"Loading model from: {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
print("Model loaded successfully.")

tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer loaded successfully.")

# Example usage:
##########################################################################################################################################

input_text = "Here I go again After going 13 days without engaging in my food addiction I gave in and went back into binge eating and once again I was eating food from out the trash. I always feel like crap every time I do that. It has been difficult for me to just get past one day without eating any processed foods and sweets but the fact the I went 13 days without was great I have never been able to do that before. Now I'm back at day one again. Its been almost 4 months since I have had coffee and I have no plans to go back to it, I'm a coffee addict and would sometimes drink 8 to 10 cups of coffee a day from the morning to the am. I would always try to make the coffee as strong as possible and to extend its duration. I had previously went 5 months without coffee since I almost had a heart attack but I had relapse and almost had another heart attack this past Christmas"
print(f"Input text: {input_text}")

inputs = tokenizer(input_text, return_tensors="pt")
print("Tokenized input:", inputs)

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


print("____________________________________________________________________________________________________________________")

##########################################################################################################################################

input_text = "recently one of my friend died, he used to keep himself isolated from everyone,I have ben trying to forget about him but I am unable to,he was a good friend of mine,I feel sad whenever I remember about him, I don't feel like eating"
print(f"Input text: {input_text}")

inputs = tokenizer(input_text, return_tensors="pt")
print("Tokenized input:", inputs)

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


print("____________________________________________________________________________________________________________________")

##########################################################################################################################################

input_text = "i was attacked by a herd of dogs when i was a kid, i am terrified of them, whenever i see one as an adult i get really scared"
print(f"Input text: {input_text}")

inputs = tokenizer(input_text, return_tensors="pt")
print("Tokenized input:", inputs)

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

print("____________________________________________________________________________________________________________________")

##########################################################################################################################################
input_text = "whenever i think about how bad my college is and i probably wont get  placed, my heart starts pounding very fast and i start seating heavily and start shaking and trembling and feel shortness of breath"
print(f"Input text: {input_text}")

inputs = tokenizer(input_text, return_tensors="pt")
print("Tokenized input:", inputs)

outputs = model(**inputs)
print("Model outputs:", outputs)

# Assuming 'outputs' is the SequenceClassifierOutput object as shown
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

print("____________________________________________________________________________________________________________________")