import numpy as np
import json

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

total = 1
model_loaded = False
global userScore
userScore = 0


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
    model_path = "diagnostics_model/working/saved_model"
    print(f"Loading model from: {model_path}")
    global model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Model loaded successfully.")

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
@app.route("/diagnosis")
def home():
    print("flag bruh")
    global total
    global model_loaded
    if not model_loaded:
        load_model()
        model_loaded=True
    total = 1
    return render_template("index.html")

userTextforPrediction = " "

@app.route('/get')
def get_bot_response():
    
    print("flag 1")
    global userTextforPrediction
    userText = request.args.get('msg')
    global total
    global predicted_class
    global userScore
    
    # first we ask questions to get the symptoms
    if total == 1:
        print("flag 2")
        total += 1
        userTextforPrediction += userText
        return "Question 2?"
    if total == 2:
        print("flag 3")
        
        total += 1
        userTextforPrediction += userText
        return "Question 3?"
    if total == 3:
        print("flag 4")
        
        total += 1
        userTextforPrediction += userText
    
        # GET PREDICTION FROM THE MODEL #
        input_text = userTextforPrediction
        
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
        if predicted_class == "Anxiety":
            return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> Answer 0 : For Not At All <br> Answer 1: If several days <br> Answer 2: If More than half the days <br> Answer 3: If Nearly every day <br> <br>  Feeling nervous, anxious, or on edge ?"
        if predicted_class == "PTSD":
            return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> Answer in 'Yes' or 'No' <br><br> Sometimes things happen to people that are unusually or especially frightening, horrible, or traumatic. <br> For example: <br> - A serious accident or fire <br> - A physical or sexual assualt or abuse <br> - Seeing someone getting killed or get seriously injured <br> - Having a loved one die through homicide or sucide <br><br> Have you ever experienced this kind of event? "
        if predicted_class == "Depression":
            return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> Answer 0 : For Not At All <br> Answer 1: If several days <br> Answer 2: If More than half the days <br> Answer 3: If Nearly every day <br> <br>Little interest or pleasure in doing things ?"
        if predicted_class == "Addiction":
            return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> Answer 0 : For Not At All <br> Answer 1: For Sometimes <br> Answer 2: For Often <br> Answer 3: For Always <br> <br>  How often do you have strong urges or cravings to use the substance or engage in the behavior?"

    # now we dive into the questionaire
    if total >= 4:
        if predicted_class == "Anxiety":
            if total == 4:  
                userScore+= (int)(userText)
                total += 1
                return "Not being able to stop or control worrying"
            if total == 5:  
                userScore+= (int)(userText)
                total += 1
                return "Worrying too much about different things"            
            if total == 6:  
                userScore+= (int)(userText)
                total += 1
                return " Trouble relaxing "
            if total == 7:  
                userScore+= (int)(userText)
                total += 1
                return "Being so restless that it is hard to sit still"
            if total == 8:  
                userScore+= (int)(userText)
                total += 1
                return "Becoming easily annoyed or irritable"            
            if total == 9   :  
                userScore+= (int)(userText)
                total += 1
                return "Feeling afraid, as if something awful might happen "
            if total == 10:
                if userScore >=0 and userScore <= 4:
                    return "The Severity of your Anxiety Disorder is Minimal.<br>You can refer to professional help or You can also try our Anxiety Health Counsellor"
                if userScore >=5 and userScore <= 9:
                    return "The Severity of your Anxiety Disorder is Mild.<br>You can refer to professional help or You can also try our Anxiety Health Counsellor"
                if userScore >=10 and userScore <= 14:
                    return "The Severity of your Anxiety Disorder is Moderate.<br>I recommend taking professional help. Along with that You can also try our Anxiety Health Counsellor"
                if userScore >=15 and userScore <= 21:
                    return "The Severity of your Anxiety Disorder is Severe.<br>You can try our Anxiety Health Counsellor, but I strongly recommend taking professional help"
    
        if predicted_class == "PTSD":
            if total == 4:  
                if userText == "NO" or userText == "No" or userText == "nO" or userText == "no":
                    total = 15
                    return "PTSDs are generally caused due to a traumatic event. I would recommend you to consult our PTSD Mental Health Counsellor or Consider taking Professional Help."
                if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                    total += 1
                    return " Had nightmares about it or thought about it when you did not want to?"
            if total == 5:  
                if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                    userScore += 1
                total += 1
                return "Tried hard not to think about it or went out of your way to avoid situations that reminded you of it?"            
            if total == 6:  
                if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                    userScore += 1
                total += 1
                return "Were constantly on guard, watchful or easily startled?"
            if total == 7:
                if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                    userScore += 1
                total += 1
                return "Felt numb or detached from others, activities, or your surroundings?"
            if total == 8:  
                if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                    userScore += 1
                total += 1
                return "Felt guilty or unable to stop blaming yourself or others for the event or any problems the event may have caused"
            if total == 9:
                if userScore >=0 and userScore < 3:
                    return "The Severity of your PTSD Disorder is Mild.<br>You can refer to professional help or You can also try our PTSD Health Counsellor"
                if userScore >=3:
                    return "The Severity of your PTSD Disorder is Moderate/Severe.<br>You can try our PTSD Health Counsellor, but I strongly recommend taking professional help"

        if predicted_class == "Depression":
            if total == 4:  
                userScore+= (int)(userText)
                total += 1
                return "Feeling down, depressed, or hopeless"
            if total == 5:  
                userScore+= (int)(userText)
                total += 1
                return "Trouble falling or staying asleep, or sleeping too much"            
            if total == 6:  
                userScore+= (int)(userText)
                total += 1
                return " Feeling tired or having little energy "
            if total == 7:  
                userScore+= (int)(userText)
                total += 1
                return "Poor appetite or overeating"
            if total == 8:  
                userScore+= (int)(userText)
                total += 1
                return "Feeling bad about yourself or that you are a failure or have let yourself or your family down"            
            if total == 9   :  
                userScore+= (int)(userText)
                total += 1
                return "Trouble concentrating on things, such as reading the newspaper or watching television "
            if total == 10:  
                userScore+= (int)(userText)
                total += 1
                return "Moving or speaking so slowly that other people could not have noticed. Or the opposite being so figety or restless that you have been moving around a lot more than usual "            
            if total == 11:  
                userScore+= (int)(userText)
                total += 1
                return "Thoughts that you would be better off dead, or of hurting yourself"
            if total == 12:
                userScore+= (int)(userText)
                if userScore >=0 and userScore <= 4:
                    return "The Severity of your Depression Disorder is Minimal.<br>You can refer to professional help or You can also try our Depression Health Counsellor"
                if userScore >=5 and userScore <= 9:
                    return "The Severity of your Depression Disorder is Mild.<br>You can refer to professional help or You can also try our Depression Health Counsellor"
                if userScore >=10 and userScore <= 14:
                    return "The Severity of your Depression Disorder is Moderate.<br>I recommend taking professional help. Along with that You can also try our Depression Health Counsellor"
                if userScore >=15 and userScore <= 19:
                    return "The Severity of your Depression Disorder is Moderately Severe.<br>You can try our Depression Health Counsellor, but I strongly recommend taking professional help"
                if userScore >=20 and userScore <= 27:
                    return "The Severity of your Depression Disorder is Quite Severe.<br>YI strongly recommend taking professional help however you can also try our Depression Health Counsellor"                
       
        if predicted_class == "Addiction":
            if total == 4:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you find it difficult to control or stop using the substance or engaging in the behavior?"
            if total == 5:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you need to use more of the substance or engage more in the behavior to achieve the same effect?"            
            if total == 6:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you experience physical or emotional withdrawal symptoms when you try to stop using the substance or engaging in the behavior?"
            if total == 7:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you neglect your responsibilities at work, school, or home due to your use of the substance or engagement in the behavior?"
            if total == 8:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you continue to use the substance or engage in the behavior despite knowing it causes problems in your life?"            
            if total == 9   :  
                userScore+= (int)(userText)
                total += 1
                return "How often do you spend a lot of time obtaining, using, or recovering from the substance or behavior?"
            if total == 10:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you lose interest in other activities or hobbies because of your use of the substance or engagement in the behavior?"
            if total == 11:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you continue to use the substance or engage in the behavior in situations where it is physically dangerous (e.g., driving, operating machinery)?"
            if total == 12:  
                userScore+= (int)(userText)
                total += 1
                return "How often do you feel guilty or ashamed about your use of the substance or engagement in the behavior?"                        
            if total == 13:
                userScore += (int)(userText)
                if userScore >=0 and userScore <= 6:
                    return "The Severity of your Addiction Disorder is Mild.<br>You can refer to professional help or You can also try our Addiction Health Counsellor"
                if userScore >=7 and userScore <= 15:
                    return "The Severity of your Addiction` Disorder is Moderate.<br>You can refer to professional help or You can also try our Addiction Health Counsellor"
                if userScore >=16 and userScore <= 24:
                    return "The Severity of your Addiction` Disorder is Mildy Severe.<br>I recommend taking professional help. Along with that You can also try our Addiction Health Counsellor"
                if userScore >=25 and userScore <= 30:
                    return "The Severity of your Addiction` Disorder is Severe.<br>You can try our Addiction Health Counsellor, but I strongly recommend taking professional help"
    
    
    return "Thankyou for using out website. Refresh the page for another diagnosis"                  
                

if __name__ == "__main__":
    app.run()
    
    
    
    
    