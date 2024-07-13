import os
from flask import Flask, render_template, request, redirect, session
import sqlite3
import hashlib
import numpy as np
from datetime import date
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)
app.static_folder = 'static'

app.config['total'] = 0
app.config['model_loaded'] = False
app.config['userScore'] = 0
app.config['InCounselor'] = False
app.config['InDiagnosis'] = False
app.config['userText_diagnosis'] = " "
app.config['Vector_DB_loaded'] = False
app.config['username'] = "NA"

print("Flag: started flask app")

salt = '4f3d2e5b9a7c1d8e6f2b4a8c3e5d7f1a0b9c2d3e6a5b4c8d7f9a0e6c5b8d7f1'

# ------------------------------------------------------------- Diagnostics Model ------------------------------------------------------------------------------------- #

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

# --------------------------------------------------------------- Addiction Counselor ------------------------------------------------------------------------#
# import streamlit as st


## load the GROQ And OpenAI API KEY 
groq_api_key="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"
os.environ["GOOGLE_API_KEY"]="AIzaSyAT2tXl0kKimGzB7VvLS5Ln7jaN_827xgA"

# st.title("Gemma Model Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-70b-8192")

global store
store = {}

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
#########################################################

def vector_embedding():

    # if "vectors" not in st.session_state:

    global embeddings
    embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    global loader
    loader=PyPDFDirectoryLoader("context_for_RAG/") ## Data Ingestion
    
    global docs
    docs=loader.load() ## Document Loading
    global text_splitter
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
    global final_documents
    final_documents=text_splitter.split_documents(docs[:20]) #splitting
    global vectors
    vectors=FAISS.from_documents(final_documents,embeddings) #vector OpenAI embeddings

def counselor(prompt1):
### ANSWER QUESTION ####
    system_prompt=(

    "assume you are a mental health counselor, learn from the given sample conversation given to you as context"
    "and as the patient the right questions about their situation"
    "If you don't know the answer, say that you "
    "don't know. Use two sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"

    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
        
    retriever=vectors.as_retriever()
    
    ## History Aware retriever
    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)   
    # Question Answer Chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    #Final Rag Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    ### Statefully manage chat history ###

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
    
    
    ### generate responses
    response = conversational_rag_chain.invoke(
    {"input": prompt1},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)
    return response['answer']


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #

@app.route("/",  methods=["GET", "POST"])
def landing():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        connection = sqlite3.connect("database.db")
        cursor = connection.cursor()
        
        name = request.form['name']
        password = request.form['password']
        
        password = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
        
        query = "SELECT username, password FROM users where username= '"+name+"' and password='"+password+"' "
        cursor.execute(query)
        
        results = cursor.fetchall()
        
        print(results)
        
        if len(results) == 0:
            return render_template("login.html", error=True)
        else:
            app.config['username'] = name
            return redirect("/home")
        
    return render_template("login.html", error=False)

@app.route("/signup",  methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        connection = sqlite3.connect("database.db")
        cursor = connection.cursor()
        
        firstname = request.form['firstn']
        lastname = request.form['lastn']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT username FROM users WHERE username = '"+username+"'";
        cursor.execute(query)
        results = cursor.fetchall()
        
        if len(results) != 0:
            return render_template("signup.html", error=True)
        
        print(firstname, lastname, username, email, password)
        
        password = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
        
        query = " INSERT INTO users VALUES ('"+username+"' , '"+password+"' , 'date', '"+firstname+"', '"+lastname+"', 'NA', 'Not-Diagnosed', 'Not-Diagnosed') "
        cursor.execute(query)
        connection.commit()
        
        app.config['username'] = username
        return redirect("/home")
    return render_template("signup.html", error=False)

@app.route("/home",  methods=["GET", "POST"])
def home():
    if app.config['username'] == "NA":
        return redirect("/home")
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    
    name = app.config['username']
    query = "SELECT disorder, severity,date FROM users WHERE username='"+name+"'"
    cursor.execute(query)
    
    data = cursor.fetchall()
    print(data) 
    
    disorder = data[0][0]
    severity = data[0][1]
    date = data[0][2]
    
    if disorder != "Not-Diagnosed":
        diagnosed = True
    else:
        diagnosed = False
        
    if date == 'date':
        counsled = False
    else:
        counsled = True
    
    return render_template("home.html", username=app.config['username'], date=date, diagnosed = diagnosed, severity=severity, disorder=disorder, counsled=counsled)


@app.route("/diagnosis",  methods=["GET", "POST"])
def diagnosis():
    if app.config['username'] == "NA":
        return redirect("/home")
    app.config['InDiagnosis'] = True
    app.config['InCounselor'] = False
    app.config['userScore'] = 0
    app.config['userText_diagnosis'] = " "
    app.config['total'] = 1
    
    print("loading model")
    if not app.config['model_loaded']:
        load_model()
        print("loaded the model")
        app.config['model_loaded']=True
        print("updated the load_model variable")

    return render_template("diagnosis.html", username = app.config['username'])


@app.route("/counsel")
def counsel():
    if app.config['username'] == "NA":
        return redirect("/home")
    app.config['InCounselor'] = True
    app.config['InDiagnosis'] = False
    app.config['Vector_DB_loaded'] = False
    global store
    store = {}
    
    
    if not app.config['Vector_DB_loaded']:
        print("Making Vector Store DB")
        vector_embedding()
        print("Vector Store DB Is Ready")
        app.config['Vector_DB_loaded'] = True
        
    today = str(date.today())
    print(type(today))
    print("Today's Date: "+ today)
    
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    
    query = "UPDATE users SET date = ? WHERE username = ?"
    cursor.execute(query, (today, app.config['username']))
    connection.commit()
        
    return render_template("counsel.html", username = app.config['username'])

@app.route("/history")
def history():
    if app.config['username'] == "NA":
        return redirect("/home")
    
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    
    query = "SELECT history FROM users WHERE username = '"+app.config['username']+"'"
    cursor.execute(query)
    
    input_string = str(cursor.fetchall())
    # Sample input string with brackets
    cleaned_string = input_string[3:-4]
    print(cleaned_string)
    return render_template('history.html', input_string=cleaned_string)

@app.route("/logout")
def logout():
    app.config['username'] = "NA"
    return redirect("/")

@app.route("/doctor")
def doctor():
    return render_template("doctor.html")

@app.route('/get')
def get_bot_response():
    
    print("Flag: Getting user response")

    userText = request.args.get('msg')
    global predicted_class
    
    # for diagnosis
    if app.config['InDiagnosis'] and not app.config['InCounselor']:
        # first we ask questions to get the symptoms
        if app.config['total'] == 1:
            print("flag 2")
            app.config['total'] += 1
            app.config['userText_diagnosis'] += userText
            return "Can you share any recent events or experiences that might have triggered these feelings or symptoms?"
        if app.config['total'] == 2:
            print("flag 3")
            
            app.config['total'] += 1
            app.config['userText_diagnosis'] += userText
            return "Have you experienced any significant traumas in the past, or do you have any habits or behaviors that you think might be affecting your mental health ?"
        if app.config['total'] == 3:
            print("flag 4")
            
            app.config['total'] += 1
            app.config['userText_diagnosis'] += userText
        
            # GET PREDICTION FROM THE MODEL #
            input_text = app.config['userText_diagnosis']
            
            print("User Input: "+input_text)
            
            inputs = tokenizer(input_text, return_tensors="pt")
            
            outputs = model(**inputs)
            
            logits = outputs.logits
            print(logits)

            # Convert the logits tensor to a NumPy array
            logits_np = logits.detach().numpy()

            # Find the index with the maximum value using NumPy
            max_index = np.argmax(logits_np, axis=1)[0]
            print(max_index)

            # Define the class labels
            class_labels = ["Addiction", "Anxiety", "Depression", "PTSD"]

            # Get the class label based on the index
            predicted_class = class_labels[max_index]

            print(f"The predicted Disorder is: {predicted_class}")
            
            connection = sqlite3.connect("database.db")
            cursor = connection.cursor()
            
            query = "UPDATE users SET disorder = ? WHERE username = ?"
            cursor.execute(query, (predicted_class, app.config['username']))
            connection.commit()
            
            
            # RETURN THE PREDICTOIN
            if predicted_class == "Anxiety":
                return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> Answer 0 : For Not At All <br> Answer 1: If several days <br> Answer 2: If More than half the days <br> Answer 3: If Nearly every day <br> <br>  Feeling nervous, anxious, or on edge ? <br><br> NOTE: Please provide Numerical Input !"
            if predicted_class == "PTSD":
                return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> NOTE: Please Answer in 'yes' or 'no' ! <br><br> Sometimes things happen to people that are unusually or especially frightening, horrible, or traumatic. <br> For example: <br> - A serious accident or fire <br> - A physical or sexual assualt or abuse <br> - Seeing someone getting killed or get seriously injured <br> - Having a loved one die through homicide or sucide <br><br> Have you ever experienced this kind of event? "
            if predicted_class == "Depression":
                return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> Answer 0 : For Not At All <br> Answer 1: If several days <br> Answer 2: If More than half the days <br> Answer 3: If Nearly every day <br> <br>Little interest or pleasure in doing things ? <br><br> NOTE: Please provide Numerical Input !"
            if predicted_class == "Addiction":
                return "You have: " + predicted_class +"<br>"+ "Now could please answer a few symptoms related questions, to help me diagnose the severity of you disorder? <br> Answer 0 : For Not At All <br> Answer 1: For Sometimes <br> Answer 2: For Often <br> Answer 3: For Always <br> <br>  How often do you have strong urges or cravings to use the substance or engage in the behavior? <br><br> NOTE: Please provide Numerical input !"

        # now we dive into the questionaire
        if app.config['total'] >= 4:
            if predicted_class == "Anxiety":
                if userText == '0' or userText == '1' or userText == '2' or userText == '3':
                    if app.config['total'] == 4:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Not being able to stop or control worrying"
                    if app.config['total'] == 5:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Worrying too much about different things"            
                    if app.config['total'] == 6:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return " Trouble relaxing "
                    if app.config['total'] == 7:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Being so restless that it is hard to sit still"
                    if app.config['total'] == 8:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Becoming easily annoyed or irritable"            
                    if app.config['total'] == 9   :  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Feeling afraid, as if something awful might happen "
                    if app.config['total'] == 10:
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] = -1
                        
                        connection = sqlite3.connect("database.db")
                        cursor = connection.cursor()
                                                
                        if app.config['userScore'] >=0 and app.config['userScore'] <= 4:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Minimal", app.config['username']))
                            connection.commit()
                            return "The Severity of your Anxiety Disorder is Minimal.<br>You can refer to professional help or You can also try our Anxiety Health Counsellor"
                        if app.config['userScore'] >=5 and app.config['userScore'] <= 9:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Mild", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Anxiety Disorder is Mild.<br>You can refer to professional help or You can also try our Anxiety Health Counsellor"
                        if app.config['userScore'] >=10 and app.config['userScore'] <= 14:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Moderate", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Anxiety Disorder is Moderate.<br>I recommend taking professional help. Along with that You can also try our Anxiety Health Counsellor"
                        if app.config['userScore'] >=15 and app.config['userScore'] <= 21:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Severe", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Anxiety Disorder is Severe.<br>You can try our Anxiety Health Counsellor, but I strongly recommend taking professional help"
                else:
                    return "Please Provide Answers in Required Format !"
                
            if predicted_class == "PTSD":
                if app.config['total'] == 4:  
                    if userText == "NO" or userText == "No" or userText == "nO" or userText == "no":
                        app.config['total'] = 15
                        connection = sqlite3.connect("database.db")
                        cursor = connection.cursor()
                        query = "UPDATE users SET severity = ? WHERE username = ?"
                        cursor.execute(query, ("Minimal", app.config['username']))
                        connection.commit()
                        return "PTSDs are generally caused due to a traumatic event. I would recommend you to consult our PTSD Mental Health Counsellor or Consider taking Professional Help."
                    elif userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                        app.config['total'] += 1
                        return " Had nightmares about it or thought about it when you did not want to?"
                    else:
                        return "Please Provide Answer in Required Format"
                    
                if app.config['total'] == 5:  
                    if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                        app.config['userScore'] += 1
                        app.config['total'] += 1
                        return "Tried hard not to think about it or went out of your way to avoid situations that reminded you of it?"            
                    elif userText == "NO" or userText == "No" or userText == "nO" or userText == "no":
                        app.config['total'] += 1
                        return "Tried hard not to think about it or went out of your way to avoid situations that reminded you of it?"             
                    else:
                        return "Please Provide Answer in Required Format" 
                if app.config['total'] == 6:  
                    if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                        app.config['userScore'] += 1
                        app.config['total'] += 1
                        return "Were constantly on guard, watchful or easily startled?"            
                    elif userText == "NO" or userText == "No" or userText == "nO" or userText == "no":
                        app.config['total'] += 1
                        return "Were constantly on guard, watchful or easily startled?"
                    else:
                        return "Please Provide Answer in Required Format" 
                if app.config['total'] == 7:
                    if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                        app.config['userScore'] += 1
                        app.config['total'] += 1
                        return "Felt numb or detached from others, activities, or your surroundings?"            
                    elif userText == "NO" or userText == "No" or userText == "nO" or userText == "no":
                        app.config['total'] += 1
                        return "Felt numb or detached from others, activities, or your surroundings?"
                    else:
                        return "Please Provide Answer in Required Format" 
                if app.config['total'] == 8:  
                    if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                        app.config['userScore'] += 1
                        app.config['total'] += 1
                        return "Felt guilty or unable to stop blaming yourself or others for the event or any problems the event may have caused"      
                    elif userText == "NO" or userText == "No" or userText == "nO" or userText == "no":
                        app.config['total'] += 1
                        return "Felt guilty or unable to stop blaming yourself or others for the event or any problems the event may have caused"
                    else:
                        return "Please Provide Answer in Required Format" 
                if app.config['total'] == 9:
                    if userText == "YES" or userText =="Yes" or userText == "YEs" or userText == "yES" or userText == "yEs" or userText == "yeS" or userText == "YeS" or userText == "yes":    
                        app.config['userScore'] += 1
                        app.config['total'] = -1      
                    elif userText == "NO" or userText == "No" or userText == "nO" or userText == "no":
                        app.config['total'] = -1
                    else:
                        return "Please Provide Answer in Required Format"
                    
                    connection = sqlite3.connect("database.db")
                    cursor = connection.cursor()
                    
                    if app.config['userScore'] >=0 and app.config['userScore'] < 3:
                        query = "UPDATE users SET severity = ? WHERE username = ?"
                        cursor.execute(query, ("Mild", app.config['username']))
                        connection.commit()                        
                        return "The Severity of your PTSD Disorder is Mild.<br>You can refer to professional help or You can also try our PTSD Health Counsellor"
                    if app.config['userScore'] >=3:
                        query = "UPDATE users SET severity = ? WHERE username = ?"
                        cursor.execute(query, ("Moderate/Severe", app.config['username']))
                        connection.commit()
                        return "The Severity of your PTSD Disorder is Moderate/Severe.<br>You can try our PTSD Health Counsellor, but I strongly recommend taking professional help"

            if predicted_class == "Depression":
                if userText == '0' or userText == '1' or userText == '2' or userText == '3':                
                    if app.config['total'] == 4:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Feeling down, depressed, or hopeless"
                    if app.config['total'] == 5:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Trouble falling or staying asleep, or sleeping too much"            
                    if app.config['total'] == 6:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return " Feeling tired or having little energy "
                    if app.config['total'] == 7:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Poor appetite or overeating"
                    if app.config['total'] == 8:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Feeling bad about yourself or that you are a failure or have let yourself or your family down"            
                    if app.config['total'] == 9   :  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Trouble concentrating on things, such as reading the newspaper or watching television "
                    if app.config['total'] == 10:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Moving or speaking so slowly that other people could not have noticed. Or the opposite being so figety or restless that you have been moving around a lot more than usual "            
                    if app.config['total'] == 11:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "Thoughts that you would be better off dead, or of hurting yourself"
                    if app.config['total'] == 12:
                        app.config['total'] = -1
                        app.config['userScore']+= (int)(userText)
                        
                        connection = sqlite3.connect("database.db")
                        cursor = connection.cursor()                        
                        
                        if app.config['userScore'] >=0 and app.config['userScore'] <= 4:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Minimal", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Depression Disorder is Minimal.<br>You can refer to professional help or You can also try our Depression Health Counsellor"
                        if app.config['userScore'] >=5 and app.config['userScore'] <= 9:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Mild", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Depression Disorder is Mild.<br>You can refer to professional help or You can also try our Depression Health Counsellor"
                        if app.config['userScore'] >=10 and app.config['userScore'] <= 14:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Moderate", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Depression Disorder is Moderate.<br>I recommend taking professional help. Along with that You can also try our Depression Health Counsellor"
                        if app.config['userScore'] >=15 and app.config['userScore'] <= 19:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Moderately Severe", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Depression Disorder is Moderately Severe.<br>You can try our Depression Health Counsellor, but I strongly recommend taking professional help"
                        if app.config['userScore'] >=20 and app.config['userScore'] <= 27:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Severe", app.config['username']))
                            connection.commit()                            
                            return "The Severity of your Depression Disorder is Quite Severe.<br>I strongly recommend taking professional help however you can also try our Depression Health Counsellor"                
                else:
                    return "Please Provide Answers in Required Format !"
                
            if predicted_class == "Addiction":
                if userText == '0' or userText == '1' or userText == '2' or userText == '3':
                    if app.config['total'] == 4:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you find it difficult to control or stop using the substance or engaging in the behavior?"
                    if app.config['total'] == 5:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you need to use more of the substance or engage more in the behavior to achieve the same effect?"            
                    if app.config['total'] == 6:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you experience physical or emotional withdrawal symptoms when you try to stop using the substance or engaging in the behavior?"
                    if app.config['total'] == 7:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you neglect your responsibilities at work, school, or home due to your use of the substance or engagement in the behavior?"
                    if app.config['total'] == 8:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you continue to use the substance or engage in the behavior despite knowing it causes problems in your life?"            
                    if app.config['total'] == 9   :  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you spend a lot of time obtaining, using, or recovering from the substance or behavior?"
                    if app.config['total'] == 10:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you lose interest in other activities or hobbies because of your use of the substance or engagement in the behavior?"
                    if app.config['total'] == 11:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you continue to use the substance or engage in the behavior in situations where it is physically dangerous (e.g., driving, operating machinery)?"
                    if app.config['total'] == 12:  
                        app.config['userScore']+= (int)(userText)
                        app.config['total'] += 1
                        return "How often do you feel guilty or ashamed about your use of the substance or engagement in the behavior?"                        
                    if app.config['total'] == 13:
                        app.config['total'] = -1
                        app.config['userScore'] += (int)(userText)
                        
                        connection = sqlite3.connect("database.db")
                        cursor = connection.cursor()                        
                        
                        if app.config['userScore'] >=0 and app.config['userScore'] <= 6:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Mild", app.config['username']))
                            connection.commit()                             
                            return "The Severity of your Addiction Disorder is Mild.<br>You can refer to professional help or You can also try our Addiction Health Counsellor"
                        if app.config['userScore'] >=7 and app.config['userScore'] <= 15:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Moderate", app.config['username']))
                            connection.commit()                             
                            return "The Severity of your Addiction` Disorder is Moderate.<br>You can refer to professional help or You can also try our Addiction Health Counsellor"
                        if app.config['userScore'] >=16 and app.config['userScore'] <= 24:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Mildly Severe", app.config['username']))
                            connection.commit()                             
                            return "The Severity of your Addiction` Disorder is Mildy Severe.<br>I recommend taking professional help. Along with that You can also try our Addiction Health Counsellor"
                        if app.config['userScore'] >=25 and app.config['userScore'] <= 30:
                            query = "UPDATE users SET severity = ? WHERE username = ?"
                            cursor.execute(query, ("Severe", app.config['username']))
                            connection.commit()                             
                            return "The Severity of your Addiction` Disorder is Severe.<br>You can try our Addiction Health Counsellor, but I strongly recommend taking professional help"
                else:
                    return "Please Provide Answers in Required Format !"            
            
        return "Thankyou for using out website. Refresh the page for another diagnosis"                  
        
    # for counseling
    elif not app.config['InDiagnosis'] and app.config['InCounselor']:
        rag_model_output = counselor(userText)
        
        messages = store['abc123'].messages
        # Separate human and AI messages
        message_list = [msg.content for msg in messages]

        # Combine the messages with '|'
        history = ' | '.join(message_list)
        print(history)
        
        connection = sqlite3.connect("database.db")
        cursor = connection.cursor()
        
        query = "UPDATE users SET history = ? WHERE username = ?"
        cursor.execute(query, (history, app.config['username']))
        
        connection.commit()
        
        return rag_model_output
    
                

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    
    
    
    
