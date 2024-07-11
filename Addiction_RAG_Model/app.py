# import streamlit as st
import os
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
import os
print("flag 1")

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

##############################################################


print("flag 2")

def vector_embedding():

    # if "vectors" not in st.session_state:

    global embeddings
    embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    global loader
    loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
    global docs
    docs=loader.load() ## Document Loading
    global text_splitter
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
    global final_documents
    final_documents=text_splitter.split_documents(docs[:20]) #splitting
    global vectors
    vectors=FAISS.from_documents(final_documents,embeddings) #vector OpenAI embeddings


print("Making Vector Store DB")
vector_embedding()
print("Vector Store DB Is Ready")

import time

while(True):
    
    prompt1=input("Enter Your Question From Doduments: ")
    if prompt1 == "END":
        print("_____________________________ CHAT HISTORY _______________________________________")
        print(store)
        break
    print("flag 3")
    
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
    print(response['answer'])
    
    print("---------------------------------------------------------------------------------------------------")

    # Terminal equivalent of the Streamlit expander
    print("Document Similarity Search\n")

    # Find the relevant chunks
    for i, doc in enumerate(response["context"]):
        print(doc.page_content)
        print("--------------------------------")