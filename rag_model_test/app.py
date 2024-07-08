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
from dotenv import load_dotenv
import os
load_dotenv()
print("flag 1")

## load the GROQ And OpenAI API KEY 
groq_api_key="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"
os.environ["GOOGLE_API_KEY"]="AIzaSyAT2tXl0kKimGzB7VvLS5Ln7jaN_827xgA"

# st.title("Gemma Model Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
Assume that you are a addiction health counsellor and counsel a patient based on the given context.
Try to give as accurate answers as you can with respect to the context
<context>
{context}
<context>
Questions:{input}

"""
)
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

    print("flag 3")

    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    print(response['answer'])
    
    print("---------------------------------------------------------------------------------------------------")

    # # Terminal equivalent of the Streamlit expander
    # print("Document Similarity Search\n")

    # # Find the relevant chunks
    # for i, doc in enumerate(response["context"]):
    #     print(doc.page_content)
    #     print("--------------------------------")

