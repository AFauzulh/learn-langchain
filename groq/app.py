import os

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    st.session_state.loader = WebBaseLoader("http://docs.smith.langchain.com")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("ChatGroq Demo")

llm = ChatGroq(
    groq_api_key = groq_api_key,
    model_name="mixtral-8x7b-32768"
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions:{input}

""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)
prompt = st.text_input("Input your prompt here")

if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])