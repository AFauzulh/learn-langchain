import os
import shutil
import time

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_objectbox.vectorstores import ObjectBox

from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Objectbox VectorstoreDB with Groq and Ollama")

# Vector Embedding and Objectbox VectorStoresDB
def vector_embedding():
    if os.path.exists("./objectbox"):
        shutil.rmtree("./objectbox")
        
    if "vectordb" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
        
        st.session_state.loader = PyPDFDirectoryLoader("./rag_papers") # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        st.session_state.vectordb = ObjectBox.from_documents(st.session_state.final_documents, 
                                                             st.session_state.embeddings, 
                                                             embedding_dimensions=768
                                                             )
        
# llm_model = ChatGroq(
#     api_key=groq_api_key,
#     model="Llama3-8b-8192"
# )

llm_model = ChatOllama(
    model="llama3.2"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate answer based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    """
) 
       
input_prompt = st.text_input("Ask your PDF")

if (st.button("Create PDF Embedding")):
    vector_embedding()
    st.write("ObjectBox database ready !")

if input_prompt:
    document_chain = create_stuff_documents_chain(llm = llm_model, prompt=prompt)
    
    retriever = st.session_state.vectordb.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    
    response = retrieval_chain.invoke({"input": input_prompt})

    print(f"Response Time: {time.process_time() - start}")
    st.write(response["answer"])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------------------------")