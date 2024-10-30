import requests
import streamlit as st

def get_essay_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
    json = {'input':{'topic':input_text}}
    )

    return response.json()['output']

def get_poem_response(input_text):
    response = requests.post(
    "http://localhost:8000/poem/invoke",
    json = {'input':{'topic':input_text}}
    )

    return response.json()['output']


st.title("LangChain Demo with llama2 API")

input_text_essay = st.text_input("Write an essay on")
input_text_poem = st.text_input("Write an poem on")


if input_text_essay:
    st.write(get_essay_response(input_text_essay))

if input_text_poem:
    st.write(get_poem_response(input_text_poem))