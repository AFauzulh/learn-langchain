import os 

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama

import uvicorn
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="false"

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="a simple API server"
)

llm = Ollama(model="llama2")

prompt_essay = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt_poem = ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child")

add_routes(
    app,
    prompt_essay | llm,
    path="/essay"
)

add_routes(
    app,
    prompt_poem | llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)