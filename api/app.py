from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from fastapi import FastAPI
import uvicorn
from langserve import add_routes

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)
model=ChatOpenAI()
##ollama llama2
llm=Ollama(model="llama2")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"


)

add_routes(
    app,
    prompt2|llm,
    path="/poem"


)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)




#this is the server this server is listening on the request of the client side requestS
#first the server import the chatPromptTemplate and and the llm model either the openAI or the ollama model
#secound the server to listen the client side it call the unicorn and the Fastapi for router  parameters and add_routes to connect to the clinet side
#third import the dotenv file to load the data form the envirment variables 
#fourth instantiate the model and prompt 
#fivth instantiate the the route 

