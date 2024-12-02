## Data Ingestion
from langchain_community.document_loaders import TextLoader 
loader=TextLoader("speech.txt")
text_documents=loader.load()
text_documents



#load the data from the enviroment file 
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")





# web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")

                     )))
text_documents=loader.load()




## Pdf reader
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('attention.pdf')
docs=loader.load()


# change the document in to chuncks and overlap the documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
documents[:5]



## Vector Embedding 
from langchain_openai import OpenAIEmbeddings


# And Vector Store Chroma database
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(documents,OpenAIEmbeddings())



## FAISS Vector Database
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(documents[:15], OpenAIEmbeddings())


#make quary to the model 

query = "who are the authors of attention is you need reasearch paper"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)



# RAG MODEL FUNCTIONS
# data injesttion -> chunck and overlap the data ->Vector embedding  -> vector store -> query