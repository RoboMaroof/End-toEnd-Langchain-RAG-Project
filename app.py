from langchain_core.prompts import ChatPromptTemplate

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langserve import add_routes

from langchain.schema.runnable import RunnableLambda

from fastapi import FastAPI
import uvicorn

import os
from operator import itemgetter
from dotenv import load_dotenv


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
VECTORDB_PATH = "faiss_index"

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"
)

@app.post("/vectordb/create")
def create_vector_store():
    save_path = "faiss_index"
    embeddings=OpenAIEmbeddings()

    loader=WebBaseLoader("https://docs.smith.langchain.com/") ## Data Ingestion
    docs=loader.load() ## Document Loading

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
    final_documents=text_splitter.split_documents(docs[:50])    #splitting
    vectors=FAISS.from_documents(final_documents,embeddings)    #vector OpenAI embeddings
    
    vectors.save_local(save_path)

    return {"message": "Vector Store Created Successfully"}

def load_vector_store(load_path):
    """Load FAISS vector store from disk if available, otherwise return None."""
    # global vectors  # Ensure vectors is accessible globally
    if os.path.exists(load_path):
        print("path exists")
        vectors = FAISS.load_local(load_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        return vectors
    else:
        print(f"FAISS vector store not found at {load_path}. Please create it first.")
        return None


model = ChatOpenAI(model="gpt-4o-mini")   #gpt-3.5-turbo"
##ollama 
llm = Ollama(model="gemma3:1b")
# ToDo Groq

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
""")



def create_rag_chain(model):
    vectors = load_vector_store("faiss_index")
    
    if vectors is None:
        print("⚠️ Warning: FAISS Vector Store is missing. Please run /vectordb/create first.")
        return RunnableLambda(lambda x: {"output": "Error: Vector Store not initialized. Please run /vectordb/create."})
    
    retriever = vectors.as_retriever()
    
    return (
        {
            "context": lambda x: retriever.get_relevant_documents(x["input"]),
            "input": itemgetter("input"),
        }
        | prompt
        | model
    )

create_vector_store()

rag_chain_openai = create_rag_chain(model)
rag_chain_ollama = create_rag_chain(llm)

# API Endpoints
add_routes(
    app,
    rag_chain_openai,
    path="/openai"
)

add_routes(
    app,
    rag_chain_ollama,
    path="/ollama"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
