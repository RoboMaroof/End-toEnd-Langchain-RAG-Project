from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

from langchain_community.document_loaders import WebBaseLoader

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools.retriever import create_retriever_tool

from langchain.schema.runnable import RunnableLambda

from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

from langserve import add_routes

from fastapi import FastAPI, Body
import uvicorn

import os
from dotenv import load_dotenv
import logging



load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
VECTORDB_PATH = "faiss_index"

logging.basicConfig(level=logging.INFO)

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"
)


@app.post("/vectordb/create")
def create_vector_store(url: str = Body(..., embed=True)):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local(VECTORDB_PATH)

        return {"message": "Vector Store created from URL."}
    except Exception as e:
        return {"error": str(e)}


def load_vector_store():
    if os.path.exists(VECTORDB_PATH):
        return FAISS.load_local(VECTORDB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return None


prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
<context>
{context}
</context>
Question: {input}
""")

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


# Models
openai_model = ChatOpenAI(model="gpt-4o-mini")
ollama_model = Ollama(model="gemma3:1b")


def create_rag_chain(model):
    vectors = load_vector_store()
    if not vectors:
        return RunnableLambda(lambda x: {
            "output": "Vector Store not initialized. Please run /vectordb/create.",
            "retrieved_sections": []
        })
    
    retriever = vectors.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 sections

    def retrieve_and_generate_response(inputs):
        docs = retriever.get_relevant_documents(inputs["input"])    # Rerank retrieved docs
        context = "\n\n".join(doc.page_content for doc in docs)  # Extract content
        
        response = model.invoke(prompt.format(context=context, input=inputs["input"]))
        return {
            "output": response,
            "retrieved_sections": [doc.page_content for doc in docs]
        }

    return RunnableLambda(retrieve_and_generate_response)


def create_agent_chain(model):
    vectors = load_vector_store()

    # Create tools
    tools = [
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)),
        ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)),
    ]
    
    if vectors:
        retriever_tool = create_retriever_tool(
            retriever=vectors.as_retriever(),
            name="langsmith_search",
            description="Search for information about LangSmith. Use this tool for LangSmith-related queries."
        )
        tools.append(retriever_tool)

    # Create agent
    agent = create_openai_tools_agent(model, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    
    def run_agent(inputs):
        response = agent_executor.invoke({"input": inputs["input"]})
        print(response)
        tool_invocations = response.get("intermediate_steps", [])
        tools_used = [action.tool for action, _ in tool_invocations]
        return {
            "output": response["output"],
            "tool_used": tools_used
        }
    
    return RunnableLambda(run_agent)

create_vector_store()

# Create chains
rag_chain_openai = create_rag_chain(openai_model)
rag_chain_ollama = create_rag_chain(ollama_model)
agent_chain_openai = create_agent_chain(openai_model)
agent_chain_ollama = create_agent_chain(ollama_model)

# Register routes
add_routes(app, rag_chain_openai, path="/openai")
add_routes(app, rag_chain_ollama, path="/ollama")
add_routes(app, agent_chain_openai, path="/openai/agent")
add_routes(app, agent_chain_ollama, path="/ollama/agent")


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)