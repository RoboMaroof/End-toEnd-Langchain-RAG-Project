from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from ingestion.index_builder import load_index
from langchain.agents import Tool

def get_tools():
    tools = []

    tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)))
    tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)))

    index = load_index()
    if index:
        retriever_tool = Tool(
            name="vector_retriever",
            func=lambda q: index.as_retriever().retrieve(q),
            description="Useful for answering questions from ingested sources."
        )
        tools.append(retriever_tool)

    return tools
