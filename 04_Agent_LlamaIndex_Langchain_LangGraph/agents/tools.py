from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import Tool
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from ingestion.index_builder import load_index


def get_tools(llm):
    tools = []

    tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)))
    tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)))

    index = load_index()
    if index:
        reranker = LLMRerank(top_n=5, llm=llm)

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
            postprocessors=[reranker]
        )

        retriever_tool = Tool(
            name="vector_retriever",
            func=lambda q: [
                {
                    "text": node.node.text,
                    "score": round(node.score, 3) if hasattr(node, "score") else None
                }
                for node in retriever.retrieve(q)
            ],
            description="Useful for retrieving custom knowledge from docs, websites, and SQL."
        )
        tools.append(retriever_tool)

    return tools

