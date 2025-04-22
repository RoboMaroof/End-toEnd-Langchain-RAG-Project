from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import Tool
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from ingestion.index_builder import load_index


def get_tools():
    tools = []

    tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)))
    tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)))

    index = load_index()

    if index:

        llm = OpenAI(model="gpt-4o-mini")

        reranker = LLMRerank(
            top_n=5,
            llm=llm
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
            postprocessors=[reranker]  # ← reranker applied here
        )

        query_engine = RetrieverQueryEngine.from_args(retriever)

        retriever_tool = Tool(
            name="vector_retriever",
            func=lambda q: [
                {
                    "text": node.node.text,
                    "score": round(node.score, 3) if hasattr(node, "score") else None
                }
                for node in retriever.retrieve(q)
            ],
            description="Useful for answering questions from uploaded documents, websites, or SQL databases such as FAQs, company data, policies, etc."
        )
        tools.append(retriever_tool)

    return tools
