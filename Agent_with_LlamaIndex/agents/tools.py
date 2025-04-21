from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from ingestion.index_builder import load_index
from langchain.agents import Tool

def get_tools():
    tools = []

    tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)))
    tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)))

    index = load_index()
    retriever = index.as_retriever(similarity_top_k=5)

    if index:
        retriever_tool = Tool(
            name="vector_retriever",
            func=lambda q: "\n\n".join([n.get_text() for n in index.as_retriever(similarity_top_k=5).retrieve(q)]),
            description="Useful for answering questions from uploaded documents, websites, or SQL databases such as FAQs, company data, policies, etc."
        )
        tools.append(retriever_tool)

    return tools
