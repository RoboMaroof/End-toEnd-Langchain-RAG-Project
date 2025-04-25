from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from agents.tools import get_tools
from agents.llm_loader import get_llm


# Define a simple agent node that uses ReAct agent logic
def get_langgraph_agent(model_name: str):
    llm = get_llm(model_name)
    tools = get_tools(llm)

    # Create ReAct agent from tools and LLM
    react_agent = create_react_agent(llm=llm, tools=tools)

    # Define a graph
    builder = StateGraph()
    builder.add_node("agent", react_agent)
    builder.set_entry_point("agent")

    # Compile the graph
    app = builder.compile()

    return app
