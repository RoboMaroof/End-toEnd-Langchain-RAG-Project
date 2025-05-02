from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from typing import Annotated 
from langgraph.graph.message import add_messages 

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from fastapi import APIRouter, Body, HTTPException

import os
from pathlib import Path
from dotenv import load_dotenv

from .tools import get_tools
from ingestion.index_builder import load_index

env_path = Path(__file__).resolve().parents[1]/'.env'
load_dotenv(dotenv_path=env_path)
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

router = APIRouter()
tools = get_tools()

llm = ChatOpenAI(model="gpt-4o-mini")
# llm=ChatGroq(model="qwen-qwq-32b")
llm_with_tools=llm.bind_tools(tools=tools)

#State Schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

#Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

## Edgess
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

def parse_graph_response(response: dict):
    messages = response.get("messages", [])

    final_output = None
    tools_used = []
    retrieved_chunks = []
    intermediate_steps = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            intermediate_steps.append({
                "type": "human",
                "content": msg.content
            })

        elif isinstance(msg, AIMessage):
            tool_calls = msg.additional_kwargs.get("tool_calls", [])
            if tool_calls:
                for call in tool_calls:
                    function = call.get("function", {})
                    tool_name = function.get("name")
                    args = function.get("arguments")
                    tools_used.append(tool_name)
                    intermediate_steps.append({
                        "type": "ai_tool_call",
                        "tool": tool_name,
                        "args": args
                    })
            else:
                final_output = msg.content
                intermediate_steps.append({
                    "type": "ai_final_response",
                    "content": msg.content
                })

        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None)
            tool_content = msg.content
            artifact = getattr(msg, "artifact", {})

            intermediate_steps.append({
                "type": "tool_response",
                "tool": tool_name,
                "content": tool_content
            })

            if isinstance(artifact, dict) and "results" in artifact:
                retrieved_chunks.extend([
                    {
                        "tool": tool_name,
                        "type": "result",
                        "data": result
                    }
                    for result in artifact["results"]
                ])
            else:
                retrieved_chunks.append({
                    "tool": tool_name,
                    "type": "text",
                    "data": tool_content
                })

    print("#" * 50)
    print("FINAL OUTPUT:")
    print(final_output)
    print("#" * 50)
    print("TOOLS USED:")
    print(tools_used)
    print("#" * 50)
    print("RETRIEVED CHUNKS:")
    for chunk in retrieved_chunks:
        print(chunk)
    print("#" * 50)
    print("INTERMEDIATE STEPS:")
    for step in intermediate_steps:
        print(step)
    print("#" * 50)

    return {
        "final_output": final_output,
        "tools_used": tools_used,
        "retrieved_chunks": retrieved_chunks,
        "intermediate_steps": intermediate_steps,
    }


@router.post("/openai/agent/invoke")
def run_agent(inputs: dict = Body(...)):

    user_input_data = inputs.get("input")

    # Handle nested 'input': {'input': ...}
    if isinstance(user_input_data, dict) and "input" in user_input_data:
        user_input = user_input_data["input"]
    else:
        user_input = user_input_data

    # Validate
    if not isinstance(user_input, str):
        raise HTTPException(status_code=400, detail="Field 'input' must be a string.")
    
    messages = [HumanMessage(content=user_input)]
    
    response = graph.invoke({"messages": messages})
    print("$"*50)
    print(response)
    print("$"*50)
    
    return parse_graph_response(response)