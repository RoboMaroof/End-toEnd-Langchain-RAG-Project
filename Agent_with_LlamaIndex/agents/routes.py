from fastapi import APIRouter, Body
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from .tools import get_tools
from ingestion.index_builder import load_index


router = APIRouter()
model = ChatOpenAI(model="gpt-4o-mini")

@router.post("/openai/agent/invoke")
def run_agent(inputs: dict):
    tools = get_tools()
    agent = initialize_agent(
        tools,
        model,
        agent_type="openai-tools",
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    result = agent.invoke({"input": inputs["input"]})

    tool_names = []
    retrieved_chunks = []
    intermediate_steps = []

    for action, observation in result.get("intermediate_steps", []):
        tool_names.append(action.tool)
        step_detail = {
            "tool": action.tool,
            "input": action.tool_input,
            "observation": observation
        }

        if action.tool == "vector_retriever":
            retrieved_chunks.extend(action.tool_input.split("\n\n"))

        intermediate_steps.append(step_detail)

    return {
        "output": result["output"],
        "tool_used": tool_names,
        "retrieved_chunks": retrieved_chunks,
        "intermediate_steps": intermediate_steps,
    }
