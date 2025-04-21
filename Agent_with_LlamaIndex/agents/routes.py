from fastapi import APIRouter
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from .tools import get_tools

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
    tool_names = [action.tool for action, _ in result.get("intermediate_steps", [])]
    return {"output": result["output"], "tool_used": tool_names}
