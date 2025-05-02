from fastapi import APIRouter, Body, HTTPException
from langchain_core.messages import HumanMessage
from .graph_builder import GraphBuilder

router = APIRouter()

@router.post("/agent/invoke")
def run_agent(inputs: dict = Body(...)):
    user_input = inputs.get("input", "")
    model_config = inputs.get("model", "openai:gpt-4o-mini")

    if isinstance(user_input, dict):
        user_input = user_input.get("input", "")

    if not isinstance(user_input, str):
        raise HTTPException(status_code=400, detail="Field 'input' must be a string.")

    try:
        graph_runner = GraphBuilder(model_config=model_config)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_config}")

    messages = [HumanMessage(content=user_input)]
    return graph_runner.invoke_and_parse(messages)
