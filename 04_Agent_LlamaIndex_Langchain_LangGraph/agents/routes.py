from fastapi import APIRouter, Body
from langchain_core.agents import AgentFinish
from agents.langgraph_agent import get_langgraph_agent

router = APIRouter()

@router.post("/{model_choice}/agent/invoke")
def invoke_langgraph_agent(inputs: dict, model_choice: str):
    try:
        app = get_langgraph_agent(model_choice)
        result = app.invoke({"input": inputs["input"]})

        tool_names = []
        intermediate_steps = []
        retrieved_chunks = []
        output_text = None

        if isinstance(result, AgentFinish):
            output_text = result.return_values.get("output")

        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                action = step.get("action")
                observation = step.get("observation")

                if action:
                    tool_names.append(action.tool)
                    intermediate_steps.append({
                        "tool": action.tool,
                        "input": action.tool_input,
                        "observation": observation
                    })

                    if action.tool == "vector_retriever" and isinstance(observation, list):
                        retrieved_chunks.extend([
                            {"text": chunk["text"], "score": chunk.get("score")}
                            for chunk in observation if isinstance(chunk, dict)
                        ])

        return {
            "output": output_text,
            "tool_used": tool_names,
            "intermediate_steps": intermediate_steps,
            "retrieved_chunks": retrieved_chunks
        }

    except Exception as e:
        return {"error": str(e)}
