from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import (
    AnyMessage, HumanMessage, AIMessage, ToolMessage
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from .tools import get_tools


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class GraphBuilder:
    def __init__(self, model_config: str = "openai:gpt-4o-mini"):
        model_type, model_name = model_config.split(":")
        self.tools = get_tools()
        self.llm = self._init_llm(model_type, model_name).bind_tools(tools=self.tools)
        self.graph = self._build_graph()

    def _init_llm(self, model_type: str, model_name: str):
        if model_type == "groq":
            return ChatGroq(model=model_name)
        return ChatOpenAI(model=model_name)

    def _llm_tool_node(self, state: AgentState):
        return {"messages": [self.llm.invoke(state["messages"])]}

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("tool_calling_llm", self._llm_tool_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "tool_calling_llm")
        builder.add_conditional_edges("tool_calling_llm", tools_condition)
        builder.add_edge("tools", "tool_calling_llm")

        return builder.compile()

    def invoke(self, messages: List[AnyMessage]) -> dict:
        return self.graph.invoke({"messages": messages})

    def invoke_and_parse(self, messages: List[AnyMessage]) -> dict:
        raw_response = self.invoke(messages)
        return self._parse_response(raw_response)

    def _parse_response(self, response: dict) -> dict:
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

        return {
            "final_output": final_output,
            "tools_used": tools_used,
            "retrieved_chunks": retrieved_chunks,
            "intermediate_steps": intermediate_steps,
        }
