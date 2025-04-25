from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.llms import HuggingFaceHub
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.groq import Groq
# from llama_index.llms import Groq, OpenAI
import os

def get_llm(model_name: str):
    if model_name.startswith("openai"):
        return ChatOpenAI(model=model_name.replace("openai:", ""), temperature=0.7)

    elif model_name.startswith("huggingface"):
        return HuggingFaceHub(repo_id=model_name.split(":")[1], model_kwargs={"temperature": 0.7})

    elif model_name.startswith("ollama"):
        return ChatOllama(model=model_name.split(":")[1])

    elif model_name.startswith("anthropic"):
        return Anthropic(model=model_name.split(":")[1])

    elif model_name.startswith("groq"):
        return Groq(model=model_name.split(":")[1])

    elif model_name.startswith("local"):
        # Placeholder for local models; use langchain.llms.LocalLLM or similar
        raise NotImplementedError("Local model loading not implemented yet.")

    else:
        raise ValueError("Unsupported model provider.")
