# 🔗 End-to-End LangChain RAG + Agent Project

> A progressive suite of apps demonstrating Retrieval-Augmented Generation (RAG) and Agent-based orchestration using LangChain, LlamaIndex, and LangGraph.

---

## 🧠 Tech Stack

- **LLMs**: OpenAI, Groq, Ollama
- **Frameworks**: LangChain, LangGraph, LlamaIndex, FastAPI, Streamlit
- **Data Sources**: Web (URLs), PDFs, SQL
- **Vector Store**: FAISS
- **Reranking**: Cohere ReRank, LLMRerank
- **APIs**: Wikipedia, Arxiv, Tavily
- **Environment Management**: `python-dotenv`

---

## 📚 Project Progression

| Folder                          | Description                                               |
|---------------------------------|-----------------------------------------------------------|
| `01_Basic_ChatApp_Langchain`    | Basic RAG with LangChain + Streamlit                     |
| `02_Agent_Langchain`            | Adds LangChain agent tools (Wikipedia, Arxiv)            |
| `03_Agent_LlamaIndex_Langchain` | Uses LlamaIndex for ingestion & LangChain for agents     |
| `04_Agent_LangGraph`            | Full LangGraph-based agent flow with advanced orchestration |

---

## 🚀 Final Project: `04_Agent_LangGraph`

### Features

- Fully modular LLM agent using LangGraph and LangChain
- Tools: Wikipedia, Arxiv, Tavily, FAISS-based Vector Retrieval
- Ingest from web, files, or SQL using FastAPI endpoints
- Reranking with `LLMRerank` from LlamaIndex
- Streamlit UI with:
  - File uploader or web URL input
  - Multi-tool invocation
  - Display of retrieved chunks, tools used, and intermediate reasoning steps

### Run Locally

```bash
cd 04_Agent_LangGraph
uvicorn app:app --reload
streamlit run client.py
```

Ensure .env file has:
```env
OPENAI_API_KEY=your_key
API_BASE_URL=http://localhost:8000
VECTORDB_PATH=./faiss_index/index_store.json
DEFAULT_DOCS_FOLDER=./data/docs
SQL_DB_PATH=./data/faq_knowledge.db
```

---

## 📸 Screenshots

![App Screenshot](/mnt/04_LangGraph_1.png)
![App Screenshot](/mnt/04_LangGraph_2.png)
![App Screenshot](/mnt/04_LangGraph_3.png)
![App Screenshot](/mnt/04_LangGraph_4.png)

---