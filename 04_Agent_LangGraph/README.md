# 04_Agent_LangGraph

> A production-style, tool-augmented LLM agent framework powered by **LangGraph**, **LangChain**, and **LlamaIndex**, with support for multi-step reasoning, advanced tool chaining, and model routing (OpenAI + Groq).

---

## Additions to `03_Agent_LlamaIndex_Langchain`

| Feature                                             | `03_LlamaIndex` | `04_LangGraph` |
|-----------------------------------------------------|------------------|----------------|
| LlamaIndex ingestion and vector retrieval           | ✅               | ✅              |
| LLM-based reranking (`LLMRerank`)                   | ✅               | ✅              |
| Tool-augmented agent with Wikipedia, Arxiv, etc.    | ✅               | ✅              |
| **LangGraph state-machine agent framework**         | ❌               | ✅              |
| Agent memory + multi-turn message state             | ❌               | ✅              |
| ToolNode with dynamic routing via `tools_condition` | ❌             | ✅              |
| Model routing (OpenAI & Groq support)               | ❌               | ✅              |
| Fine-grained intermediate step visualization        | Limited        | ✅              |

---

## Overview

This is the most advanced agent implementation in the suite. It uses:

- **LangGraph** to build a persistent, message-driven agent workflow
- **ToolNode** and `tools_condition` for dynamic tool invocation
- Integration of multiple tools (Wikipedia, Arxiv, Tavily, VectorRetriever)
- Flexible ingestion from websites, PDFs, SQL
- Streamlit UI with tool usage breakdown, reasoning trace, and chunk scores

---

## Tech Stack

- **LLMs**: OpenAI (GPT-4o-mini), Groq (Qwen 32B)
- **Agent Framework**: LangGraph (for stateful agent execution)
- **Tools**: Wikipedia, Arxiv, Tavily Search, LlamaIndex-powered retriever
- **Reranker**: LLMRerank via LlamaIndex
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector DB**: JSON-based FAISS via LlamaIndex
- **Env Management**: `.env` + `python-dotenv`

---

## Project Structure

```plaintext
04_Agent_LangGraph/
├── app.py               # FastAPI server and startup ingestion
├── client.py            # Streamlit frontend for querying and data ingestion
├── ingestion/           # Ingestion from files, URLs, or SQL
│   ├── index_builder.py
│   ├── routes.py
│   ├── sources.py
│   └── upload_handler.py
├── agents/              # LangGraph-based agent logic
│   ├── graph_builder.py
│   ├── tools.py
│   └── routes.py
├── data/                # Document folder, uploads, and SQLite FAQ
├── faiss_index/         # LlamaIndex-based vector storage (JSON format)
└── README.md
```

## Quickstart

### 1. Install Dependencies

```bash
cd 04_Agent_LangGraph
pip install -r ../requirements.txt
```

### 2. Configure `.env` File
```env
OPENAI_API_KEY=openai_api_key
VECTORDB_PATH=./faiss_index
DEFAULT_DOCS_FOLDER=./data/docs
UPLOADED_DOCS_FOLDER=./data/uploads
SQL_DB_PATH=./data/faq_knowledge.db
```

### 3. Run the Backend (FastAPI)
```bash
python app.py
```
On first run, it will:
- Loads existing vector index if present
- Optionally ingests docs/SQL data on startup
- Save a FAISS vector store locally in `faiss_index/`
- Hosts endpoints for ingestion and querying
- Start the API server at `http://localhost:8000`

### 4. Launch the Frontend (Streamlit)
```bash
streamlit run client.py
```
- Choose mode: RAG or Agent
- Select model: openai or ollama
- Enter a query
- (Optional) Provide a custom URL for document ingestion

## Agent Features
- Stateful message flow via LangGraph
- Real-time tool invocation based on model output
- Support for multiple model backends (OpenAI & Groq)
- Visual breakdown of:
  - Tools used
  - Retrieved data chunks (with tool/source type)
  - Intermediate reasoning steps
---

## API Endpoints
| Method | Endpoint                   | Description                                |
|--------|----------------------------|--------------------------------------------|
| POST   | `/vectordb/create`         | Ingest data from URL, file, or SQL         |
| POST   | `/vectordb/upload`         | Upload and ingest a local document         |
| POST   | `/agent/invoke`     | Trigger LangGraph agent w/ model ID        |

## Tools Used by Agent

- **WikipediaQueryRun**: Answer general knowledge questions  
- **ArxivQueryRun**: Fetch scientific papers and summaries  
- **TavilySearchResults**: Fetch results through online search
- **vector_retriever** (custom):
  - Retrieves relevant docs from FAISS via LlamaIndex  
  - Applies LLM reranking (`LLMRerank`)  
  - Returns top chunks with scores  

---

## Screenshot
Below are screenshots from the final LangGraph-powered agent app:

![App Screenshot](/mnt/04_LangGraph_1.png)
![App Screenshot](/mnt/04_LangGraph_2.png)
![App Screenshot](/mnt/04_LangGraph_3.png)
![App Screenshot](/mnt/04_LangGraph_4.png)
---