# 02_Agent_Langchain

> A Retrieval-Augmented Generation (RAG) + Tool-Enabled Agent app using LangChain and Streamlit with dual model support (OpenAI & Ollama).

---

## Additions to `01_Basic_ChatApp_Langchain`

| Feature                             | `01_Basic` | `02_Agent` |
|-------------------------------------|------------|------------|
| Basic RAG with LangChain            | ✅         | ✅         |
| Custom URL-based document ingestion | ❌         | ✅         |
| Tool-augmented LLM agent            | ❌         | ✅         |
| Wikipedia & Arxiv API integration   | ❌         | ✅         |
| Agent reasoning via `AgentExecutor` | ❌         | ✅         |
| LangSmith retriever tool (if vector exists) | ❌  | ✅         |
| Mode toggle: RAG vs Agent           | ❌         | ✅         |
| Tool usage breakdown in UI          | ❌         | ✅         |

---

## Overview

This version expands the basic RAG app into a dynamic, tool-augmented LLM agent system. It supports:

- Querying using either **RAG** or **Agent** mode
- Integration with **Wikipedia** and **Arxiv** as external tools
- Live ingestion of documents via **custom URLs**
- Model selection between **OpenAI GPT-4o-mini** and **Ollama (Gemma 3B)**

---

### Two Modes

#### RAG Mode (Retrieval-Augmented Generation)

- Uses a **vector database** (FAISS) to fetch relevant chunks from ingested documents
- Constructs a response **purely based on retrieved content**
- Ideal for **contextual question answering** over known sources like PDFs, websites, or databases

#### 🤖 Agent Mode (LLM Agent with Tools)

- Leverages LangChain's **AgentExecutor** to dynamically choose and invoke external tools
- Tools include:
  - `WikipediaQueryRun`
  - `ArxivQueryRun`
  - `LangSmithRetriever` (if vector store is present)
- Allows the LLM to **reason, plan, and fetch live knowledge**

---

## Tech Stack

- **LLMs**: OpenAI GPT-4o-mini, Ollama (Gemma 3B)
- **Frameworks**: LangChain, FastAPI, Streamlit
- **Tools**: WikipediaQueryRun, ArxivQueryRun, LangSmith (custom retriever)
- **Data Source**: Live document loading from URLs
- **Vector Store**: FAISS
- **Agent Engine**: `AgentExecutor` with intermediate step tracking
- **Environment**: `.env` + `python-dotenv`

---

## Project Structure

```plaintext
02_Agent_Langchain/
│
├── app.py           # FastAPI app with RAG and Agent chains
├── client.py        # Streamlit UI supporting both RAG and Agent modes
├── faiss_index/     # Saved FAISS vector store from URL ingestion
└── README.md        # This file
```

## Quickstart

### 1. Install Dependencies

```bash
cd 02_Agent_Langchain
pip install -r ../requirements.txt
```

### 2. Configure `.env` File
```env
OPENAI_API_KEY=openai_api_key
```

### 3. Run the Backend (FastAPI)
```bash
python app.py
```
On first run, it will:
- Load LangChain docs
- Chunk and embed them
- Save a FAISS vector store locally in `faiss_index/`
- Start the API server at `http://localhost:8000`

### 4. Launch the Frontend (Streamlit)
```bash
streamlit run client.py
```
- Choose mode: RAG or Agent
- Select model: openai or ollama
- Enter a query
- (Optional) Provide a custom URL for document ingestion

## API Endpoints
| Method | Endpoint                     | Description                           |
|--------|------------------------------|---------------------------------------|
| POST   | `/vectordb/create`           | Ingest docs from a custom URL         |
| POST   | `/openai/invoke`             | RAG with OpenAI                       |
| POST   | `/ollama/invoke`             | RAG with Ollama                       |
| POST   | `/openai/agent/invoke`       | Agent with OpenAI + tools             |
| POST   | `/ollama/agent/invoke`       | Agent with Ollama (limited)        |

## UI Features
- **Switch** between RAG and Agent **mode**
- View top **retrieved sections** in RAG mode
- See **tools used** (Wikipedia, Arxiv, etc.) in Agent mode
- **Ingest** new documents live via URL

## Project Structure
```
01_Basic_ChatApp_Langchain/
│
├── app.py           # FastAPI backend with vector indexing and RAG endpoints
├── client.py        # Streamlit frontend for asking questions
├── faiss_index/     # Stores vector index (auto-created)
│   ├── index.faiss
│   └── index.pkl
└── README.md        
```

---

## Screenshot

Below is screenshot from the Langchain-powered chat app:

![App Screenshot](/mnt/02_Langchain_Agent_1.png)
![App Screenshot](/mnt/02_Langchain_Agent_2.png)
---