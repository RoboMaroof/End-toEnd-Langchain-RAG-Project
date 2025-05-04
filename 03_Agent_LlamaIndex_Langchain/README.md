# 03_Agent_LlamaIndex_Langchain

> A more advanced agent-based RAG application combining **LlamaIndex** (for ingestion & retrieval) and **LangChain** (for agent orchestration) with full support for tool chaining, reranking, and custom document uploads.

---

## Additions to `03_Agent_LlamaIndex_Langchain`

| Feature                                      | `02_Agent` | `03_Agent_LlamaIndex` |
|----------------------------------------------|------------|------------------------|
| Static vector store from URL                 | ✅         | ✅                     |
| Dynamic file ingestion (PDF, TXT, DOCX, DB)  | ❌         | ✅                     |
| SQL-based data ingestion (FAQs, etc.)        | ❌         | ✅                     |
| Multi-source ingestion UI                    | ❌         | ✅                     |
| LlamaIndex-based vector index + retrieval    | ❌         | ✅                     |
| LLM-powered reranking via `LLMRerank`        | ❌         | ✅                     |
| Enhanced chunk scoring + filtering           | ❌         | ✅                     |
| Intermediate agent reasoning breakdown       | Limited | ✅ Detailed            |

---

## Overview

This app combines **LlamaIndex** for flexible ingestion and retrieval with **LangChain agents** to form a powerful, multi-tool LLM system. You can:

- Upload and ingest PDFs, DOCX, TXT, or SQLite DBs
- Use website URLs for live indexing
- Query using OpenAI-based agents
- View retrieved document chunks with scores
- Explore full agent reasoning steps and tool invocations

---

## Tech Stack

- **LLMs**: OpenAI (GPT-4o-mini)
- **Frameworks**: LangChain, LlamaIndex, FastAPI, Streamlit
- **Retrieval**: LlamaIndex + FAISS backend
- **Reranking**: LLM-based reranker via `LLMRerank`
- **Tools**: Wikipedia, Arxiv, custom vector retriever
- **Ingestion Sources**: Web, local files, SQL databases
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Environment**: `.env` with `python-dotenv`

---

## Project Structure

```plaintext
03_Agent_LlamaIndex_Langchain/
│
├── app.py                   # FastAPI app startup and router integration
├── client.py                # Streamlit UI with ingestion + chat
├── ingestion/               # All ingestion-related code (PDF, Web, SQL, Upload)
│   ├── index_builder.py
│   ├── routes.py
│   ├── sources.py
│   ├── upload_handler.py
├── agents/                  # Agent tools and logic using LangChain + LlamaIndex
│   ├── routes.py
│   └── tools.py
├── data/                    # Uploaded files, SQL databases, etc.
├── faiss_index/             # JSON-based vector store (via LlamaIndex)
└── README.md
```

## Quickstart

### 1. Install Dependencies

```bash
cd 03_Agent_LlamaIndex_Langchain
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

## Features in Streamlit UI

- **Ingestion Panel**:
  - Upload files: PDF, TXT, DOCX, SQLite DB
  - Ingest data from a website URL

- **Agent Chat**:
  - Enter a query
  - View:
    - Final agent response
    - Tools used (Wikipedia, Arxiv, VectorRetriever)
    - Retrieved document chunks with scores
    - Step-by-step intermediate reasoning
---

## API Endpoints
| Method | Endpoint                   | Description                                |
|--------|----------------------------|--------------------------------------------|
| POST   | `/vectordb/create`         | Ingest data from URL, file, or SQL         |
| POST   | `/vectordb/upload`         | Upload and ingest a local document         |
| POST   | `/openai/agent/invoke`     | Query agent with tools + retrieval         |

## Tools Used by Agent

- **WikipediaQueryRun**: Answer general knowledge questions  
- **ArxivQueryRun**: Fetch scientific papers and summaries  
- **vector_retriever** (custom):
  - Retrieves relevant docs from FAISS via LlamaIndex  
  - Applies LLM reranking (`LLMRerank`)  
  - Returns top chunks with scores  

---

## Screenshot

Below is screenshot from the Langchain-powered chat app:

![App Screenshot](/mnt/03_LlamaIndex_1.png)
![App Screenshot](/mnt/03_LlamaIndex_2.png)
---