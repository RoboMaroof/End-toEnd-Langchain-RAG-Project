# 01_Basic_ChatApp_Langchain

> A minimal Retrieval-Augmented Generation (RAG) app using LangChain, Streamlit, and FAISS vector store, with dual model support for OpenAI and Ollama.

---

## Overview

This basic RAG chat app demonstrates how to:
- Load and chunk documents from a website
- Embed them into a FAISS vector store
- Retrieve top-k relevant chunks with reranking (via Cohere)
- Generate answers using OpenAI or Ollama models
- Interact via a simple Streamlit UI

---

## Tech Stack

- **LLMs**: OpenAI GPT-4o-mini, Ollama (Gemma 3B)
- **Frameworks**: LangChain, FastAPI, Streamlit
- **Vector Store**: FAISS
- **Embeddings**: OpenAI Embeddings
- **Reranking**: Cohere ReRank API (`rerank-english-v3.0`)
- **Ingestion Source**: [LangChain documentation](https://docs.smith.langchain.com/)
- **Environment**: `.env` managed with `python-dotenv`

---

## Quickstart

### 1. Install Dependencies

```bash
cd 01_Basic_ChatApp_Langchain
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
- Choose a model (`openai` or `ollama`)
- Ask questions
- View top 5 retrieved chunks with source context

## API Endpoints
| Method | Endpoint            | Description                                 |
|--------|---------------------|---------------------------------------------|
| POST   | `/vectordb/create`  | Creates FAISS vector store (on startup)     |
| POST   | `/openai/invoke`    | Query with OpenAI model                     |
| POST   | `/ollama/invoke`    | Query with Ollama (Gemma 3B)                |

## UI Features
- **Model selector** (`OpenAI` or `Ollama`)
- **Query input** with retrieval and response
- **Expandable views** of top 5 document chunks retrieved
- Requires vector store to be created (automatically done on first run)

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

![App Screenshot](/mnt/01_Langchain.png)
---