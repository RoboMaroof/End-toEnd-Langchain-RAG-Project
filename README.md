# 🔗 End-to-End LangChain RAG + Agent Project

> Personal project showcasing Retrieval-Augmented Generation (RAG) and Agent-based architectures using LangChain. Built for learning, experimentation, and to demonstrate real-world LLM orchestration techniques.

---

## 🧰 Tech Stack

- **🧠 LLMs**: OpenAI, Ollama, Grok
- **🛠 Frameworks**: LangChain, FastAPI, Streamlit
- **🔍 Retrieval**: FAISS (vector store), OpenAI Embeddings
- **📄 Data Ingestion**: Web(`WebBaseLoader`), SQL, PDF
- **📚 Reranking**: Cohere Rerank API
- **🤖 APIs**: Wikipedia, Arxiv
- **🤖 Agent Tools**: Langchain, LangGraph, LlamaIndex, LangSmith retriever
- **🔐 Secrets**: `dotenv`

---

## 🚀 Project Overview

This is an end-to-end Retrieval-Augmented Generation (RAG) and Agent system using [LangChain](https://www.langchain.com/). It supports:

- Dual model backends (OpenAI + Ollama)
- Custom document ingestion
- Dynamic agent tool usage (Wikipedia, Arxiv, Vector retriever)
- Web API and GUI frontend using Streamlit

---

## 📁 Project Structure

```plaintext
END-TOEND-LANGCHAIN-RAG-PROJECT/
│
├── Agent/
│   ├── app.py               # RAG + Agent API server with tool integrations
│   ├── client.py            # Streamlit frontend to interact with RAG & Agent endpoints
│   └── faiss_index/         # Folder to store FAISS vector DB files
│
├── WebURL/
│   ├── app.py               # RAG API server using static LangChain Docs
│   ├── client.py            # Streamlit frontend for RAG over LangChain docs
│   └── faiss_index/         # Folder to store FAISS vector DB files
│
├── .env                     # Environment variables
├── .gitignore
├── requirements.txt
└── README.md                
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/langchain-rag-agent.git
cd langchain-rag-agent
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a .env file in the root:

```env
OPENAI_API_KEY=your_openai_api_key
```

## 🌐 Running the Applications

### 🔹 WebURL App

**Start API Server**

```bash
cd WebURL
python app.py
```

**Run Client**

```bash
streamlit run client.py
```
🔹 Use this to query LangChain documentation via RAG.

### 🔸 Agent App

**Start API Server**

```bash
cd Agent
python app.py
```

**Run Client**

```bash
streamlit run client.py
```
🔸 This supports both RAG and Agent modes. Agent mode includes external tools (Wikipedia, Arxiv, etc).


## 📡 API Endpoints

### Shared Endpoints

| Method | Endpoint             | Description                              |
|--------|----------------------|------------------------------------------|
| POST   | `/vectordb/create`   | Ingest docs and create FAISS vectorstore |
| POST   | `/openai/invoke`     | Query using OpenAI + RAG                 |
| POST   | `/ollama/invoke`     | Query using Ollama + RAG                 |

### Agent-Specific (Agent App Only)

| Method | Endpoint                 | Description                           |
|--------|--------------------------|---------------------------------------|
| POST   | `/openai/agent/invoke`   | Use OpenAI Agent with tools           |
| POST   | `/ollama/agent/invoke`   | ⚠️ Not supported currently             |

---

## 🖥 Streamlit UI Features

### WebURL Client

- Static RAG pipeline using LangChain Docs
- Choose between OpenAI / Ollama
- View response + top 5 retrieved sections

### Agent Client

- Ingest any public web page
- Choose RAG or Agent mode
- Agent mode uses:
  - Wikipedia
  - Arxiv
  - LangSmith (via vector store)
- Displays:
  - Final answer
  - Retrieved docs or tools used

---

## 📸 Screenshot

![App Screenshot](/mnt/Agent_App.png)

---

## 🧠 Notes

- 🔐 Don't forget to set your `.env` file with API keys.
- ⚠️ Vector store must be created before querying.
- 🧪 Ollama is only available for RAG (not Agent mode).
- Designed for local experimentation, research, and demos.

---

## 🙋 Why This Project?

This repo demonstrates key skills in:

- LangChain application design
- RAG pipeline engineering
- Tool-enabled LLM agents
- Backend + frontend LLM deployment

Ideal for showcasing your experience in **LLM-based systems**, **LangChain**, and **real-world ML pipelines** to potential employers.

---

## 📬 Contributions

Pull requests welcome! Feel free to fork, customize, and extend.

---

