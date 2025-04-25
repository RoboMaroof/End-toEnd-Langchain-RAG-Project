from fastapi import FastAPI
from ingestion.routes import router as ingestion_router
from agents.routes import router as agent_router
from ingestion import create_index

import os
import logging

app = FastAPI(title="Langchain RAG API", version="1.0")

app.include_router(ingestion_router, prefix="/vectordb")
app.include_router(agent_router, prefix="")

VECTORDB_PATH = os.getenv("VECTORDB_PATH")
DEFAULT_DOCS_FOLDER = os.getenv("DEFAULT_DOCS_FOLDER")
UPLOADED_DOCS_FOLDER = os.getenv("UPLOADED_DOCS_FOLDER")
SQL_DB_PATH = os.getenv("SQL_DB_PATH")

logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
def initialize_vector_index():
    index_exists = os.path.exists(VECTORDB_PATH)
    if index_exists:
        logging.info("📦 Existing vector index found. Skipping index creation.")
        return

    sources_ingested = []

    # Ingest documents folder if it exists
    if DEFAULT_DOCS_FOLDER and os.path.exists(DEFAULT_DOCS_FOLDER):
        try:
            logging.info(f"📁 Creating index from local documents in: {DEFAULT_DOCS_FOLDER}")
            create_index("docs", DEFAULT_DOCS_FOLDER)
            sources_ingested.append("docs")
        except Exception as e:
            logging.error(f"❌ Failed to index documents: {e}")

    # Ingest SQL database if it exists
    if SQL_DB_PATH and os.path.exists(SQL_DB_PATH):
        try:
            logging.info(f"🗄️ Creating index from FAQ database at: {SQL_DB_PATH}")
            create_index("sql", SQL_DB_PATH)
            sources_ingested.append("sql")
        except Exception as e:
            logging.error(f"❌ Failed to index FAQ database: {e}")

    if not sources_ingested:
        logging.warning("⚠️ No data sources found to index at startup.")
    else:
        logging.info(f"✅ Indexed sources at startup: {', '.join(sources_ingested)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)