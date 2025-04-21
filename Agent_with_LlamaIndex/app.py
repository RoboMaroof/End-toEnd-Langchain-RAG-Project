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

logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
def initialize_vector_index():
    if os.path.exists(VECTORDB_PATH):
        logging.info("📦 Existing vector index found. Skipping index creation.")
    elif os.path.exists(DEFAULT_DOCS_FOLDER):
        logging.info(f"📁 No index found. Creating index from local documents in: {DEFAULT_DOCS_FOLDER}")
        try:
            create_index("docs", DEFAULT_DOCS_FOLDER)
            logging.info("✅ Initial vector index created successfully.")
        except Exception as e:
             logging.error(f"❌ Failed to initialize index from local documents: {e}")
    else:
        logging.warning(f"⚠️ No index or local document folder found. Please ingest data via API.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
