from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.database import DatabaseReader
from llama_index.core import SimpleDirectoryReader

def get_documents(source_type, source_path):
    if source_type == "website":
        return SimpleWebPageReader().load_data(urls=[source_path])
    elif source_type == "docs":
        return SimpleDirectoryReader(source_path).load_data()
    elif source_type == "sql":
        query = "SELECT * FROM faq"
        config = {
            "scheme": "sqlite",
            "database": source_path
        }
        return DatabaseReader(**config).load_data(query=query)
    else:
        raise ValueError("Unsupported source type")
