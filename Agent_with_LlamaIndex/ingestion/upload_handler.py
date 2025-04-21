import os
from fastapi import UploadFile
from pathlib import Path
import shutil

UPLOADED_DOCS_FOLDER = os.getenv("UPLOADED_DOCS_FOLDER")

def save_uploaded_file(uploaded_file: UploadFile) -> str:
    Path(UPLOADED_DOCS_FOLDER).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(UPLOADED_DOCS_FOLDER, uploaded_file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    return file_path
