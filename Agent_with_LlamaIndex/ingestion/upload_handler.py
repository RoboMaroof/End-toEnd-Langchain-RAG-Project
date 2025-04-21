import os
from fastapi import UploadFile
from pathlib import Path
import shutil

UPLOAD_DIR = "uploads"

def save_uploaded_file(uploaded_file: UploadFile) -> str:
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    return file_path
