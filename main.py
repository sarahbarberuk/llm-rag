from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from rag_chain import get_rag_chain
from ingest import ingest
from typing import List
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()
qa = get_rag_chain()

app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def serve_upload_page():
    with open("static/upload.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask")
def ask_question(query: Query):
    result = qa(query.question)
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
    }

@app.post("/upload")
async def upload_pdfs(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...)

    ):
    uploaded = []

    for file in files:
        # Uploads to data/ folder
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        # Ingests to vectordb/
        # Run in the background (non-blocking)
        background_tasks.add_task(ingest, file_path)

        uploaded.append(file.filename)

    return {
        "status": "uploaded",
        "message": f"Uploaded and started ingesting: {', '.join(uploaded)}"
    }