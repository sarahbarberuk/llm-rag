from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from rag_chain import get_rag_chain
from ingest import ingest
import shutil

app = FastAPI()
qa = get_rag_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    result = qa(query.question)
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
    }

@app.post("/upload")
async def upload_pdf(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
    ):
    # Uploads to data/ folder
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingests to vectordb/
    # Run in the background (non-blocking)
    background_tasks.add_task(ingest, file_path)

    return {"status": "uploaded", "filename": file.filename}