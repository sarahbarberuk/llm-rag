from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import get_rag_chain

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
