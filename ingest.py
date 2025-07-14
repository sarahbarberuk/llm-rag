from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def ingest():
    loader = PyPDFLoader("data/sample.pdf")  # Or any file you put in /data
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="vectordb"
    )
    vectordb.persist()
    print("✅ Ingested and saved to vectordb!")

if __name__ == "__main__":
    ingest()
