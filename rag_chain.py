from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def get_rag_chain():
    # load vector DB
    db = Chroma(persist_directory="vectordb", embedding_function=OpenAIEmbeddings())
    # create retriever wrapper around the vector DB so LangChain chains can use it
    # (return the top 3 most similar chunks to the question)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    # creates a chain like: 
    # question → [retrieve top 3 chunks] → [send to GPT-3.5] → answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
