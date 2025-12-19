import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = "data/docs"
DB_PATH = "vectorstore"

def ingest():
    print("INGEST STARTED")

    if not os.path.exists(DATA_PATH):
        print("❌ data/docs not found")
        return

    files = os.listdir(DATA_PATH)
    print("FILES FOUND:", files)

    documents = []
    for file in files:
        if file.lower().endswith(".pdf"):
            print("LOADING PDF:", file)
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        print("❌ No documents loaded")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    print(f"CREATED {len(chunks)} CHUNKS")

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    print("✅ VECTORSTORE CREATED")

if __name__ == "__main__":
    ingest()
