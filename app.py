
import os
import io
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# =============================
# 1. Setup
# =============================
load_dotenv()

st.set_page_config(page_title="PDF Chat", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in your .env file.")
    st.stop()

# Session state initialization
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =============================
# 2. PDF Processing
# =============================
def build_vectorstore(uploaded_files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    all_chunks = []

    for file in uploaded_files:
        reader = PdfReader(io.BytesIO(file.read()))
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"page": i + 1}
                    )
                )

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    return FAISS.from_documents(all_chunks, embeddings)

# =============================
# 3. UI & Sidebar (Security Controls)
# =============================
st.title("📄 Simple PDF Chat")

with st.sidebar:
    st.header("Control Panel")
    
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.vectorstore is None:
        with st.spinner("Processing PDFs..."):
            st.session_state.vectorstore = build_vectorstore(uploaded_files)
            st.success("PDFs indexed successfully.")
    
    st.markdown("---")
    st.subheader("Security Protocols")
    
    # This button fulfills your "automated/manual purge" claim
    if st.button("Purge Session & Data", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.caption("Clearing the session wipes the in-memory FAISS index and all chat history.")

# =============================
# 4. Chat Interface
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask something about your PDF..."):

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.write("Please upload a PDF first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(k=5)

        docs = retriever.invoke(question)
        context = "\n\n".join(
            [f"(Page {d.metadata.get('page')}) {d.page_content}" for d in docs]
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        prompt = f"""
        Use the context below to answer the question.
        Mention page numbers when possible.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        with st.chat_message("assistant"):
            st.write(response.content)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )