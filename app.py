import os
import io
import time
import streamlit as st

from datetime import datetime
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

# ================================================
# CONFIG
# ================================================
st.set_page_config(
    page_title="RAG Chatbot (Secure AI-First Prototype)",
    page_icon="🤖",
    layout="wide"
)

# ================================================
# SIDEBAR: SETTINGS & SECURITY FEATURES
# ================================================
st.sidebar.header("🔐 Session & Developer Settings")

# Hybrid key entry: Default from secrets, optional user override
DEFAULT_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
user_api_key = st.sidebar.text_input(
    "🔑 Enter your OpenAI API key (optional)",
    type="password"
)
OPENAI_API_KEY = user_api_key.strip() if user_api_key else DEFAULT_API_KEY

if not OPENAI_API_KEY:
    st.error("No OpenAI API key found. Please add it to Streamlit Secrets or input one in the sidebar.")
    st.stop()

# Developer mode toggle
DEV_MODE = st.sidebar.checkbox("👩‍💻 Developer Mode", value=False)

# Manual purge button
if st.sidebar.button("🧹 Purge Session & Clear Memory"):
    st.session_state.clear()
    st.experimental_rerun()

# ================================================
# APP HEADER
# ================================================
st.title("📄 Secure RAG Document Chatbot")
st.caption("In-memory, privacy-first document Q&A system 🧠")

INACTIVITY_LIMIT = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")) * 60  # seconds


# ================================================
# SESSION STATE INITIALIZATION
# ================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

# Timeout logic
now = time.time()
if now - st.session_state.last_activity > INACTIVITY_LIMIT:
    st.warning("Session expired due to inactivity. Clearing data for privacy.")
    st.session_state.clear()
    st.experimental_rerun()
else:
    st.session_state.last_activity = now  # refresh activity timestamp


# ================================================
# CALLBACK HANDLER FOR STREAMING OUTPUT
# ================================================
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.streamed_text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.streamed_text += token
        self.placeholder.markdown(self.streamed_text)


# ================================================
# RAG PIPELINE BUILDER
# ================================================
def build_history() -> str:
    lines = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_vectorstore_from_pdfs(uploaded_files, api_key: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    all_chunks = []

    for file in uploaded_files:
        try:
            bytes_data = io.BytesIO(file.read())
            loader = PyPDFLoader(bytes_data)
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            if DEV_MODE:
                st.error(f"Error reading {file.name}: {e}")
            else:
                st.error(f"Failed to read {file.name}. It may be corrupted.")

    if not all_chunks:
        st.error("No readable text found in the uploaded PDFs.")
        return None

    return FAISS.from_documents(all_chunks, embeddings)


def build_rag_chain(vectorstore, api_key: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant answering questions about the user's uploaded documents.

Conversation so far:
{history}

Use ONLY the context below to answer. If the answer is not grounded in the context, say "I don't know."

Context:
{context}

Question:
{question}

After replying, list each referenced source with page number or snippet for citation.
        """
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True,
        callbacks=[],
        openai_api_key=api_key,
    )

    rag_chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
            "history": lambda _: build_history(),
        }
        | RunnablePassthrough.assign(
            context=lambda x: "\n\n".join(
                f"(Page {d.metadata.get('page', 'N/A')}) {d.page_content[:500]}" for d in x["docs"]
            )
        )
        | prompt
        | llm
    )

    return rag_chain


# ================================================
# FILE UPLOAD & VECTOR STORE BUILD
# ================================================
uploaded_files = st.file_uploader(
    "📥 Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents and generating embeddings..."):
        st.session_state.vectorstore = build_vectorstore_from_pdfs(uploaded_files, OPENAI_API_KEY)
    if st.session_state.vectorstore:
        st.success("✅ Documents processed in memory. You can now chat.")


# ================================================
# LOAD RAG CHAIN
# ================================================
rag_chain = None
if st.session_state.vectorstore:
    rag_chain = build_rag_chain(st.session_state.vectorstore, OPENAI_API_KEY)

# ================================================
# CHAT INTERFACE
# ================================================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    st.session_state.last_activity = time.time()
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if not rag_chain:
        response_text = "Please upload PDFs first."
    else:
        try:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                stream_handler = StreamHandler(placeholder)

                rag_chain.llm.callbacks = [stream_handler]
                result = rag_chain.invoke(question)
                response_text = stream_handler.streamed_text or result.content
        except Exception as e:
            if DEV_MODE:
                st.exception(e)
            response_text = (
                "⚠️ An error occurred while generating the response. Try again later."
            )

    st.session_state.messages.append({"role": "assistant", "content": response_text})


# ================================================
# CITATION DISPLAY
# ================================================
if rag_chain and DEV_MODE and st.session_state.vectorstore:
    st.write("---")
    st.subheader("📚 Retrieved Source Snippets (Developer View)")
    retriever_preview = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
    sample_docs = retriever_preview.get_relevant_documents("Sample inspection prompt")
    for doc in sample_docs:
        with st.expander(f"File: {doc.metadata.get('source', 'unknown')} - Page: {doc.metadata.get('page', 'N/A')}"):
            st.write(doc.page_content[:800])
