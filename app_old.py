# import os
# import streamlit as st


# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# # Trigger rebuild

# # --------------------------------------------------
# # SESSION STATE (MUST BE FIRST STREAMLIT OPERATION)
# # --------------------------------------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []



# st.set_page_config(
#     page_title="RAG Chatbot",
#     page_icon="🤖",
#     layout="centered"
# )
# st.write("KEY LOADED:", os.getenv("OPENAI_API_KEY") is not None)

# # --------------------------------------------------
# # STYLING
# # --------------------------------------------------
# st.markdown(
#     """
#     <style>
#         .block-container {
#             max-width: 750px;
#             margin: auto;
#             padding-top: 2rem;
#         }
#         .stChatMessage {
#             padding: 0.8rem 1rem;
#             border-radius: 12px;
#             margin-bottom: 0.8rem;
#         }
#         .stChatMessage[data-testid="stChatMessage-user"] {
#             background: #DCF2FF;
#             border: 1px solid #B6E3FF;
#         }
#         .stChatMessage[data-testid="stChatMessage-assistant"] {
#             background: #F4F4F8;
#             border: 1px solid #E0E0E7;
#         }
#         h1 {
#             text-align: center;
#             font-weight: 700;
#         }
#         .caption {
#             text-align: center;
#             color: #666;
#             margin-bottom: 2rem;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --------------------------------------------------
# # HEADER
# # --------------------------------------------------
# st.markdown("<h1>📄 RAG Document Chatbot</h1>", unsafe_allow_html=True)
# st.markdown(
#     "<p class='caption'>Ask questions grounded in your documents</p>",
#     unsafe_allow_html=True
# )

# DB_PATH = "vectorstore"

# # --------------------------------------------------
# # CHAT HISTORY (DEFENSIVE)
# # --------------------------------------------------
# def build_history() -> str:
#     if "messages" not in st.session_state:
#         return ""

#     history_lines = []
#     for msg in st.session_state.messages:
#         role = "User" if msg["role"] == "user" else "Assistant"
#         history_lines.append(f"{role}: {msg['content']}")

#     return "\n".join(history_lines)

# # --------------------------------------------------
# # LOAD RAG PIPELINE
# # --------------------------------------------------
# @st.cache_resource
# def load_rag():
#     embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


#     if not os.path.exists(DB_PATH):
#         return None

#     db = FAISS.load_local(
#         DB_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     retriever = db.as_retriever(search_kwargs={"k": 12})

#     prompt = ChatPromptTemplate.from_template(
#         """You are a helpful assistant answering questions about the user's documents.

# Conversation so far:
# {history}

# Answer the question using ONLY the context below.
# If the answer is not in the context, say "I don't know."

# Context:
# {context}

# Question:
# {question}
# """
#     )

#     llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))


#     rag_chain = (
#         {
#             "docs": retriever,
#             "question": RunnablePassthrough(),
#             "history": lambda _: build_history(),
#         }
#         | RunnablePassthrough.assign(
#             context=lambda x: "\n\n".join(
#                 doc.page_content for doc in x["docs"]
#             )
#         )
#         | prompt
#         | llm
#     )

#     return rag_chain

# # --------------------------------------------------
# # PDF UPLOAD + INGESTION
# # --------------------------------------------------
# st.markdown("### 📥 Upload a PDF")
# uploaded_file = st.file_uploader("", type=["pdf"])

# if uploaded_file:
#     with open("uploaded.pdf", "wb") as f:
#         f.write(uploaded_file.read())

#     st.info("Processing PDF...")

#     from langchain_community.document_loaders import PyPDFLoader
#     from langchain_text_splitters import RecursiveCharacterTextSplitter

#     loader = PyPDFLoader("uploaded.pdf")
#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks = splitter.split_documents(docs)

#     embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


#     if not chunks:
#         st.error("No readable text found in the PDF.")
#     else:
#         db = FAISS.from_documents(chunks, embeddings)
#         db.save_local(DB_PATH)
#         st.success("Vectorstore updated. You can now chat with the document.")
#         st.cache_resource.clear()

# # --------------------------------------------------
# # LOAD RAG
# # --------------------------------------------------
# rag_chain = load_rag()

# # --------------------------------------------------
# # CHAT UI
# # --------------------------------------------------
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# question = st.chat_input("Ask a question about your document...")

# if question:
#     # Save user message FIRST
#     st.session_state.messages.append(
#         {"role": "user", "content": question}
#     )
#     st.chat_message("user").write(question)

#     if rag_chain is None:
#         response = "Please upload a PDF first."
#     else:
#         response = rag_chain.invoke(question).content

#     st.session_state.messages.append(
#         {"role": "assistant", "content": response}
#     )
#     st.chat_message("assistant").write(response)
import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ==================================================
# STREAMLIT CONFIG (MUST BE FIRST UI CALL)
# ==================================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ==================================================
# ENV GUARD (CRITICAL FOR STREAMLIT CLOUD)
# ==================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set.\n\n"
        "If running on Streamlit Cloud:\n"
        "Manage app → Settings → Secrets → add OPENAI_API_KEY"
    )
    st.stop()

# ==================================================
# SESSION STATE
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# UI HEADER
# ==================================================
st.title("📄 RAG Document Chatbot")
st.caption("Ask questions grounded in your uploaded documents")

# ==================================================
# CONSTANTS
# ==================================================
DB_PATH = "vectorstore"

# ==================================================
# HELPERS
# ==================================================
def build_history() -> str:
    messages = st.session_state.get("messages", [])
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ==================================================
# LOAD RAG PIPELINE
# ==================================================
@st.cache_resource
def load_rag_chain():
    if not os.path.exists(DB_PATH):
        return None

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant answering questions about the user's documents.

Conversation so far:
{history}

Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
    )

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    rag_chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
            "history": lambda _: build_history(),
        }
        | RunnablePassthrough.assign(
            context=lambda x: "\n\n".join(
                doc.page_content for doc in x["docs"]
            )
        )
        | prompt
        | llm
    )

    return rag_chain

# ==================================================
# PDF UPLOAD & INGESTION
# ==================================================
st.markdown("### 📥 Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing PDF...")

    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        st.error("No readable text found in the PDF.")
        st.stop()

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    st.cache_resource.clear()
    st.success("PDF processed. You can now chat with the document.")

# ==================================================
# LOAD RAG
# ==================================================
rag_chain = load_rag_chain()

# ==================================================
# CHAT HISTORY UI
# ==================================================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ==================================================
# CHAT INPUT
# ==================================================
question = st.chat_input("Ask a question about your document...")

if question:
    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    st.chat_message("user").write(question)

    if rag_chain is None:
        response = "Please upload a PDF first."
    else:
        response = rag_chain.invoke(question).content

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    st.chat_message("assistant").write(response)
