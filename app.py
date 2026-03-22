
# import os
# import io
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.documents import Document
# from PyPDF2 import PdfReader

# # =====================================================
# # SYSTEM & THEME CONFIG
# # =====================================================
# load_dotenv()
# st.set_page_config(page_title="NovaDocs AI", page_icon="🌌", layout="wide")

# # Lovable-inspired "Midnight Tech" CSS
# st.markdown("""
#     <style>
#     /* Main Background */
#     .stApp { background-color: #0B0E14; color: #94A3B8; }
    
#     /* Hide default Streamlit elements for a cleaner look */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     /* Sidebar / Left Column Styling */
#     [data-testid="stVerticalBlock"] > div:first-child {
#         background-color: #0B0E14;
#     }

#     /* Document Card Styling */
#     .doc-card {
#         background-color: #161B22;
#         border: 1px solid #1E293B;
#         border-radius: 8px;
#         padding: 12px;
#         margin-bottom: 10px;
#         display: flex;
#         align-items: center;
#         gap: 10px;
#     }
    
#     /* Custom Headers */
#     .main-header {
#         font-family: 'JetBrains Mono', monospace;
#         color: #00D1FF;
#         font-size: 1.2rem;
#         letter-spacing: -0.5px;
#         margin-bottom: 20px;
#     }

#     /* Chat Input Styling */
#     .stChatInputContainer {
#         background-color: #0B0E14 !important;
#         border-top: 1px solid #1E293B !important;
#     }

#     /* Assistant Message Styling */
#     [data-testid="stChatMessage"] {
#         background-color: #111827;
#         border: 1px solid #1E293B;
#         border-radius: 12px;
#     }
    
#     /* Metrics/Status text */
#     .tech-stat {
#         font-family: 'JetBrains Mono', monospace;
#         font-size: 0.8rem;
#         color: #00D1FF;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # =====================================================
# # LOGIC CORES
# # =====================================================
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# if "messages" not in st.session_state: st.session_state.messages = []
# if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
# if "filenames" not in st.session_state: st.session_state.filenames = []

# def build_vectorstore(uploaded_files):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
#     all_chunks = []
#     names = []
#     for file in uploaded_files:
#         names.append(file.name)
#         reader = PdfReader(io.BytesIO(file.read()))
#         docs = [Document(page_content=p.extract_text(), metadata={"source": file.name, "page": i+1}) 
#                 for i, p in enumerate(reader.pages) if p.extract_text()]
#         all_chunks.extend(splitter.split_documents(docs))
#     st.session_state.filenames = names
#     return FAISS.from_documents(all_chunks, embeddings)

# def build_chain(vs):
#     retriever = vs.as_retriever(search_kwargs={"k": 5})
#     prompt = ChatPromptTemplate.from_template("""
#     Context: {context}
#     History: {history}
#     User: {question}
#     AI: Answer in a technical, structured format. Mention page numbers.
#     """)
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
#     return (
#         {"docs": lambda x: retriever.invoke(x["question"]), "question": lambda x: x["question"], "history": lambda x: x["history"]}
#         | RunnablePassthrough.assign(context=lambda x: "\n\n".join([d.page_content for d in x["docs"]]))
#         | prompt | llm
#     )

# # =====================================================
# # LAYOUT: LOVABLE VIBE
# # =====================================================
# # Using two columns to simulate the Lovable layout
# left_col, right_col = st.columns([1, 3], gap="large")

# with left_col:
#     st.markdown('<p class="main-header">📄 PDFCHAT</p>', unsafe_allow_html=True)
    
#     uploaded_files = st.file_uploader("Upload new PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
#     if uploaded_files and st.session_state.vectorstore is None:
#         with st.spinner("Indexing..."):
#             st.session_state.vectorstore = build_vectorstore(uploaded_files)
#             st.rerun()

#     st.markdown("---")
#     st.markdown('<p class="tech-stat">DOCUMENTS</p>', unsafe_allow_html=True)
    
#     for name in st.session_state.filenames:
#         st.markdown(f"""
#             <div class="doc-card">
#                 <span style="color:#00D1FF">📄</span>
#                 <span style="font-size:0.9rem; color:#E2E8F0">{name}</span>
#             </div>
#         """, unsafe_allow_html=True)
    
#     if st.button("Purge Session", use_container_width=True):
#         st.session_state.clear()
#         st.rerun()

# with right_col:
#     # Top status bar
#     if st.session_state.filenames:
#         st.markdown(f'<p class="tech-stat">ACTIVE SESSION: {st.session_state.filenames[0]}</p>', unsafe_allow_html=True)
#     else:
#         st.markdown('<p class="tech-stat">IDLE ENGINE // WAITING FOR INGEST</p>', unsafe_allow_html=True)

#     # Chat Display
#     chat_container = st.container(height=500, border=False)
#     for msg in st.session_state.messages:
#         chat_container.chat_message(msg["role"]).write(msg["content"])

#     # Chat Input
#     if question := st.chat_input("Ask about your PDF..."):
#         st.session_state.messages.append({"role": "user", "content": question})
#         chat_container.chat_message("user").write(question)

#         if st.session_state.vectorstore:
#             chain = build_chain(st.session_state.vectorstore)
#             history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
            
#             with chat_container.chat_message("assistant"):
#                 response = chain.invoke({"question": question, "history": history})
#                 st.markdown(response.content)
#                 st.session_state.messages.append({"role": "assistant", "content": response.content})
#         else:
#             st.warning("Please upload a document on the left to start.")
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

# Session state
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
# 3. UI
# =============================
st.title("📄 Simple PDF Chat")

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing PDFs..."):
        st.session_state.vectorstore = build_vectorstore(uploaded_files)
        st.success("PDFs indexed successfully.")

# =============================
# 4. Chat
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