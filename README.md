# 📄 Secure Enterprise RAG Pipeline (Production-Grade)

### 📌 Project Overview
This repository contains a **containerized, security-first Retrieval-Augmented Generation (RAG)** system. Developed as a technical proof-of-concept for **AI-First Engineering**, this project demonstrates a "Spec-First" development workflow to solve the problem of LLM hallucination through strict document grounding.

The system was architected to prove that high-performance AI tools can be built with **zero data retention**, making it suitable for enterprise-grade privacy standards.

---

### 🧠 System Architecture & Data Flow
The application follows a modular RAG architecture designed for **privacy-preserving inference**.



1.  **Ingestion:** Multi-PDF batch upload processing with **RecursiveCharacterTextSplitter** (1000-character chunks / 200-character overlap) to preserve deep semantic context.
2.  **Vector Store:** **In-Memory FAISS** (Facebook AI Similarity Search) index for high-speed similarity retrieval without disk persistence.
3.  **Security Layer:** Integrated 30-minute automated session-purge logic and manual memory-wipe protocols.
4.  **Generation:** OpenAI **GPT-4o** orchestrated via **LangChain** with real-time streaming output.
5.  **Attribution:** Collapsible source citations displaying exact page numbers and text snippets for factual verification.

---

### 🚀 Technical Stack
- **Orchestration:** LangChain
- **Vector DB:** FAISS (In-memory CPU-optimized)
- **Models:** OpenAI GPT-4o & `text-embedding-3-small`
- **Environment:** Docker-ready (Ubuntu/Python 3.11 base)
- **Frontend:** Streamlit

---

### 🔐 Security & Privacy Features
- **Ephemeral Memory:** No local PDFs or vector indexes are written to the server's disk; all data is discarded upon session termination.
- **Session Isolation:** Strictly segregated `st.session_state` prevents cross-user data leakage in multi-user environments.
- **Inactivity Purge:** Automated 30-minute timeout monitor that deletes all session-related embeddings.
- **Hybrid Key Policy:** Supports both `st.secrets` and sidebar User-Entry (Bring-Your-Own-Key) for maximum user privacy.

---

### 🏗️ Deployment & Local Setup

**Docker Deployment (Recommended for Production Testing):**
```bash
docker build -t rag-enterprise .
docker run -p 8501:8501 --env-file .env rag-enterprise
