
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

DB_PATH = "vectorstore"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    if not os.path.exists(DB_PATH):
        print("❌ vectorstore not found. Run ingest.py first.")
        return

    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 12})

    llm = ChatOpenAI(temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    print("✅ RAG chatbot ready\n")

    while True:
        question = input("Ask a question (or type 'exit'): ").strip()
        if question.lower() == "exit":
            break

        response = rag_chain.invoke(question)
        print("\nANSWER:")
        print(response.content)
        print("\n" + "-" * 40)

if __name__ == "__main__":
    main()
