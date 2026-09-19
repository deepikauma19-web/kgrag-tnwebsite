"""Streamlit RAG chatbot for the Tamil Nadu Schemes FAISS index.

Run the ingestion script first, then start this app with:
    streamlit run streamlit_app.py

Required .env setting:
    OPENAI_API_KEY=your_key
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# This must match the shared location created by the backend ingestion service.
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/processed/tn_schemes_faiss"))
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
INGEST_TIMEOUT_SECONDS = int(os.getenv("INGEST_TIMEOUT_SECONDS", "1800"))
STOP_WORDS = {
    "about", "after", "also", "and", "are", "can", "did", "does", "for",
    "from", "have", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "the", "their", "these", "this", "to", "under", "what", "where", "which",
    "who", "with", "would", "you",
}
SYSTEM_PROMPT = """You are a helpful Tamil Nadu government schemes assistant.
Answer using only the retrieved scheme records below. If the records do not
contain the answer, say that you could not find it in the indexed scheme-list
data. Do not invent eligibility, benefits, application steps, or amounts.
Always finish with a short 'Sources:' line containing the source URLs provided.
"""


@st.cache_resource(show_spinner="Loading the scheme knowledge base...")
def load_vectorstore() -> FAISS:
    """Load the local index created by the ingestion program."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{INDEX_PATH}'. Run tn_schemes_ingest.py first."
        )

    # Only load an index created by your own ingestion script. FAISS uses a
    # pickle metadata file, so never enable this for an untrusted downloaded index.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def keywords(text: str) -> set[str]:
    """Meaningful words used for a lightweight exact-match retrieval signal."""
    return {
        word
        for word in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if word not in STOP_WORDS
    }


def lexical_score(document: Document, question_words: set[str]) -> int:
    """Reward a page that contains the user's important question words."""
    searchable = (
        document.page_content
        + " "
        + document.metadata.get("scheme_name", "")
        + " "
        + document.metadata.get("department", "")
    ).lower()
    return sum(word in searchable for word in question_words)


def retrieve(question: str) -> list[Document]:
    """Hybrid retrieval: FAISS semantic matches plus exact terms over all chunks.

    The exact-term pass matters for questions such as "funding pattern for
    farmers during training": it promotes the detail page that contains those
    field labels even if FAISS initially returns broadly related farmer pages.
    """
    vectorstore = load_vectorstore()
    semantic_results = vectorstore.similarity_search(question, k=24)
    candidates: dict[tuple[str, str], Document] = {}

    def add(document: Document) -> None:
        key = (document.metadata.get("source", ""), document.page_content)
        candidates[key] = document

    for document in semantic_results:
        add(document)

    question_words = keywords(question)
    for document_id in vectorstore.index_to_docstore_id.values():
        document = vectorstore.docstore.search(document_id)
        if isinstance(document, Document) and lexical_score(document, question_words):
            add(document)

    semantic_rank = {
        (document.metadata.get("source", ""), document.page_content): position
        for position, document in enumerate(semantic_results)
    }
    ranked = sorted(
        candidates.values(),
        key=lambda document: (
            lexical_score(document, question_words),
            24 - semantic_rank.get(
                (document.metadata.get("source", ""), document.page_content), 24
            ),
        ),
        reverse=True,
    )
    return ranked[:4]


def format_context(question: str) -> tuple[str, list[Document]]:
    """Build answer context from hybrid-retrieved official scheme pages."""
    results = retrieve(question)
    if not results:
        return "No matching scheme records were found.", []

    blocks = []
    for number, document in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown source")
        blocks.append(f"[Record {number}]\n{document.page_content}\nSource: {source}")
    return "\n\n".join(blocks), results


def answer_question(question: str) -> tuple[str, list[Document]]:
    """Ask the model to answer strictly from retrieved FAISS context."""
    context, documents = format_context(question)
    model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Retrieved scheme records:\n\n{context}\n\nQuestion: {question}"),
        ]
    )
    return response.content, documents


def refresh_knowledge_base() -> None:
    """Ask the backend to refresh Neo4j and the shared FAISS index."""
    response = requests.post(f"{BACKEND_URL}/ingest", timeout=INGEST_TIMEOUT_SECONDS)
    response.raise_for_status()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Tamil Nadu Schemes RAG", page_icon="🏛️")
    st.title("🏛️ Tamil Nadu Schemes RAG Chatbot")
    st.caption("Answers are based on the individual official scheme pages in your local FAISS index.")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Add it to your .env file, then restart Streamlit.")
        st.stop()

    with st.sidebar:
        st.header("Knowledge base")
        st.caption("Refresh after changing SCHEME_START_URL or when source data changes.")
        if st.button("Refresh Neo4j and FAISS", type="primary"):
            with st.spinner("Fetching schemes, updating Neo4j, and rebuilding FAISS..."):
                try:
                    refresh_knowledge_base()
                    load_vectorstore.clear()
                    st.success("The scheme knowledge base was refreshed.")
                except requests.RequestException as error:
                    st.error(f"Refresh failed: {error}")

    try:
        load_vectorstore()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Use the sidebar Refresh button to create the initial index.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me about the Tamil Nadu schemes currently in the index."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("For example: What schemes are available?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the scheme knowledge base..."):
                try:
                    answer, retrieved_documents = answer_question(question)
                except Exception as error:
                    answer = f"I could not answer that question: {error}"
                    retrieved_documents = []
            st.markdown(answer)
            if retrieved_documents:
                with st.expander("Retrieved official pages"):
                    for document in retrieved_documents:
                        st.markdown(
                            f"- **{document.metadata.get('scheme_name', 'Scheme')}**: "
                            f"{document.metadata.get('source', 'Unknown source')}"
                        )
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
