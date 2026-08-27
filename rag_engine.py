"""
rag_engine.py
--------------
Core RAG (Retrieval-Augmented Generation) logic for the PDF Chatbot.

Pipeline:
 1. Load PDF(s) and split into overlapping text chunks
 2. Embed chunks with a local, free HuggingFace sentence-transformer
    (no API key required for this step -> keeps running costs near zero)
 3. Store embeddings in a FAISS vector index for fast similarity search
 4. On each user question: retrieve the most relevant chunks, then send
    them + the question to Groq's LLM API to generate a grounded answer
 5. Return the answer along with the source chunks used, so the UI can
    show citations (page number + snippet) -- this builds client trust.

This module has no Streamlit-specific code so it can be reused in a CLI,
an API, or a different frontend later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq periodically retires older model IDs, so instead of hardcoding one
# string that can break overnight, we fetch the live list of available
# chat models from Groq at runtime (see list_available_groq_models below)
# and let the caller pick one. This is a sensible fallback if that lookup
# ever fails.
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"


def list_available_groq_models(api_key: str) -> list[str]:
    """Return the chat-capable model IDs currently available on this Groq
    account. Filters out audio/whisper and guard/safety models, which use
    a different calling convention and aren't relevant here."""
    from groq import Groq

    client = Groq(api_key=api_key)
    models = client.models.list().data
    excluded_markers = ("whisper", "guard", "prompt-guard", "tts")
    chat_models = [
        m.id for m in models
        if not any(marker in m.id.lower() for marker in excluded_markers)
    ]
    return sorted(chat_models)

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions \
using ONLY the provided document excerpts as your source of truth.

Rules:
- If the answer is not contained in the excerpts, say clearly that the \
document does not contain that information. Never invent facts.
- Keep answers concise and well-organized. Use bullet points for lists.
- When you use information from an excerpt, you may reference its page \
number in parentheses, e.g. (p. 4).
"""


@dataclass
class ChatResult:
    answer: str
    sources: List[Document] = field(default_factory=list)


class PDFChatEngine:
    """Wraps the full ingest -> retrieve -> generate pipeline for one session."""

    def __init__(self, groq_api_key: str | None = None, chunk_size: int = 1000,
                 chunk_overlap: int = 150, top_k: int = 4,
                 groq_model: str | None = None):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.groq_model = groq_model or FALLBACK_GROQ_MODEL

        self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._vectorstore: FAISS | None = None
        self._llm: ChatGroq | None = None
        self.indexed_files: List[str] = []

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def add_pdfs(self, file_paths: List[str]) -> int:
        """Load, chunk, and embed one or more PDF files. Returns chunk count."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: List[Document] = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            # Tag each chunk with a clean source filename for citations
            filename = os.path.basename(path)
            for c in chunks:
                c.metadata["source_file"] = filename
            all_chunks.extend(chunks)
            self.indexed_files.append(filename)

        if not all_chunks:
            return 0

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(all_chunks, self._embeddings)
        else:
            self._vectorstore.add_documents(all_chunks)

        return len(all_chunks)

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None

    # ------------------------------------------------------------------ #
    # Retrieval + generation
    # ------------------------------------------------------------------ #
    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not self.groq_api_key:
                raise ValueError(
                    "No Groq API key found. Set GROQ_API_KEY as an environment "
                    "variable or Streamlit secret."
                )
            self._llm = ChatGroq(
                api_key=self.groq_api_key,
                model=self.groq_model,
                temperature=0.2,
            )
        return self._llm

    def ask(self, question: str, chat_history: List[tuple[str, str]] | None = None) -> ChatResult:
        """Answer a question grounded in the indexed PDFs."""
        if not self.is_ready:
            return ChatResult(answer="Please upload and process a PDF first.")

        retriever = self._vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        docs = retriever.invoke(question)

        context = "\n\n---\n\n".join(
            f"[{d.metadata.get('source_file', 'document')}, "
            f"p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
            for d in docs
        )

        history_text = ""
        if chat_history:
            history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in chat_history[-4:])

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",
             "Conversation so far:\n{history}\n\n"
             "Document excerpts:\n{context}\n\n"
             "Question: {question}"),
        ])

        chain = prompt | self._get_llm()
        response = chain.invoke({
            "history": history_text or "(none yet)",
            "context": context or "(no relevant excerpts found)",
            "question": question,
        })

        return ChatResult(answer=response.content, sources=docs)

    def reset(self):
        self._vectorstore = None
        self.indexed_files = []
