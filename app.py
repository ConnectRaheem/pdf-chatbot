"""
app.py
------
Streamlit frontend for the RAG PDF Chatbot.

Client pitch: "Upload any PDF -- a contract, manual, report, policy
document -- and get instant, grounded, cited answers instead of
manually searching through pages."

Run locally:
    streamlit run app.py

Deploy free:
    https://share.streamlit.io  (Streamlit Community Cloud)
See README.md for full deployment steps.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from rag_engine import PDFChatEngine, list_available_groq_models, FALLBACK_GROQ_MODEL

# Load variables from a local .env file (GROQ_API_KEY=...) into the
# environment so the sidebar field below is pre-filled automatically
# when running locally.
load_dotenv()

# On Streamlit Community Cloud, secrets are exposed via st.secrets instead
# of a .env file. Bridge it into os.environ so the same code path works
# both locally and once deployed. Wrapped in try/except because st.secrets
# raises if no secrets.toml exists at all (e.g. fresh local clone).
try:
    if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(
    page_title="DocChat AI — Talk to Your Documents",
    page_icon="📄",
    layout="wide",
)


def _pick_best_default_model(options: list[str]) -> str:
    """Choose the strongest general-purpose chat model available, in
    order of preference, instead of just grabbing whatever sorts first
    alphabetically (which could land on a small/specialized model)."""
    ranked_preferences = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
    ]
    for preferred in ranked_preferences:
        if preferred in options:
            return preferred
    # None of the known-good models matched -- fall back to the largest
    # parameter-count model we can detect from the name, else just the
    # first option so the app never crashes on an empty selection.
    for opt in options:
        if "70b" in opt or "72b" in opt:
            return opt
    return options[0]

# --------------------------------------------------------------------- #
# Sidebar — branding, uploader, and controls
# --------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📄 DocChat AI")
    st.caption("Retrieval-Augmented Generation demo — built by CogniticSolutions")

    st.markdown("---")

    # If a Groq key is already configured on the backend (via Streamlit
    # Secrets or a local .env), don't show the raw API key field to the
    # end user at all -- that's an implementation detail, not something
    # a client should have to see or understand. Only fall back to asking
    # for it manually if nothing is configured yet (e.g. first local run).
    env_key_present = bool(os.environ.get("GROQ_API_KEY"))

    with st.expander("⚙️ Advanced settings", expanded=not env_key_present):
        groq_key_input = st.text_input(
            "Groq API key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Get a free key at console.groq.com. Not stored anywhere.",
        )

        # Fetch the live list of models this Groq key can actually use.
        # Groq occasionally retires model IDs, so we ask the API directly
        # instead of hardcoding a name that can silently go stale.
        model_options = [FALLBACK_GROQ_MODEL]
        if groq_key_input:
            try:
                live_models = list_available_groq_models(groq_key_input)
                if live_models:
                    model_options = live_models
            except Exception:
                pass  # keep the fallback silently; the key may just be incomplete

        preferred_default = _pick_best_default_model(model_options)
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(preferred_default),
            help="Live list fetched from your Groq account. Larger/'versatile' "
                 "models give better answers; smaller 'instant' models are faster.",
        )

    if not groq_key_input:
        st.warning("Add a Groq API key above to enable the assistant.")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("Process documents", type="primary", use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ Clear session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption(
        "**Stack:** LangChain · Groq · FAISS · "
        "HuggingFace embeddings · Streamlit"
    )

# --------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------- #
if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, sources

# --------------------------------------------------------------------- #
# Process uploaded PDFs
# --------------------------------------------------------------------- #
if process_clicked:
    if not groq_key_input:
        st.sidebar.error("Add a Groq API key first.")
    elif not uploaded_files:
        st.sidebar.error("Upload at least one PDF first.")
    else:
        with st.spinner("Reading and indexing document(s)..."):
            engine = st.session_state.engine or PDFChatEngine(
                groq_api_key=groq_key_input, groq_model=selected_model
            )
            engine.groq_model = selected_model  # keep in sync if user changes it
            tmp_paths = []
            for f in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.read())
                tmp.close()
                tmp_paths.append(tmp.name)

            chunk_count = engine.add_pdfs(tmp_paths)
            st.session_state.engine = engine

        st.sidebar.success(f"Indexed {chunk_count} chunks from {len(uploaded_files)} file(s).")

# --------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------- #
st.title("Chat with your documents")

if not st.session_state.engine or not st.session_state.engine.is_ready:
    st.info(
        "👈 Upload a PDF in the sidebar, add a Groq API key, and click "
        "**Process documents** to get started. Try a policy document, "
        "a contract, a research paper, or a product manual."
    )
else:
    files_label = ", ".join(st.session_state.engine.indexed_files)
    st.caption(f"📚 Ready — indexed: {files_label}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View sources"):
                for i, doc in enumerate(msg["sources"], start=1):
                    page = doc.metadata.get("page", 0) + 1
                    src = doc.metadata.get("source_file", "document")
                    st.markdown(f"**{i}. {src} — page {page}**")
                    st.caption(doc.page_content[:400] + "...")

question = st.chat_input("Ask a question about your document(s)...")

if question:
    if not st.session_state.engine or not st.session_state.engine.is_ready:
        st.warning("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history = [
            (m["content"], st.session_state.messages[i + 1]["content"])
            for i, m in enumerate(st.session_state.messages[:-1])
            if m["role"] == "user" and i + 1 < len(st.session_state.messages) - 1
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.engine.ask(question, chat_history=history)
                    st.markdown(result.answer)
                    if result.sources:
                        with st.expander("View sources"):
                            for i, doc in enumerate(result.sources, start=1):
                                page = doc.metadata.get("page", 0) + 1
                                src = doc.metadata.get("source_file", "document")
                                st.markdown(f"**{i}. {src} — page {page}**")
                                st.caption(doc.page_content[:400] + "...")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    })
                except Exception as e:
                    st.error(f"Something went wrong: {e}")