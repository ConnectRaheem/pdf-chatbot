# DocChat AI — Chat With Your Documents

A Retrieval-Augmented Generation (RAG) app that lets anyone upload a PDF —
a contract, policy, manual, or report — and get instant, **cited** answers
instead of manually searching through pages.

**Live demo:** [raheem-pdf-chatbot.streamlit.app](https://raheem-pdf-chatbot.streamlit.app)

![screenshot placeholder](docs/screenshot.png)

---

## Why this matters for clients

Most teams have hundreds of pages of PDFs — contracts, SOPs, compliance
docs, product manuals — that staff waste hours searching manually. This
app turns any document into a searchable, conversational knowledge base
in under a minute, with every answer traceable back to the exact page it
came from. That traceability is the difference between a toy chatbot demo
and something a business can actually trust.

**Example use cases to pitch:**
- Legal/contract review — "What's the termination clause?"
- HR — employees ask policy questions instead of emailing HR
- Customer support — agents query product manuals instantly
- Compliance — auditors query regulatory documents with page citations

## How it works

```
PDF upload → text extraction & chunking → embeddings → FAISS vector index
                                                              │
User question ──────────────────────────────────────────────┤
                                                              ▼
                                   retrieve top-k relevant chunks
                                                              │
                                                              ▼
                                  Groq (Llama 3.3 70B) generates a
                                  grounded answer with page citations
```

**Stack:**
| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free, no API key) |
| Vector store | FAISS |
| LLM | Groq — Llama 3.3 70B (fast, cheap, generous free tier) |

## Run it locally

```bash
git clone https://github.com/ConnectRaheem/pdf-chatbot.git
cd pdf-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your free Groq API key from console.groq.com

streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), upload a
PDF, click **Process documents**, and start asking questions.

## Deploy it for free (so you have a live client-facing link)

### Option A — Streamlit Community Cloud (recommended, easiest)
1. Push this repo to GitHub (already there if you're reading this on `ConnectRaheem/pdf-chatbot`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Deploy. You'll get a public URL like `https://your-app.streamlit.app` —
   put this in your portfolio.

### Option B — Hugging Face Spaces
1. Create a new Space → SDK: Streamlit.
2. Upload these files (or connect the GitHub repo).
3. Add `GROQ_API_KEY` under Space **Settings → Repository secrets**.

## Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) → sign up.
2. Create an API key (free tier is generous and plenty for demos).
3. Add it as `GROQ_API_KEY` — either in `.env` locally, or as a secret on
   whichever platform you deploy to.

## Project structure

```
pdf-chatbot/
├── app.py           # Streamlit UI
├── rag_engine.py     # Core RAG pipeline (ingest, retrieve, generate)
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap / ideas for extending this into a bigger client offering
- Swap FAISS for a persistent vector DB (e.g. Chroma, Pinecone) to support
  multiple users and saved knowledge bases
- Add authentication so each client gets a private, isolated instance
- Support Word docs, web pages, and spreadsheets, not just PDFs
- Add usage analytics (questions asked, most-cited pages) as a client
  dashboard — this becomes a strong upsell

---

Built by **CogniticSolutions**.
