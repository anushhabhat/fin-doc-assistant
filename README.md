# Financial Document Intelligence Assistant

A RAG (Retrieval-Augmented Generation) powered chatbot that lets you upload financial PDFs — annual reports, regulatory filings, loan documents — and ask questions about them in natural language. Built with LangChain, ChromaDB, Gemini embeddings, and Groq LLM.

---

## Demo

> Upload any financial PDF → click Ingest → ask questions → get grounded answers with source citations

Example questions you can ask:
- *"What was the net profit for FY2024?"*
- *"What are the key risk factors mentioned?"*
- *"Summarize the CEO's message to shareholders"*
- *"What is the capital adequacy ratio?"*

---

## How It Works

This project implements a two-stage RAG pipeline:

**Ingestion pipeline** — runs once per document:
1. PDFs are parsed page-by-page using PyMuPDF
2. Text is split into ~500 token chunks with overlap using LangChain's `RecursiveCharacterTextSplitter`
3. Each chunk is converted into a vector embedding using Google's `gemini-embedding-001` model
4. Embeddings and chunks are stored locally in ChromaDB

**Query pipeline** — runs on every question:
1. The user's question is embedded using the same model
2. ChromaDB performs cosine similarity search to find the top 5 most relevant chunks
3. Retrieved chunks are injected into a prompt template alongside the question
4. Groq's `llama-3.1-8b-instant` generates a grounded answer with source citations

---

## Tech Stack

| Layer | Tool |
|---|---|
| PDF parsing | PyMuPDF |
| Chunking | LangChain Text Splitters |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector store | ChromaDB (local) |
| LLM | Groq (`llama-3.1-8b-instant`) |
| UI | Streamlit |

---

## Project Structure

```
fin-doc-assistant/
├── app.py            # Streamlit UI
├── ingest.py         # PDF → chunks → embeddings → ChromaDB
├── retriever.py      # Query → similarity search → top chunks
├── chain.py          # Chunks + question → LLM → answer
├── requirements.txt
├── .env              # API keys (never committed)
├── data/
│   └── pdfs/         # Drop PDFs here
└── chroma_db/        # Auto-created, vector store
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- A free [Gemini API key](https://aistudio.google.com/app/apikey)
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/fin-doc-assistant.git
cd fin-doc-assistant

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-gemini-key-here
GROQ_API_KEY=your-groq-key-here
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

1. Upload one or more financial PDFs via the sidebar
2. Click **Ingest documents** and wait for processing
3. Ask questions in the chat input

---

## Key Concepts Demonstrated

- **RAG architecture** — grounding LLM responses in retrieved document context to eliminate hallucination
- **Vector embeddings** — semantic search over financial text using dense vector representations
- **Chunking strategy** — recursive character splitting with overlap to preserve context across chunk boundaries
- **Prompt engineering** — structured prompts that force citation and prevent out-of-context answers
- **Source attribution** — every answer includes the document name and page number it was derived from

---

## Limitations & Future Improvements

- [ ] Add support for scanned PDFs via OCR (pytesseract)
- [ ] Implement hybrid search (BM25 + vector) for better retrieval on financial tables
- [ ] Add document management — delete/update individual documents from the vector store
- [ ] Persist ChromaDB across sessions on deployment using cloud storage
- [ ] Add evaluation metrics (faithfulness, answer relevancy) using RAGAS
