# AI Chat with Article — URL / PDF RAG Summarizer

A **RAG (Retrieval-Augmented Generation)** app that lets you paste an article URL or direct PDF link, then get a clean summary or Q&A grounded in the source content.

Built with **Python · Streamlit · LangChain · FAISS · OpenAI API**

> **Note:** Optimized for text-heavy content. Pages that are mostly tables, images, or scanned PDFs may extract incompletely.

---

## Features

- Paste any article URL and get an instant **summary**
- Ask follow-up questions such as:
  - *"What are the key takeaways?"*
  - *"What does the author claim about X?"*
  - *"List the main arguments and evidence"*
- Answers are grounded in **retrieved source chunks** — not hallucinated

---

## Architecture

![Architecture](./image.png)

**Pipeline:**

```
User (Browser)
  → Streamlit UI (app_UI.py)
  → RAG Orchestrator (rag_core.py)
  → Fetch URL / PDF + Extract Text
  → Clean + Chunk
  → Embeddings API Call
  → FAISS Vector Store (persistent)
  → Retriever (Top-K chunks)
  → Map-Reduce Summarizer (LLM calls)
  → Final Answer shown in UI
```

**Map-Reduce summarization strategy:**
- **MAP** — the LLM summarizes each retrieved chunk individually
- **REDUCE** — those summaries are merged into one final consolidated response

---

## Repo Structure

```
.
├── app_UI.py
├── rag_core.py
├── requirements.txt
├── .env
├── image.png
├── faiss_store/              # persisted FAISS artifacts
├── vector_index.pkl
├── faiss_store_openai.pkl
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vishnu1121/AI-Chat-with-Article.git
cd AI-Chat-with-Article
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root (same folder as `app_UI.py`):

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the app

```bash
streamlit run app_UI.py
```

---

## Usage

1. Paste an article URL or direct PDF link into the input field (or upload a local file)
2. Click the button to load, process, and index the content
3. Ask a question or request a summary
4. The UI returns a final consolidated answer

---

## How It Works

| Step | What happens |
|---|---|
| **Text extraction** | The app fetches and parses content from the URL or PDF |
| **Chunking** | Extracted text is cleaned and split into smaller chunks for efficient embedding and retrieval |
| **Embeddings + FAISS** | Chunks are embedded via the OpenAI Embeddings API and stored locally (`vector_index.pkl`, `faiss_store_openai.pkl`) |
| **Retrieval** | Your query is embedded and FAISS retrieves the Top-K most relevant chunks |
| **Map-Reduce** | The LLM summarizes each chunk (MAP), then merges them into one answer (REDUCE) |

---

## Known Limitations

- Works best on **text-heavy** content
- Web pages with heavy tables or images may extract poorly
- **Scanned PDFs** (image-based) won't work without adding OCR
- Some websites block scraping or require a login
- First run may be slower while the FAISS index is being built

---

## Troubleshooting

**"No text extracted" / empty output**
- Try a different URL — some sites block automated extraction
- For cleaner results, use a *reader view* or *print view* URL if the site offers one
- For PDFs, ensure the file is text-based and not a scanned image

**Dependency conflicts**
```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**API key not found**
- Confirm `.env` exists in the same folder as `app_UI.py`
- Confirm it contains `OPENAI_API_KEY=...`
- Restart Streamlit after creating or editing `.env`

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python | Core language |
| Streamlit | Web UI |
| LangChain | RAG orchestration |
| FAISS | Vector store & retrieval |
| OpenAI API | Embeddings + LLM |