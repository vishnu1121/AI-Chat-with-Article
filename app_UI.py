import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import rag_core as core

# Optional: drag reorder
HAS_SORTABLES = False
try:
    from streamlit_sortables import sort_items
    HAS_SORTABLES = True
except Exception:
    HAS_SORTABLES = False

load_dotenv()

st.set_page_config(page_title="younameit", layout="wide")

# Notion-dark CSS (UI-only)
st.markdown(
    """
<style>
.stApp { background:#0f1115; color:#e8e8e8; }
section[data-testid="stSidebar"] { background:#121521; border-right:1px solid #23283a; }
input, textarea { background:#151a28 !important; color:#e8e8e8 !important; border:1px solid #2a3147 !important; border-radius:10px !important; }
.stButton button { background:#1a2032; color:#e8e8e8; border:1px solid #2a3147; border-radius:10px; padding:0.55rem 0.9rem; }
.stButton button:hover { border-color:#3a4566; }
.n-card { background:#121726; border:1px solid #232a3f; border-radius:14px; padding:14px; }
.pill { display:inline-block; padding:0.2rem 0.55rem; border-radius:999px; border:1px solid #2a3147; background:#151a28; font-size:0.85rem; }
.muted { color:#a7afc2; }
a { color:#9fb4ff !important; }
</style>
""",
    unsafe_allow_html=True,
)

INDEX_DIR = "faiss_store"

LLM_MODEL = "bytedance-seed/seed-2.0-mini"
EMBED_MODEL = "openai/text-embedding-3-small"

llm = ChatOpenAI(temperature=0.2, model=LLM_MODEL)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# Session state
if "sources" not in st.session_state:
    st.session_state.sources = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "question" not in st.session_state:
    st.session_state.question = ""
if "preview" not in st.session_state:
    st.session_state.preview = ""

# Header
st.markdown(
    f"""
<div class="n-card">
  <div style="font-size:2rem; font-weight:700;">younameit</div>
  <div class="muted">URLs + PDFs</div>
  <div style="margin-top:8px;"><span class="pill">{"useyourllm"}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### Sources")

    mode = st.radio("Add mode", ["Single", "Bulk"], horizontal=True)

    if mode == "Single":
        new_src = st.text_input("Paste URL / local path / file://")
        if st.button("＋ Add", use_container_width=True) and new_src.strip():
            st.session_state.sources.append(
                {
                    "value": new_src.strip(),
                    "label": core.pretty_label(new_src),
                    "kind": core.detect_kind(new_src),
                    "domain": core.domain_of(new_src),
                    "status": "pending",
                    "error": "",
                }
            )
            st.rerun()
    else:
        bulk = st.text_area("Paste multiple (one per line)", height=140)
        if st.button("＋ Add all", use_container_width=True):
            for ln in [x.strip() for x in bulk.splitlines() if x.strip()]:
                st.session_state.sources.append(
                    {
                        "value": ln,
                        "label": core.pretty_label(ln),
                        "kind": core.detect_kind(ln),
                        "domain": core.domain_of(ln),
                        "status": "pending",
                        "error": "",
                    }
                )
            st.rerun()

    up = st.file_uploader("Upload PDF", type=["pdf"])
    if up is not None:
        if st.button("＋ Add uploaded PDF", use_container_width=True):
            st.session_state.sources.append(
                {
                    "value": "",
                    "label": f"Uploaded · {up.name}",
                    "kind": "uploaded_pdf",
                    "domain": "uploaded",
                    "status": "pending",
                    "error": "",
                    "uploaded_name": up.name,
                    "uploaded_bytes": up.getvalue(),
                }
            )
            st.rerun()

    st.markdown("---")
    st.markdown("#### Reorder")
    if st.session_state.sources and HAS_SORTABLES:
        labels = [s["label"] for s in st.session_state.sources]
        new_order = sort_items(labels, direction="vertical", key="sort_sources")
        # reorder safely (handles duplicates)
        used = set()
        reordered = []
        for lbl in new_order:
            for i, s in enumerate(st.session_state.sources):
                if i in used:
                    continue
                if s["label"] == lbl:
                    used.add(i)
                    reordered.append(s)
                    break
        for i, s in enumerate(st.session_state.sources):
            if i not in used:
                reordered.append(s)
        st.session_state.sources = reordered

    st.markdown("---")
    st.markdown("#### Source list (grouped)")
    groups = {}
    for idx, s in enumerate(st.session_state.sources):
        groups.setdefault(s["domain"], []).append((idx, s))

    for dom in sorted(groups.keys()):
        st.markdown(f"**{dom}**")
        for idx, s in groups[dom]:
            icon = "⏳" if s["status"] == "pending" else ("✅" if s["status"] == "ok" else "❌")
            st.caption(f"{icon} {s['label']}  ·  {s['kind']}  ·  {s['status']}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Remove", key=f"rm_{idx}", use_container_width=True):
                    st.session_state.sources.pop(idx)
                    st.rerun()
            with c2:
                if s["status"] == "fail":
                    if st.button("Retry", key=f"rt_{idx}", use_container_width=True):
                        st.session_state.sources[idx]["status"] = "pending"
                        st.session_state.sources[idx]["error"] = ""
                        st.rerun()

            if s["status"] == "fail" and s.get("error"):
                st.caption(f"Error: {s['error'][:160]}")

    st.markdown("---")
    process_clicked = st.button("▶ Process sources", type="primary", use_container_width=True)
    clear_clicked = st.button("🧹 Clear", use_container_width=True)
    if clear_clicked:
        st.session_state.sources = []
        st.session_state.last_answer = ""
        st.session_state.last_sources = []
        st.session_state.question = ""
        st.session_state.preview = ""
        st.rerun()

# Main tabs
tabs = st.tabs(["Ask", "Preview", "Export", "Settings"])

with tabs[3]:
    st.markdown('<div class="n-card">', unsafe_allow_html=True)
    k = st.slider("Top-k chunks", 2, 10, 4)
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=25)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[0]:
    st.markdown('<div class="n-card">', unsafe_allow_html=True)
    chip_cols = st.columns(5)
    chips = [
        ("Summarize", "Summarize the content clearly in bullet points."),
        ("Key claims", "List the key claims and supporting evidence. Cite sources."),
        ("Extract numbers", "Extract important numbers, dates, and metrics with context."),
        ("Compare", "Compare sources: agreements, contradictions, missing info."),
        ("TL;DR", "Give a 5-line TL;DR + 5 takeaways."),
    ]
    for col, (label, text) in zip(chip_cols, chips):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.question = text
                st.rerun()

    st.text_input("Question", key="question", placeholder="Ask anything…")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_answer:
        st.markdown('<div class="n-card" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("### Answer")
        st.write(st.session_state.last_answer)
        if st.session_state.last_sources:
            st.markdown("**Sources**")
            for s in st.session_state.last_sources:
                st.write(s)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="n-card">', unsafe_allow_html=True)
    st.write("**Extracted preview**")
    if st.session_state.preview:
        st.text(st.session_state.preview[:2500])
    else:
        st.caption("No preview yet. Process sources first.")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="n-card">', unsafe_allow_html=True)
    md = core.notion_markdown(st.session_state.last_answer, st.session_state.last_sources)
    st.code(md, language="markdown")
    st.download_button(
        "Download markdown",
        data=md.encode("utf-8"),
        file_name="export.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Processing
if process_clicked:
    if not st.session_state.sources:
        st.error("Add at least one source first.")
        st.stop()

    prog = st.progress(0, text="Loading sources…")
    status_box = st.empty()

    all_docs = []
    for i, s in enumerate(st.session_state.sources, start=1):
        try:
            status_box.markdown(f'<div class="n-card">Loading <b>{s["label"]}</b> ({i}/{len(st.session_state.sources)})…</div>', unsafe_allow_html=True)
            docs = core.load_any_source(s)
            if not docs:
                raise RuntimeError("No text extracted")
            all_docs.extend(docs)
            s["status"] = "ok"
            s["error"] = ""
        except Exception as e:
            s["status"] = "fail"
            s["error"] = str(e)
        prog.progress(int(i / len(st.session_state.sources) * 100), text="Loading sources…")

    if not all_docs:
        st.error("No usable text extracted from any source.")
        st.stop()

    if not core.extraction_quality_ok(all_docs):
        st.session_state.preview = (all_docs[0].page_content or "")[:2500]
        st.error("Extraction looks blocked/low quality (paywall/JS). Not building index.")
        st.stop()

    st.session_state.preview = (all_docs[0].page_content or "")[:2500]
    status_box.markdown('<div class="n-card">Building index…</div>', unsafe_allow_html=True)
    stats = core.build_index(all_docs, embeddings, INDEX_DIR, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    prog.progress(100, text="Done!")
    status_box.success(f"Index saved. Docs: {stats['doc_count']} · Chunks: {stats['chunk_count']}")
    time.sleep(0.15)

# Ask
if st.session_state.question.strip():
    if not os.path.exists(INDEX_DIR):
        st.warning("No index found. Process sources first.")
    else:
        vs = core.load_index(INDEX_DIR, embeddings)
        answer, srcs = core.ask(vs, llm, st.session_state.question, k=k)
        st.session_state.last_answer = answer
        st.session_state.last_sources = srcs
        st.rerun()