import os
import json
import time
import tempfile
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


BAD_MARKERS = [
    "enable javascript",
    "disable ad blocker",
    "subscribe",
    "sign in",
    "verify you are human",
    "access denied",
    "we noticed you’re using",
]


def is_http_url(u: str) -> bool:
    u = (u or "").strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def is_file_url(u: str) -> bool:
    return (u or "").strip().lower().startswith("file://")


def file_url_to_path(file_url: str) -> str:
    parsed = urlparse(file_url)
    local_path = unquote(parsed.path)
    if os.name == "nt" and local_path.startswith("/"):
        local_path = local_path[1:]
    return local_path


def looks_like_pdf_http_url(url: str) -> bool:
    if urlparse(url).path.lower().endswith(".pdf"):
        return True
    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        ctype = (resp.headers.get("Content-Type") or "").lower()
        return "application/pdf" in ctype
    except Exception:
        return False


def domain_of(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return "uploaded"
    if is_file_url(v) or os.path.exists(v):
        return "local"
    if is_http_url(v):
        host = urlparse(v).netloc.replace("www.", "")
        return host if host else "web"
    return "other"


def detect_kind(inp: str) -> str:
    inp = (inp or "").strip()
    if not inp:
        return "uploaded_pdf"
    if is_file_url(inp) or os.path.exists(inp):
        return "local_pdf"
    if is_http_url(inp) and looks_like_pdf_http_url(inp):
        return "pdf_url"
    if is_http_url(inp):
        return "web_url"
    return "unknown"


def pretty_label(inp: str) -> str:
    inp = (inp or "").strip()
    if not inp:
        return "Uploaded PDF"
    if is_http_url(inp):
        host = urlparse(inp).netloc.replace("www.", "")
        short = inp if len(inp) <= 60 else inp[:60] + "…"
        return f"{host} · {short}"
    if is_file_url(inp) or os.path.exists(inp):
        p = file_url_to_path(inp) if is_file_url(inp) else inp
        return f"Local PDF · {Path(p).name}"
    return inp[:60] + ("…" if len(inp) > 60 else "")


def load_pdf_from_http(url: str):
    r = requests.get(url, stream=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
        pdf_path = f.name
    try:
        docs = PyMuPDFLoader(pdf_path).load()
        for d in docs:
            d.metadata["source"] = url
        return docs
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass


def load_pdf_from_local(local_path: str, source_label: str):
    docs = PyMuPDFLoader(local_path).load()
    for d in docs:
        d.metadata["source"] = source_label
    return docs


def load_web_page(url: str):
    loader = WebBaseLoader(
        web_paths=[url],
        requests_kwargs={
            "timeout": 20,
            "headers": {
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        },
        raise_for_status=False,
    )
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = url
    return docs


def load_any_source(source: dict):
    """
    source dict supports:
      - {"value": "..."} for URL/path/file://
      - {"kind": "uploaded_pdf", "uploaded_bytes": b"...", "uploaded_name": "..."}
    """
    kind = source.get("kind", detect_kind(source.get("value", "")))

    if kind == "uploaded_pdf":
        b = source.get("uploaded_bytes")
        name = source.get("uploaded_name", "uploaded.pdf")
        if not b:
            raise ValueError("Uploaded PDF bytes missing.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b)
            temp_path = f.name
        try:
            docs = PyMuPDFLoader(temp_path).load()
            for d in docs:
                d.metadata["source"] = f"uploaded:{name}"
            return docs
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    val = (source.get("value") or "").strip()
    if not val:
        raise ValueError("Empty source value.")

    if is_file_url(val):
        local_path = file_url_to_path(val)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        return load_pdf_from_local(local_path, val)

    if os.path.exists(val):
        return load_pdf_from_local(val, val)

    if is_http_url(val):
        if looks_like_pdf_http_url(val):
            return load_pdf_from_http(val)
        return load_web_page(val)

    raise ValueError(f"Unsupported source format: {val}")


def extraction_quality_ok(docs) -> bool:
    if not docs:
        return False
    combined = "\n".join((d.page_content or "").lower() for d in docs)
    if len(combined.strip()) < 900:
        return False
    return not any(m in combined for m in BAD_MARKERS)


def build_index(
    all_docs,
    embeddings,
    index_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)

    preview = (all_docs[0].page_content or "")[:2500]
    stats = {"doc_count": len(all_docs), "chunk_count": len(split_docs), "preview": preview}
    return stats


def load_index(index_dir: str, embeddings):
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def ask(vectorstore, llm, question: str, k: int = 4):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
    )
    out = qa.invoke({"query": question})
    answer = out.get("result", "")

    uniq, seen = [], set()
    for d in out.get("source_documents", []):
        src = d.metadata.get("source") or d.metadata.get("url") or ""
        if src and src not in seen:
            seen.add(src)
            uniq.append(src)
    return answer, uniq


def notion_markdown(answer: str, sources: list[str]) -> str:
    md = ["# Notes", "", "## Answer", answer.strip() if answer else "", "", "## Sources"]
    if sources:
        md.extend([f"- {s}" for s in sources])
    else:
        md.append("- (none)")
    md.append("")
    return "\n".join(md)