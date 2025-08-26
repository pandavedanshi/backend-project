import os
import io
import json
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ----------------- Patch NumPy 2.x -----------------
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "uint"):
    np.uint = np.uint64

# Vector DB + Embeddings
import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ----------------- Config & Setup -----------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Set GROQ_API_KEY in .env")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_DIR = os.getenv("INDEX_DIR", "vectorstore")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ----------------- FastAPI App -----------------

app = FastAPI(title="Groq RAG Server with Web Ingest", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Groq Client -----------------

client = Groq(api_key=GROQ_API_KEY)

# ----------------- Chroma Vector Store -----------------

chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
sentence_embedder = SentenceTransformer(EMBED_MODEL_NAME)

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, convert_to_numpy=True).tolist()

embed_fn = SentenceTransformerEmbeddingFunction(sentence_embedder)

collection = chroma_client.get_or_create_collection(
    name="rag_store",
    embedding_function=embed_fn
)

# ----------------- File Reading Helpers -----------------

def read_txt_or_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def load_corpus_from_dir(root: str) -> List[Tuple[str, str]]:
    corpus = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            lower = fn.lower()
            try:
                if lower.endswith(".txt") or lower.endswith(".md"):
                    corpus.append((p, read_txt_or_md(p)))
                elif lower.endswith(".pdf"):
                    corpus.append((p, read_pdf(p)))
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")
    return corpus

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    if n == 0:
        return []
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - chunk_overlap, 1)
    return chunks

# ----------------- RAG Engine -----------------

class RAGEngine:
    def __init__(self, collection):
        self.collection = collection

    def build_or_load(self, data_dir: str) -> str:
        corpus = load_corpus_from_dir(data_dir)
        if not corpus:
            return "empty"

        for src, content in corpus:
            chunks = chunk_text(content)
            if not chunks:
                continue
            ids = [f"{src}-{i}" for i in range(len(chunks))]
            self.collection.add(
                documents=chunks,
                metadatas=[{"source": src, "chunk_id": i} for i in range(len(chunks))],
                ids=ids
            )
        return "built"

    def add_documents(self, docs: List[Tuple[str, str]]) -> int:
        total_chunks = 0
        for src, content in docs:
            chunks = chunk_text(content)
            if not chunks:
                continue
            ids = [f"{src}-{int(time.time())}-{i}" for i in range(len(chunks))]
            self.collection.add(
                documents=chunks,
                metadatas=[{"source": src, "chunk_id": i} for i in range(len(chunks))],
                ids=ids
            )
            total_chunks += len(chunks)
        return total_chunks

    def retrieve(self, query: str, k: int = TOP_K) -> List[Tuple[float, str, Dict]]:
        results = self.collection.query(query_texts=[query], n_results=k)
        out = []
        for score, text, meta in zip(results["distances"][0], results["documents"][0], results["metadatas"][0]):
            out.append((float(score), text, meta))
        return out

rag = RAGEngine(collection)
status = rag.build_or_load(DATA_DIR)
print(f"[RAG] Index status: {status}")

# ----------------- Conversation Memory -----------------

class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful, accurate assistant. Use supplied context to answer and cite sources."}
        ]
        self.active: bool = True

conversations: Dict[str, Conversation] = {}

def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]

# ----------------- Prompting Helpers -----------------

RAG_SYSTEM_INSTRUCTION = (
    "You are a domain assistant. Use ONLY the supplied context below to answer. "
    "If the answer is not present in the context, say you don't know. "
    "When you quote facts, cite their source in the form [source:filename#chunk]. Be concise and correct."
)

def build_context_block(hits: List[Tuple[float, str, Dict]]) -> str:
    blocks = []
    for _, text, meta in hits:
        src = os.path.basename(meta.get("source", "unknown"))
        cid = meta.get("chunk_id", -1)
        blocks.append(f"[{src}#{cid}] {text}")
    return "\n\n".join(blocks)

def make_augmented_messages(conv: Conversation, user_query: str) -> List[Dict[str, str]]:
    hits = rag.retrieve(user_query, k=TOP_K)
    context = build_context_block(hits) if hits else "No relevant context found."

    msgs = []
    msgs.append({"role": "system", "content": RAG_SYSTEM_INSTRUCTION})
    msgs.append({"role": "system", "content": f"Context:\n{context}"})
    last_history = [m for m in conv.messages if m["role"] in ("user", "assistant")][-6:]
    msgs.extend(last_history)
    msgs.append({"role": "user", "content": user_query})
    return msgs

# ----------------- Groq Chat -----------------

def query_groq(messages: List[Dict[str, str]]) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            top_p=1.0,
            stream=True,
        )
        buf = []
        for chunk in completion:
            buf.append(chunk.choices[0].delta.content or "")
        return "".join(buf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

# ----------------- Web Page Fetcher -----------------

def fetch_and_clean(url: str) -> str:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def safe_filename_for_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.replace(":", "_")
    path = parsed.path.strip("/").replace("/", "_") or "root"
    fname = f"{host}_{path}.txt"
    return "".join(c for c in fname if c.isalnum() or c in ("_", ".", "-"))

# ----------------- Auto-Ingest Apple Watch URLs -----------------

APPLE_WATCH_URLS = [
    "https://www.apple.com/apple-watch-series-10/",
    "https://www.apple.com/apple-watch-ultra-2/",
    "https://www.apple.com/watchos/"
]

def auto_ingest_apple_watch():
    for url in APPLE_WATCH_URLS:
        try:
            fname = safe_filename_for_url(url)
            dest = os.path.join(DATA_DIR, fname + ".txt")
            if not os.path.exists(dest):  # only ingest once
                print(f"[Auto-Ingest] Fetching {url}")
                text = fetch_and_clean(url)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(text)
                rag.add_documents([(dest, text)])
                print(f"[Auto-Ingest] Added {url}")
            else:
                print(f"[Auto-Ingest] Skipping {url} (already ingested)")
        except Exception as e:
            print(f"[Auto-Ingest] Failed {url}: {e}")

@app.on_event("startup")
def startup_event():
    auto_ingest_apple_watch()
    print("[Startup] Auto-ingest completed.")

# ----------------- Routes -----------------

@app.get("/health")
def health():
    count = collection.count()
    return {"ok": True, "index_status": status, "docs_indexed": count}

@app.post("/reindex")
def reindex():
    chroma_client.delete_collection("rag_store")
    global collection
    collection = chroma_client.get_or_create_collection(
        name="rag_store",
        embedding_function=embed_fn
    )
    stat = rag.build_or_load(DATA_DIR)
    return {"status": stat, "docs_indexed": collection.count()}

class IngestURLRequest(BaseModel):
    url: str
    save_filename: Optional[str] = None

@app.post("/ingest_url")
def ingest_url(payload: IngestURLRequest):
    try:
        text = fetch_and_clean(payload.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Fetched page contained no text.")

    fname = payload.save_filename or safe_filename_for_url(payload.url)
    dest = os.path.join(DATA_DIR, fname)
    if not dest.lower().endswith(".txt"):
        dest = dest + ".txt"

    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)

    added = rag.add_documents([(dest, text)])
    return {"status": "ingested", "file": dest, "chunks_added": added}

@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="The chat session has ended. Please start a new session.")

    rag_messages = make_augmented_messages(conversation, input.message)
    response = query_groq(rag_messages)

    conversation.messages.append({"role": input.role, "content": input.message})
    conversation.messages.append({"role": "assistant", "content": response})

    return {"response": response, "conversation_id": input.conversation_id}
