import os
import io
import json
import time
import pickle
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# Embeddings + Vector DB
import faiss
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

# ----------------- Vector Store Wrapper -----------------

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # use inner product on normalized vectors = cosine
        self.texts: List[str] = []
        self.metas: List[Dict] = []
        self._is_normalized = False

    def add(self, embeddings: np.ndarray, texts: List[str], metas: List[Dict]):
        assert embeddings.shape[0] == len(texts) == len(metas)
        # normalize to use cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)
        self.metas.extend(metas)
        self._is_normalized = True

    def search(self, query_emb: np.ndarray, k: int) -> List[Tuple[float, str, Dict]]:
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        faiss.normalize_L2(query_emb)
        scores, idxs = self.index.search(query_emb.astype(np.float32), k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((float(score), self.texts[idx], self.metas[idx]))
        return out

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metas": self.metas, "dim": self.dim}, f)

    @staticmethod
    def load(path: str) -> "VectorStore":
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            obj = pickle.load(f)
        vs = VectorStore(obj["dim"])
        vs.index = faiss.read_index(os.path.join(path, "faiss.index"))
        vs.texts = obj["texts"]
        vs.metas = obj["metas"]
        vs._is_normalized = True
        return vs

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
    def __init__(self, embed_model_name: str, index_dir: str):
        self.embedder = SentenceTransformer(embed_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index_dir = index_dir
        self.vs: Optional[VectorStore] = None

    def build_or_load(self, data_dir: str) -> str:
        faiss_file = os.path.join(self.index_dir, "faiss.index")
        store_file = os.path.join(self.index_dir, "store.pkl")

        if os.path.exists(faiss_file) and os.path.exists(store_file):
            self.vs = VectorStore.load(self.index_dir)
            return "loaded"

        corpus = load_corpus_from_dir(data_dir)
        if not corpus:
            self.vs = VectorStore(self.dim)
            self.vs.save(self.index_dir)
            return "built-empty"

        texts, metas = [], []
        for src, content in corpus:
            chunks = chunk_text(content)
            for i, ch in enumerate(chunks):
                texts.append(ch)
                metas.append({"source": src, "chunk_id": i})

        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False, batch_size=64, show_progress_bar=True
        )
        vs = VectorStore(self.dim)
        vs.add(embeddings, texts, metas)
        vs.save(self.index_dir)
        self.vs = vs
        return "built"

    def add_documents(self, docs: List[Tuple[str, str]]) -> int:
        """
        Add (source_path, content) pairs to existing vector store (or create new).
        Returns number of chunks added.
        """
        if self.vs is None:
            self.vs = VectorStore(self.dim)

        new_texts, new_metas = [], []
        for src, content in docs:
            chunks = chunk_text(content)
            for i, ch in enumerate(chunks):
                new_texts.append(ch)
                new_metas.append({"source": src, "chunk_id": i})

        if not new_texts:
            return 0

        embeddings = self.embedder.encode(new_texts, convert_to_numpy=True, normalize_embeddings=False, batch_size=64)
        self.vs.add(embeddings, new_texts, new_metas)
        self.vs.save(self.index_dir)
        return len(new_texts)

    def retrieve(self, query: str, k: int = TOP_K) -> List[Tuple[float, str, Dict]]:
        if self.vs is None:
            raise RuntimeError("Vector store not initialized")
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)[0]
        return self.vs.search(q_emb, k=k)

# ----------------- Instantiate RAG -----------------

rag = RAGEngine(EMBED_MODEL_NAME, INDEX_DIR)
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
    for score, text, meta in hits:
        src = os.path.basename(meta.get("source", "unknown"))
        cid = meta.get("chunk_id", -1)
        blocks.append(f"[{src}#{cid}] {text}")
    return "\n\n".join(blocks)

def make_augmented_messages(conv: Conversation, user_query: str) -> List[Dict[str, str]]:
    hits = rag.retrieve(user_query, k=TOP_K) if rag.vs and len(rag.vs.texts) > 0 else []
    context = build_context_block(hits) if hits else "No relevant context found."

    msgs = []
    msgs.append({"role": "system", "content": RAG_SYSTEM_INSTRUCTION})
    msgs.append({"role": "system", "content": f"Context:\n{context}"})
    # keep a bit of recent history for coherence
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
    # remove illegal chars
    return "".join(c for c in fname if c.isalnum() or c in ("_", ".", "-"))

# ----------------- Routes -----------------

@app.get("/health")
def health():
    docs_indexed = len(rag.vs.texts) if (rag.vs and hasattr(rag.vs, "texts")) else 0
    return {"ok": True, "index_status": status, "docs_indexed": docs_indexed}

@app.post("/reindex")
def reindex():
    # remove current index files and rebuild from DATA_DIR
    for fn in ("faiss.index", "store.pkl"):
        try:
            os.remove(os.path.join(INDEX_DIR, fn))
        except FileNotFoundError:
            pass
    stat = rag.build_or_load(DATA_DIR)
    docs_indexed = len(rag.vs.texts) if rag.vs else 0
    return {"status": stat, "docs_indexed": docs_indexed}

class IngestURLRequest(BaseModel):
    url: str
    save_filename: Optional[str] = None  # optional custom name inside data/

@app.post("/ingest_url")
def ingest_url(payload: IngestURLRequest):
    """
    Fetch a webpage, save cleaned text to DATA_DIR, then chunk+index it immediately.
    Returns number of chunks added.
    """
    try:
        text = fetch_and_clean(payload.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Fetched page contained no text.")

    fname = payload.save_filename or safe_filename_for_url(payload.url)
    dest = os.path.join(DATA_DIR, fname)
    # ensure .txt extension
    if not dest.lower().endswith(".txt"):
        dest = dest + ".txt"

    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)

    # add to vector store immediately
    added = rag.add_documents([(dest, text)])
    return {"status": "ingested", "file": dest, "chunks_added": added}

@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(status_code=400, detail="The chat session has ended. Please start a new session.")

    rag_messages = make_augmented_messages(conversation, input.message)
    response = query_groq(rag_messages)

    # Update conversation memory (only keep the short chat history)
    conversation.messages.append({"role": input.role, "content": input.message})
    conversation.messages.append({"role": "assistant", "content": response})

    return {"response": response, "conversation_id": input.conversation_id}

# ----------------- Run Instructions -----------------
# Start with:
#   uvicorn app:app --reload
#
# Example ingestion:
#   POST /ingest_url  { "url": "https://www.apple.com/apple-watch-series-10/" }
#
# Example chat:
#   POST /chat/ { "message": "What's the battery life of the Apple Watch Series 10?", "conversation_id": "apple_demo" }
#