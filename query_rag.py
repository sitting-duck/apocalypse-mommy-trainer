# query_rag.py
import os, json
from pathlib import Path
import numpy as np, requests
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
EMB_PATH = DATA_DIR / "embeddings.npy"
TEXTS_PATH = DATA_DIR / "texts.jsonl"
META_PATH = DATA_DIR / "meta.jsonl"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
TOP_K = int(os.getenv("TOP_K", 6))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 2200))

SYSTEM_PROMPT = """You are PrepAdvisor, a practical disaster-preparedness coach.
Be concise, source-grounded, and legally safe.
Format:
1) Short prioritized checklist (3–8 bullets)
2) One-sentence reason for each
3) Cite sources by filename
If asked for illegal/violent/harmful instructions, refuse politely and offer safe, legal alternatives."""

def load_store():
    if not EMB_PATH.exists():
        raise RuntimeError("No embeddings found. Run build_index.py first.")
    embs = np.load(EMB_PATH)                 # [N, d], L2-normalized
    texts = [json.loads(l)["t"] for l in TEXTS_PATH.read_text().splitlines()]
    metas = [json.loads(l) for l in META_PATH.read_text().splitlines()]
    assert len(texts) == embs.shape[0] == len(metas), "Store mismatch."
    nn = NearestNeighbors(n_neighbors=min(TOP_K, embs.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(embs)
    return embs, texts, metas, nn

def embed_query(q: str, model_name=EMBED_MODEL_NAME):
    m = SentenceTransformer(model_name)
    v = m.encode([q])  # [1, d]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v.astype("float32")

def retrieve(question: str, embs, texts, metas, nn):
    v = embed_query(question)
    dist, idx = nn.kneighbors(v, return_distance=True)
    out = []
    for i in idx[0]:
        out.append({"text": texts[i], "source": metas[i]["source"], "chunk_id": metas[i]["chunk_id"]})
    return out

def build_context(passages):
    ctx = ""
    for p in passages:
        block = f"[Source: {p['source']}]\n{p['text']}\n\n"
        if len(ctx) + len(block) > MAX_CONTEXT_CHARS:
            break
        ctx += block
    return ctx

def ask_ollama(system_prompt: str, user_question: str, context: str):
    url = "http://localhost:11434/api/generate"
    prompt = f"""<|system|>
{system_prompt}
<|user|>
Use ONLY the context below. If insufficient, say what's missing and suggest safe next steps.

[CONTEXT START]
{context}
[CONTEXT END]

User: {user_question}
"""
    r = requests.post(url, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python query_rag.py "Your question"'); raise SystemExit(1)
    question = sys.argv[1]
    embs, texts, metas, nn = load_store()
    passages = retrieve(question, embs, texts, metas, nn)
    context = build_context(passages)
    answer = ask_ollama(SYSTEM_PROMPT, question, context)
    print("\n=== ANSWER ===\n", answer)
    print("\n=== SOURCES ===")
    for p in passages:
        print(f"• {p['source']} (chunk {p['chunk_id']})")
