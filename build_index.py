# build_index.py
import os, json, hashlib
from pathlib import Path
from typing import List
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "corpus"))
DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

EMB_PATH = DATA_DIR / "embeddings.npy"
TEXTS_PATH = DATA_DIR / "texts.jsonl"
META_PATH = DATA_DIR / "meta.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

def pdf_to_text(fp: Path) -> str:
    try:
        reader = PdfReader(str(fp))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        print(f"[WARN] {fp}: {e}")
        return ""

def chunk_text(text: str, size=900, overlap=150):
    text = " ".join(text.split())
    out, i, n = [], 0, len(text)
    step = max(size - overlap, 1)
    while i < n:
        ch = text[i:i+size]
        if len(ch.strip()) >= 60:
            out.append(ch)
        i += step
    return out

def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for block in iter(lambda: f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()

def load_manifest():
    m = {}
    if MANIFEST_PATH.exists():
        for line in MANIFEST_PATH.read_text().splitlines():
            try:
                rec = json.loads(line)
                m[rec["path"]] = rec
            except:
                pass
    return m

def append_manifest(rec):
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    if not CORPUS_DIR.exists():
        raise SystemExit(f"Put your PDFs under {CORPUS_DIR}/")

    files = list(CORPUS_DIR.rglob("*.pdf"))
    if not files:
        raise SystemExit("No PDFs found.")

    manifest = load_manifest()

    texts, metas = [], []
    for fp in tqdm(files, desc="Scanning PDFs"):
        key = str(fp.resolve())
        fhash = sha256_file(fp)
        if key in manifest and manifest[key].get("sha256") == fhash:
            continue

        raw = pdf_to_text(fp)
        chunks = chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP) if raw.strip() else []
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"source": str(fp.relative_to(CORPUS_DIR)), "chunk_id": idx})
        append_manifest({"path": key, "sha256": fhash, "chunks": len(chunks)})

    if not texts:
        print("Nothing new to embed. (All files already indexed?)")
        raise SystemExit(0)

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME} …")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True)  # [N, d]
    # cosine needs L2-normalized vectors so dot ~ cosine
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = (embs / norms).astype("float32")

    # append to existing store if present
    if EMB_PATH.exists():
        old = np.load(EMB_PATH)
        embs = np.concatenate([old, embs], axis=0)

    np.save(EMB_PATH, embs)

    with open(TEXTS_PATH, "a", encoding="utf-8") as ft, open(META_PATH, "a", encoding="utf-8") as fm:
        for t, m in zip(texts, metas):
            ft.write(json.dumps({"t": t}, ensure_ascii=False) + "\n")
            fm.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"✅ Saved embeddings to {EMB_PATH}, texts to {TEXTS_PATH}, meta to {META_PATH}")
