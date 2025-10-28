
### setup

```bash

python3.11 -m venv .venv_py311

# 1) Install Ollama and pull a model (Qwen 2.5 7B is a great start)
brew install ollama
ollama serve
ollama pull qwen2.5:7b

# 2) Python env + libs (CPU is fine for indexing)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

python -m pip install --upgrade pip setuptools wheel

# PyTorch first (macOS wheels include MPS by default)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# Then sentence-transformers + the rest (no faiss-cpu)
pip install "sentence-transformers==3.0.1" hnswlib pypdf tqdm requests python-dotenv


```

### directory layout
```
prep-rag/
  build_index.py
  query_rag.py
  .env                # optional config (see below)
  data/               # created automatically (index + manifests)
  corpus/             # <-- point this to your top-level PDF folder (subfolders OK)
```

First we build our index. Point corpus dir to your folder of pdfs in your .env file then run 
```bash
python build_index.py

```
after building we can query the rag: 
```bash
python query_rag.py "What should a 3-day kit include for two adults and a toddler?"

```
