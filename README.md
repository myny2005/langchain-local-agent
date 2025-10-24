# LangChain Local Agent – Article Q&A

A small pipeline that ingests articles, chunks them, builds a local vector index, and answers questions with a LangChain agent running local LLMs (Ollama/llama.cpp/vLLM).

## Features
- Fetch → Split → Index → Q&A → Auto-QA evaluation
- Local-first (no external API keys required)
- Deterministic config via `configs/config.yaml`
- Reproducible steps: `step1_fetch.py` ... `step5_autoqa.py`

## Quickstart

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or `uv pip sync`

# 2a) (Optional) Set a custom config path and API key if using OpenAI models
export LC_CONFIG=configs/config.yaml
export OPENAI_API_KEY="your_api_key_here"

# 2b) Set up your configs/config.yaml

# 3) Fetch + build index
python src/step1_fetch.py --url "https://example.com/article"
python src/step2_split.py
python src/step3_index.py

# 4) Ask questions
python src/step4_qa.py --question "What is the main claim?"

# 5) Run automated QA/eval
python src/step5_autoqa.py
```

## **Install Ollama**

### macOS (Homebrew)

```bash
brew install ollama
```
### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download and run the installer from [ollama.com/download](https://ollama.com/download) 

### Start the Ollama server

```bash
ollama serve
```

### Pull a model (one-time setup)
```bash
# choose any supported model
ollama pull mistral
ollama pull llama3
ollama pull mixtral:8x7b
```

### Configure your project
Edit configs/config.yaml:
```yaml
models:
  llm_backend: "ollama"
  llm: "mistral"       # must match the pulled model name
  temperature: 0.2
  max_tokens: 1024
```

### GPU & Performance Tips
- Ollama automatically uses your GPU if available (NVIDIA, Apple Silicon, etc.).
- Use smaller or quantized models (e.g. mistral:instruct, llama3:8b) for faster inference.
- If memory is low, stop the server and pull a lighter model variant.

## **Testing & CI info**
Run all unit tests and style checks:

```bash
pytest -q
ruff check .
mypy .
```
