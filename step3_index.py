# step3_index.py
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from step1_fetch import fetch_url  # or fetch_url_robust if needed
from step2_split import chunk_text
from settings import load_config, persist_path_for_url, load_cookies


def index_url():
    cfg = load_config()
    url = cfg["article"]["url"]
    cookies = load_cookies(cfg["article"].get("cookies_path"))
    _ = cookies  # to pass ruff tests (maybe i will need cookies in the future)
    base_dir = cfg["index"]["base_dir"]
    chunk_size = cfg["index"]["chunk_size"]
    chunk_overlap = cfg["index"]["chunk_overlap"]
    emb_model = cfg["models"]["embeddings"]

    persist_dir = persist_path_for_url(base_dir, url)

    print("[1/3] Fetching URL")
    text = fetch_url(url)

    print("[2/3] Dividing text into chunks")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

    print(f"[3/3] Creating embeddings and writing into Chroma ({persist_dir})")
    embeddings = OllamaEmbeddings(model=emb_model)
    metadatas: List[Dict[str, Any]] = [ # type: ignore[annotation-unchecked]
        {"source": url, "chunk": i} for i in range(len(chunks))
    ]

    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir,  # auto-persist Chroma 0.4+
    )
    print(f"Indexed {len(chunks)} chunks at {persist_dir}")


if __name__ == "__main__":
    index_url()
