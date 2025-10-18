from langchain_text_splitters import RecursiveCharacterTextSplitter
from step1_fetch import fetch_url


def chunk_text(text: str, chunk_size=1000, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    example_url = "https://www.theguardian.com/world/2025/oct/03/czech-election-voting-andrej-babis"  # random Guardian article
    text = fetch_url(example_url)
    chunks = chunk_text(text)

    print(f"🔹 Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # show only first 3 chunks
        print(f"\\n--- CHUNK {i + 1} ({len(chunk)} characters) ---\\n{chunk[:500]}...")
