from pathlib import Path

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import (
    add_documents,
    reset_collection
)


PROJECT_ROOT = Path(__file__).resolve().parent

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"


def build_vector_db(chunk_size=800, chunk_overlap=200):

    print("=" * 60)
    print("BUILD VECTOR DATABASE")
    print("=" * 60)

    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")

    reset_collection()
    print("ChromaDB collection reset.")

    documents = load_pdf(str(PDF_PATH))

    print(f"Number of pages: {len(documents)}")

    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print(f"Number of chunks: {len(chunks)}")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    metadatas = []

    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata.copy()
        metadata["source"] = PDF_PATH.name
        metadata["chunk_id"] = index
        metadatas.append(metadata)

    embeddings = create_embeddings(texts)

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    add_documents(
        texts,
        embeddings,
        metadatas
    )

    print("PDF successfully added to ChromaDB.")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    build_vector_db(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )