from pathlib import Path

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import (
    add_documents,
    delete_document,
    document_exists
)


def index_pdf(pdf_path: str):
    source = Path(pdf_path).name

    # Nếu document đã tồn tại thì xóa dữ liệu cũ
    if document_exists(source):
        delete_document(source)

    # Load PDF
    documents = load_pdf(pdf_path)

    # Chunk
    chunks = split_documents(documents)

    # Extract text
    texts = [chunk.page_content for chunk in chunks]

    # Prepare metadata
    metadatas = []

    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata.copy()
        metadata["source"] = source
        metadata["chunk_id"] = index
        metadatas.append(metadata)

    # Create embeddings
    embeddings = create_embeddings(texts)

    # Add to ChromaDB
    add_documents(
        texts,
        embeddings,
        metadatas
    )

    return len(documents), len(chunks)