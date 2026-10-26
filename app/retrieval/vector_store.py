import os
import uuid
from pathlib import Path
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(PROJECT_ROOT / "data" / "chroma")
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "documents"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def reset_collection():
    global collection

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def add_documents(texts, embeddings, metadata):
    ids = [str(uuid.uuid4()) for _ in texts]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadata
    )


def delete_document(source: str):
    collection.delete(
        where={
            "source": source
        }
    )
    
def get_document_stats(source: str):
    results = collection.get(
        where={
            "source": source
        }
    )

    chunk_count = len(results["ids"])

    pages = set()

    for metadata in results["metadatas"]:
        if metadata and "page" in metadata:
            pages.add(metadata["page"])

    return {
        "chunks": chunk_count,
        "pages": len(pages)
    }

def document_exists(source: str):
    results = collection.get(
        where={
            "source": source
        },
        limit=1
    )

    return len(results["ids"]) > 0


def list_documents():
    results = collection.get()

    documents = set()

    for metadata in results["metadatas"]:
        if metadata and "source" in metadata:
            documents.add(metadata["source"])

    return sorted(documents)


def search(query_embedding, n_results=3, source=None):
    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": n_results
    }

    if source is not None:
        query_params["where"] = {
            "source": source
        }

    results = collection.query(**query_params)

    return results

def get_all_chunks(source=None):
    query_params = {
        "include": ["documents", "metadatas"]
    }

    if source is not None:
        query_params["where"] = {"source": source}

    results = collection.get(**query_params)

    chunks = []

    for document, metadata in zip(
        results["documents"],
        results["metadatas"]
    ):
        if not metadata:
            continue

        chunks.append({
            "document": document,
            "metadata": metadata
        })

    # Đảm bảo thứ tự ổn định cho mỗi lần chạy
    chunks.sort(
        key=lambda item: (
            item["metadata"].get("source", ""),
            item["metadata"].get("chunk_id", -1)
        )
    )

    return chunks