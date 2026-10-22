from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search
from app.retrieval.bm25 import BM25Retriever


PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"


# ==========================================
# 1. Load PDF
# ==========================================

documents = load_pdf(str(PDF_PATH))

chunks = split_documents(
    documents,
    chunk_size=800,
    chunk_overlap=200
)

texts = [
    chunk.page_content
    for chunk in chunks
]


# ==========================================
# 2. Create BM25 retriever
# ==========================================

bm25 = BM25Retriever(texts)


# ==========================================
# 3. Query
# ==========================================

query = "What is BDD100K?"


# ==========================================
# 4. BM25 search
# ==========================================

bm25_results = bm25.search(
    query,
    n_results=20
)


# ==========================================
# 5. Vector search
# ==========================================

query_embedding = create_embeddings([query])[0]

vector_raw = search(
    query_embedding,
    n_results=20,
    source="core.pdf"
)


vector_results = []

for document, metadata in zip(
    vector_raw["documents"][0],
    vector_raw["metadatas"][0]
):
    vector_results.append({
        "index": metadata["chunk_id"],
        "document": document
    })


# ==========================================
# 6. Reciprocal Rank Fusion (RRF)
# ==========================================

rrf_k = 60

scores = {}


# Vector ranking
for rank, result in enumerate(vector_results, start=1):

    chunk_id = result["index"]

    scores[chunk_id] = scores.get(chunk_id, 0) + (
        1 / (rrf_k + rank)
    )


# BM25 ranking
for rank, result in enumerate(bm25_results, start=1):

    chunk_id = result["index"]

    scores[chunk_id] = scores.get(chunk_id, 0) + (
        1 / (rrf_k + rank)
    )


# ==========================================
# 7. Sort hybrid results
# ==========================================

ranked_chunk_ids = sorted(
    scores,
    key=scores.get,
    reverse=True
)


# ==========================================
# 8. Display results
# ==========================================

print("=" * 60)
print("HYBRID SEARCH TEST")
print("=" * 60)

print(f"Number of chunks: {len(texts)}")

print("\nQuery:")
print(query)

print("\nTop 5 Hybrid Results:")
print("-" * 60)


for rank, chunk_id in enumerate(
    ranked_chunk_ids[:5],
    start=1
):

    print(f"\nRank {rank}")
    print(f"Chunk ID: {chunk_id}")
    print(f"RRF Score: {scores[chunk_id]:.6f}")

    print("\nText:")
    print(texts[chunk_id][:500])