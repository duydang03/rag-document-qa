import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search
from app.retrieval.bm25 import BM25Retriever


# ==========================================
# Configuration
# ==========================================

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"

QUESTIONS_PATH = (
    PROJECT_ROOT / "tests" / "evaluation_questions.json"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "tests" / "retrieval_ground_truth.json"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TOP_K_VALUES = [1, 3, 5, 8]

# Retrieve more candidates before RRF fusion
CANDIDATE_K = 20

RRF_K = 60


# ==========================================
# Load evaluation data
# ==========================================


with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

questions = ground_truth

ground_truth_map = {
    item["question"]: item["relevant_pages"]
    for item in ground_truth
}


# ==========================================
# Load and chunk PDF
# ==========================================

print("=" * 60)
print("HYBRID RETRIEVAL EVALUATION")
print("=" * 60)

documents = load_pdf(str(PDF_PATH))

chunks = split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

texts = [
    chunk.page_content
    for chunk in chunks
]

print(f"PDF: {PDF_PATH.name}")
print(f"Number of pages: {len(documents)}")
print(f"Number of chunks: {len(chunks)}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunk overlap: {CHUNK_OVERLAP}")
print(f"Candidate K: {CANDIDATE_K}")
print(f"RRF K: {RRF_K}")


# ==========================================
# Initialize BM25
# ==========================================

bm25 = BM25Retriever(texts)


# ==========================================
# Hybrid Search function
# ==========================================

def hybrid_search(question, n_results=8):

    # 1. BM25 retrieval
    bm25_results = bm25.search(
        question,
        n_results=CANDIDATE_K
    )

    # 2. Vector retrieval
    query_embedding = create_embeddings([question])[0]

    vector_raw = search(
        query_embedding,
        n_results=CANDIDATE_K,
        source=PDF_PATH.name
    )

    vector_documents = vector_raw["documents"][0]
    vector_metadatas = vector_raw["metadatas"][0]

    # 3. Reciprocal Rank Fusion
    scores = {}

    # Add vector ranking scores
    for rank, metadata in enumerate(vector_metadatas, start=1):

        chunk_id = metadata.get("chunk_id")

        if chunk_id is None:
            raise ValueError(
                "Missing chunk_id in ChromaDB metadata. "
                "Rebuild the vector database first."
            )

        chunk_id = int(chunk_id)

        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    # Add BM25 ranking scores
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["index"]

        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    # 4. Sort by fused score
    ranked_chunk_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return ranked_chunk_ids[:n_results]


# ==========================================
# Evaluate Hit@K and MRR@K
# ==========================================

results = {}

for top_k in TOP_K_VALUES:

    hit_count = 0
    reciprocal_ranks = []

    print("\n" + "=" * 60)
    print(f"Hybrid Retrieval - Hit@{top_k}")
    print("=" * 60)

    for item in questions:

        question = item["question"]

        relevant_pages = ground_truth_map.get(
            question,
            []
        )

        # Retrieve the ranked chunk IDs once per question
        ranked_chunk_ids = hybrid_search(
            question,
            n_results=CANDIDATE_K
        )

        # Convert chunk IDs into 1-based page numbers
        retrieved_pages = [
            chunks[chunk_id].metadata.get("page", 0) + 1
            for chunk_id in ranked_chunk_ids[:top_k]
        ]

        reciprocal_rank = 0.0

        for rank, page in enumerate(retrieved_pages, start=1):

            if page in relevant_pages:
                reciprocal_rank = 1.0 / rank
                break

        hit = reciprocal_rank > 0

        if hit:
            hit_count += 1

        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {question}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Relevant pages:  {relevant_pages}")
        print(f"Hit@{top_k}: {'YES' if hit else 'NO'}")

    total = len(questions)

    score = hit_count / total if total else 0.0
    mrr = (
        sum(reciprocal_ranks) / total
        if total else 0.0
    )

    results[top_k] = {
        "hits": hit_count,
        "total": total,
        "score": score,
        "mrr": mrr
    }


# ==========================================
# Summary
# ==========================================

print("\n" + "=" * 60)
print("HYBRID RETRIEVAL SUMMARY")
print("=" * 60)

for top_k in TOP_K_VALUES:

    result = results[top_k]

    print(
        f"Hit@{top_k}: "
        f"{result['hits']}/{result['total']} "
        f"({result['score'] * 100:.2f}%)"
    )

    print(f"MRR@{top_k}: {result['mrr']:.4f}")


# ==========================================
# Save results
# ==========================================

output_path = (
    PROJECT_ROOT / "tests" / "hybrid_retrieval_results.json"
)

output = {
    "method": "Hybrid Search (BM25 + Vector + RRF)",
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "candidate_k": CANDIDATE_K,
    "rrf_k": RRF_K,
    "num_questions": len(questions),
    "results": {
        str(k): value
        for k, value in results.items()
    }
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\nResults saved to: {output_path}")

