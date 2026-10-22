
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DocumentReranker


# ==========================================
# Configuration
# ==========================================

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"

GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "tests" / "retrieval_chunk_ground_truth.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "tests"
    / "reranking_chunk_evaluation_results.json"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TOP_K_VALUES = [1, 3, 5, 8]
CANDIDATE_K = 20
RRF_K = 60


# ==========================================
# Load ground truth and PDF
# ==========================================

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)

if not questions:
    raise ValueError("Ground-truth dataset is empty.")

documents = load_pdf(str(PDF_PATH))

chunks = split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

texts = [chunk.page_content for chunk in chunks]

print("=" * 65)
print("HYBRID SEARCH VS HYBRID + RERANKING")
print("=" * 65)
print(f"PDF: {PDF_PATH.name}")
print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")
print(f"Questions: {len(questions)}")
print(f"Chunk size / overlap: {CHUNK_SIZE} / {CHUNK_OVERLAP}")
print(f"Candidate K: {CANDIDATE_K}")
print(f"RRF K: {RRF_K}")


# ==========================================
# Initialize retrievers
# ==========================================

bm25 = BM25Retriever(texts)
reranker = DocumentReranker()


# ==========================================
# Hybrid Search
# ==========================================

def hybrid_search(question):
    candidate_k = min(CANDIDATE_K, len(chunks))

    # 1. BM25
    bm25_results = bm25.search(
        question,
        n_results=candidate_k,
    )

    # 2. Vector Search
    query_embedding = create_embeddings([question])[0]

    vector_raw = search(
        query_embedding=query_embedding,
        n_results=candidate_k,
        source=PDF_PATH.name,
    )

    vector_metadatas = vector_raw["metadatas"][0]

    # 3. Reciprocal Rank Fusion
    scores = {}

    for rank, metadata in enumerate(vector_metadatas, start=1):
        chunk_id = metadata.get("chunk_id")

        if chunk_id is None:
            raise ValueError(
                "A vector result has no chunk_id in its metadata."
            )

        chunk_id = int(chunk_id)

        if not 0 <= chunk_id < len(chunks):
            raise ValueError(
                f"Chunk ID {chunk_id} does not match the current PDF "
                "chunking. Rebuild the vector database using the "
                "same PDF and chunk configuration before evaluating."
            )

        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["index"]

        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return ranked_ids


# ==========================================
# Rerank Hybrid candidates
# ==========================================

def rerank_candidates(question, hybrid_ids):
    candidate_ids = hybrid_ids[:min(CANDIDATE_K, len(hybrid_ids))]
    candidate_texts = [texts[chunk_id] for chunk_id in candidate_ids]

    reranked = reranker.rerank(
        query=question,
        documents=candidate_texts,
        top_k=len(candidate_texts),
    )

    print("\n--- Top 5 BEFORE reranking ---")
    for rank, chunk_id in enumerate(candidate_ids[:5], start=1):
        page = get_page_number(chunk_id)
        print(f"\nRank {rank} | Chunk {chunk_id} | Page {page}")
        print(texts[chunk_id][:500])

    print("\n--- Top 5 AFTER reranking ---")
    for rank, item in enumerate(reranked[:5], start=1):
        chunk_id = candidate_ids[item["index"]]
        page = get_page_number(chunk_id)
        print(
            f"\nRank {rank} | Chunk {chunk_id} | Page {page} "
            f"| Score {item.get('score')}"
        )
        print(texts[chunk_id][:500])

    return [candidate_ids[item["index"]] for item in reranked]


# ==========================================
# Evaluation helpers
# ==========================================

def get_page_number(chunk_id):
    page_index = chunks[chunk_id].metadata.get("page")

    if not isinstance(page_index, int):
        return None

    # PDF loader uses zero-based page indices.
    return page_index + 1

# def unique_pages_in_rank_order(pages):
#     """Keep each page once, preserving its first ranked occurrence."""
#     return list(dict.fromkeys(
#         page for page in pages if page is not None
#     ))


def evaluate_rankings(rankings, ground_truth, ks):
    metrics = {}

    if len(rankings) != len(ground_truth):
        raise ValueError(
            "Rankings and ground truth must have the same length."
        )

    if not ground_truth:
        raise ValueError("Ground truth is empty.")

    for k in ks:
        hits = 0
        reciprocal_ranks = []

        for ranked_ids, relevant_ids in zip(
            rankings, ground_truth
        ):
            first_relevant_rank = next(
                (
                    rank
                    for rank, chunk_id in enumerate(
                        ranked_ids[:k], start=1
                    )
                    if chunk_id in relevant_ids
                ),
                None,
            )

            if first_relevant_rank is None:
                reciprocal_ranks.append(0.0)
            else:
                hits += 1
                reciprocal_ranks.append(
                    1.0 / first_relevant_rank
                )

        metrics[f"Hit@{k}"] = hits / len(ground_truth)
        metrics[f"MRR@{k}"] = (
            sum(reciprocal_ranks) / len(ground_truth)
        )

    return metrics


# ==========================================
# Run evaluation
# ==========================================


# ==========================================
# Run chunk-level evaluation
# ==========================================

hybrid_rankings = []
reranked_rankings = []
ground_truth_chunks = []

for number, item in enumerate(questions, start=1):
    question = item["question"]
    relevant_ids = set(item["relevant_chunk_ids"])

    # Validate ground-truth chunk IDs.
    invalid_ids = [
        chunk_id
        for chunk_id in relevant_ids
        if not isinstance(chunk_id, int)
        or not 0 <= chunk_id < len(chunks)
    ]

    if invalid_ids:
        raise ValueError(
            f"Invalid relevant chunk IDs for question "
            f"{number}: {invalid_ids}"
        )

    hybrid_ids = hybrid_search(question)
    reranked_ids = rerank_candidates(question, hybrid_ids)

    hybrid_rankings.append(hybrid_ids)
    reranked_rankings.append(reranked_ids)
    ground_truth_chunks.append(relevant_ids)

    print(f"\n[{number}/{len(questions)}] {question}")
    print(f"Relevant chunks: {sorted(relevant_ids)}")
    print(f"Hybrid Top 5 chunks: {hybrid_ids[:5]}")
    print(f"Reranked Top 5 chunks: {reranked_ids[:5]}")


hybrid_metrics = evaluate_rankings(
    hybrid_rankings,
    ground_truth_chunks,
    TOP_K_VALUES,
)

reranked_metrics = evaluate_rankings(
    reranked_rankings,
    ground_truth_chunks,
    TOP_K_VALUES,
)


# ==========================================
# Print summary
# ==========================================

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

print(
    f"{'Metric':<12}"
    f"{'Hybrid':>14}"
    f"{'Hybrid+Rerank':>20}"
    f"{'Change (pp)':>16}"
)

for k in TOP_K_VALUES:
    for metric_name in (f"Hit@{k}", f"MRR@{k}"):
        before = hybrid_metrics[metric_name]
        after = reranked_metrics[metric_name]
        change_pp = (after - before) * 100

        print(
            f"{metric_name:<12}"
            f"{before:>14.4f}"
            f"{after:>20.4f}"
            f"{change_pp:>15.2f}"
        )


# ==========================================
# Save results
# ==========================================

output = {
    "pdf": PDF_PATH.name,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "candidate_k": CANDIDATE_K,
    "rrf_k": RRF_K,
    "num_questions": len(questions),
    "evaluation_unit": "chunk",
    "methods": {
        "hybrid": hybrid_metrics,
        "hybrid_reranking": reranked_metrics,
    },
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(output, file, indent=4, ensure_ascii=False)

print(f"\nResults saved to: {OUTPUT_PATH}")
