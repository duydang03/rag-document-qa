
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search


QUESTIONS_PATH = (
    PROJECT_ROOT / "tests" / "evaluation_questions.json"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "tests" / "retrieval_ground_truth.json"
)

TOP_K_VALUES = [1, 3, 5, 8]
CANDIDATE_K = 20
SOURCE = "core.pdf"



with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

questions = ground_truth

ground_truth_map = {
    item["question"]: item["relevant_pages"]
    for item in ground_truth
}



# Retrieve the same number of candidates as Hybrid.
# Reuse the ranking to calculate all Top-K metrics.
rankings = {}

for item in questions:
    question = item["question"]

    query_embedding = create_embeddings([question])[0]

    retrieved = search(
        query_embedding,
        n_results=CANDIDATE_K,
        source=SOURCE
    )

    metadatas = retrieved["metadatas"][0]

    rankings[question] = [
        metadata.get("page", 0) + 1
        for metadata in metadatas
    ]


results = {}

for top_k in TOP_K_VALUES:
    hit_count = 0
    reciprocal_ranks = []

    for item in questions:
        question = item["question"]

        relevant_pages = ground_truth_map.get(
            question, []
        )

        retrieved_pages = rankings[question][:top_k]

        reciprocal_rank = 0.0

        for rank, page in enumerate(
            retrieved_pages, start=1
        ):
            if page in relevant_pages:
                reciprocal_rank = 1.0 / rank
                break

        if reciprocal_rank > 0:
            hit_count += 1

        reciprocal_ranks.append(reciprocal_rank)

    total = len(questions)

    results[top_k] = {
        "hits": hit_count,
        "total": total,
        "score": hit_count / total if total else 0.0,
        "mrr": (
            sum(reciprocal_ranks) / total
            if total else 0.0
        )
    }


print("=" * 60)
print("VECTOR SEARCH BASELINE")
print("=" * 60)
print(f"Questions: {len(questions)}")
print(f"Candidate K: {CANDIDATE_K}")
print(f"Source: {SOURCE}")

for top_k, result in results.items():
    print(
        f"Hit@{top_k}: "
        f"{result['hits']}/{result['total']} "
        f"({result['score'] * 100:.2f}%)"
    )
    print(f"MRR@{top_k}: {result['mrr']:.4f}")

