import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search



with open(
    PROJECT_ROOT / "tests" / "retrieval_ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    ground_truth = json.load(f)


K_VALUES = [3, 5, 8, 10, 15]

hit_results = {
    k: 0
    for k in K_VALUES
}

mrr_results = {
    k: 0
    for k in K_VALUES
}


total_questions = len(ground_truth)


for item in ground_truth:

    question = item["question"]
    relevant_pages = set(item["relevant_pages"])

    print("=" * 80)
    print(question)

    query_embedding = create_embeddings(
        [question]
    )[0]

    results = search(
        query_embedding,
        n_results=15,
        source="core.pdf"
    )

    retrieved_pages = [
        metadata.get("page", 0) + 1
        for metadata in results["metadatas"][0]
    ]

    print("Retrieved pages:")
    print(retrieved_pages)

    for k in K_VALUES:

        top_k_pages = retrieved_pages[:k]

        # Hit@K
        if any(
            page in relevant_pages
            for page in top_k_pages
        ):
            hit_results[k] += 1

        # MRR@K
        reciprocal_rank = 0

        for rank, page in enumerate(
            top_k_pages,
            start=1
        ):
            if page in relevant_pages:
                reciprocal_rank = 1 / rank
                break

        mrr_results[k] += reciprocal_rank


print()
print("=" * 80)
print("RETRIEVAL RESULTS")
print("=" * 80)

for k in K_VALUES:

    hit = hit_results[k] / total_questions

    mrr = mrr_results[k] / total_questions

    print(
        f"Hit@{k}: {hit:.4f}"
    )

    print(
        f"MRR@{k}: {mrr:.4f}"
    )

    print()