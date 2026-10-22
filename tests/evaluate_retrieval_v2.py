import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
import json

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search


# ==========================================
# Load evaluation questions
# ==========================================

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


# ==========================================
# Load retrieval ground truth
# ==========================================

with open(
    "tests/retrieval_ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    ground_truth = json.load(f)


ground_truth_map = {
    item["question"]: item["relevant_pages"]
    for item in ground_truth
}


# ==========================================
# Evaluation settings
# ==========================================

top_k_values = [1, 3, 5, 8]

results = {}


# ==========================================
# Evaluate
# ==========================================

for top_k in top_k_values:

    hit_count = 0
    reciprocal_ranks = []

    print("\n" + "=" * 60)
    print(f"Retrieval Evaluation - Hit@{top_k}")
    print("=" * 60)

    for item in questions:

        question = item["question"]

        relevant_pages = ground_truth_map.get(
            question,
            []
        )

        # Create query embedding
        query_embedding = create_embeddings(
            [question]
        )[0]

        # Retrieve chunks
        retrieved = search(
            query_embedding,
            n_results=top_k
        )

        metadatas = retrieved["metadatas"][0]

        retrieved_pages = [
            metadata.get("page", 0) + 1
            for metadata in metadatas
        ]

        # Check whether any relevant page
        # appears in retrieved results
        hit = False
        reciprocal_rank = 0.0

        for rank, page in enumerate(
            retrieved_pages,
            start=1
        ):

            if page in relevant_pages:

                hit = True

                reciprocal_rank = 1 / rank

                break


        if hit:
            hit_count += 1

        reciprocal_ranks.append(
            reciprocal_rank
        )

        print(
            f"{question}"
        )

        print(
            f"Retrieved pages: {retrieved_pages}"
        )

        print(
            f"Relevant pages:  {relevant_pages}"
        )

        print(
            f"Hit@{top_k}: "
            f"{'YES' if hit else 'NO'}"
        )


    score = hit_count / len(questions)

    mrr = sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )

    results[top_k] = {
        "hits": hit_count,
        "total": len(questions),
        "score": score,
        "mrr": mrr
    }


# ==========================================
# Summary
# ==========================================

print("\n" + "=" * 60)
print("Retrieval Summary")
print("=" * 60)

for top_k in top_k_values:

    result = results[top_k]

    print(
        f"Hit@{top_k}: "
        f"{result['hits']}/{result['total']} "
        f"({result['score'] * 100:.2f}%)"
    )

    print(
        f"MRR@{top_k}: "
        f"{result['mrr']:.4f}"
    )