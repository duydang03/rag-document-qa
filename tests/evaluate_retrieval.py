import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import json

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search


# ==============================
# Load evaluation data
# ==============================

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


with open(
    "tests/ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    ground_truth = json.load(f)


# ==============================
# Build ground truth map
# ==============================

ground_truth_map = {
    item["question"]: {
        "answer": item["answer"],
        "keywords": item["keywords"]
    }
    for item in ground_truth
}


# ==============================
# Top-K values to benchmark
# ==============================

top_k_values = [1, 3, 5, 8]


results = {}


# ==============================
# Evaluate each Top-K
# ==============================

for top_k in top_k_values:

    success_count = 0

    print("\n================================")
    print(f"Retrieval Evaluation - Top-{top_k}")
    print("================================")

    for item in questions:

        question = item["question"]

        ground_truth_item = ground_truth_map.get(
            question,
            {}
        )

        keywords = ground_truth_item.get(
            "keywords",
            []
        )

        # Create query embedding
        query_embedding = create_embeddings(
            [question]
        )[0]

        # Retrieve Top-K chunks
        retrieved = search(
            query_embedding,
            n_results=top_k
        )

        documents = retrieved["documents"][0]

        # Check whether any keyword appears
        # in any retrieved chunk
        success = False

        for document in documents:

            document_lower = document.lower()

            for keyword in keywords:

                if keyword.lower() in document_lower:
                    success = True
                    break

            if success:
                break

        if success:
            success_count += 1

        print(
            f"{question}"
        )

        print(
            f"Top-{top_k}: "
            f"{'YES' if success else 'NO'}"
        )

    score = success_count / len(questions)

    results[top_k] = {
        "success": success_count,
        "total": len(questions),
        "score": score
    }


# ==============================
# Summary
# ==============================

print("\n================================")
print("Top-K Benchmark Summary")
print("================================")

for top_k in top_k_values:

    result = results[top_k]

    print(
        f"Top-{top_k}: "
        f"{result['success']}/{result['total']} "
        f"({result['score'] * 100:.2f}%)"
    )