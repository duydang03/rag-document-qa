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


for item in ground_truth:

    question = item["question"]
    relevant_pages = set(item["relevant_pages"])

    query_embedding = create_embeddings([question])[0]

    results = search(
        query_embedding,
        n_results=15,
        source="core.pdf"
    )

    retrieved_pages = [
        metadata.get("page", 0) + 1
        for metadata in results["metadatas"][0]
    ]

    hit = any(
        page in relevant_pages
        for page in retrieved_pages
    )

    if not hit:
        print("=" * 80)
        print("RETRIEVAL FAILURE")
        print("=" * 80)

        print("Question:")
        print(question)

        print("\nRelevant pages:")
        print(sorted(relevant_pages))

        print("\nRetrieved pages:")
        print(retrieved_pages)

        print()