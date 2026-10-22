import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
import json

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search


# ==============================
# Load evaluation questions
# ==============================

with open(
    "tests/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


# ==============================
# Inspect Top-3 retrieval
# ==============================

for item in questions:

    question = item["question"]

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    # Create query embedding
    query_embedding = create_embeddings(
        [question]
    )[0]

    # Retrieve Top-3 chunks
    results = search(
        query_embedding,
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ==============================
    # Display retrieved chunks
    # ==============================

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print("\n" + "-" * 80)

        print(f"Rank: {rank}")

        print(
            f"Source: "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page', 0) + 1}"
        )

        print(
            f"Distance: "
            f"{distance:.4f}"
        )

        print("\nChunk:")

        print(document[:1500])