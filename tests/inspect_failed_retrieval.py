import sys
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search


QUESTIONS = [
    "What are the scene categories represented in BDD100K?",
    "What are the time-of-day categories represented in BDD100K?"
]


for question in QUESTIONS:

    print("=" * 100)
    print("QUESTION:")
    print(question)
    print("=" * 100)

    query_embedding = create_embeddings([question])[0]

    results = search(
        query_embedding,
        n_results=15,
        source="core.pdf"
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        page = metadata.get("page", 0) + 1

        print()
        print(f"Rank {rank}")
        print(f"Page: {page}")
        print(f"Distance: {distance:.4f}")
        print("-" * 80)
        print(document[:1000])
        print("-" * 80)