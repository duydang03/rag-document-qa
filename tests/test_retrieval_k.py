import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search


questions = [
    "What are the scene categories represented in BDD100K?",
    "What are the time-of-day categories represented in BDD100K?",
    "What are the weather categories represented in BDD100K?"
]


for question in questions:

    print("=" * 80)
    print("QUESTION:")
    print(question)
    print("=" * 80)

    query_embedding = create_embeddings([question])[0]

    results = search(
        query_embedding,
        n_results=15,
        source="core.pdf"
    )

    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for i, (metadata, document) in enumerate(
        zip(metadatas, documents),
        start=1
    ):
        page = metadata.get("page", 0) + 1
        distance = results["distances"][0][i - 1]

        print()
        print(f"Rank: {i}")
        print(f"Page: {page}")
        print(f"Distance: {distance}")
        print("-" * 60)
        print(document[:500])