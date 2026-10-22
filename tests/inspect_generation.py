import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search
from app.generation.llm import generate_answer


# Load evaluation questions
with open(
    PROJECT_ROOT / "tests" / "ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    ground_truth = json.load(f)


for item in ground_truth:

    question = item["question"]

    print("=" * 70)
    print("QUESTION")
    print(question)

    # Create query embedding
    query_embedding = create_embeddings([question])[0]

    # Retrieve top 3 chunks
    results = search(
        query_embedding,
        n_results=8,
        source="core.pdf"
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Build context
    context = "\n\n".join(documents)

    # Generate answer
    answer = generate_answer(
        context=context,
        question=question
    )

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):

        print(f"\n--- Chunk {i} ---")
        print(
            f"Source: {metadata.get('source')}"
        )
        print(
            f"Page: {metadata.get('page', 0) + 1}"
        )

        print(document)

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print(answer)

    print()