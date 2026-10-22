from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.retrieval.reranker import DocumentReranker


def main():
    reranker = DocumentReranker()

    question = "What is BDD100K?"

    documents = [
        "Drivable area segmentation divides a road scene into regions.",
        "BDD100K is a large-scale driving video dataset containing 100,000 videos.",
        "Lane markings have direction and continuity attributes."
    ]

    results = reranker.rerank(
        query=question,
        documents=documents,
        top_k=3
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result['score']:.4f}")
        print(result["document"])


if __name__ == "__main__":
    main()
