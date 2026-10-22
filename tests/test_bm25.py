from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.bm25 import BM25Retriever
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents


PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"


documents = load_pdf(str(PDF_PATH))

chunks = split_documents(
    documents,
    chunk_size=800,
    chunk_overlap=200
)

texts = [
    chunk.page_content
    for chunk in chunks
]

print("=" * 60)
print("BM25 TEST")
print("=" * 60)

print(f"Number of chunks: {len(texts)}")

bm25 = BM25Retriever(texts)

query = "What is BDD100K?"

results = bm25.search(
    query,
    n_results=5
)

print("\nQuery:")
print(query)

print("\nTop 5 BM25 results:")
print("-" * 60)

for rank, result in enumerate(results, start=1):

    print(f"\nRank {rank}")
    print(f"Chunk index: {result['index']}")
    print(f"Score: {result['score']:.4f}")

    print("\nText:")
    print(result["document"][:500])