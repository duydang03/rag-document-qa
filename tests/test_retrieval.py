import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search


query = "What is BDD100K?"


# Create embedding for the question
query_embedding = create_embeddings([query])[0]


# Search ChromaDB
results = search(query_embedding, n_results=3)


print("Question:")
print(query)

print("\nRetrieved documents:")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\n--- Document {i} ---")
    print(document)