
import sys
from pathlib import Path

# Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import add_documents, search

texts = [
    "BDD100K is a large-scale driving dataset.",
    "Lane segmentation is an important task for autonomous driving.",
    "DeepLabV3+ is a semantic segmentation model."
]

# Tạo embeddings
embeddings = create_embeddings(texts)

print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))

# Metadata tương ứng với từng văn bản
metadata = [
    {"source": "test_embedding.pdf", "page": 1, "chunk_id": 0},
    {"source": "test_embedding.pdf", "page": 1, "chunk_id": 1},
    {"source": "test_embedding.pdf", "page": 1, "chunk_id": 2},
]

# Lưu vào vector store
add_documents(texts, embeddings, metadata)
print("Documents added to the vector store.")

# Kiểm tra truy xuất
query = "What is lane segmentation?"
query_embedding = create_embeddings([query])[0]

results = search(query_embedding, n_results=2)

print("\nSearch results:")
for document in results["documents"][0]:
    print("-", document)
