from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import add_documents, search

texts = [
    "BDD100K is a large-scale driving dataset.",
    "Lane segmentation is an important task for autonomous driving.",
    "DeepLabV3+ is a semantic segmentation model."
]


embeddings = create_embeddings(texts)

print("Number of embeddings:", len(embeddings))

print("Embedding dimension:", len(embeddings[0]))

add_documents(texts, embeddings)
print("Documents added to the vector store.")

query = "What is lane segmentation?"
query_embedding = create_embeddings([query])[0]

results = search(query_embedding, n_result=2)
print("\nSearch results:")
for document in results['documents'][0]:
    print("-",document)