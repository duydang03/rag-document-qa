from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search

query ="What is BDD100K?"

query_embedding = create_embeddings([query])[0]

results = search(
    query_embedding=query_embedding,
    n_results=3
)

documents= results['documents'][0]
metadatas= results['metadatas'][0]

print("Question:")
print(query)

print("\nRetrieved Documents:")

for i,(document, metadata) in enumerate(zip(documents, metadatas),start=1):
    print (f"\n---Document {i} ---")
    print ("Content:")
    print(document[:300])
    print("\nMetadata:")
    print(metadata) 