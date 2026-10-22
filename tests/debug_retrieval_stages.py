from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search, get_all_chunks
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DocumentReranker
from app.retrieval.query_transform import rewrite_query
QUESTIONS = [
    (
        "What are the weather categories represented in BDD100K?",
        [74],
    ),
    (
        "What are the scene categories represented in BDD100K?",
        [18],
    ),
    (
        "What are the time-of-day categories represented in BDD100K?",
        [74],
    ),
]

chunks = get_all_chunks(source="core.pdf")
documents = [item["document"] for item in chunks]
metadatas = [item["metadata"] for item in chunks]

bm25 = BM25Retriever(documents)
candidate_k = len(chunks)
def print_chunk_content(target_chunk_ids, documents, metadatas):
    for target_id in target_chunk_ids:
        print(f"\n{'=' * 25} CHUNK {target_id} {'=' * 25}")

        found = False

        for index, metadata in enumerate(metadatas):
            if metadata.get("chunk_id") == target_id:
                print(f"Page: {metadata.get('page')}")
                print(documents[index])
                found = True
                break

        if not found:
            print(f"Không tìm thấy chunk_id={target_id}")

def describe(item):
    metadata = item["metadata"]
    return (
        f"chunk_id={metadata.get('chunk_id')}, "
        f"page={metadata.get('page')}"
    )

reranker = DocumentReranker()

# In nội dung các chunk cần điều tra
print_chunk_content(
    target_chunk_ids=[69, 74, 18],
    documents=documents,
    metadatas=metadatas,
)
for question, target_chunk_ids in QUESTIONS:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("TARGET CHUNKS:", target_chunk_ids)
    
    retrieval_query = rewrite_query(question)

    print("Original query:", question)
    print("Retrieval query:", retrieval_query)

    # Use the expanded query for vector search.
    embedding = create_embeddings([retrieval_query])[0]

    # 1. Vector search
    vector_data = search(
        query_embedding=embedding,
        n_results=candidate_k,
        source="core.pdf",
    )


    vector_documents = vector_data["documents"][0]
    vector_metadatas = vector_data["metadatas"][0]

    vector_results = []
    for document, metadata in zip(
        vector_documents, vector_metadatas
    ):
        vector_results.append({
            "document": document,
            "metadata": metadata,
        })

    print("\n--- VECTOR TOP 20 ---")
    for rank, item in enumerate(vector_results, start=1):
        print(rank, describe(item))

    # 2. BM25
    bm25_results = bm25.search(
        retrieval_query,
        n_results=candidate_k,
    )

    
    print("\n--- BM25 TOP 20 ---")
    for rank, item in enumerate(bm25_results, start=1):
        print(rank, describe({
            "metadata": metadatas[item["index"]]
        }))

    # 3. Weighted Reciprocal Rank Fusion (RRF)
    
    # 3. Reciprocal Rank Fusion (RRF) và Weighted RRF
    index_by_key = {}
    for index, metadata in enumerate(metadatas):
        key = (
            metadata.get("source", ""),
            metadata.get("chunk_id"),
        )
        index_by_key[key] = index
    
    rrf_k = 60
    BM25_WEIGHT = 4.0

    baseline_scores = {}
    weighted_scores = {}

    # Vector Search: đóng góp vào cả hai phương pháp
    for rank, item in enumerate(vector_results, start=1):
        metadata = item["metadata"]
        key = (
            metadata.get("source", ""),
            metadata.get("chunk_id"),
        )
        index = index_by_key.get(key)

        if index is not None:
            score = 1.0 / (rrf_k + rank)

            baseline_scores[index] = (
                baseline_scores.get(index, 0) + score
            )
            weighted_scores[index] = (
                weighted_scores.get(index, 0) + score
            )

    # BM25: trọng số 1.0 cho baseline
    # và trọng số BM25_WEIGHT cho Weighted RRF
    for rank, result in enumerate(bm25_results, start=1):
        index = result["index"]
        score = 1.0 / (rrf_k + rank)

        baseline_scores[index] = (
            baseline_scores.get(index, 0) + score
        )
        weighted_scores[index] = (
            weighted_scores.get(index, 0)
            + BM25_WEIGHT * score
        )

    baseline_indices = sorted(
        baseline_scores,
        key=baseline_scores.get,
        reverse=True,
    )

    ranked_indices = sorted(
        weighted_scores,
        key=weighted_scores.get,
        reverse=True,
    )

    print("\n--- BASELINE RRF TOP 3 ---")
    for rank, index in enumerate(baseline_indices[:3], start=1):
        metadata = metadatas[index]
        print(
            f"{rank}. chunk_id={metadata.get('chunk_id')}, "
            f"page={metadata.get('page')}, "
            f"score={baseline_scores[index]:.6f}"
        )

    print("\n--- WEIGHTED RRF TOP 3 ---")
    for rank, index in enumerate(ranked_indices[:3], start=1):
        metadata = metadatas[index]
        print(
            f"{rank}. chunk_id={metadata.get('chunk_id')}, "
            f"page={metadata.get('page')}, "
            f"score={weighted_scores[index]:.6f}"
        )

    # 4. Rerank the top 20 RRF candidates
    candidate_indices = ranked_indices[:20]
    candidate_documents = [
        documents[index] for index in candidate_indices
    ]

    reranked = reranker.rerank(
        query=question,
        documents=candidate_documents,
        top_k=len(candidate_documents),
    )

    print("\n--- RERANK TOP 20 ---")

    for rank, result in enumerate(reranked, start=1):
        original_index = candidate_indices[result["index"]]
        metadata = metadatas[original_index]

        print(
            rank,
            f"chunk_id={metadata.get('chunk_id')}, "
            f"page={metadata.get('page')}, "
            f"score={result['score']:.4f}",
        )

    for target_id in target_chunk_ids:
        target_indices = [
            i for i, metadata in enumerate(metadatas)
            if metadata.get("chunk_id") == target_id
        ]

        for target_index in target_indices:
            print(f"\nTARGET chunk_id={target_id}")
            print(
                "Vector rank:",
                next(
                    (
                        rank for rank, item
                        in enumerate(vector_results, start=1)
                        if (
                            item["metadata"].get("source", ""),
                            item["metadata"].get("chunk_id"),
                        ) == (
                            metadatas[target_index].get("source", ""),
                            target_id,
                        )
                    ),
                    "NOT FOUND",
                ),
            )
            print(
                "BM25 rank:",
                next(
                    (
                        rank for rank, result
                        in enumerate(bm25_results, start=1)
                        if result["index"] == target_index
                    ),
                    "NOT FOUND",
                ),
            )
            print(
                "RRF rank:",
                next(
                    (
                        rank for rank, index
                        in enumerate(ranked_indices, start=1)
                        if index == target_index
                    ),
                    "NOT FOUND",
                ),
            )
            print(
                "Reranker rank:",
                next(
                    (
                        rank
                        for rank, result in enumerate(reranked, start=1)
                        if candidate_indices[result["index"]] == target_index
                    ),
                    "NOT FOUND",
                ),
            )
