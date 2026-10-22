
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DocumentReranker

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
CANDIDATE_K = 20
RRF_K = 60

documents = load_pdf(str(PDF_PATH))
chunks = split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
texts = [chunk.page_content for chunk in chunks]

bm25 = BM25Retriever(texts)
reranker = DocumentReranker()


def hybrid_search(question):
    candidate_k = min(CANDIDATE_K, len(chunks))

    bm25_results = bm25.search(
        question,
        n_results=candidate_k,
    )

    query_embedding = create_embeddings([question])[0]

    vector_raw = search(
        query_embedding=query_embedding,
        n_results=candidate_k,
        source=PDF_PATH.name,
    )

    scores = {}

    for rank, metadata in enumerate(
        vector_raw["metadatas"][0], start=1
    ):
        chunk_id = int(metadata["chunk_id"])
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["index"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


def rerank_candidates(question, hybrid_ids):
    candidate_ids = hybrid_ids[:CANDIDATE_K]
    candidate_texts = [texts[i] for i in candidate_ids]

    reranked = reranker.rerank(
        query=question,
        documents=candidate_texts,
        top_k=len(candidate_texts),
    )

    return [
        candidate_ids[item["index"]]
        for item in reranked
    ]


def show_results(label, ranked_ids, limit=5):
    print(f"\n--- {label} ---")

    for rank, chunk_id in enumerate(ranked_ids[:limit], start=1):
        page = chunks[chunk_id].metadata.get("page", 0) + 1
        content = " ".join(texts[chunk_id].split())

        print(
            f"\nRank {rank} | Chunk {chunk_id} | Page {page}"
        )
        print(content[:700])


if __name__ == "__main__":
    question = "How many videos does BDD100K contain?"

    hybrid_ids = hybrid_search(question)
    reranked_ids = rerank_candidates(question, hybrid_ids)

    print(f"Question: {question}")
    show_results("BEFORE RERANKING", hybrid_ids)
    show_results("AFTER RERANKING", reranked_ids)
