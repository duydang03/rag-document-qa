from app.ingestion.embedder import create_embeddings
from app.generation.llm import generate_answer
from app.retrieval.vector_store import search, get_all_chunks
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_transform import rewrite_query
CANDIDATE_K = 88
DEFAULT_TOP_K = 8
DEBUG = True


def retrieve(question: str, n_results: int = DEFAULT_TOP_K, source: str = None):
    """Hybrid retrieval (vector + BM25 + RRF) with query expansion."""
    chunks = get_all_chunks(source=source)

    if not chunks:
        return []

    documents = [item["document"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]

    candidate_k = min(max(CANDIDATE_K, n_results), len(chunks))

    # Expand the query for retrieval only.
    retrieval_query = rewrite_query(question)

    question_embedding = create_embeddings([retrieval_query])[0]

    vector_data = search(
        query_embedding=question_embedding,
        n_results=candidate_k,
        source=source,
    )

    index_by_key = {
        (m.get("source", ""), m.get("chunk_id")): i
        for i, m in enumerate(metadatas)
    }

    vector_results = []
    for document, metadata in zip(
        vector_data["documents"][0],
        vector_data["metadatas"][0],
    ):
        key = (metadata.get("source", ""), metadata.get("chunk_id"))
        index = index_by_key.get(key)
        if index is not None:
            vector_results.append({"index": index, "document": document})

    hybrid = HybridRetriever(
        documents,
        rrf_k=60,
        vector_weight=1.0,
        bm25_weight=2.0,
    )
    results = hybrid.search(
        retrieval_query,
        vector_results,
        n_results=n_results,
        candidate_k=candidate_k,
    )

    return [
        {
            "index": r["index"],
            "document": documents[r["index"]],
            "metadata": metadatas[r["index"]],
            "score": r["score"],
        }
        for r in results
    ]



def answer_question(
    question: str,
    n_results: int = DEFAULT_TOP_K,
    source: str = None,
):
    results = retrieve(question, n_results=n_results, source=source)

    if not results:
        return "I don't know based on the provided document.", []

    context_parts = []
    sources_by_key = {}

    for item in results:
        metadata = item["metadata"]
        source_name = metadata.get("source", "Unknown source")
        page = metadata.get("page")

        try:
            page_number = int(page) + 1 if page is not None else "Unknown"
        except (TypeError, ValueError):
            page_number = "Unknown"

        context_parts.append(
            f"Source: {source_name}\n"
            f"Page: {page_number}\n"
            f"Content:\n{item['document']}"
        )

        key = (source_name, page_number)

        if key in sources_by_key:
            sources_by_key[key]["content"] += "\n\n" + item["document"]
        else:
            sources_by_key[key] = {
                "source": source_name,
                "page": page_number,
                "content": item["document"],
            }

    sources = list(sources_by_key.values())
    context = "\n\n---\n\n".join(context_parts)

    # if DEBUG:
    #     # print("\n" + "=" * 80)
    #     # print("DEBUG QUESTION:", question)
    #     # print("=" * 80)
    #     # print(context)
    #     # print("=" * 80 + "\n")

    answer = generate_answer(context=context, question=question)

    return answer, sources