
from app.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    """Combine BM25 and vector search using weighted RRF."""

    def __init__(
        self,
        documents,
        rrf_k=60,
        vector_weight=1.0,
        bm25_weight=4.0,
    ):
        self.documents = documents
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25 = BM25Retriever(documents)

    def search(self, query, vector_results, n_results=5, candidate_k=20):
        bm25_results = self.bm25.search(
            query,
            n_results=candidate_k,
        )

        scores = {}

        for rank, result in enumerate(
            vector_results[:candidate_k],
            start=1,
        ):
            index = result["index"]
            scores[index] = scores.get(index, 0.0) + (
                self.vector_weight / (self.rrf_k + rank)
            )

        for rank, result in enumerate(bm25_results, start=1):
            index = result["index"]
            scores[index] = scores.get(index, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )

        ranked = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )[:n_results]

        return [
            {
                "index": index,
                "document": self.documents[index],
                "score": scores[index],
            }
            for index in ranked
        ]
