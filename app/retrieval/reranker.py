
from sentence_transformers import CrossEncoder


class DocumentReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=3):
        if not documents:
            return []

        pairs = [
            (query, document)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        results = []
        for index, score in ranked[:top_k]:
            results.append({
                "index": index,
                "document": documents[index],
                "score": float(score)
            })

        return results
