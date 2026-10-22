import re

from rank_bm25 import BM25Okapi


STOPWORDS = {
    "what", "are", "is", "was", "were", "the", "a", "an", "of", "in", "on",
    "to", "and", "or", "for", "by", "with", "as", "be", "how", "many",
    "does", "do", "did", "why", "which", "who", "when", "where", "this",
    "that", "it", "its", "from", "at", "can", "we", "our",
}


def tokenize(text):
    text = text.lower()

    tokens = re.findall(
        r"\b[\w]+\b",
        text
    )

    return [token for token in tokens if token not in STOPWORDS]


class BM25Retriever:

    def __init__(self, documents):
        self.documents = documents

        tokenized_documents = [
            tokenize(document)
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(self, query, n_results=5):

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        top_indices = ranked_indices[:n_results]

        results = []

        for index in top_indices:
            results.append({
                "index": index,
                "document": self.documents[index],
                "score": scores[index]
            })

        return results