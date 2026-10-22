import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents, clean_text
from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DocumentReranker


# ============================================================
# Configuration / CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Evaluate Hybrid Search, Hybrid + Reranking, and "
        "Hybrid/Reranker Reciprocal Rank Fusion."
    )
)
parser.add_argument("--candidate-k", type=int, default=10)
parser.add_argument(
    "--reranker-model",
    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
parser.add_argument("--verbose", action="store_true")
parser.add_argument(
    "--skip-vector-text-check",
    action="store_true",
    help="Skip comparing Chroma text with freshly generated chunks (not recommended).",
)
args = parser.parse_args()

if args.candidate_k < 1:
    parser.error("--candidate-k must be >= 1")

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"
GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "tests" / "retrieval_evidence_ground_truth.json"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_VALUES = [1, 3, 5, 8]
CANDIDATE_K = args.candidate_k
RRF_K = 60
MODEL_NAME = args.reranker_model

model_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", MODEL_NAME.split("/")[-1])
OUTPUT_PATH = (
    PROJECT_ROOT
    / "tests"
    / f"reranking_eval_k{CANDIDATE_K}_{model_tag}.json"
)


# ============================================================
# Normalization and ground-truth evidence matching
# ============================================================

def norm(text: str) -> str:
    """Normalize text consistently for evidence matching and DB checks."""
    return re.sub(r"\s+", "", clean_text(text).casefold())


# ============================================================
# Load data
# ============================================================

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

if not GROUND_TRUTH_PATH.exists():
    raise FileNotFoundError(
        f"Evidence ground-truth file not found: {GROUND_TRUTH_PATH}"
    )

with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)

if not isinstance(questions, list) or not questions:
    raise ValueError("Ground-truth dataset must be a non-empty JSON list.")

for index, item in enumerate(questions, start=1):
    if not isinstance(item, dict):
        raise ValueError(f"Ground-truth item {index} must be a JSON object.")
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        raise ValueError(f"Ground-truth item {index} has no valid question.")
    if not isinstance(item.get("evidence"), list) or not item["evidence"]:
        raise ValueError(
            f"Ground-truth item {index} must have a non-empty evidence list."
        )
    if item.get("match", "any") not in ("any", "all"):
        raise ValueError(
            f"Ground-truth item {index}: match must be 'any' or 'all'."
        )

documents = load_pdf(str(PDF_PATH))
chunks = split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
texts = [chunk.page_content for chunk in chunks]
norm_chunks = [norm(text) for text in texts]


def find_relevant_ids(item: dict[str, Any]) -> set[int]:
    """
    Return current chunk indices matching the evidence strings.
    'any': a chunk is relevant if it contains at least one evidence string.
    'all': a chunk is relevant if it contains every evidence string.
    """
    evidence = [norm(value) for value in item["evidence"] if isinstance(value, str)]
    evidence = [value for value in evidence if value]

    if not evidence:
        raise ValueError(
            f"Question has no usable evidence strings: {item['question']}"
        )

    match_mode = item.get("match", "any")
    relevant = set()

    for chunk_id, normalized_chunk in enumerate(norm_chunks):
        if match_mode == "any":
            matched = any(value in normalized_chunk for value in evidence)
        else:
            matched = all(value in normalized_chunk for value in evidence)

        if matched:
            relevant.add(chunk_id)

    return relevant


# Resolve evidence labels before running expensive models, so label problems
# are detected early.
ground_truth_chunks: list[set[int]] = []
for question_number, item in enumerate(questions, start=1):
    relevant = find_relevant_ids(item)
    if not relevant:
        raise ValueError(
            f"No chunk matches evidence for question {question_number}: "
            f"{item['question']}\n"
            f"Evidence: {item['evidence']}\n"
            "Check the evidence text or normalization."
        )
    ground_truth_chunks.append(relevant)


# ============================================================
# Initialize retrievers
# ============================================================

bm25 = BM25Retriever(texts)
reranker = DocumentReranker(model_name=MODEL_NAME)


# ============================================================
# Hybrid Search: BM25 + Vector Search + RRF
# ============================================================

def hybrid_search(question: str) -> tuple[list[int], list[int], list[int]]:
    candidate_k = min(CANDIDATE_K, len(chunks))

    bm25_results = bm25.search(question, n_results=candidate_k)

    query_embedding = create_embeddings([question])[0]
    vector_raw = search(
        query_embedding=query_embedding,
        n_results=candidate_k,
        source=PDF_PATH.name,
    )

    metadatas = (vector_raw.get("metadatas") or [[]])[0]
    vector_docs = (vector_raw.get("documents") or [[]])[0]

    if not metadatas:
        raise ValueError(
            f"Vector search returned no results for question: {question}"
        )

    scores: dict[int, float] = {}
    vector_ids: list[int] = []

    for rank, metadata in enumerate(metadatas, start=1):
        raw_chunk_id = metadata.get("chunk_id")
        if raw_chunk_id is None:
            raise ValueError(
                "A vector result has no chunk_id metadata. "
                "Rebuild the vector database with chunk_id metadata."
            )

        try:
            chunk_id = int(raw_chunk_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid chunk_id in vector metadata: {raw_chunk_id!r}"
            ) from exc

        if not 0 <= chunk_id < len(chunks):
            raise ValueError(
                f"Vector DB chunk_id {chunk_id} is outside current range "
                f"0..{len(chunks) - 1}. Rebuild the vector database."
            )

        # Verify the vector DB contains the same chunk text as this run.
        if (
            not args.skip_vector_text_check
            and rank - 1 < len(vector_docs)
            and norm(vector_docs[rank - 1]) != norm_chunks[chunk_id]
        ):
            raise ValueError(
                f"Vector DB text for chunk_id {chunk_id} differs from the "
                "current chunker output. Rebuild the vector database after "
                "changing chunker.py. Do not use --skip-vector-text-check "
                "unless you understand this mismatch."
            )

        vector_ids.append(chunk_id)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    bm25_ids: list[int] = []
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = int(result["index"])

        if not 0 <= chunk_id < len(chunks):
            raise ValueError(
                f"BM25 index {chunk_id} is outside current chunk range."
            )

        bm25_ids.append(chunk_id)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (RRF_K + rank)
        )

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return ranked_ids, bm25_ids, vector_ids


# ============================================================
# Reranking
# ============================================================

def rerank_candidates(
    question: str,
    hybrid_ids: list[int],
) -> tuple[list[int], list[int], dict[int, float]]:
    candidate_ids = hybrid_ids[: min(CANDIDATE_K, len(hybrid_ids))]

    if not candidate_ids:
        return [], [], {}

    candidate_texts = [texts[chunk_id] for chunk_id in candidate_ids]

    reranked = reranker.rerank(
        query=question,
        documents=candidate_texts,
        top_k=len(candidate_texts),
    )

    reranked_ids = [
        candidate_ids[item["index"]]
        for item in reranked
    ]
    score_by_id = {
        candidate_ids[item["index"]]: float(item["score"])
        for item in reranked
    }

    return candidate_ids, reranked_ids, score_by_id


# ============================================================
# Rank fusion: preserve Hybrid candidates and blend ranks
# ============================================================

def fuse(
    hybrid_ids: list[int],
    reranked_ids: list[int],
    k: int = 60,
    w_h: float = 1.0,
) -> list[int]:
    """
    Weighted Reciprocal Rank Fusion.

    Only candidates present in both lists are scored with both ranks.
    Hybrid-only candidates are retained at the end in Hybrid order.
    """
    hybrid_rank = {
        chunk_id: rank
        for rank, chunk_id in enumerate(hybrid_ids, start=1)
    }
    rerank_rank = {
        chunk_id: rank
        for rank, chunk_id in enumerate(reranked_ids, start=1)
    }

    scores: dict[int, float] = {}

    for chunk_id, rank_r in rerank_rank.items():
        rank_h = hybrid_rank.get(chunk_id)
        if rank_h is None:
            continue

        scores[chunk_id] = (
            w_h / (k + rank_h)
            + 1.0 / (k + rank_r)
        )

    head = sorted(scores, key=scores.get, reverse=True)
    tail = [chunk_id for chunk_id in hybrid_ids if chunk_id not in scores]

    return head + tail


# ============================================================
# Metrics
# ============================================================

def rank_of(ids: list[int], relevant: set[int]) -> int | None:
    return next(
        (rank for rank, chunk_id in enumerate(ids, start=1)
         if chunk_id in relevant),
        None,
    )


def evaluate_rankings(
    rankings: list[list[int]],
    ground_truth: list[set[int]],
    ks: list[int],
) -> dict[str, float]:
    if not rankings or len(rankings) != len(ground_truth):
        raise ValueError(
            "Rankings and ground truth must have the same non-zero length."
        )

    metrics: dict[str, float] = {}

    for k in ks:
        hits = 0
        reciprocal_ranks: list[float] = []

        for ranked_ids, relevant_ids in zip(rankings, ground_truth):
            first_relevant_rank = rank_of(ranked_ids[:k], relevant_ids)

            if first_relevant_rank is None:
                reciprocal_ranks.append(0.0)
            else:
                hits += 1
                reciprocal_ranks.append(1.0 / first_relevant_rank)

        metrics[f"Hit@{k}"] = hits / len(ground_truth)
        metrics[f"MRR@{k}"] = sum(reciprocal_ranks) / len(ground_truth)

    return metrics


def chunk_candidate_recall(
    candidate_sets: list[list[int]],
    ground_truth: list[set[int]],
) -> float:
    total_relevant = sum(len(relevant) for relevant in ground_truth)
    if total_relevant == 0:
        return 0.0

    found_relevant = sum(
        len(set(candidates) & relevant)
        for candidates, relevant in zip(candidate_sets, ground_truth)
    )
    return found_relevant / total_relevant


# ============================================================
# Run evaluation
# ============================================================

hybrid_rankings: list[list[int]] = []
reranked_rankings: list[list[int]] = []
fused_rankings_by_weight: dict[float, list[list[int]]] = {
    weight: [] for weight in (1.0, 2.0, 3.0)
}
candidate_sets: list[list[int]] = []
per_question: list[dict[str, Any]] = []

print("=" * 72)
print("HYBRID SEARCH VS RERANKING VS WEIGHTED RANK FUSION")
print("=" * 72)
print(f"PDF: {PDF_PATH.name}")
print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")
print(f"Questions: {len(questions)}")
print(f"Chunk size / overlap: {CHUNK_SIZE} / {CHUNK_OVERLAP}")
print(f"Candidate K: {CANDIDATE_K}")
print(f"RRF K: {RRF_K}")
print(f"Reranker: {MODEL_NAME}")
print("Evidence matching: normalized substring matching")

for number, (item, relevant_ids) in enumerate(
    zip(questions, ground_truth_chunks),
    start=1,
):
    question = item["question"]

    hybrid_ids, bm25_ids, vector_ids = hybrid_search(question)
    candidate_ids, reranked_ids, score_by_id = rerank_candidates(
        question, hybrid_ids
    )

    if not candidate_ids:
        raise ValueError(f"No candidates retrieved for question: {question}")

    hybrid_rankings.append(hybrid_ids)
    reranked_rankings.append(reranked_ids)
    candidate_sets.append(candidate_ids)

    for weight in fused_rankings_by_weight:
        fused_rankings_by_weight[weight].append(
            fuse(
                hybrid_ids,
                reranked_ids,
                k=RRF_K,
                w_h=weight,
            )
        )

    relevant_scores = [
        score_by_id[chunk_id]
        for chunk_id in candidate_ids
        if chunk_id in relevant_ids and chunk_id in score_by_id
    ]

    candidate_question_hit = bool(set(candidate_ids) & relevant_ids)

    record = {
        "question": question,
        "match_mode": item.get("match", "any"),
        "relevant_chunk_ids": sorted(relevant_ids),
        "num_relevant_chunks": len(relevant_ids),
        "candidate_question_hit": candidate_question_hit,
        "hybrid_first_relevant_rank": rank_of(hybrid_ids, relevant_ids),
        "reranked_first_relevant_rank": rank_of(reranked_ids, relevant_ids),
        "fused_first_relevant_rank": {
            str(weight): rank_of(
                fused_rankings_by_weight[weight][-1], relevant_ids
            )
            for weight in fused_rankings_by_weight
        },
        "hybrid_top5_chunk_ids": hybrid_ids[:5],
        "reranked_top5_chunk_ids": reranked_ids[:5],
        "reranker_top_chunk_id": reranked_ids[0],
        "reranker_top_score": round(score_by_id[reranked_ids[0]], 4),
        "best_relevant_reranker_score": (
            round(max(relevant_scores), 4) if relevant_scores else None
        ),
        "candidate_relevant_chunk_ids": sorted(
            set(candidate_ids) & relevant_ids
        ),
        "bm25_top5_chunk_ids": bm25_ids[:5],
        "vector_top5_chunk_ids": vector_ids[:5],
    }
    per_question.append(record)

    print(
        f"[{number:>2}/{len(questions)}] "
        f"relevant={len(relevant_ids):>2} "
        f"candidate_hit={str(candidate_question_hit):<5} "
        f"hybrid_rank={str(record['hybrid_first_relevant_rank']):>4} "
        f"rerank_rank={str(record['reranked_first_relevant_rank']):>4} "
        f"top={record['reranker_top_chunk_id']} "
        f"score={record['reranker_top_score']}"
    )

    if args.verbose:
        print(f"    Question: {question}")
        print(f"    Relevant IDs: {sorted(relevant_ids)}")
        print(f"    Hybrid top 10: {hybrid_ids[:10]}")
        print(f"    Reranked top 10: {reranked_ids[:10]}")
        print(f"    Evidence: {item['evidence']}")


# ============================================================
# Calculate and print results
# ============================================================

hybrid_metrics = evaluate_rankings(
    hybrid_rankings, ground_truth_chunks, TOP_K_VALUES
)
reranked_metrics = evaluate_rankings(
    reranked_rankings, ground_truth_chunks, TOP_K_VALUES
)

fusion_metrics = {
    weight: evaluate_rankings(
        rankings, ground_truth_chunks, TOP_K_VALUES
    )
    for weight, rankings in fused_rankings_by_weight.items()
}

candidate_recall = chunk_candidate_recall(
    candidate_sets, ground_truth_chunks
)
candidate_question_hit_rate = sum(
    record["candidate_question_hit"] for record in per_question
) / len(per_question)

print("\n" + "=" * 72)
print("FINAL COMPARISON")
print("=" * 72)
print(
    f"{'Metric':<10}"
    f"{'Hybrid':>12}"
    f"{'Reranked':>12}"
    f"{'Fusion w=1':>14}"
    f"{'Fusion w=2':>14}"
    f"{'Fusion w=3':>14}"
)

for k in TOP_K_VALUES:
    for metric_name in (f"Hit@{k}", f"MRR@{k}"):
        values = [
            hybrid_metrics[metric_name],
            reranked_metrics[metric_name],
            fusion_metrics[1.0][metric_name],
            fusion_metrics[2.0][metric_name],
            fusion_metrics[3.0][metric_name],
        ]
        print(
            f"{metric_name:<10}"
            + "".join(f"{value:>14.4f}" for value in values)
        )

print(f"\nCandidate chunk recall@{CANDIDATE_K}: {candidate_recall:.4f}")
print(
    "Candidate question hit rate: "
    f"{candidate_question_hit_rate:.4f}"
)

# ============================================================
# Save results
# ============================================================

output = {
    "pdf": PDF_PATH.name,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "num_pages": len(documents),
    "num_chunks": len(chunks),
    "candidate_k": CANDIDATE_K,
    "reranker_model": MODEL_NAME,
    "rrf_k": RRF_K,
    "num_questions": len(questions),
    "evidence_matching": "normalized_substring",
    "candidate_chunk_recall": candidate_recall,
    "candidate_question_hit_rate": candidate_question_hit_rate,
    "top_k_values": TOP_K_VALUES,
    "methods": {
        "hybrid": hybrid_metrics,
        "hybrid_reranking": reranked_metrics,
        "hybrid_rrf_rerank": {
            f"hybrid_weight_{weight}": metrics
            for weight, metrics in fusion_metrics.items()
        },
    },
    "per_question": per_question,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(output, file, indent=4, ensure_ascii=False)

print(f"\nResults saved to: {OUTPUT_PATH}")
