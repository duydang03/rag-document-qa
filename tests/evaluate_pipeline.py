import importlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from app.ingestion.chunker import clean_text
from app.retrieval.vector_store import get_all_chunks


# ---------------------------------------------------------------
# Tìm module chứa hàm retrieve() (sửa danh sách nếu file của bạn
# nằm ở chỗ khác)
# ---------------------------------------------------------------
CANDIDATE_MODULES = [
    "app.rag_pipeline",
    "app.pipeline.rag_pipeline",
    "app.generation.rag_pipeline",
    "app.services.rag_pipeline",
    "app.core.rag_pipeline",
]

retrieve = None
for module_name in CANDIDATE_MODULES:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        continue
    if hasattr(module, "retrieve"):
        retrieve = module.retrieve
        print(f"Using retrieve() from {module_name}")
        break

if retrieve is None:
    raise ImportError(
        "Could not find retrieve(). Make sure rag_pipeline.py has been "
        "updated with the retrieve() function, and add its module path "
        "to CANDIDATE_MODULES in this script."
    )


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
SOURCE = "core.pdf"
GT_PATH = (
    PROJECT_ROOT / "tests" / "retrieval_evidence_ground_truth.json"
)
KS = [1, 3, 5, 8]


def norm(text):
    return re.sub(r"\s+", "", clean_text(text).lower())


# Chunk đúng được tính từ chính văn bản đang nằm trong vector DB.
chunks = get_all_chunks(source=SOURCE)
if not chunks:
    raise RuntimeError(
        f"No chunks found for source '{SOURCE}'. Run build_vector_db.py."
    )

norm_docs = [norm(c["document"]) for c in chunks]

with open(GT_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)

print(f"Chunks in DB: {len(chunks)} | Questions: {len(questions)}\n")

hits = {k: 0 for k in KS}
reciprocal = {k: 0.0 for k in KS}

for number, item in enumerate(questions, start=1):
    relevant = {
        i for i, doc in enumerate(norm_docs)
        if any(norm(e) in doc for e in item["evidence"])
    }

    if not relevant:
        raise ValueError(
            f"No chunk matches the evidence of question {number}: "
            f"{item['question']}"
        )

    ranked = [
        r["index"]
        for r in retrieve(item["question"], n_results=20, source=SOURCE)
    ]

    rank = next(
        (r for r, c in enumerate(ranked, start=1) if c in relevant),
        None,
    )

    print(
        f"[{number:>2}] rank={rank} relevant={sorted(relevant)} "
        f"top3={ranked[:3]}"
    )

    for k in KS:
        if rank is not None and rank <= k:
            hits[k] += 1
            reciprocal[k] += 1.0 / rank

total = len(questions)
print()
for k in KS:
    print(
        f"Hit@{k}: {hits[k] / total:.4f}   "
        f"MRR@{k}: {reciprocal[k] / total:.4f}"
    )