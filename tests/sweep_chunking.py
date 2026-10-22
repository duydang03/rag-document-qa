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
from build_vector_db import build_vector_db

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
        break

if retrieve is None:
    raise ImportError("Could not find retrieve(). Update CANDIDATE_MODULES.")

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
SOURCE = "core.pdf"
GT_PATH = PROJECT_ROOT / "tests" / "retrieval_evidence_ground_truth.json"
FINAL_CONFIG = (800, 200)  # cấu hình được dựng lại ở cuối

CONFIGS = [
    (800, 100),
    (800, 150),
    (800, 200),
    (800, 250),
    (800, 300),
]
KS = [1, 3, 5]


def norm(text):
    return re.sub(r"\s+", "", clean_text(text).lower())


with open(GT_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)

rows = []

try:
    for chunk_size, overlap in CONFIGS:
        print(f"\n##### Building {chunk_size}/{overlap} #####")
        build_vector_db(chunk_size=chunk_size, chunk_overlap=overlap)

        chunks = get_all_chunks(source=SOURCE)
        norm_docs = [norm(c["document"]) for c in chunks]

        hits = {k: 0 for k in KS}
        mrr5 = 0.0
        evaluated = 0
        skipped = []

        for number, item in enumerate(questions, start=1):
            relevant = {
                i for i, doc in enumerate(norm_docs)
                if any(norm(e) in doc for e in item["evidence"])
            }

            # Cụm evidence có thể bị cắt đôi ở chunk nhỏ: bỏ qua và báo lại.
            if not relevant:
                skipped.append(number)
                continue

            ranked = [
                r["index"]
                for r in retrieve(item["question"], n_results=20, source=SOURCE)
            ]
            rank = next(
                (r for r, c in enumerate(ranked, start=1) if c in relevant),
                None,
            )

            evaluated += 1
            for k in KS:
                if rank is not None and rank <= k:
                    hits[k] += 1
            if rank is not None and rank <= 5:
                mrr5 += 1.0 / rank

        rows.append({
            "config": f"{chunk_size}/{overlap}",
            "chunks": len(chunks),
            "evaluated": evaluated,
            "skipped": skipped,
            "Hit@1": hits[1] / evaluated if evaluated else 0.0,
            "Hit@3": hits[3] / evaluated if evaluated else 0.0,
            "Hit@5": hits[5] / evaluated if evaluated else 0.0,
            "MRR@5": mrr5 / evaluated if evaluated else 0.0,
        })
finally:
    print(f"\n##### Restoring {FINAL_CONFIG[0]}/{FINAL_CONFIG[1]} #####")
    build_vector_db(chunk_size=FINAL_CONFIG[0], chunk_overlap=FINAL_CONFIG[1])

print("\n" + "=" * 78)
print(
    f"{'Config':<12}{'Chunks':>8}{'Eval':>6}"
    f"{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}{'MRR@5':>9}  Skipped"
)
for r in rows:
    print(
        f"{r['config']:<12}{r['chunks']:>8}{r['evaluated']:>6}"
        f"{r['Hit@1']:>9.4f}{r['Hit@3']:>9.4f}{r['Hit@5']:>9.4f}"
        f"{r['MRR@5']:>9.4f}  {r['skipped'] or '-'}"
    )

out_path = PROJECT_ROOT / "tests" / "chunking_sweep_results.json"
with open(out_path, "w", encoding="utf-8") as file:
    json.dump(rows, file, indent=4, ensure_ascii=False)
print(f"\nSaved: {out_path}")