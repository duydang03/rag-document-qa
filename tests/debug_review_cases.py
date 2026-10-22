import importlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from app.ingestion.chunker import clean_text

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

SOURCE = "core.pdf"

# (câu hỏi, cụm văn bản chứng tỏ chunk chứa đáp án)
CASES = [
    (
        "How were the driving videos collected?",
        ["crowd-sourcing manner uploaded by tens of thousands of drivers"],
    ),
    (
        "What is an example of the long-tail distribution in semantic "
        "instance segmentation?",
        ["almost 60 thousand car instances"],
    ),
]


def norm(text):
    return re.sub(r"\s+", "", clean_text(text).lower())


for question, key_phrases in CASES:
    print("=" * 78)
    print(question)
    print("=" * 78)

    results = retrieve(question, n_results=8, source=SOURCE)

    for rank, item in enumerate(results, start=1):
        has_key = any(
            norm(phrase) in norm(item["document"]) for phrase in key_phrases
        )
        page = item["metadata"].get("page")
        page_number = page + 1 if isinstance(page, int) else "?"
        marker = "  <== CHỨA ĐÁP ÁN" if has_key else ""
        preview = item["document"][:110].replace("\n", " ")
        print(
            f"{rank}. chunk={item['index']:>3} page={page_number} "
            f"| {preview}{marker}"
        )
    print()
