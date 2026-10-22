from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")
CASES = ROOT / "tests" / "rag_quality_cases.json"
REPORT_DIR = ROOT / "tests" / "reports"

ABSTAIN = [
    "i don't know based on the provided document",
    "not mentioned in the document",
    "does not provide",
    "cannot determine from the provided document",
    "the provided document does not",
    "không có thông tin trong tài liệu",
    "không thể xác định từ tài liệu",
    "tài liệu không đề cập",
]


def norm(x):
    return re.sub(r"\s+", " ", str(x or "")).strip().lower()


def term_found(term, answer_norm, aliases):
    """True nếu term hoặc một alias của nó xuất hiện trong câu trả lời."""
    candidates = [term] + list(aliases.get(term, []))
    return any(norm(c) in answer_norm for c in candidates)

def normalize_text(text: str) -> str:
    return " ".join(str(text or "").casefold().split())

def find_missing_terms(answer: str, required_terms: list[str]) -> list[str]:
    normalized_answer = normalize_text(answer)

    return [
        term
        for term in required_terms
        if normalize_text(term) not in normalized_answer
    ]
def get_pages(sources):
    """Lấy số trang hiển thị từ danh sách sources."""
    pages = set()

    if not isinstance(sources, (list, tuple)):
        return pages

    for item in sources:
        if not isinstance(item, dict):
            continue

        meta = (
            item.get("metadata")
            if isinstance(item.get("metadata"), dict)
            else item
        )

        for key in ("page", "page_number", "Page"):
            val = meta.get(key)

            if isinstance(val, int) or (
                isinstance(val, str) and val.isdigit()
            ):
                pages.add(int(val))
                break

    return pages



def get_cited_pages(answer):
    """Extract cited pages from Page: 3, Pages: 3, 5, or Pages: 3–4."""
    pages = set()

    pattern = r"\bPages?\s*:\s*([0-9][0-9,\u2013\- ]*)"

    groups = re.findall(
        pattern,
        str(answer or ""),
        flags=re.IGNORECASE,
    )

    for group in groups:
        for part in re.split(r"[,\s]+", group.strip()):
            if not part:
                continue

            match = re.fullmatch(r"(\d+)[\u2013\-](\d+)", part)

            if match:
                start = int(match.group(1))
                end = int(match.group(2))

                if start <= end <= start + 20:
                    pages.update(range(start, end + 1))

            elif part.isdigit():
                pages.add(int(part))

    return pages

assert get_cited_pages("[Source: core.pdf, Page: 4]") == {4}
assert get_cited_pages("[Source: core.pdf, Pages: 3, 5]") == {3, 5}
assert get_cited_pages("[Source: core.pdf, Pages: 3–4]") == {3, 4}
assert get_cited_pages("[Source: core.pdf, Pages: 3-4]") == {3, 4}

print("Citation parser tests passed.")

def is_abstain(answer):
    a = norm(answer)
    if len(a) > 200:
        return False
    return any(p in a for p in ABSTAIN)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Số chunk đưa vào LLM (mặc định: dùng giá trị của pipeline).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Danh sách id cách nhau bởi dấu phẩy, ví dụ Q06,Q15.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from app.rag_pipeline import answer_question
    except Exception as e:
        print("Không import được app.rag_pipeline.answer_question:", repr(e))
        print("Hãy chạy ở thư mục gốc và kích hoạt đúng virtual environment.")
        return 2

    cases = json.loads(CASES.read_text(encoding="utf-8"))

    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"].upper() in wanted]
        if not cases:
            print("Không có case nào khớp --only:", args.only)
            return 2

    call_kwargs = {"n_results": args.top_k} if args.top_k else {}

    suffix = ""
    if args.top_k:
        suffix += f"_top{args.top_k}"
    if args.only:
        suffix += "_subset"
    out_path = REPORT_DIR / f"rag_quality_report{suffix}.json"

    results = []
    total_start = time.perf_counter()

    for c in cases:
        t = time.perf_counter()
        answer = ""
        pages = set()
        cited_pages = set()

        try:
            answer, sources = answer_question(c["question"], **call_kwargs)
            answer = str(answer or "")

            pages = get_pages(sources)
            cited_pages = get_cited_pages(answer)
            abstained = is_abstain(answer)

            if c["type"] == "answerable":
                aliases = c.get("term_aliases", {})
                answer_norm = norm(answer)

                missing = [
                    term
                    for term in c.get("expected_terms", [])
                    if not term_found(term, answer_norm, aliases)
                ]

                expected = set(c.get("expected_pages", []))
                page_ok = bool(cited_pages & expected) if expected else True

                checks = {
                    "answered": bool(answer_norm) and not abstained,
                    "required_terms_present": not missing,
                    "expected_page_cited_any": page_ok,
                }
                status = "PASS" if all(checks.values()) else "REVIEW"
            else:
                missing = []
                checks = {"abstained": abstained}
                status = "PASS" if abstained else "FAIL"

            results.append({
                "id": c["id"],
                "type": c["type"],
                "question": c["question"],
                "status": status,
                "checks": checks,
                "missing_terms": missing,
                "expected_pages": c.get("expected_pages", []),
                "returned_pages": sorted(pages),
                "cited_pages": sorted(cited_pages),
                "latency_seconds": round(time.perf_counter() - t, 3),
                "answer": answer,
                "notes": c.get("notes", ""),
            })

        except Exception as e:
            results.append({
                "id": c["id"],
                "type": c["type"],
                "question": c["question"],
                "status": "ERROR",
                "error": repr(e),
                "latency_seconds": round(time.perf_counter() - t, 3),
                "cited_pages": sorted(cited_pages),
            })

    ans = [r for r in results if r["type"] == "answerable"]
    unans = [r for r in results if r["type"] == "unanswerable"]

    report = {
        "top_k": args.top_k,
        "total": len(results),
        "answerable_total": len(ans),
        "unanswerable_total": len(unans),
        "pass": sum(r["status"] == "PASS" for r in results),
        "review": sum(r["status"] == "REVIEW" for r in results),
        "fail": sum(r["status"] == "FAIL" for r in results),
        "errors": sum(r["status"] == "ERROR" for r in results),
        "answerable_pass_rate": (
            round(sum(r["status"] == "PASS" for r in ans) / len(ans), 3)
            if ans else None
        ),
        "abstention_rate": (
            round(
                sum(r.get("checks", {}).get("abstained", False) for r in unans)
                / len(unans),
                3,
            )
            if unans else None
        ),
        "average_latency_seconds": (
            round(sum(r.get("latency_seconds", 0) for r in results) / len(results), 3)
            if results else None
        ),
        "total_runtime_seconds": round(time.perf_counter() - total_start, 3),
        "warning": (
            "Heuristic checks only; manually verify correctness, "
            "entailment, and citation accuracy."
        ),
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== RAG QUALITY TEST ===")
    print(
        f"Total {report['total']} | PASS {report['pass']} | "
        f"REVIEW {report['review']} | FAIL {report['fail']} | "
        f"ERROR {report['errors']}"
    )
    print(
        f"Answerable pass rate: {report['answerable_pass_rate']} | "
        f"Abstention rate: {report['abstention_rate']} | "
        f"Avg latency: {report['average_latency_seconds']}s"
    )
    print("Report:", out_path)

    print("\n--- Cases needing attention ---")
    for r in results:
        if r["status"] == "PASS":
            continue

        line = f"[{r['status']}] {r['id']}: {r['question']}"
        if r.get("missing_terms"):
            line += f" | missing={r['missing_terms']}"
        print(line)

        if r["status"] == "ERROR":
            print("    error:", r.get("error"))
            continue

        print(
            f"    expected_pages={r['expected_pages']} "
            f"cited_pages={r['cited_pages']} "
            f"returned_pages={r['returned_pages']}"
        )
        snippet = r["answer"].replace("\n", " ")[:500]
        print(f"    answer: {snippet}")

    print("\nLưu ý: từ khóa/trang nguồn là heuristic, hãy xem lại các ca REVIEW thủ công.")
    return 1 if report["fail"] or report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())