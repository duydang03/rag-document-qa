
import json
import re
import time
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.rag_pipeline import answer_question

GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "tests" / "retrieval_evidence_ground_truth.json"
)
OUTPUT_DIR = PROJECT_ROOT / "tests" / "results"

ABSTENTION = "I don't know based on the provided document."


def check_citations(answer, sources):
    """Check citation format and whether cited pages were returned."""
    if ABSTENTION.lower() in answer.lower():
        return {
            "status": "abstained",
            "citation_check": "not_required",
        }

    citation_pattern = r"\[Source: (.+?), Page: (\d+)\]"
    citations = re.findall(citation_pattern, answer)

    if not citations:
        return {
            "status": "answered",
            "citation_check": "missing",
        }

    returned_sources = {
        (item.get("source"), str(item.get("page")))
        for item in sources
    }

    valid_citations = [
        (source, page)
        for source, page in citations
        if (source, page) in returned_sources
    ]

    return {
        "status": "answered",
        "citation_check": (
            "pass" if valid_citations else "mismatch"
        ),
        "citation_count": len(citations),
        "valid_citation_count": len(valid_citations),
    }


def main():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"\n[{index}/{len(questions)}] {question}")

        start = time.perf_counter()

        try:
            answer, sources = answer_question(
                question=question,
                n_results=5,
                source="core.pdf",
            )

            latency_seconds = time.perf_counter() - start
            citation_result = check_citations(answer, sources)

            result = {
                "question": question,
                "evidence": item.get("evidence", []),
                "match": item.get("match", "any"),
                "answer": answer,
                "sources": sources,
                "latency_seconds": round(latency_seconds, 3),
                **citation_result,
                # Điền thủ công sau khi đọc câu trả lời:
                "correctness": None,
                "groundedness": None,
                "citation_accuracy": None,
                "notes": "",
            }

            print("Answer:", answer)
            print("Sources:", sources)
            print("Latency:", result["latency_seconds"], "seconds")
            print("Citation check:", result["citation_check"])

        except Exception as error:
            result = {
                "question": question,
                "error": str(error),
                "status": "error",
            }
            print("ERROR:", error)

        results.append(result)

    successful = [
        item for item in results
        if item.get("status") != "error"
    ]

    answered = [
        item for item in successful
        if item.get("status") == "answered"
    ]

    abstained = [
        item for item in successful
        if item.get("status") == "abstained"
    ]

    latencies = [
        item["latency_seconds"]
        for item in successful
        if "latency_seconds" in item
    ]

    summary = {
        "total_questions": len(questions),
        "successful_runs": len(successful),
        "errors": len(results) - len(successful),
        "answered": len(answered),
        "abstained": len(abstained),
        "citation_missing": sum(
            item.get("citation_check") == "missing"
            for item in successful
        ),
        "citation_mismatch": sum(
            item.get("citation_check") == "mismatch"
            for item in successful
        ),
        "average_latency_seconds": (
            round(sum(latencies) / len(latencies), 3)
            if latencies else None
        ),
    }

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": results,
    }

    output_path = OUTPUT_DIR / (
        "rag_e2e_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
