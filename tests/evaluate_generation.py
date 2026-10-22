import json
import sys
import re
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_pipeline import answer_question

def normalize_text(text):
    text = text.lower()

    # Normalize common numeric expressions
    text = text.replace("100,000", "100k")
    text = text.replace("100000", "100k")

    # Remove markdown formatting
    text = text.replace("**", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# Load ground truth
with open(
    PROJECT_ROOT / "tests" / "ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    ground_truth = json.load(f)


print("=" * 60)
print("Generation Evaluation")
print("=" * 60)


for item in ground_truth:

    question = item["question"]
    expected_answer = item["answer"]
    keywords = item["keywords"]

    print("\nQuestion:")
    print(question)

    # Run the complete RAG pipeline
    answer, sources = answer_question(
        question,
        n_results=3,
        source="core.pdf"
    )

    print("\nExpected answer:")
    print(expected_answer)

    print("\nGenerated answer:")
    print(answer)

    # Check keywords
    normalized_answer = normalize_text(answer)

    matched_keywords = []

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)

        if normalized_keyword in normalized_answer:
            matched_keywords.append(keyword)

    coverage = len(matched_keywords) / len(keywords)

    print("\nMatched keywords:")
    print(matched_keywords)

    print(
        f"Keyword coverage: "
        f"{coverage * 100:.2f}%"
    )

    print("\nSources:")
    for source in sources:
        print(
            f"- {source['source']}, "
            f"page {source['page']}"
        )

    print("-" * 60)