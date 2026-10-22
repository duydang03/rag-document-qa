
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_pipeline import answer_question
def main():
    questions = [
        "What is BDD100K?",
        "What tasks does BDD100K support?",
        "What is drivable area segmentation?"
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        answer, sources = answer_question(
            question=question,
            n_results=3,
            source="core.pdf"
        )

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES:")
        for item in sources:
            print(
                f"- {item['source']}, page {item['page']}"
            )


if __name__ == "__main__":
    main()
