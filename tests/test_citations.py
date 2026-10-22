from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.rag_pipeline import answer_question


def main():
    question = "What is the population of Paris?"

    answer, sources = answer_question(
        question=question,
        n_results=3,
        source="core.pdf",
    )

    print("=" * 60)
    print("QUESTION")
    print(question)

    print("\nANSWER")
    print(answer)

    print("\nRETURNED SOURCES")
    for source in sources:
        print(source)

    
    print("\nCITATION CHECK")

    abstention = "I don't know based on the provided document."

    if not sources:
        print("WARNING: No sources returned.")
    elif abstention.lower() in answer.lower():
        print("PASS: System abstained because evidence was insufficient.")
        print("Citation is not required for this answer.")
    else:
        cited_count = 0

        for source in sources:
            name = source.get("source")
            page = source.get("page")
            expected = f"[Source: {name}, Page: {page}]"

            if expected in answer:
                print(f"PASS: {expected}")
                cited_count += 1
            else:
                print(f"WARNING: Citation not found: {expected}")

        if cited_count > 0:
            print("PASS: At least one returned source was cited.")
        else:
            print("FAIL: No returned sources were cited.")



if __name__ == "__main__":
    main()
