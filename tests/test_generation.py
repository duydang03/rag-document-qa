from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.generation.llm import generate_answer


def main():
    context = """
Source: core.pdf
Page: 1
Content:
BDD100K is a diverse and large-scale driving video dataset.
It contains 100K driving videos.
"""

    question = "What is BDD100K?"

    answer = generate_answer(
        context=context,
        question=question,
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
