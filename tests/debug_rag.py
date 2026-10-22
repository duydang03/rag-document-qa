from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.rag_pipeline import answer_question

questions = [
    "What are the weather categories represented in BDD100K?",
    "What are the scene categories represented in BDD100K?",
    "What are the time-of-day categories represented in BDD100K?",
]

for question in questions:
    answer, sources = answer_question(
        question=question,
        n_results=5,
        source="core.pdf",
    )

    print("\nQUESTION:", question)
    print("ANSWER:", answer)
    print("SOURCES:", sources)