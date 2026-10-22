from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.generation.llm import generate_answer

from app.rag_pipeline import answer_question

answer, sources = answer_question(
    question="What is BDD100K?",
    n_results=3,
    source="core.pdf",
)

print("Answer:")
print(answer)

print("\nSources:")
for item in sources:
    print(item)