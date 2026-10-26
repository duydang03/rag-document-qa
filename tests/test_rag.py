import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.rag_pipeline import answer_question

question = "What is BDD100K?"

answer, sources = answer_question(question)

print("Question:")
print(question)

print("\nAnswer:")
print(answer)

print("\nSources:")

for source in sources:
    print(f"- {source['source']} — Page {source['page']}")