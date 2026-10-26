import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.generation.llm import generate_answer

context="""
BDD100K is a large-scale driving video dataset with 100K videos
and 10 tasks for autonomous driving. The dataset has geographic,
environmental, and weather diversity.
"""
question = "What is BDD100K?"

answer = generate_answer(context, question)

print("Question:")
print(question)
print("\nAnswer:")
print(answer)
