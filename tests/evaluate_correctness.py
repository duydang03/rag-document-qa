import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from app.ingestion.embedder import create_embeddings
from app.retrieval.vector_store import search
from app.generation.llm import generate_answer


client = OpenAI()


def evaluate_correctness(
    question: str,
    expected_answer: str,
    generated_answer: str
):
    prompt = f"""
You are evaluating the correctness of an answer in a document
question-answering system.

Question:
{question}

Expected answer:
{expected_answer}

Generated answer:
{generated_answer}

Evaluation rules:
1. Mark PASS if the generated answer correctly answers the question.
2. The wording does not need to exactly match the expected answer.
3. Equivalent wording is acceptable.
4. Additional information is acceptable only if it does not contradict
   the expected answer.
5. Mark FAIL if the answer is incorrect, contradicts the expected answer,
   or does not answer the question.

Return exactly this format:

VERDICT: PASS or FAIL
REASON: <short explanation>
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text


with open(
    PROJECT_ROOT / "tests" / "ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    ground_truth = json.load(f)


print("=" * 60)
print("Answer Correctness Evaluation")
print("=" * 60)


for item in ground_truth:

    question = item["question"]
    expected_answer = item["answer"]

    print("\n" + "=" * 60)
    print("Question:")
    print(question)

    query_embedding = create_embeddings([question])[0]

    results = search(
        query_embedding,
        n_results=3,
        source="core.pdf"
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    generated_answer = generate_answer(
        context=context,
        question=question
    )

    evaluation = evaluate_correctness(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer
    )

    print("\nExpected answer:")
    print(expected_answer)

    print("\nGenerated answer:")
    print(generated_answer)

    print("\nCorrectness evaluation:")
    print(evaluation)