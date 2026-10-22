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


def evaluate_faithfulness(context: str, answer: str):
    prompt = f"""
You are evaluating the faithfulness of an answer in a RAG system.

Your task is to determine whether every factual claim in the answer
is supported by the provided context.

Context:
{context}

Answer:
{answer}

Evaluation rules:
1. Mark PASS if all factual claims in the answer are supported by the context.
2. Mark FAIL if any factual claim is not supported by the context.
3. Do not use outside knowledge.
4. Ignore wording differences if the meaning is clearly supported.
5. Keep the explanation short.

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
print("Faithfulness Evaluation")
print("=" * 60)


for item in ground_truth:

    question = item["question"]

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
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)

    answer = generate_answer(
        context=context,
        question=question
    )

    evaluation = evaluate_faithfulness(
        context=context,
        answer=answer
    )

    print("\nGenerated answer:")
    print(answer)

    print("\nFaithfulness evaluation:")
    print(evaluation)

    print("\nSources:")

    for metadata in metadatas:
        print(
            f"- {metadata.get('source', 'Unknown')}, "
            f"page {metadata.get('page', 0) + 1}"
        )