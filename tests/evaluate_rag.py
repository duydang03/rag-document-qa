import json
import sys
import re
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

def normalize_text(text):
    text = text.lower()
    text = text.replace("100,000", "100k")
    text = text.replace("100000", "100k")
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

Return only valid JSON in this exact structure:

{{
  "verdict": "PASS",
  "reason": "short explanation"
}}

The "verdict" value must be exactly "PASS" or "FAIL".
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return json.loads(response.output_text)


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

Return only valid JSON in this exact structure:

{{
  "verdict": "PASS",
  "reason": "short explanation"
}}

The "verdict" value must be exactly "PASS" or "FAIL".
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return json.loads(response.output_text)


with open(
    PROJECT_ROOT / "tests" / "ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    ground_truth = json.load(f)

with open(
    PROJECT_ROOT / "tests" / "retrieval_ground_truth.json",
    "r",
    encoding="utf-8"
) as f:
    retrieval_ground_truth = json.load(f)

total_questions = len(ground_truth)

generation_scores = []
faithfulness_pass = 0
correctness_pass = 0

retrieval_hits = {
    1: 0,
    3: 0,
    5: 0,
    8: 0
}

retrieval_mrr = {
    1: 0,
    3: 0,
    5: 0,
    8: 0
}

print("=" * 60)
print("RAG EVALUATION")
print("=" * 60)

for item in ground_truth:

    question = item["question"]
    expected_answer = item["answer"]
    keywords = item["keywords"]
    retrieval_item = next(
    item
    for item in retrieval_ground_truth
    if item["question"] == question
    )

    relevant_pages = retrieval_item["relevant_pages"]
    print("\n" + "=" * 60)
    print("Question:")
    print(question)

    query_embedding = create_embeddings([question])[0]

    results = search(
        query_embedding,
        n_results=8,
        source="core.pdf"
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)
    retrieved_pages = [
    metadata.get("page", 0) + 1
    for metadata in metadatas
    ]
    for k in retrieval_hits:

        top_k_pages = retrieved_pages[:k]

        if any(
            page in relevant_pages
            for page in top_k_pages
        ):
            retrieval_hits[k] += 1

        reciprocal_rank = 0

        for rank, page in enumerate(top_k_pages, start=1):
            if page in relevant_pages:
                reciprocal_rank = 1 / rank
                break

        retrieval_mrr[k] += reciprocal_rank

    generated_answer = generate_answer(
    context=context,
    question=question
    )
    normalized_answer = normalize_text(generated_answer)

    matched_keywords = []

    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)

        if normalized_keyword in normalized_answer:
            matched_keywords.append(keyword)

    coverage = (
        len(matched_keywords) / len(keywords) * 100
    )
    generation_scores.append(coverage)

    faithfulness_result = evaluate_faithfulness(
        context=context,
        answer=generated_answer
    )

    if faithfulness_result["verdict"] == "PASS":
        faithfulness_pass += 1


    correctness_result = evaluate_correctness(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer
    )

    if correctness_result["verdict"] == "PASS":
        correctness_pass += 1

    print("\nGenerated answer:")
    print(generated_answer)

    print("\nKeyword coverage:")
    print(f"{coverage:.2f}%")

    print("\nFaithfulness:")
    print(f"VERDICT: {faithfulness_result['verdict']}")
    print(f"REASON: {faithfulness_result['reason']}")

    print("\nCorrectness:")
    print(f"VERDICT: {correctness_result['verdict']}")
    print(f"REASON: {correctness_result['reason']}")

    print("\nSources:")

    for metadata in metadatas:
        print(
            f"- {metadata.get('source', 'Unknown')}, "
            f"page {metadata.get('page', 0) + 1}"
        )

retrieval_hit_scores = {}

for k in retrieval_hits:
    retrieval_hit_scores[k] = (
        retrieval_hits[k] / total_questions * 100
    )

retrieval_mrr_scores = {}

for k in retrieval_mrr:
    retrieval_mrr_scores[k] = (
        retrieval_mrr[k] / total_questions
    )

average_generation = (
    sum(generation_scores) / len(generation_scores)
)

faithfulness_score = (
    faithfulness_pass / total_questions * 100
)

correctness_score = (
    correctness_pass / total_questions * 100
)

print("\n" + "=" * 60)
print("RAG EVALUATION SUMMARY")
print("=" * 60)
print("\nRetrieval")
print("-" * 60)

for k in [1, 3, 5, 8]:
    print(
        f"Hit@{k}: "
        f"{retrieval_hit_scores[k]:.2f}%"
    )

for k in [1, 3, 5, 8]:
    print(
        f"MRR@{k}: "
        f"{retrieval_mrr_scores[k]:.4f}"
    )
print("\nGeneration")
print("-" * 60)
print(
    f"Average keyword coverage: "
    f"{average_generation:.2f}%"
)

print("\nFaithfulness")
print("-" * 60)
print(
    f"PASS: {faithfulness_pass}/{total_questions}"
)
print(
    f"Faithfulness: {faithfulness_score:.2f}%"
)

print("\nAnswer Correctness")
print("-" * 60)
print(
    f"PASS: {correctness_pass}/{total_questions}"
)
print(
    f"Correctness: {correctness_score:.2f}%"
)

evaluation_results = {
    "retrieval": {
        "hit@1": retrieval_hit_scores[1],
        "hit@3": retrieval_hit_scores[3],
        "hit@5": retrieval_hit_scores[5],
        "hit@8": retrieval_hit_scores[8],
        "mrr@1": retrieval_mrr_scores[1],
        "mrr@3": retrieval_mrr_scores[3],
        "mrr@5": retrieval_mrr_scores[5],
        "mrr@8": retrieval_mrr_scores[8]
    },
    "generation": {
        "average_keyword_coverage": average_generation
    },
    "faithfulness": {
        "pass": faithfulness_pass,
        "total": total_questions,
        "score": faithfulness_score
    },
    "correctness": {
        "pass": correctness_pass,
        "total": total_questions,
        "score": correctness_score
    }
}
with open(
    PROJECT_ROOT / "tests" / "evaluation_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        evaluation_results,
        f,
        indent=4
    )