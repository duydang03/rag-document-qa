import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

EMPTY_ANSWER = "I don't know based on the provided document."


INSTRUCTIONS = """
You are a document question-answering assistant.

Answer the user's question using only the supplied document context.

Rules:
1. Answer the question directly and clearly.
2. Include all important aspects explicitly requested by the question.
3. For questions asking about multiple categories, attributes,
   methods, reasons, or examples, provide a concise list of the
   relevant items supported by the context.
4. Preserve important technical terms used in the document.
   Do not replace a specific term with a broader synonym when
   the distinction matters to the question.
5. For questions about data collection, describe the collection
   method and the relevant participants or contributors when
   these details appear in the context.
6. For questions about dataset diversity, identify the relevant
   dimensions explicitly, such as geographic, environmental,
   weather, scene, and time-of-day diversity, but include only
   dimensions supported by the context.
7. For questions involving quantities or comparisons, preserve
   the quantities and their scope accurately. Do not confuse
   per-image counts with counts across the entire dataset.
8. Do not use outside knowledge or invent facts.
9. If the context does not provide enough evidence to answer,
   say exactly:
   "I don't know based on the provided document."
10. Cite factual claims using the source filename and page number
    provided in the context.
11. Use this citation format: [Source: filename, Page: N].
12. Only cite source names and page numbers present in the context.
13. Do not invent citations or cite a page that does not support
    the associated claim.
14. If different context passages support different claims,
    cite the appropriate source for each claim.
15. Use clear, natural language and proper spacing.
16. When describing how data was collected, distinguish the
    collection method, contributors, and supporting organizations.
    Do not attribute all data collection directly to a supporting
    organization unless the context explicitly says so.
17. Preserve important details from the context even when they
    are expressed using technical terms such as "crowd-sourcing".
"""


_client = None


def _get_client() -> OpenAI:
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. "
                "Please configure it in your .env file."
            )

        _client = OpenAI(api_key=api_key)

    return _client


def generate_answer(context: str, question: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    if not context or not context.strip():
        return EMPTY_ANSWER

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    user_input = f"""Document context:
<context>
{context}
</context>

User question:
{question}

Answer:
"""

    request = {
        "model": model,
        "instructions": INSTRUCTIONS.strip(),
        "input": user_input,
    }

    # Tùy chọn: đặt OPENAI_TEMPERATURE=0 trong .env nếu model hỗ trợ.
    temperature = os.getenv("OPENAI_TEMPERATURE")
    if temperature:
        request["temperature"] = float(temperature)

    response = _get_client().responses.create(**request)

    answer = response.output_text.strip()

    return answer or EMPTY_ANSWER