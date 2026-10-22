from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.retrieval.vector_store import get_all_chunks

keywords = [
    "partly cloudy",
    "dawn",
    "dusk",
    "city street",
    "gas station",
    "weather",
    "time of day",
]

chunks = get_all_chunks(source="core.pdf")

print("Total chunks:", len(chunks))

for keyword in keywords:
    matches = []

    for item in chunks:
        document = item["document"]
        metadata = item["metadata"]

        if keyword.lower() in document.lower():
            matches.append({
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "text": document,
            })

    print(f"\n{'=' * 70}")
    print("KEYWORD:", keyword)
    print("MATCHES:", len(matches))

    for match in matches[:5]:
        print(
            f"\nPage: {match['page']} | "
            f"Chunk: {match['chunk_id']}"
        )
        print(match["text"][:1200])
