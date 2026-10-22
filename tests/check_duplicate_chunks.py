import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from app.retrieval.vector_store import get_all_chunks

chunks = get_all_chunks()

seen = {}
duplicates = []

for item in chunks:
    document = item["document"].strip()
    metadata = item["metadata"]

    if document in seen:
        duplicates.append({
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "duplicate_of": seen[document]
        })
    else:
        seen[document] = {
            "source": metadata.get("source"),
            "page": metadata.get("page")
        }

print("Total chunks:", len(chunks))
print("Exact duplicate chunks:", len(duplicates))

for item in duplicates[:10]:
    print(item)
