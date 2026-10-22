
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_documents

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "core.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Page numbers are 1-based, like the PDF viewer.
TARGET_PAGES = [1, 3, 4, 5, 11, 12, 13]

documents = load_pdf(str(PDF_PATH))
chunks = split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

print(f"PDF: {PDF_PATH.name}")
print(f"Total pages: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

for chunk_id, chunk in enumerate(chunks):
    page = chunk.metadata.get("page", 0) + 1

    if page not in TARGET_PAGES:
        continue

    text = " ".join(chunk.page_content.split())

    print("\n" + "=" * 80)
    print(f"Chunk ID: {chunk_id} | Page: {page}")
    print("-" * 80)
    print(text)
