from pathlib import Path

from app.retrieval.vector_store import (
    list_documents,
    get_document_stats,
    delete_document
)


def get_document_list():
    documents_folder = Path("data/documents")

    if not documents_folder.exists():
        return []

    pdf_files = {
        file.name
        for file in documents_folder.glob("*.pdf")
    }

    indexed_documents = set(list_documents())

    documents = []

    for filename in sorted(pdf_files):

        stats = {
            "chunks": 0,
            "pages": 0
        }

        if filename in indexed_documents:
            stats = get_document_stats(filename)

        documents.append({
            "name": filename,
            "indexed": filename in indexed_documents,
            "pages": stats["pages"],
            "chunks": stats["chunks"]
        })

    return documents

def remove_document(filename: str):
    documents_folder = Path("data/documents")

    file_path = documents_folder / filename

    # Xóa dữ liệu khỏi ChromaDB
    delete_document(filename)

    # Xóa file PDF
    if file_path.exists():
        file_path.unlink()