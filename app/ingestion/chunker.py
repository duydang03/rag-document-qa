import re
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)     # ﬁ -> fi, ﬂ -> fl
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # "di- verse" -> "diverse"
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_documents(documents, chunk_size=800, chunk_overlap=200):
    cleaned = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=dict(doc.metadata),
        )
        for doc in documents
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(cleaned)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks

