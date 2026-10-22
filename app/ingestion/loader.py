from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    return documents