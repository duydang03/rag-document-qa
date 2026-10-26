# 📄 RAG Document QA

**A Retrieval-Augmented Generation (RAG) application for document-grounded question answering.**

Upload PDF documents, ask questions about their content, and receive answers based on retrieved document context, with source references when available.

The project combines semantic retrieval, vector search, LLM-based answer generation, document management, and evaluation scripts in an interactive Streamlit application.

## ✨ Features

- **📑 PDF Ingestion** — Load, process, and index PDF documents.
- **💬 Interactive Q&A** — Ask questions and follow up through a chat interface with persistent session history.
- **🔎 Semantic Retrieval** — Retrieve relevant document chunks using vector representations.
- **📚 Source References** — Display supporting document names and page numbers when available.
- **🛡️ Context-Aware Responses** — Abstain from answering when the retrieved context does not provide sufficient information.
- **🗂️ Document Management** — Select, inspect, and delete indexed documents.
- **📊 Evaluation** — Evaluate retrieval performance, answer quality, abstention behavior, and response latency.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| LangChain document loaders | PDF loading and document processing |
| ChromaDB | Vector storage and similarity search |
| Embedding models | Convert text into vector representations |
| OpenAI API | LLM-based answer generation |
| Retrieval and reranking components | Improve context selection |
| Git & GitHub | Version control and project hosting |

## 🏗️ System Architecture

```text
             ┌──────────────────────┐
             │   PDF Documents      │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   Document Loading    │
             │   & Chunking          │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Embedding Generation │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │      ChromaDB        │
             │   Vector Storage     │
             └──────────┬───────────┘
                        │
 User Question ─────────┤
                        ▼
             ┌──────────────────────┐
             │ Retrieval / Reranking│
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Context-Based Answer │
             │   Generation (LLM)   │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Answer + Sources     │
             │ Streamlit Interface  │
             └──────────────────────┘
```

### Processing Pipeline

1. **Ingestion:** Load PDF documents and split their content into manageable chunks.
2. **Embedding:** Transform document chunks into numerical vector representations.
3. **Indexing:** Store vectors and associated metadata in ChromaDB.
4. **Retrieval:** Retrieve relevant chunks for the user's question.
5. **Generation:** Use the retrieved context to generate a grounded answer.
6. **Source Attribution:** Display supporting document references when available.
7. **Evaluation:** Run test scripts to assess retrieval and answer behavior.

## 📁 Project Structure

```text
rag-document-qa/
├── app/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── document_manager.py
│   └── rag_pipeline.py
├── frontend/
│   └── streamlit_app.py
├── tests/    
├── data/
│   └── documents/       # Local PDF documents
├── build_vector_db.py
├── requirements.txt
├── .gitignore
└── README.md
```

> The structure above describes the intended organization. Local environment files, generated vector databases, and private documents should not be committed to the repository.

## 🚀 Getting Started

### Prerequisites

- Python installed on your machine
- An OpenAI API key for LLM-based answer generation
- Git
- A PDF document you are authorized to process

### 1. Clone the Repository

```bash
git clone https://github.com/duydang03/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and configure the OpenAI API key using the environment variable name expected by the application.

Example:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

Make sure the application loads this variable correctly. Never commit `.env` or expose API keys in source code.

### 5. Add a PDF Document

You can either:

- Upload a PDF through the Streamlit interface, or
- Place an authorized PDF in the project's configured document directory.

### 6. Build the Vector Database (If Needed)

```bash
python build_vector_db.py
```

Run this step when you need to build or rebuild the vector index from the configured documents. If document indexing is already handled through the application, follow the existing workflow instead.

### 7. Launch the Application

```bash
streamlit run frontend/streamlit_app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 🧪 Evaluation

The project includes scripts for testing embeddings, LLM integration, metadata, retrieval, and the end-to-end RAG pipeline.

### Retrieval: Chunking Strategy

Several chunk-size and overlap configurations were evaluated to investigate their impact on retrieval performance.

| Chunk Size / Overlap | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR@3 | Keyword Coverage |
|---|---:|---:|---:|---:|---:|---:|
| 500 / 100 | 132 | 0% | 80% | 100% | 0.3833 | 95.00% |
| 750 / 150 | 92 | 20% | 100% | 100% | 0.6000 | **100.00%** |
| **800 / 200** | **90** | **80%** | **100%** | **100%** | **0.9000** | 95.00% |
| 1000 / 200 | 69 | 40% | 100% | 100% | 0.7000 | 73.33% |

**Key observations**

- **800 / 200** achieved the strongest retrieval ranking results in the reported evaluation, with an 80% Hit@1 and an MRR@3 of 0.90.
- **750 / 150** achieved the highest keyword coverage at 100%.
- **1000 / 200** produced fewer chunks but lower Hit@1 and keyword coverage than 800 / 200.

Based on these results, 800 / 200 is a promising configuration when prioritizing retrieval ranking performance, while 750 / 150 may be preferable when keyword coverage is the primary consideration.

### Answer Quality

The reported Faithfulness and Correctness scores were both 100% across the evaluated chunking configurations.

These scores should be interpreted in the context of the test dataset, evaluation criteria, and scoring methodology.

### End-to-End RAG Evaluation

In the current local evaluation, the configuration with `top_k = 8` achieved the following results across three runs:

| Metric | Reported Result |
|---|---:|
| PASS results | 26/26 |
| Answerable cases | 15/15 |
| Abstention cases | 11/11 |
| Average response latency | Approximately 2.005 seconds |

These results represent the project's current evaluation setup. They do not guarantee equivalent performance on arbitrary documents or unseen questions.

## 🔒 Security Considerations

- Keep API keys and `.env` files out of version control.
- Do not publish private PDFs or sensitive document contents.
- Avoid committing generated vector databases and local environment folders.
- Only process documents you have permission to use.
- Configure secrets securely before deploying the application.

## 🔮 Future Improvements

- [ ] Expand evaluation to more documents and question types.
- [ ] Improve retrieval confidence and abstention logic.
- [ ] Add automated test execution and continuous integration.
- [ ] Improve document versioning and re-indexing.
- [ ] Deploy a public demo with secure secret management.
- [ ] Document evaluation methodology and metric definitions.

## 👨‍💻 Author

Developed as a hands-on project exploring Retrieval-Augmented Generation, semantic search, vector databases, and document-grounded question answering.

---

**Note:** Evaluation results are specific to the tested dataset and configuration. Broader validation is needed before drawing conclusions about general-purpose performance.
