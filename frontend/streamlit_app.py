import sys
from pathlib import Path

import streamlit as st


# ============================================================
# 1. PROJECT PATH
# ============================================================

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from app.rag_pipeline import answer_question
from app.ingestion.indexer import index_pdf
from app.document_manager import (
    get_document_list,
    remove_document
)

# ============================================================
# 3. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_document" not in st.session_state:
    st.session_state.current_document = None

# ============================================================
# CHECK WHETHER THE ANSWER IS A REFUSAL
# ============================================================

def is_refusal_answer(answer):
    refusal_phrases = [
        "tôi không biết dựa trên tài liệu được cung cấp",
        "tôi không thể trả lời dựa trên tài liệu được cung cấp",
        "i don't know based on the provided document",
        "i cannot answer based on the provided document",
        "i don't know based on the provided context",
        "i cannot answer based on the provided context",
    ]

    normalized_answer = answer.strip().casefold()

    return any(
        phrase in normalized_answer
        for phrase in refusal_phrases
    )
# ============================================================
# 5. TITLE
# ============================================================

st.title("📚 RAG Document QA")

st.caption(
    "Upload documents and ask questions using Retrieval-Augmented Generation."
)


# ============================================================
# 6. DOCUMENT LIST
# ============================================================

documents = get_document_list()

indexed_documents = [
    document["name"]
    for document in documents
    if document["indexed"]
]


# ============================================================
# DOCUMENT SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 Documents")

    documents = get_document_list()

    indexed_documents = [
        document["name"]
        for document in documents
        if document["indexed"]
    ]

    if indexed_documents:

        if st.session_state.current_document is None:
            st.session_state.current_document = (
                indexed_documents[0]
            )

        elif (
            st.session_state.current_document
            not in indexed_documents
        ):
            st.session_state.current_document = (
                indexed_documents[0]
            )

        selected_document = st.selectbox(
            "Select document",
            indexed_documents,
            index=indexed_documents.index(
                st.session_state.current_document
            )
        )

        st.session_state.current_document = (
            selected_document
        )

        # --------------------------------------------
        # Current document information
        # --------------------------------------------

        current_info = next(
            (
                document
                for document in documents
                if document["name"]
                == st.session_state.current_document
            ),
            None
        )

        if current_info:

            st.divider()

            st.caption("Current document")

            st.write(
                f"📄 {current_info['name']}"
            )

            st.caption(
                f"{current_info['pages']} pages · "
                f"{current_info['chunks']} chunks"
            )

    else:

        st.info(
            "No documents available."
        )

    if indexed_documents:

        st.divider()

        st.subheader("Manage")

        document_to_delete = st.selectbox(
            "Document",
            indexed_documents,
            key="delete_document_select"
        )

        if st.button(
            "🗑 Delete document",
            use_container_width=True
        ):

            from app.document_manager import remove_document

            remove_document(
                document_to_delete
            )

            # Nếu xóa document đang sử dụng
            if (
                st.session_state.current_document
                == document_to_delete
            ):

                st.session_state.current_document = None

            st.success(
                f"{document_to_delete} deleted."
            )

            st.rerun()

# ============================================================
# 8. SHOW CURRENT DOCUMENT
# ============================================================

if st.session_state.current_document:

    st.caption(
        f"Current document: "
        f"**{st.session_state.current_document}**"
    )


# ============================================================
# 9. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show which document was used for this message.
        if message.get("document"):
            st.caption(
                f"Document: {message['document']}"
            )

        # Show sources only for answerable responses.
        if message["role"] == "assistant":
            answer = message["content"]
            sources = message.get("sources", [])

            if sources and not is_refusal_answer(answer):
                st.markdown("**Sources:**")

                for source in sources:
                    st.write(
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )



# ============================================================
# 10. CHAT INPUT + PDF UPLOAD
# ============================================================

prompt = st.chat_input(
    "Ask a question or attach a PDF...",
     accept_file=True,
     file_type=["pdf"],
     key="main_chat_input"
)

# ============================================================
# 11. HANDLE USER INPUT
# ============================================================

if prompt:
    
# # ============================================================
# # 10. CHAT INPUT + PDF UPLOAD
# # ============================================================

#     prompt = st.chat_input(
#     "Ask a question or attach a PDF...",
#     accept_file=True,
#     file_type=["pdf"],
#     key="main_chat_input"
# )


    # ============================================================
    # 11. HANDLE USER INPUT
    # ============================================================

    if prompt:
        uploaded_files = prompt.files
        question = (prompt.text or "").strip()

        # Default: use the document selected in the sidebar.
        active_document = st.session_state.current_document

        # --------------------------------------------------------
        # 11A. HANDLE PDF UPLOAD
        # --------------------------------------------------------

        if uploaded_files:
            uploaded_file = uploaded_files[0]

            documents_folder = project_root / "data" / "documents"
            documents_folder.mkdir(parents=True, exist_ok=True)

            file_path = documents_folder / uploaded_file.name

            # Save the uploaded PDF.
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # Index the uploaded PDF.
            try:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    pages, chunks = index_pdf(str(file_path))

                # Use the uploaded PDF for THIS question.
                # Do not overwrite the sidebar selection.
                active_document = uploaded_file.name

                st.success(
                    f"Document indexed: {pages} pages, {chunks} chunks."
                )

            except Exception as e:
                st.error(f"Could not index PDF: {e}")
                st.stop()

        # --------------------------------------------------------
        # 11B. HANDLE QUESTION
        # --------------------------------------------------------

        if question:
            if not active_document:
                st.warning("Please upload or select a document first.")

            else:
                # Save the user message with its document.
                st.session_state.messages.append({
                    "role": "user",
                    "content": question,
                    "document": active_document
                })

                with st.chat_message("user"):
                    st.write(question)
                    st.caption(f"Document: {active_document}")

                # Generate answer using the explicitly selected source.
                with st.chat_message("assistant"):
                    with st.spinner(
                        f"Searching {active_document}..."
                    ):
                        try:
                            answer, sources = answer_question(
                                question,
                                source=active_document
                            )
                        except Exception as e:
                            st.error(f"Could not answer the question: {e}")
                            st.stop()

                    st.write(answer)

                    st.caption(f"Document used: {active_document}")

                    # Hide sources when the RAG system cannot answer.
                    is_refusal = is_refusal_answer(answer)

                    # Do not save irrelevant sources for refusal answers.
                    sources_to_save = [] if is_refusal else sources

                    if sources_to_save:
                        st.markdown("**Sources:**")

                        for source in sources_to_save:
                            st.write(
                                f"📄 {source['source']} "
                                f"— Page {source['page']}"
                            )

                # Save the answer with its document.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_to_save,
                    "document": active_document
                })
