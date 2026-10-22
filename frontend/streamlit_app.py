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

        # Hiển thị sources nếu message là assistant
        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            st.markdown("**Sources:**")

            for source in message["sources"]:

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
    file_type=["pdf"]
)


# ============================================================
# 11. HANDLE USER INPUT
# ============================================================

if prompt:

    uploaded_files = prompt.files
    question = prompt.text.strip()


    # ========================================================
    # 11A. HANDLE PDF UPLOAD
    # ========================================================

    if uploaded_files:

        uploaded_file = uploaded_files[0]

        documents_folder = (
            project_root / "data" / "documents"
        )

        documents_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path = (
            documents_folder / uploaded_file.name
        )


        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        with open(file_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # ----------------------------------------------------
        # Index PDF
        # ----------------------------------------------------

        with st.spinner(
            f"Indexing {uploaded_file.name}..."
        ):

            pages, chunks = index_pdf(
                str(file_path)
            )


        # ----------------------------------------------------
        # Set uploaded document as current document
        # ----------------------------------------------------

        st.session_state.current_document = (
            uploaded_file.name
        )


        st.success(
            f"Document indexed successfully: "
            f"{pages} pages, {chunks} chunks."
        )


    # ========================================================
    # 11B. HANDLE QUESTION
    # ========================================================

    if question:

        # ----------------------------------------------------
        # Check document
        # ----------------------------------------------------

        if st.session_state.current_document is None:

            st.warning(
                "Please upload or select a document first."
            )

        else:

            # ------------------------------------------------
            # Add user message
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )


            # ------------------------------------------------
            # Display user message
            # ------------------------------------------------

            with st.chat_message("user"):

                st.write(question)


            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching the document..."
                ):

                    answer, sources = answer_question(
                        question,
                        source=(
                            st.session_state.current_document
                        )
                    )


                # --------------------------------------------
                # Display answer
                # --------------------------------------------

                st.write(answer)


                # --------------------------------------------
                # Display sources
                # --------------------------------------------

                st.markdown("**Sources:**")

                for source in sources:

                    st.write(
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )


            # ------------------------------------------------
            # Save assistant message
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )