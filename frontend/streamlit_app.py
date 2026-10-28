import sys
from pathlib import Path
import re
import streamlit as st


# ============================================================
# 1. PROJECT PATH AND IMPORTS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from app.rag_pipeline import answer_question
from app.ingestion.indexer import index_pdf
from app.document_manager import (
    get_document_list,
    remove_document,
)


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# 3. HELPERS
# ============================================================

REFUSAL_PHRASES = [
    "tôi không biết dựa trên tài liệu được cung cấp",
    "tôi không thể trả lời dựa trên tài liệu được cung cấp",
    "i don't know based on the provided document",
    "i cannot answer based on the provided document",
    "i don't know based on the provided context",
    "i cannot answer based on the provided context",
]


def is_refusal_answer(answer):
    normalized_answer = answer.strip().casefold()
    return any(phrase in normalized_answer for phrase in REFUSAL_PHRASES)


CITATION_RE = re.compile(r"\[Source:[^\]]*\]")


def strip_citations(text):
    return CITATION_RE.sub("", text)


def compact_citations(text):
    """[Source: core.pdf, Page: 1; Source: core.pdf, Page: 3] -> [p.1, 3]. Chỉ để hiển thị."""
    def repl(match):
        numbers = []
        for group in re.findall(r"Pages?:\s*([\d,\u2013\- ]+)", match.group(0)):
            numbers.extend(re.findall(r"\d+(?:\u2013\d+)?", group))
        return "[p." + ", ".join(dict.fromkeys(numbers)) + "]" if numbers else match.group(0)

    return CITATION_RE.sub(repl, text)


def order_by_citation(sources, answer):
    cited = list(dict.fromkeys(
        int(page) for page in re.findall(r"Pages?:\s*(\d+)", answer)
    ))
    return sorted(
        sources,
        key=lambda s: cited.index(s["page"]) if s["page"] in cited else len(cited),
    )


def focus_snippet(text, query, max_chars=360):
    """Chọn câu liên quan nhất với câu hỏi và câu trả lời, thay vì lấy đầu chunk."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    query_tokens = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 3}
    if not query_tokens:
        return text[:max_chars] + " …"

    overlap = [
        len(query_tokens & set(re.findall(r"\w+", s.lower())))
        for s in sentences
    ]
    best = max(range(len(sentences)), key=lambda i: (overlap[i], -i))

    lo = hi = best
    size = len(sentences[best])
    while size < max_chars:
        if hi + 1 < len(sentences) and size + len(sentences[hi + 1]) <= max_chars:
            hi += 1
            size += len(sentences[hi]) + 1
        elif lo > 0 and size + len(sentences[lo - 1]) <= max_chars:
            lo -= 1
            size += len(sentences[lo]) + 1
        else:
            break

    return (
        ("… " if lo > 0 else "")
        + " ".join(sentences[lo:hi + 1])
        + (" …" if hi < len(sentences) - 1 else "")
    )

def cited_page_numbers(answer):
    """Các trang được dẫn trong câu trả lời: 'Page: 3', 'Pages: 3, 5', 'Pages: 3–4'."""
    pages = set()
    for group in re.findall(r"Pages?:\s*([\d,\u2013\- ]+)", answer):
        for part in re.split(r"[,\s]+", group.strip()):
            match = re.fullmatch(r"(\d+)[\u2013\-](\d+)", part)
            if match:
                low, high = int(match.group(1)), int(match.group(2))
                if low <= high <= low + 20:
                    pages.update(range(low, high + 1))
            elif part.isdigit():
                pages.add(int(part))
    return pages

def render_sources(sources):
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            st.markdown(f"**{index}. 📄 {source['source']} · p.{source['page']}**")
            if source.get("snippet"):
                st.markdown(f"> {source['snippet']}")


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
# 6. DOCUMENT LIST AND SELECTION STATE
# ============================================================

documents = get_document_list()

indexed_documents = [
    document["name"]
    for document in documents
    if document["indexed"]
]

if indexed_documents:
    if st.session_state.current_document not in indexed_documents:
        st.session_state.current_document = indexed_documents[0]
else:
    st.session_state.current_document = None


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📚 Documents")

    if indexed_documents:
        selected_document = st.selectbox(
            "Select document",
            indexed_documents,
            index=indexed_documents.index(
                st.session_state.current_document
            ),
        )

        st.session_state.current_document = selected_document

        current_info = next(
            (
                document
                for document in documents
                if document["name"] == selected_document
            ),
            None,
        )

        if current_info:
            st.caption(
                f"{current_info['pages']} pages · "
                f"{current_info['chunks']} chunks"
            )

        st.divider()
        st.subheader("Manage")

        # Chỉ xóa tài liệu đang chọn, có bước xác nhận.
        confirm_delete = st.checkbox(
            f"Confirm deleting {selected_document}",
            key=f"confirm_delete_{selected_document}",
        )

        if st.button(
            f"🗑 Delete {selected_document}",
            use_container_width=True,
            disabled=not confirm_delete,
        ):
            remove_document(selected_document)
            st.session_state.current_document = None
            st.toast(f"{selected_document} deleted.")
            st.rerun()

    else:
        st.info("No documents available.")

    st.divider()

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# 8. CURRENT DOCUMENT
# ============================================================

if st.session_state.current_document:
    st.caption(
        f"Current document: **{st.session_state.current_document}**"
    )


# ============================================================
# 9. CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(compact_citations(message["content"]))

        if message.get("document"):
            st.caption(f"Document: {message['document']}")

        if message["role"] == "assistant":
            if not is_refusal_answer(message["content"]):
                render_sources(message.get("sources", []))


# ============================================================
# 10. CHAT INPUT + PDF UPLOAD
# ============================================================

prompt = st.chat_input(
    "Ask a question or attach a PDF...",
    accept_file=True,
    file_type=["pdf"],
    key="main_chat_input",
)


# ============================================================
# 11. HANDLE USER INPUT
# ============================================================

if prompt:
    uploaded_files = prompt.files
    question = (prompt.text or "").strip()

    active_document = st.session_state.current_document

    # --------------------------------------------------------
    # 11A. HANDLE PDF UPLOAD
    # --------------------------------------------------------

    if uploaded_files:
        uploaded_file = uploaded_files[0]

        # Chỉ lấy phần tên file, tránh đường dẫn do người dùng gửi.
        file_name = Path(uploaded_file.name).name

        if file_name in indexed_documents:
            # Lưu ý: so sánh theo tên file. Nếu bạn tải bản đã chỉnh sửa
            # cùng tên, hãy xóa bản cũ trước rồi tải lại.
            st.toast(f"{file_name} is already indexed. Using the existing index.")

        else:
            documents_folder = project_root / "data" / "documents"
            documents_folder.mkdir(parents=True, exist_ok=True)

            file_path = documents_folder / file_name

            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            try:
                with st.spinner(f"Indexing {file_name}..."):
                    pages, chunks = index_pdf(str(file_path))

                st.toast(f"Document indexed: {pages} pages, {chunks} chunks.")

            except Exception as e:
                st.error(f"Could not index PDF: {e}")
                st.stop()

        # Chuyển tài liệu đang dùng sang file vừa tải lên.
        active_document = file_name
        st.session_state.current_document = file_name

    # --------------------------------------------------------
    # 11B. HANDLE QUESTION
    # --------------------------------------------------------

    if question:
        if not active_document:
            st.warning("Please upload or select a document first.")

        else:
            st.session_state.messages.append({
                "role": "user",
                "content": question,
                "document": active_document,
            })

            with st.chat_message("user"):
                st.write(question)
                st.caption(f"Document: {active_document}")

            with st.chat_message("assistant"):
                with st.spinner(f"Searching {active_document}..."):
                    try:
                        answer, sources = answer_question(
                            question,
                            source=active_document,
                        )
                    except Exception as e:
                        st.error(f"Could not answer the question: {e}")
                        st.stop()

                st.write(compact_citations(answer))

                if is_refusal_answer(answer):
                    sources_to_save = []
                else:
                    ordered = order_by_citation(sources, answer)
                    cited = cited_page_numbers(answer)
                    # Chỉ hiện các trang được trích dẫn; nếu mô hình không dẫn trang nào thì hiện 3 nguồn đầu.
                    shown = [s for s in ordered if s["page"] in cited] or ordered[:3]

                    focus_query = f"{question} {strip_citations(answer)}"
                    sources_to_save = [
                        {
                            "source": s["source"],
                            "page": s["page"],
                            "snippet": focus_snippet(s.get("content", ""), focus_query),
                        }
                        for s in shown
                    ]

                render_sources(sources_to_save)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_to_save,
                "document": active_document,
            })

            # Vẽ lại để thanh bên cập nhật đúng tài liệu đang dùng.
            st.rerun()

    elif uploaded_files:
        # Chỉ tải file, không có câu hỏi: cập nhật thanh bên ngay.
        st.rerun()