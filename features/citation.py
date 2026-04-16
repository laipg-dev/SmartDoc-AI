"""
features/citation.py — Câu 5: Citation & source tracking
Thành viên C
"""
import streamlit as st
from langchain.schema import Document


def extract_citations(docs: list[Document]) -> list[dict]:
    """
    Trích xuất thông tin nguồn từ retrieved documents.
    Trả về list[{page, source, snippet, char_count}]
    """
    citations = []
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        citations.append({
            "index":      i + 1,
            "source":     meta.get("source", "Không rõ nguồn"),
            "page":       meta.get("page", "?"),
            "snippet":    doc.page_content[:300],
            "full_text":  doc.page_content,
            "char_count": len(doc.page_content),
        })
    return citations


def format_citation_for_prompt(citations: list[dict]) -> str:
    """
    Format citations thành chuỗi để đưa vào prompt,
    giúp LLM biết nguồn nào để tham chiếu.
    """
    lines = []
    for c in citations:
        lines.append(
            f"[Nguồn {c['index']}] Trang {c['page']} ({c['source']}):\n{c['snippet']}"
        )
    return "\n\n".join(lines)


def render_citations(citations: list[dict]):
    """
    Câu 5: Hiển thị citations trong Streamlit expander.
    Cho phép xem full text của từng chunk.
    """
    if not citations:
        return

    with st.expander(f"📎 Nguồn tham khảo ({len(citations)} đoạn)", expanded=False):
        for c in citations:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(
                    f"<div style='background:#e8f4f8;border-radius:8px;"
                    f"padding:8px;text-align:center'>"
                    f"<b>[{c['index']}]</b><br>"
                    f"<small>Trang {c['page']}</small></div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.caption(f"📄 **{c['source']}** — {c['char_count']} ký tự")
                st.markdown(f"> {c['snippet']}{'...' if len(c['full_text']) > 300 else ''}")

                # Toggle xem full text
                if len(c["full_text"]) > 300:
                    with st.popover("Xem đầy đủ"):
                        st.text(c["full_text"])
            st.divider()
