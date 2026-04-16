"""
features/history.py — Câu 2 + Câu 3
Câu 2: Lưu & hiển thị lịch sử hội thoại
Câu 3: Nút xóa lịch sử + xóa vector store
Thành viên B
"""
import json
import streamlit as st
from datetime import datetime


# ── Câu 2: Quản lý session ─────────────────────────────────────────────

def init_history():
    """Khởi tạo session state khi app load lần đầu."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_clear_confirm" not in st.session_state:
        st.session_state.show_clear_confirm = False
    if "show_doc_confirm" not in st.session_state:
        st.session_state.show_doc_confirm = False


def add_message(role: str, content: str, sources: list[dict] | None = None):
    """
    Thêm tin nhắn vào lịch sử.
    role: 'user' | 'assistant'
    sources: list citation dùng cho câu 5
    """
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "sources": sources or [],
        "time": datetime.now().strftime("%H:%M"),
    })


def get_history() -> list[dict]:
    return st.session_state.get("chat_history", [])


def get_history_as_text() -> str:
    """Trả về lịch sử dạng text để đưa vào prompt (Câu 6 dùng)."""
    lines = []
    for msg in get_history():
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines[-6:])  # chỉ giữ 3 lượt gần nhất


# ── Câu 3: Xóa lịch sử với confirmation ───────────────────────────────

def clear_history():
    """Xóa toàn bộ lịch sử chat."""
    st.session_state.chat_history = []
    st.session_state.show_clear_confirm = False


def clear_vector_store():
    """Xóa tài liệu đã upload khỏi session."""
    for key in ["vector_store", "raw_docs", "current_filename"]:
        st.session_state.pop(key, None)
    st.session_state.show_doc_confirm = False


def render_sidebar():
    """
    Render toàn bộ sidebar:
    - Lịch sử hội thoại (Câu 2)
    - Nút xóa có confirmation dialog (Câu 3)
    """
    with st.sidebar:
        st.header("📄 SmartDoc AI")
        st.caption("Intelligent Document Q&A")
        st.divider()

        # ── Lịch sử (Câu 2) ────────────────────────────
        st.subheader("💬 Lịch sử hội thoại")
        history = get_history()

        if not history:
            st.caption("_Chưa có câu hỏi nào._")
        else:
            # Hiển thị tối đa 10 tin gần nhất
            for msg in history[-10:]:
                icon = "🧑" if msg["role"] == "user" else "🤖"
                time = msg.get("time", "")
                content = msg["content"]
                short = content[:60] + "..." if len(content) > 60 else content
                st.markdown(
                    f"<small style='color:gray'>{time}</small> {icon} {short}",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Nút xóa (Câu 3) ────────────────────────────
        st.subheader("🗑️ Xóa dữ liệu")

        # Xóa lịch sử chat
        if not st.session_state.get("show_clear_confirm"):
            if st.button("Xóa lịch sử chat", use_container_width=True):
                st.session_state.show_clear_confirm = True
                st.rerun()
        else:
            st.warning("⚠️ Xác nhận xóa toàn bộ lịch sử chat?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Xóa", use_container_width=True):
                    clear_history()
                    st.rerun()
            with col2:
                if st.button("❌ Huỷ", use_container_width=True):
                    st.session_state.show_clear_confirm = False
                    st.rerun()

        st.markdown("")

        # Xóa vector store
        if not st.session_state.get("show_doc_confirm"):
            if st.button("Xóa tài liệu đã tải", use_container_width=True):
                st.session_state.show_doc_confirm = True
                st.rerun()
        else:
            st.warning("⚠️ Xác nhận xóa tài liệu? Cần upload lại để hỏi tiếp.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Xóa ", use_container_width=True):
                    clear_vector_store()
                    st.rerun()
            with col2:
                if st.button("❌ Huỷ ", use_container_width=True):
                    st.session_state.show_doc_confirm = False
                    st.rerun()

        # Thông tin tài liệu hiện tại
        if "current_filename" in st.session_state:
            st.divider()
            st.caption(f"📂 **{st.session_state.current_filename}**")
            chunks = st.session_state.get("num_chunks", "?")
            st.caption(f"Chunks: {chunks}")
