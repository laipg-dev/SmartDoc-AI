"""
app.py — SmartDoc AI: Tích hợp tất cả 10 câu hỏi
Chạy: streamlit run app.py
"""
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama

from core.loader import load_file
from core.chunker import split_documents, benchmark_chunk_configs, PRESETS
from core.retriever import build_vector_store, get_retriever, get_hybrid_retriever
from features.history import init_history, add_message, get_history, get_history_as_text, render_sidebar
from features.citation import extract_citations, format_citation_for_prompt, render_citations
from features.memory import build_context_with_history, build_conversational_prompt, is_followup_question
from features.multi_doc import MultiDocStore
from features.reranker import rerank_with_comparison
from features.self_rag import self_rag_pipeline

load_dotenv()

# ── Cấu hình ────────────────────────────────────────────────────────────
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
RETRIEVER_K   = int(os.getenv("RETRIEVER_K", 3))

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session init ────────────────────────────────────────────────────────
init_history()
if "multi_doc_store" not in st.session_state:
    st.session_state.multi_doc_store = MultiDocStore()
if "use_hybrid"   not in st.session_state: st.session_state.use_hybrid   = False
if "use_rerank"   not in st.session_state: st.session_state.use_rerank   = False
if "use_self_rag" not in st.session_state: st.session_state.use_self_rag = False
if "use_conv_rag" not in st.session_state: st.session_state.use_conv_rag = True

# ── Sidebar ─────────────────────────────────────────────────────────────
render_sidebar()  # Câu 2 + 3

# Cấu hình nâng cao trong sidebar
with st.sidebar:
    st.divider()
    st.subheader("🔧 Tính năng nâng cao")

    # Câu 4: chunk config
    chunk_preset = st.selectbox(
        "Chunk strategy (Câu 4)",
        options=list(PRESETS.keys()),
        index=1,
    )
    selected_cfg = PRESETS[chunk_preset]

    # Câu 6: Conversational RAG
    st.session_state.use_conv_rag = st.toggle(
        "Conversational RAG (Câu 6)", value=st.session_state.use_conv_rag
    )
    # Câu 7: Hybrid search
    st.session_state.use_hybrid = st.toggle(
        "Hybrid Search BM25 (Câu 7)", value=st.session_state.use_hybrid
    )
    # Câu 9: Re-ranking
    st.session_state.use_rerank = st.toggle(
        "Re-ranking Cross-Encoder (Câu 9)", value=st.session_state.use_rerank
    )
    # Câu 10: Self-RAG
    st.session_state.use_self_rag = st.toggle(
        "Self-RAG + Query Rewrite (Câu 10)", value=st.session_state.use_self_rag
    )

    # Câu 8: filter theo tài liệu
    store: MultiDocStore = st.session_state.multi_doc_store
    st.divider()
    st.subheader("📂 Tài liệu (Câu 8)")
    doc_list = store.list_documents()
    if doc_list:
        st.caption(f"{len(doc_list)} tài liệu đã tải")
        filter_options = ["Tất cả"] + [d["filename"] for d in doc_list]
        selected_filter = st.selectbox("Lọc theo tài liệu:", filter_options)
        filter_source = None if selected_filter == "Tất cả" else selected_filter

        for d in doc_list:
            st.caption(f"• {d['filename']} ({d['num_chunks']} chunks, {d['upload_time']})")
    else:
        filter_source = None

# ── Main area ────────────────────────────────────────────────────────────
st.title("📄 SmartDoc AI — Intelligent Document Q&A")
st.caption("RAG System với LLMs | OSSD Spring 2026")

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_qa, tab_bench, tab_history = st.tabs(["💬 Hỏi & Đáp", "📊 Benchmark (Câu 4)", "📜 Lịch sử"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Q&A chính
# ══════════════════════════════════════════════════════════════════════════
with tab_qa:
    # Upload (Câu 1: hỗ trợ PDF + DOCX, Câu 8: nhiều file)
    st.subheader("1. Tải tài liệu")
    uploaded_files = st.file_uploader(
        "Chọn PDF hoặc DOCX (có thể chọn nhiều file — Câu 8)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        store: MultiDocStore = st.session_state.multi_doc_store
        for uf in uploaded_files:
            if uf.name not in [d["filename"] for d in store.list_documents()]:
                with st.spinner(f"Đang xử lý **{uf.name}**..."):
                    suffix = "." + uf.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.getbuffer())
                        tmp_path = tmp.name
                    try:
                        info = store.add_document(
                            tmp_path,
                            selected_cfg.chunk_size,
                            selected_cfg.chunk_overlap,
                        )
                        st.success(f"✅ **{info['filename']}** — {info['num_chunks']} chunks")
                    except NotImplementedError as e:
                        st.error(f"⚠️ {e}")
                    finally:
                        os.unlink(tmp_path)

    # Q&A
    st.divider()
    st.subheader("2. Đặt câu hỏi")

    if not store.has_documents:
        st.info("⬆️ Hãy tải ít nhất một tài liệu lên.")
    else:
        question = st.text_input(
            "Câu hỏi của bạn:",
            placeholder="Nội dung chính của tài liệu là gì?",
        )

        if question:
            with st.spinner("🔍 Đang tìm kiếm và tạo câu trả lời..."):
                t_start = time.time()

                # ── Self-RAG pipeline (Câu 10) ──────────────────────────
                if st.session_state.use_self_rag:
                    filter_src = locals().get("filter_source")
                    retriever = store.get_retriever(k=RETRIEVER_K * 2, filter_source=filter_src)
                    llm = Ollama(model=OLLAMA_MODEL)
                    rag_result = self_rag_pipeline(
                        question, retriever, llm,
                        use_query_rewrite=True,
                        use_self_eval=True,
                    )
                    answer = rag_result["answer"]
                    relevant_docs = rag_result.get("docs", [])

                    # Hiển thị thông tin Self-RAG
                    if rag_result["rewritten_query"] != question:
                        st.info(f"🔄 **Query rewrite:** _{rag_result['rewritten_query']}_")
                    if rag_result["retried"]:
                        st.warning("♻️ Self-RAG đã retry do câu trả lời ban đầu không đủ tốt.")
                    if rag_result["evaluation"]:
                        ev = rag_result["evaluation"]
                        conf = ev["confidence"]
                        color = "green" if conf >= 0.7 else "orange" if conf >= 0.4 else "red"
                        st.markdown(
                            f"<small>🧠 Self-eval: confidence = "
                            f"<b style='color:{color}'>{conf:.0%}</b> | {ev['reason']}</small>",
                            unsafe_allow_html=True,
                        )

                else:
                    # ── Retrieval bình thường ────────────────────────────
                    filter_src = locals().get("filter_source")

                    if st.session_state.use_hybrid and store.vector_store:
                        raw_chunks = [d for d in (store.vector_store.docstore._dict.values())]
                        try:
                            retriever = get_hybrid_retriever(
                                list(store.vector_store.docstore._dict.values()),
                                store.vector_store,
                                k=RETRIEVER_K * 2,
                            )
                            st.caption("🔀 Hybrid Search đang dùng")
                        except Exception:
                            retriever = store.get_retriever(k=RETRIEVER_K * 2, filter_source=filter_src)
                    else:
                        retriever = store.get_retriever(k=RETRIEVER_K * 2, filter_source=filter_src)

                    relevant_docs = retriever.get_relevant_documents(question)

                    # ── Re-ranking (Câu 9) ───────────────────────────────
                    if st.session_state.use_rerank and relevant_docs:
                        try:
                            rerank_result = rerank_with_comparison(question, relevant_docs, RETRIEVER_K)
                            relevant_docs = rerank_result["docs"]
                            if rerank_result["changed"]:
                                st.caption("🔁 Re-ranking đã thay đổi thứ tự kết quả")
                        except Exception as e:
                            st.caption(f"⚠️ Re-ranking không khả dụng: {e}")
                            relevant_docs = relevant_docs[:RETRIEVER_K]
                    else:
                        relevant_docs = relevant_docs[:RETRIEVER_K]

                    # ── Build context (Câu 6: conversational) ───────────
                    citations = extract_citations(relevant_docs)  # Câu 5
                    history_text = get_history_as_text() if st.session_state.use_conv_rag else ""
                    context = build_context_with_history(relevant_docs, history_text)

                    # ── Detect ngôn ngữ & tạo prompt ────────────────────
                    vn_chars = "àáảãạăắặâầấđèéêìíòóôùúư"
                    is_vn = any(c in question.lower() for c in vn_chars)
                    is_followup = is_followup_question(question, get_history())

                    if is_followup and st.session_state.use_conv_rag:
                        st.caption("💡 Phát hiện câu hỏi tiếp theo — đang dùng lịch sử hội thoại")

                    prompt = build_conversational_prompt(question, context, is_vn)

                    llm = Ollama(model=OLLAMA_MODEL)
                    answer = llm.invoke(prompt).strip()

                # ── Lưu lịch sử (Câu 2) ─────────────────────────────────
                citations = extract_citations(relevant_docs)
                add_message("user", question)
                add_message("assistant", answer, sources=citations)

            # ── Hiển thị kết quả ─────────────────────────────────────────
            elapsed = round(time.time() - t_start, 1)
            st.markdown("### 💬 Câu trả lời")
            st.write(answer)
            st.caption(f"⏱️ {elapsed}s | {len(relevant_docs)} chunks retrieved")

            # ── Citation (Câu 5) ─────────────────────────────────────────
            render_citations(citations)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Benchmark chunk strategy (Câu 4)
# ══════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.subheader("📊 So sánh Chunk Strategy (Câu 4)")
    st.caption("Thử tất cả cấu hình chunk và so sánh số lượng + kích thước chunks.")

    store: MultiDocStore = st.session_state.multi_doc_store
    if not store.has_documents:
        st.info("Tải tài liệu lên trước để benchmark.")
    else:
        if st.button("▶️ Chạy Benchmark"):
            # Lấy docs từ store để benchmark
            # Dùng tài liệu đầu tiên trong registry
            first_doc_name = store.list_documents()[0]["filename"]
            st.caption(f"Đang benchmark trên: **{first_doc_name}**")

            # Cần raw docs — lấy từ vector store (approximate)
            import pandas as pd
            # Tạo dummy docs từ vector store để benchmark splitter
            raw_texts = list(store.vector_store.docstore._dict.values())
            results = benchmark_chunk_configs(raw_texts)
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("**Ghi chú:**")
            st.markdown("- **Chunk nhỏ** (500): nhiều chunks, tìm kiếm chính xác hơn nhưng mất ngữ cảnh")
            st.markdown("- **Chunk lớn** (2000): ít chunks, giữ ngữ cảnh tốt hơn nhưng nhiễu hơn")
            st.markdown("- **Khuyến nghị:** 1000–1500 cho tài liệu tiếng Việt")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Lịch sử đầy đủ (Câu 2)
# ══════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("📜 Lịch sử hội thoại đầy đủ (Câu 2)")
    history = get_history()
    if not history:
        st.info("Chưa có hội thoại nào.")
    else:
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            time_str = msg.get("time", "")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(f"**{time_str}** — {content}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
                    sources = msg.get("sources", [])
                    if sources:
                        st.caption(f"📎 {len(sources)} nguồn tham khảo")
