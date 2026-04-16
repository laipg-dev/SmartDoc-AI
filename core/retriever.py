"""
core/retriever.py
-----------------
Hybrid Search (BM25 + FAISS) cho RAG pipeline.

Chuyển đổi qua env:
  RETRIEVER_K        — số kết quả trả về (mặc định: 3)
  VECTOR_STORE_PATH  — đường dẫn lưu/đọc FAISS index (tuỳ chọn)
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from core.embedder import get_embedder

if TYPE_CHECKING:
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.vectorstores import VectorStoreRetriever

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số / cấu hình mặc định
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ưu tiên: env var → đường dẫn tính từ vị trí file (không phụ thuộc cwd)
VECTOR_STORE_PATH: str = os.getenv(
    "VECTOR_STORE_PATH",
    os.path.join(_MODULE_DIR, "vector_store", "faiss_index"),
)

DEFAULT_K: int = int(os.getenv("RETRIEVER_K", 3))


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def build_vector_store(
    docs: list[Document],
    save_path: str = VECTOR_STORE_PATH,
) -> FAISS:
    """
    Tạo FAISS index từ danh sách Document và lưu xuống ổ cứng.

    Args:
        docs: Danh sách chunk đã qua split_documents().
        save_path: Thư mục lưu index (ghi đè VECTOR_STORE_PATH nếu truyền vào).

    Returns:
        FAISS instance đã được lưu.

    Raises:
        ValueError: Nếu docs rỗng.
    """
    if not docs:
        raise ValueError("Không thể tạo Vector Store từ danh sách docs rỗng.")

    embedder = get_embedder()
    logger.info("Mã hoá và tạo Vector DB cho %d chunks...", len(docs))

    t0 = time.perf_counter()
    vector_store = FAISS.from_documents(docs, embedder)
    logger.info("FAISS index tạo xong sau %.2f s.", time.perf_counter() - t0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vector_store.save_local(save_path)
    logger.info("Đã lưu Vector DB tại: %s", save_path)

    return vector_store


def load_vector_store(load_path: str = VECTOR_STORE_PATH) -> FAISS:
    """
    Tải FAISS index từ ổ cứng.

    Args:
        load_path: Đường dẫn thư mục chứa index.

    Returns:
        FAISS instance đã tải.

    Raises:
        FileNotFoundError: Nếu không tìm thấy index.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"Không tìm thấy Vector DB tại '{load_path}'. "
            "Hãy chạy build_vector_store() trước."
        )

    embedder = get_embedder()
    logger.info("Đang tải Vector DB từ: %s", load_path)

    t0 = time.perf_counter()
    store = FAISS.load_local(load_path, embedder, allow_dangerous_deserialization=True)
    logger.info("Tải xong sau %.2f s.", time.perf_counter() - t0)

    return store


# ---------------------------------------------------------------------------
# Semantic retriever (FAISS)
# ---------------------------------------------------------------------------

def get_retriever(
    vector_store: FAISS,
    k: int = DEFAULT_K,
) -> VectorStoreRetriever:
    """
    Retriever cơ bản dùng cosine similarity.

    Args:
        vector_store: FAISS instance đã build hoặc load.
        k: Số kết quả trả về.

    Returns:
        LangChain VectorStoreRetriever tương thích với LCEL.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "fetch_k": max(k * 4, 20)},
    )


# ---------------------------------------------------------------------------
# BM25 cache (tránh build lại mỗi lần gọi)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_bm25_retriever(doc_fingerprint: tuple[str, ...], k: int) -> BM25Retriever:
    """
    Build và cache BM25Retriever.
    Cache key là tuple nội dung các chunk + k để tránh build lại
    khi cùng bộ tài liệu được truy vấn nhiều lần.

    Args:
        doc_fingerprint: Tuple page_content của tất cả chunks (làm cache key).
        k: Số kết quả trả về.

    Returns:
        BM25Retriever đã được cache.
    """
    from langchain_community.retrievers import BM25Retriever  # noqa: PLC0415

    docs = [Document(page_content=text) for text in doc_fingerprint]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    logger.debug("BM25Retriever: build mới cho %d docs, k=%d.", len(docs), k)
    return retriever


def reset_bm25_cache() -> None:
    """Xoá cache BM25 — hữu ích khi bộ tài liệu thay đổi hoặc trong unit test."""
    _get_bm25_retriever.cache_clear()
    logger.info("BM25 cache đã được xoá.")


# ---------------------------------------------------------------------------
# HybridRetriever — implement đúng BaseRetriever của LangChain
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """
    Kết hợp BM25 (keyword) + FAISS (semantic), dedup và trả về top-k.

    Tương thích hoàn toàn với LCEL pipeline và LangChain callback system.

    Attributes:
        docs: Danh sách chunk gốc (dùng để build BM25).
        vector_store: FAISS instance đã được build/load.
        k: Số kết quả trả về sau khi merge.
    """

    docs: list[Document]
    vector_store: FAISS
    k: int = DEFAULT_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # Tạo fingerprint để cache BM25 theo nội dung thực tế
        fingerprint = tuple(d.page_content for d in self.docs)
        bm25 = _get_bm25_retriever(fingerprint, self.k)
        faiss_ret = get_retriever(self.vector_store, k=self.k)

        res_bm25  = bm25.invoke(query)
        res_faiss = faiss_ret.invoke(query)

        # Dedup theo (source, page_content) — tránh loại nhầm chunk cùng nội dung khác nguồn
        seen: set[tuple[str, str]] = set()
        unique: list[Document] = []
        for doc in res_faiss + res_bm25:
            key = (doc.metadata.get("source", ""), doc.page_content)
            if key not in seen:
                unique.append(doc)
                seen.add(key)

        logger.debug(
            "HybridRetriever: bm25=%d, faiss=%d, merged=%d, returned=%d",
            len(res_bm25), len(res_faiss), len(unique), min(len(unique), self.k),
        )
        return unique[: self.k]


def get_hybrid_retriever(
    docs: list[Document],
    vector_store: FAISS,
    k: int = DEFAULT_K,
) -> HybridRetriever:
    """
    Factory cho HybridRetriever.

    Args:
        docs: Toàn bộ chunks (dùng để build BM25 index).
        vector_store: FAISS instance.
        k: Số kết quả trả về.

    Returns:
        HybridRetriever tương thích LCEL.
    """
    return HybridRetriever(docs=docs, vector_store=vector_store, k=k)


# ---------------------------------------------------------------------------
# Semantic search kèm score (dùng cho re-ranking)
# ---------------------------------------------------------------------------

def retrieve_with_scores(
    vector_store: FAISS,
    query: str,
    k: int = DEFAULT_K,
) -> list[tuple[Document, float]]:
    """
    Truy xuất kèm điểm similarity đã chuẩn hoá về [0, 1].

    Score được tính: `1 / (1 + l2_distance)` — càng gần 1 càng liên quan.
    Kết quả sắp xếp giảm dần theo score.

    Args:
        vector_store: FAISS instance.
        query: Câu truy vấn.
        k: Số kết quả trả về.

    Returns:
        List các tuple (Document, normalized_score).
    """
    raw = vector_store.similarity_search_with_score(query, k=k)
    scored = [(doc, 1.0 / (1.0 + dist)) for doc, dist in raw]
    return sorted(scored, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# CLI test độc lập
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    project_root = _MODULE_DIR
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from core.chunker import split_documents  # noqa: E402

    print("=" * 60)
    print("RETRIEVER — TEST LUỒNG: CHUNKER → EMBEDDER → RETRIEVER")
    print("=" * 60)

    # [1] Dữ liệu mẫu
    logger.info("[1/4] Tạo dữ liệu mẫu...")
    sample_text = (
        "Trí tuệ nhân tạo (AI) đang thay đổi thế giới. RAG là một hệ thống rất hữu ích.\n\n"
        "Spring Boot 3 kết hợp với PostgreSQL tạo ra các backend mạnh mẽ.\n\n"
        "Thanh đang xây dựng ứng dụng Flight Booking Backend dùng Clean Architecture.\n\n"
        "Ollama cho phép chạy LLM ngay trên máy tính cục bộ."
    )
    raw_docs = [Document(page_content=sample_text, metadata={"source": "test_doc.txt"})]

    # [2] Chunking
    logger.info("[2/4] Chunking...")
    chunks = split_documents(raw_docs, chunk_size=100, chunk_overlap=20)
    logger.info("Tạo được %d chunks.", len(chunks))

    # [3] Build Vector Store
    logger.info("[3/4] Build Vector Store...")
    try:
        vs = build_vector_store(chunks)
    except Exception as exc:
        logger.error("Lỗi build vector store: %s", exc)
        sys.exit(1)

    # [4] Tìm kiếm
    query = "Ai đang xây dựng backend đặt vé máy bay?"
    logger.info("[4/4] Tìm kiếm: '%s'", query)

    print("\n--- SEMANTIC SEARCH (kèm score) ---")
    for i, (doc, score) in enumerate(retrieve_with_scores(vs, query, k=2), 1):
        print(f"  {i}. [score={score:.3f}] {doc.page_content.strip()}")

    print("\n--- HYBRID SEARCH (BM25 + FAISS) ---")
    hybrid = get_hybrid_retriever(chunks, vs, k=2)
    for i, doc in enumerate(hybrid.invoke(query), 1):
        print(f"  {i}. {doc.page_content.strip()}")

    print("\nHoàn tất test Retriever!")