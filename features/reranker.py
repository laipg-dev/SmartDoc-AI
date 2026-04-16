"""
features/reranker.py — Câu 9: Re-ranking với Cross-Encoder
Leader
"""
import time
from functools import lru_cache
from langchain.schema import Document


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load cross-encoder model một lần duy nhất (cache)."""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(CROSS_ENCODER_MODEL)
    except ImportError:
        raise ImportError("Cài sentence-transformers: pip install sentence-transformers")


def rerank(
    query: str,
    docs: list[Document],
    top_k: int = 3,
) -> tuple[list[Document], list[float]]:
    """
    Câu 9: Re-rank retrieved chunks bằng Cross-Encoder.

    Tại sao cần re-rank?
    - Bi-encoder (FAISS) nhanh nhưng kém chính xác → lấy top-20
    - Cross-encoder so sánh query với từng chunk → chính xác hơn nhiều
    - Chỉ chạy cross-encoder trên top-20 → cân bằng tốc độ & chất lượng

    Trả về (docs_ranked, scores)
    """
    if not docs:
        return [], []

    model = _get_cross_encoder()

    # Tạo pairs (query, chunk_content)
    pairs = [(query, doc.page_content) for doc in docs]

    t0 = time.time()
    scores = model.predict(pairs).tolist()
    latency_ms = round((time.time() - t0) * 1000, 1)

    # Sắp xếp theo score giảm dần
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_scores = [s for s, _ in ranked[:top_k]]
    top_docs = [d for _, d in ranked[:top_k]]

    # Gắn thêm rerank score vào metadata để hiển thị
    for doc, score in zip(top_docs, top_scores):
        doc.metadata["rerank_score"] = round(score, 4)
        doc.metadata["rerank_latency_ms"] = latency_ms

    return top_docs, top_scores


def rerank_with_comparison(
    query: str,
    docs: list[Document],
    top_k: int = 3,
) -> dict:
    """
    Câu 9: So sánh kết quả trước và sau re-ranking.
    Trả về dict để hiển thị trong UI.
    """
    before_order = [d.metadata.get("source", "?") + f" (trang {d.metadata.get('page','?')})"
                    for d in docs[:top_k]]

    reranked_docs, scores = rerank(query, docs, top_k)

    after_order = [d.metadata.get("source", "?") + f" (trang {d.metadata.get('page','?')})"
                   for d in reranked_docs]

    return {
        "before": before_order,
        "after":  after_order,
        "scores": [round(s, 4) for s in scores],
        "docs":   reranked_docs,
        "changed": before_order != after_order,
    }
