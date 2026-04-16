"""
embedder.py
-----------
Factory + Singleton để khởi tạo và quản lý embedding models.
Hỗ trợ: Google Generative AI, Ollama (local), HuggingFace (local).
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số provider
# ---------------------------------------------------------------------------

PROVIDER_GOOGLE = "google"
PROVIDER_OLLAMA = "ollama"
PROVIDER_HF     = "huggingface"

_SUPPORTED_PROVIDERS = {PROVIDER_GOOGLE, PROVIDER_OLLAMA, PROVIDER_HF}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class EmbedderFactory:
    """
    Tạo embedder theo provider.
    Mỗi method chỉ import thư viện khi thực sự cần (lazy import),
    tránh load thư viện nặng không cần thiết.
    """

    @staticmethod
    def create_google() -> Embeddings:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: PLC0415

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Thiếu GOOGLE_API_KEY trong file .env")

        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        logger.info("Google embedder — model: %s", model)
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    @staticmethod
    def create_ollama() -> Embeddings:
        from langchain_community.embeddings import OllamaEmbeddings  # noqa: PLC0415

        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        logger.info("Ollama embedder — model: %s", model)
        return OllamaEmbeddings(model=model)

    @staticmethod
    def create_huggingface() -> Embeddings:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415

        model  = os.getenv("HF_EMBEDDING_MODEL",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        logger.info("HuggingFace embedder — model: %s | device: %s", model, device)

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

    # Map tên provider → method tương ứng
    _CREATORS = {
        PROVIDER_GOOGLE: create_google.__func__,   # type: ignore[attr-defined]
        PROVIDER_OLLAMA: create_ollama.__func__,   # type: ignore[attr-defined]
        PROVIDER_HF:     create_huggingface.__func__,  # type: ignore[attr-defined]
    }

    @classmethod
    def create(cls, provider: str) -> Embeddings:
        """
        Tạo embedder theo tên provider.

        Args:
            provider: Một trong 'google', 'ollama', 'huggingface'.

        Raises:
            ValueError: Nếu provider không được hỗ trợ.
        """
        creator = cls._CREATORS.get(provider)
        if creator is None:
            raise ValueError(
                f"Provider '{provider}' không được hỗ trợ. "
                f"Chọn một trong: {sorted(_SUPPORTED_PROVIDERS)}"
            )
        return creator()


# ---------------------------------------------------------------------------
# Singleton (lru_cache)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder() -> Embeddings:
    """
    Trả về embedder singleton theo biến môi trường EMBEDDING_PROVIDER.

    Kết quả được cache sau lần gọi đầu tiên. Nếu cần reload
    (ví dụ: đổi provider trong test), gọi reset_embedder_cache() trước.

    Returns:
        Instance Embeddings đang được dùng.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", PROVIDER_HF).lower().strip()
    logger.info("Khởi tạo embedder — provider: %s", provider.upper())

    t0 = time.perf_counter()
    embedder = EmbedderFactory.create(provider)
    logger.info("Embedder sẵn sàng sau %.2f s", time.perf_counter() - t0)

    return embedder


def reset_embedder_cache() -> None:
    """
    Xoá cache singleton, cho phép get_embedder() tạo instance mới.
    Hữu ích khi đổi EMBEDDING_PROVIDER trong runtime hoặc trong unit test.
    """
    get_embedder.cache_clear()
    logger.info("Embedder cache đã được xoá.")


# ---------------------------------------------------------------------------
# CLI test độc lập
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 50)
    print("EMBEDDER FACTORY — SMOKE TEST")
    print("=" * 50)

    try:
        embedder = get_embedder()

        test_text = "Hệ thống RAG cho phép đọc tài liệu thông minh."
        t0 = time.perf_counter()
        vector = embedder.embed_query(test_text)
        elapsed = round((time.perf_counter() - t0) * 1_000, 1)

        print(f"\nVăn bản:         {test_text}")
        print(f"Số chiều vector: {len(vector)}")
        print(f"3 giá trị đầu:   {[round(v, 4) for v in vector[:3]]}")
        print(f"Thời gian embed: {elapsed} ms")
        print("\nSmoke test thành công!")

    except Exception as exc:
        logger.error("Lỗi: %s", exc)
        sys.exit(1)