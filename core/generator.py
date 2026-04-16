"""
core/generator.py
-----------------
Quản lý LLM và sinh câu trả lời cho RAG pipeline.

Chuyển đổi qua env:
  LLM_PROVIDER   — gemini | ollama  (mặc định: auto-detect)
  GEMINI_MODEL   — tên model Gemini  (mặc định: models/gemini-2.5-flash)
  OLLAMA_MODEL   — tên model Ollama  (mặc định: qwen2.5:3b)
  OLLAMA_BASE_URL — URL Ollama server (mặc định: http://localhost:11434)
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số provider
# ---------------------------------------------------------------------------

PROVIDER_GEMINI = "gemini"
PROVIDER_OLLAMA = "ollama"

_SUPPORTED_PROVIDERS = {PROVIDER_GEMINI, PROVIDER_OLLAMA}

_DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"
_DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"

# ---------------------------------------------------------------------------
# Prompt — module-level constant (immutable, không cần tạo lại mỗi lần)
# ---------------------------------------------------------------------------

_RAG_PROMPT = PromptTemplate.from_template(
    "Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi CHỈ DỰA VÀO TÀI LIỆU "
    "được cung cấp dưới đây.\n"
    "Nếu tài liệu không chứa câu trả lời, hãy nói: "
    "\"Tôi không tìm thấy thông tin trong tài liệu.\"\n"
    "Tuyệt đối không sử dụng kiến thức bên ngoài.\n\n"
    "TÀI LIỆU:\n{context}\n\n"
    "CÂU HỎI: {query}\n"
    "CÂU TRẢ LỜI:"
)


# ---------------------------------------------------------------------------
# Provider resolver
# ---------------------------------------------------------------------------

def _resolve_provider() -> str:
    """
    Đọc LLM_PROVIDER từ env và validate.

    Logic 'auto': chọn gemini nếu GOOGLE_API_KEY tồn tại, ngược lại ollama.

    Returns:
        Tên provider đã được chuẩn hoá (lowercase).

    Raises:
        ValueError: Nếu provider không nằm trong _SUPPORTED_PROVIDERS.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower().strip()

    if provider == "auto":
        provider = PROVIDER_GEMINI if os.getenv("GOOGLE_API_KEY") else PROVIDER_OLLAMA
        logger.info("LLM_PROVIDER=auto → chọn: %s", provider.upper())

    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' không hợp lệ. "
            f"Chọn một trong: {sorted(_SUPPORTED_PROVIDERS)}"
        )

    return provider


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

class LLMFactory:
    """
    Tạo LLM instance theo provider.
    Lazy import để tránh load thư viện không cần thiết.
    """

    @staticmethod
    def create_gemini() -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Thiếu GOOGLE_API_KEY trong file .env")

        model = os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
        logger.info("Gemini LLM — model: %s", model)

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            google_api_key=api_key,
        )

    @staticmethod
    def create_ollama() -> BaseChatModel:
        from langchain_ollama import ChatOllama  # noqa: PLC0415

        model    = os.getenv("OLLAMA_MODEL",    _DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info("Ollama LLM — model: %s | url: %s", model, base_url)

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.1,
        )

    _CREATORS = {
        PROVIDER_GEMINI: create_gemini.__func__,  # type: ignore[attr-defined]
        PROVIDER_OLLAMA: create_ollama.__func__,  # type: ignore[attr-defined]
    }

    @classmethod
    def create(cls, provider: str) -> BaseChatModel:
        """
        Tạo LLM theo provider.

        Args:
            provider: Tên provider đã được validate bởi _resolve_provider().

        Returns:
            BaseChatModel instance.
        """
        return cls._CREATORS[provider]()


# ---------------------------------------------------------------------------
# Singleton (lru_cache) — nhất quán với get_embedder()
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Trả về LLM singleton theo biến môi trường LLM_PROVIDER.

    Kết quả được cache sau lần gọi đầu tiên. Nếu cần reload
    (ví dụ: đổi provider trong test), gọi reset_llm_cache() trước.

    Returns:
        BaseChatModel instance đang được dùng.
    """
    provider = _resolve_provider()
    logger.info("Khởi tạo LLM — provider: %s", provider.upper())

    t0 = time.perf_counter()
    llm = LLMFactory.create(provider)
    logger.info("LLM sẵn sàng sau %.2f s", time.perf_counter() - t0)

    return llm


def reset_llm_cache() -> None:
    """
    Xoá cache singleton, cho phép get_llm() tạo instance mới.
    Hữu ích khi đổi LLM_PROVIDER trong runtime hoặc trong unit test.
    """
    get_llm.cache_clear()
    logger.info("LLM cache đã được xoá.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_context(docs: list[Document]) -> str:
    """Gom các chunk thành context string với separator và nhãn nguồn."""
    return "\n\n---\n\n".join(
        f"[Nguồn: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def generate_answer(query: str, retrieved_docs: list[Document]) -> str:
    """
    Sinh câu trả lời từ câu hỏi và danh sách Document đã retrieve.

    Args:
        query: Câu hỏi của người dùng.
        retrieved_docs: Danh sách chunk liên quan từ retriever.

    Returns:
        Câu trả lời dạng string. Trả về thông báo cố định nếu không có context
        hoặc nếu LLM gặp lỗi.
    """
    if not retrieved_docs:
        logger.warning("generate_answer: retrieved_docs rỗng, bỏ qua LLM.")
        return "Tôi không tìm thấy thông tin trong tài liệu."

    context = _build_context(retrieved_docs)
    chain   = _RAG_PROMPT | get_llm() | StrOutputParser()

    try:
        t0     = time.perf_counter()
        answer = chain.invoke({"context": context, "query": query})
        logger.info("Sinh xong câu trả lời sau %.2f s.", time.perf_counter() - t0)
        return answer
    except Exception as exc:
        logger.error("Lỗi khi gọi LLM: %s", exc)
        return f"Có lỗi xảy ra khi kết nối với mô hình AI: {exc}"


# ---------------------------------------------------------------------------
# CLI test độc lập
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from core.retriever import load_vector_store  # noqa: E402

    print("=" * 55)
    print("GENERATOR — TEST SINH CÂU TRẢ LỜI")
    print("=" * 55)

    try:
        logger.info("[1/3] Tải Vector Store...")
        vector_store = load_vector_store()

        query = "Thanh đang xây dựng hệ thống gì?"
        logger.info("[2/3] Truy xuất context cho: '%s'", query)
        docs = vector_store.similarity_search(query, k=3)
        logger.info("Tìm được %d chunk.", len(docs))

        logger.info("[3/3] Sinh câu trả lời...")
        answer = generate_answer(query, docs)

        print("\n" + "-" * 40)
        print(f"Câu hỏi : {query}")
        print(f"Trả lời : {answer}")
        print("-" * 40)

    except FileNotFoundError as exc:
        logger.error("Vector Store chưa được build: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Lỗi không xác định: %s", exc)
        sys.exit(1)