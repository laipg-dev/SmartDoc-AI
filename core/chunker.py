"""
chunker.py
----------
Module chia nhỏ tài liệu (chunking) cho RAG pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkConfig:
    """Bất biến (frozen) để tránh vô tình thay đổi sau khi tạo."""
    chunk_size: int = 1_000
    chunk_overlap: int = 100

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) phải nhỏ hơn "
                f"chunk_size ({self.chunk_size})."
            )


PRESETS: dict[str, ChunkConfig] = {
    "small":  ChunkConfig(chunk_size=500,  chunk_overlap=50),
    "medium": ChunkConfig(chunk_size=1_000, chunk_overlap=100),
    "large":  ChunkConfig(chunk_size=1_500, chunk_overlap=200),
    "xlarge": ChunkConfig(chunk_size=2_000, chunk_overlap=200),
}

DEFAULT_PRESET: str = "medium"


# ---------------------------------------------------------------------------
# Factory splitter
# ---------------------------------------------------------------------------

def _build_splitter(cfg: ChunkConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Chia tài liệu thành các chunks.

    Thứ tự ưu tiên tham số:
      1. Tham số truyền trực tiếp
      2. Biến môi trường CHUNK_SIZE / CHUNK_OVERLAP
      3. Giá trị mặc định của preset 'medium'

    Args:
        docs: Danh sách Document cần chia.
        chunk_size: Kích thước tối đa mỗi chunk (tính theo ký tự).
        chunk_overlap: Số ký tự overlap giữa các chunk liền kề.

    Returns:
        Danh sách Document sau khi chia nhỏ.
    """
    if not docs:
        logger.warning("split_documents nhận danh sách rỗng, trả về [].")
        return []

    default = PRESETS[DEFAULT_PRESET]
    final_size    = chunk_size    or int(os.getenv("CHUNK_SIZE",    default.chunk_size))
    final_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", default.chunk_overlap))

    cfg = ChunkConfig(chunk_size=final_size, chunk_overlap=final_overlap)
    splitter = _build_splitter(cfg)

    chunks = splitter.split_documents(docs)
    logger.info(
        "Chia xong: %d tài liệu → %d chunks (size=%d, overlap=%d)",
        len(docs), len(chunks), cfg.chunk_size, cfg.chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class BenchmarkResult(TypedDict):
    preset: str
    num_chunks: int
    avg_chars: int
    min_chars: int
    max_chars: int
    elapsed_ms: float


def benchmark_chunk_configs(docs: list[Document]) -> list[BenchmarkResult]:
    """
    Chạy tất cả PRESETS trên cùng một bộ tài liệu và trả về thống kê.

    Args:
        docs: Bộ tài liệu dùng để benchmark.

    Returns:
        Danh sách BenchmarkResult, mỗi phần tử ứng với một preset.
    """
    if not docs:
        logger.warning("benchmark_chunk_configs nhận danh sách rỗng.")
        return []

    results: list[BenchmarkResult] = []

    for name, cfg in PRESETS.items():
        t0 = time.perf_counter()
        chunks = _build_splitter(cfg).split_documents(docs)
        elapsed_ms = round((time.perf_counter() - t0) * 1_000, 1)

        lengths = [len(c.page_content) for c in chunks]
        results.append(
            BenchmarkResult(
                preset=name,
                num_chunks=len(chunks),
                avg_chars=round(sum(lengths) / len(lengths)) if lengths else 0,
                min_chars=min(lengths) if lengths else 0,
                max_chars=max(lengths) if lengths else 0,
                elapsed_ms=elapsed_ms,
            )
        )
        logger.debug("Preset '%s': %d chunks trong %.1f ms", name, len(chunks), elapsed_ms)

    return results


# ---------------------------------------------------------------------------
# CLI test độc lập
# ---------------------------------------------------------------------------

def _print_benchmark_table(results: list[BenchmarkResult]) -> None:
    header = f"{'Preset':<10} | {'Chunks':<8} | {'TB chars':<10} | {'Min':<6} | {'Max':<6} | {'ms':<8}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['preset']:<10} | {r['num_chunks']:<8} | "
            f"{r['avg_chars']:<10} | {r['min_chars']:<6} | "
            f"{r['max_chars']:<6} | {r['elapsed_ms']:<8}"
        )
    print(sep)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 50)
    print("CHUNKER — BENCHMARK TEST")
    print("=" * 50)

    file_path = input("\n[?] Đường dẫn PDF/DOCX (Enter = dùng văn bản mẫu): ").strip()

    docs: list[Document] = []
    if file_path:
        if not os.path.exists(file_path):
            print("Không tìm thấy file. Sẽ dùng văn bản mẫu.")
            file_path = ""
        else:
            try:
                from core.loader import load_file
                logger.info("Đang đọc: %s", file_path)
                docs = load_file(file_path)
                logger.info("Đọc xong: %d phần.", len(docs))
            except Exception as exc:
                logger.error("Lỗi khi đọc file: %s", exc)
                sys.exit(1)

    if not docs:
        logger.info("Tạo văn bản mẫu...")
        sample = (
            "Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta làm việc. " * 30 + "\n\n"
            + "RAG (Retrieval-Augmented Generation) là một kỹ thuật tuyệt vời. " * 30 + "\n\n"
            + "Chunking là bước quan trọng nhất trong RAG pipeline. " * 30 + "\n\n"
            + "LangChain giúp xây dựng pipeline một cách dễ dàng. " * 30
        )
        docs = [Document(page_content=sample, metadata={"source": "sample.txt"})]

    print("\nBenchmark các preset chunking...\n")
    try:
        results = benchmark_chunk_configs(docs)
        _print_benchmark_table(results)
        print("\nHoàn tất benchmark!")
    except Exception as exc:
        logger.error("Lỗi khi chạy benchmark: %s", exc)
        sys.exit(1)