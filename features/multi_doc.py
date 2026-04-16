"""
features/multi_doc.py — Câu 8: Multi-document RAG + metadata filtering
Thành viên C
"""
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from core.embedder import get_embedder
from core.loader import load_file
from core.chunker import split_documents
from datetime import datetime


class MultiDocStore:
    """
    Câu 8: Quản lý nhiều tài liệu trong một vector store duy nhất.
    Mỗi tài liệu được gắn metadata: source, upload_time, doc_type.
    """

    def __init__(self):
        self.vector_store: FAISS | None = None
        self.doc_registry: dict[str, dict] = {}  # filename → metadata

    def add_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> dict:
        """
        Load, chunk và thêm một tài liệu mới vào store.
        Trả về thống kê: {filename, num_chunks, ...}
        """
        filename = Path(file_path).name
        ext = Path(file_path).suffix.lower().replace(".", "").upper()

        # Load + chunk
        raw_docs = load_file(file_path)
        chunks = split_documents(raw_docs, chunk_size, chunk_overlap)

        # Gắn metadata cho từng chunk
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        for chunk in chunks:
            chunk.metadata.update({
                "source":      filename,
                "doc_type":    ext,
                "upload_time": upload_time,
            })

        # Thêm vào FAISS
        embedder = get_embedder()
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, embedder)
        else:
            new_store = FAISS.from_documents(chunks, embedder)
            self.vector_store.merge_from(new_store)

        # Lưu registry
        self.doc_registry[filename] = {
            "filename":    filename,
            "doc_type":    ext,
            "num_chunks":  len(chunks),
            "upload_time": upload_time,
            "num_pages":   len(raw_docs),
        }
        return self.doc_registry[filename]

    def remove_document(self, filename: str):
        """
        Xoá tài liệu khỏi registry.
        Note: FAISS không hỗ trợ xoá từng vector → rebuild từ đầu.
        """
        if filename in self.doc_registry:
            del self.doc_registry[filename]
        # TODO: rebuild vector store không có file này nếu cần

    def get_retriever(self, k: int = 3, filter_source: str | None = None):
        """
        Trả về retriever. Nếu filter_source đặt tên file,
        chỉ lấy chunks từ tài liệu đó.
        """
        if self.vector_store is None:
            raise ValueError("Chưa có tài liệu nào được tải lên.")

        search_kwargs: dict = {"k": k}
        if filter_source:
            search_kwargs["filter"] = {"source": filter_source}

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def list_documents(self) -> list[dict]:
        """Danh sách tài liệu đã upload."""
        return list(self.doc_registry.values())

    @property
    def has_documents(self) -> bool:
        return bool(self.doc_registry)
