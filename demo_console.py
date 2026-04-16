"""
app.py
------
Console app RAG pipeline:
  Chọn tài liệu → Chunk → Embed → Index → Chat

Chạy:
    python app.py

Env (.env):
    LLM_PROVIDER        = gemini | ollama
    EMBEDDING_PROVIDER  = google | ollama | huggingface
    CHUNK_SIZE          = 1000
    CHUNK_OVERLAP       = 100
    RETRIEVER_K         = 3
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging — chỉ hiển thị WARNING trở lên từ thư viện ngoài,
#           INFO từ các core module của project
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
for _mod in ("core.loader", "core.chunker", "core.embedder", "core.retriever", "core.generator"):
    logging.getLogger(_mod).setLevel(logging.INFO)

logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_R = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_WHITE  = "\033[97m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{_R}"

def _header(text: str) -> None:
    width = 60
    print(f"\n{_BOLD}{'─' * width}{_R}")
    print(f"{_BOLD}  {text}{_R}")
    print(f"{_BOLD}{'─' * width}{_R}")

def _section(text: str) -> None:
    print(f"\n{_CYAN}{_BOLD}▸ {text}{_R}")

def _ok(text: str)   -> None: print(f"  {_GREEN}✓{_R} {text}")
def _err(text: str)  -> None: print(f"  {_RED}✗{_R} {text}")
def _info(text: str) -> None: print(f"  {_DIM}{text}{_R}")
def _warn(text: str) -> None: print(f"  {_YELLOW}⚠{_R}  {text}")

def _elapsed(ms: float) -> str:
    return _c(_DIM, f"({ms:.0f} ms)")


# ---------------------------------------------------------------------------
# State của app (giữ trong session)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.chunks:       list      = []
        self.vector_store: object    = None
        self.indexed_file: str | None = None
        self.chat_history: list[dict] = []

    @property
    def ready(self) -> bool:
        """Pipeline đã sẵn sàng để chat chưa."""
        return self.vector_store is not None and bool(self.chunks)


_state = AppState()


# ---------------------------------------------------------------------------
# Bước 1: Chọn tài liệu từ /data
# ---------------------------------------------------------------------------

def _scan_data_dir() -> list[Path]:
    """Quét thư mục /data và trả về danh sách file PDF/DOCX."""
    data_dir = _PROJECT_ROOT / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        _warn(f"Đã tạo thư mục: {data_dir}")
        _warn("Hãy thêm file PDF/DOCX vào đó rồi chạy lại.")
        return []

    files = sorted(
        f for f in data_dir.rglob("*")
        if f.suffix.lower() in {".pdf", ".docx"} and f.is_file()
    )
    return files


def _pick_file() -> Path | None:
    """Hiển thị menu chọn file, trả về Path hoặc None nếu thoát."""
    files = _scan_data_dir()

    if not files:
        _err("Không tìm thấy file PDF/DOCX nào trong thư mục data/")
        return None

    _section("Chọn tài liệu để nạp")
    for i, f in enumerate(files, 1):
        rel = f.relative_to(_PROJECT_ROOT)
        size_kb = round(f.stat().st_size / 1024, 1)
        print(f"  {_c(_BOLD, str(i))}.  {f.name}  {_c(_DIM, f'({size_kb} KB — {rel.parent})')}")

    print(f"  {_c(_DIM, '0.  Quay lại')}")

    while True:
        raw = input(f"\n{_c(_CYAN, '?')} Nhập số: ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            return files[int(raw) - 1]
        _warn("Nhập số hợp lệ.")


# ---------------------------------------------------------------------------
# Bước 2–5: Load → Chunk → Embed → Index
# ---------------------------------------------------------------------------

def _run_indexing(file_path: Path) -> bool:
    """
    Chạy toàn bộ pipeline nạp tài liệu.
    Trả về True nếu thành công.
    """
    from core.chunker import split_documents
    from core.embedder import get_embedder, reset_embedder_cache
    from core.loader import load_file
    from core.retriever import build_vector_store

    _header(f"Nạp tài liệu: {file_path.name}")

    # --- Load ---
    _section("Đọc tài liệu")
    t0 = time.perf_counter()
    try:
        raw_docs = load_file(str(file_path))
        ms = (time.perf_counter() - t0) * 1000
        total_chars = sum(len(d.page_content) for d in raw_docs)
        _ok(f"{len(raw_docs)} trang  ·  {total_chars:,} ký tự  {_elapsed(ms)}")
    except Exception as exc:
        _err(f"Không thể đọc file: {exc}")
        return False

    # --- Chunk ---
    _section("Chia chunks")
    chunk_size    = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    _info(f"chunk_size={chunk_size}  chunk_overlap={chunk_overlap}")

    t0 = time.perf_counter()
    try:
        chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ms = (time.perf_counter() - t0) * 1000
        lengths = [len(c.page_content) for c in chunks]
        _ok(
            f"{len(chunks)} chunks  ·  "
            f"TB {round(sum(lengths)/len(lengths))} ký tự  ·  "
            f"min={min(lengths)} max={max(lengths)}  {_elapsed(ms)}"
        )
    except Exception as exc:
        _err(f"Chunking thất bại: {exc}")
        return False

    # --- Embed (smoke test) ---
    _section("Khởi tạo Embedder")
    provider_emb = os.getenv("EMBEDDING_PROVIDER", "huggingface").upper()
    _info(f"EMBEDDING_PROVIDER={provider_emb}")

    t0 = time.perf_counter()
    try:
        reset_embedder_cache()
        embedder = get_embedder()
        probe    = embedder.embed_query("test")
        ms = (time.perf_counter() - t0) * 1000
        _ok(f"dims={len(probe)}  {_elapsed(ms)}")
    except Exception as exc:
        _err(f"Embedder lỗi: {exc}")
        return False

    # --- Build FAISS index ---
    _section("Tạo Vector Index (FAISS)")
    t0 = time.perf_counter()
    try:
        vs = build_vector_store(chunks)
        ms = (time.perf_counter() - t0) * 1000
        _ok(f"Index sẵn sàng  {_elapsed(ms)}")
    except Exception as exc:
        _err(f"Build index thất bại: {exc}")
        return False

    # Lưu vào state
    _state.chunks       = chunks
    _state.vector_store = vs
    _state.indexed_file = file_path.name
    _state.chat_history = []

    print(f"\n  {_GREEN}{_BOLD}Sẵn sàng!{_R}  Tài liệu «{file_path.name}» đã được nạp.\n")
    return True


# ---------------------------------------------------------------------------
# Bước 6: Chat loop
# ---------------------------------------------------------------------------

def _format_sources(docs: list) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for doc in docs:
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"{src}" + (f" trang {page}" if page else "")
        if label not in seen:
            parts.append(label)
            seen.add(label)
    return " · ".join(parts)


def _chat_loop() -> None:
    """Vòng lặp Q&A với tài liệu đang được index."""
    from core.generator import generate_answer
    from core.retriever import get_hybrid_retriever, retrieve_with_scores

    k         = int(os.getenv("RETRIEVER_K", 3))
    llm_prov  = os.getenv("LLM_PROVIDER", "auto").upper()
    emb_prov  = os.getenv("EMBEDDING_PROVIDER", "huggingface").upper()

    _header(f"Chat với «{_state.indexed_file}»")
    print(f"  {_c(_DIM, f'LLM: {llm_prov}  ·  Embedding: {emb_prov}  ·  K: {k}')}")
    print(f"  {_c(_DIM, 'Lệnh: /quit thoát  ·  /clear xoá lịch sử  ·  /info xem cấu hình')}\n")

    hybrid_retriever = get_hybrid_retriever(
        _state.chunks, _state.vector_store, k=k
    )

    while True:
        try:
            raw = input(f"{_c(_BOLD + _WHITE, 'Bạn')}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # --- Lệnh đặc biệt ---
        if raw.lower() == "/quit":
            break

        if raw.lower() == "/clear":
            _state.chat_history = []
            _ok("Đã xoá lịch sử hội thoại.")
            continue

        if raw.lower() == "/info":
            print(f"  Tài liệu   : {_state.indexed_file}")
            print(f"  Chunks     : {len(_state.chunks)}")
            print(f"  LLM        : {llm_prov}")
            print(f"  Embedding  : {emb_prov}")
            print(f"  K          : {k}")
            continue

        if raw.lower().startswith("/"):
            _warn(f"Lệnh không hợp lệ: {raw}")
            continue

        # --- Retrieve ---
        t0 = time.perf_counter()
        try:
            retrieved = hybrid_retriever.invoke(raw)
        except Exception as exc:
            _err(f"Retriever lỗi: {exc}")
            continue

        retrieve_ms = (time.perf_counter() - t0) * 1000

        if not retrieved:
            print(f"\n{_c(_YELLOW, 'AI')}: Không tìm thấy đoạn văn liên quan trong tài liệu.\n")
            continue

        # --- Generate ---
        t0 = time.perf_counter()
        try:
            answer = generate_answer(raw, retrieved)
        except Exception as exc:
            _err(f"Generator lỗi: {exc}")
            continue

        gen_ms = (time.perf_counter() - t0) * 1000

        # --- Hiển thị ---
        sources = _format_sources(retrieved)
        print(f"\n{_c(_GREEN + _BOLD, 'AI')}: {answer}")
        print(
            f"{_c(_DIM, f'  Nguồn: {sources}  ·  '
                        f'retrieve {retrieve_ms:.0f} ms  ·  generate {gen_ms:.0f} ms')}\n"
        )

        # Lưu lịch sử (dùng để hiển thị /history sau nếu cần)
        _state.chat_history.append({
            "query":    raw,
            "answer":   answer,
            "sources":  sources,
        })


# ---------------------------------------------------------------------------
# Menu chính
# ---------------------------------------------------------------------------

def _print_status_bar() -> None:
    """Hiển thị trạng thái tài liệu hiện tại."""
    if _state.ready:
        doc = _c(_GREEN, _state.indexed_file or "")
        chunk_count = len(_state.chunks)
        chunk_info = _c(_DIM, f"{chunk_count} chunks")
        print(f"  Tài liệu hiện tại: {doc}  {chunk_info}")
    else:
        print(f"  {_c(_YELLOW, 'Chưa nạp tài liệu nào.')}")


def _print_header_info() -> None:
    """Hiển thị tiêu đề và cấu hình của hệ thống RAG."""
    _header("RAG PIPELINE — CONSOLE APP")
    
    llm_provider = os.getenv("LLM_PROVIDER", "auto").upper()
    embed_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").upper()
    
    print(f"  {_c(_DIM, f'LLM: {llm_provider}')}  "
          f"{_c(_DIM, f'Embedding: {embed_provider}')}")


def _display_menu_options() -> None:
    """In ra các lựa chọn menu dựa trên trạng thái sẵn sàng."""
    print()
    _print_status_bar()
    print()
    print(f"  {_c(_BOLD, '1')}.  Chọn & nạp tài liệu mới")
    
    if _state.ready:
        print(f"  {_c(_BOLD, '2')}.  Bắt đầu chat")
        
    print(f"  {_c(_BOLD, '0')}.  Thoát")


def _main_menu() -> None:
    """Vòng lặp chính xử lý logic tương tác của người dùng."""
    _print_header_info()

    while True:
        _display_menu_options()
        
        raw = input(f"\n{_c(_CYAN, '?')} Chọn: ").strip()

        if raw == "0":
            print(_c(_DIM, "\nTạm biệt!\n"))
            sys.exit(0)

        elif raw == "1":
            chosen = _pick_file()
            if chosen:
                _run_indexing(chosen)

        elif raw == "2" and _state.ready:
            _chat_loop()

        else:
            _warn("Lựa chọn không hợp lệ hoặc chưa sẵn sàng.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        _main_menu()
    except KeyboardInterrupt:
        print(_c(_DIM, "\n\nĐã hủy bởi người dùng. Thoát chương trình.\n"))
        sys.exit(0)