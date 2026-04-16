"""
features/memory.py — Câu 6: Conversational RAG
Leader — xử lý follow-up questions bằng cách inject lịch sử vào prompt
"""
import os
from langchain_community.llms import Ollama
from langchain.schema import Document


def build_context_with_history(
    relevant_docs: list[Document],
    chat_history_text: str,
    max_history_chars: int = 1000,
) -> str:
    """
    Kết hợp context tài liệu + lịch sử hội thoại gần đây.
    """
    doc_context = "\n\n".join([d.page_content for d in relevant_docs])
    if chat_history_text:
        # Cắt bớt nếu quá dài
        if len(chat_history_text) > max_history_chars:
            chat_history_text = "..." + chat_history_text[-max_history_chars:]
        return f"=== Lịch sử hội thoại gần đây ===\n{chat_history_text}\n\n=== Nội dung tài liệu ===\n{doc_context}"
    return doc_context


def build_conversational_prompt(
    question: str,
    context: str,
    is_vietnamese: bool = True,
) -> str:
    """
    Câu 6: Prompt có nhận thức về lịch sử hội thoại,
    xử lý được follow-up questions ("nó", "ở đó", "thêm nữa"...).
    """
    if is_vietnamese:
        return f"""Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh bên dưới.
Nếu câu hỏi là câu hỏi tiếp theo (dùng từ như "nó", "điều đó", "thêm nữa"),
hãy tham khảo lịch sử hội thoại để hiểu ngữ cảnh.
Nếu không biết, hãy nói thật. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt.

{context}

Câu hỏi hiện tại: {question}

Trả lời:"""
    else:
        return f"""You are a smart AI assistant. Answer based on the context below.
If the question is a follow-up (uses words like "it", "that", "more about this"),
refer to the conversation history to understand the context.
If you don't know, say so. Be concise and clear.

{context}

Current question: {question}

Answer:"""


def is_followup_question(question: str, history: list[dict]) -> bool:
    """
    Phát hiện câu hỏi tiếp theo dựa trên từ khoá.
    """
    if not history:
        return False
    followup_keywords = [
        "nó", "điều đó", "vấn đề đó", "ý đó", "thêm", "tiếp theo",
        "cụ thể hơn", "giải thích thêm", "ví dụ", "tại sao vậy",
        "it", "that", "this", "more", "why", "how so", "explain further",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in followup_keywords)
