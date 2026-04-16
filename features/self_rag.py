"""
features/self_rag.py — Câu 10: Self-RAG + Query Rewriting
Leader
"""
import json
import re
import os
from langchain_community.llms import Ollama


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def rewrite_query(original_query: str, llm=None) -> str:
    """
    Câu 10: Query rewriting — LLM tự cải thiện câu hỏi
    để retrieval chính xác hơn.
    """
    if llm is None:
        llm = Ollama(model=OLLAMA_MODEL, temperature=0)

    prompt = f"""Viết lại câu hỏi sau đây để tìm kiếm trong tài liệu được hiệu quả hơn.
Giữ nguyên ý nghĩa, nhưng dùng từ khoá rõ ràng và cụ thể hơn.
Chỉ trả về câu hỏi đã viết lại, không giải thích thêm.

Câu hỏi gốc: {original_query}

Câu hỏi viết lại:"""

    try:
        rewritten = llm.invoke(prompt).strip()
        # Nếu LLM trả về quá dài hoặc lạ, dùng câu gốc
        if len(rewritten) > 300 or len(rewritten) < 5:
            return original_query
        return rewritten
    except Exception:
        return original_query


def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    llm=None,
) -> dict:
    """
    Câu 10: Self-RAG — LLM tự đánh giá chất lượng câu trả lời.

    Trả về:
    {
        "is_grounded": bool,     # câu trả lời có dựa trên context không?
        "is_relevant": bool,     # có trả lời đúng câu hỏi không?
        "confidence": float,     # 0.0 → 1.0
        "reason": str,           # giải thích ngắn
        "should_retry": bool,    # có nên thử lại không?
    }
    """
    if llm is None:
        llm = Ollama(model=OLLAMA_MODEL, temperature=0)

    eval_prompt = f"""Đánh giá câu trả lời sau đây dựa trên ngữ cảnh được cung cấp.
Trả về JSON với các trường sau (không thêm gì khác):

{{
  "is_grounded": true/false,
  "is_relevant": true/false,
  "confidence": 0.0-1.0,
  "reason": "giải thích ngắn gọn (tối đa 1 câu)",
  "should_retry": true/false
}}

Quy tắc:
- is_grounded = true nếu câu trả lời dựa trên thông tin trong ngữ cảnh
- is_relevant = true nếu câu trả lời trả lời đúng câu hỏi
- confidence = mức độ tin cậy từ 0 đến 1
- should_retry = true nếu confidence < 0.5 hoặc câu trả lời không liên quan

Câu hỏi: {question}

Ngữ cảnh tài liệu (tóm tắt):
{context[:500]}

Câu trả lời cần đánh giá:
{answer[:500]}

JSON:"""

    try:
        raw = llm.invoke(eval_prompt).strip()
        # Trích xuất JSON từ response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate và set defaults
            return {
                "is_grounded":  bool(result.get("is_grounded", True)),
                "is_relevant":  bool(result.get("is_relevant", True)),
                "confidence":   float(result.get("confidence", 0.7)),
                "reason":       str(result.get("reason", "")),
                "should_retry": bool(result.get("should_retry", False)),
            }
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback nếu parse thất bại
    return {
        "is_grounded": True,
        "is_relevant": True,
        "confidence": 0.6,
        "reason": "Không thể đánh giá tự động.",
        "should_retry": False,
    }


def self_rag_pipeline(
    question: str,
    retriever,
    llm=None,
    use_query_rewrite: bool = True,
    use_self_eval: bool = True,
    max_retries: int = 1,
) -> dict:
    """
    Câu 10: Pipeline Self-RAG đầy đủ:
    1. (Tuỳ chọn) Viết lại câu hỏi
    2. Retrieve
    3. Generate
    4. (Tuỳ chọn) Tự đánh giá → retry nếu kém

    Trả về dict với đầy đủ thông tin để hiển thị trong UI.
    """
    if llm is None:
        llm = Ollama(model=OLLAMA_MODEL)

    result = {
        "original_query":  question,
        "rewritten_query": question,
        "answer":          "",
        "evaluation":      None,
        "retried":         False,
        "num_retries":     0,
    }

    # Bước 1: Query rewriting
    if use_query_rewrite:
        rewritten = rewrite_query(question, llm)
        result["rewritten_query"] = rewritten
        search_query = rewritten
    else:
        search_query = question

    # Bước 2: Retrieve
    docs = retriever.get_relevant_documents(search_query)
    context = "\n\n".join([d.page_content for d in docs])
    result["docs"] = docs

    # Bước 3: Generate
    vn_chars = "àáảãạăắặâầấđèéêìíòóôùúư"
    is_vn = any(c in question.lower() for c in vn_chars)
    lang_instruction = "Trả lời bằng tiếng Việt." if is_vn else "Answer in English."

    gen_prompt = f"""Dựa trên ngữ cảnh sau, trả lời câu hỏi ngắn gọn và chính xác.
{lang_instruction}
Nếu không tìm thấy thông tin, hãy nói thật.

Ngữ cảnh:
{context}

Câu hỏi: {search_query}

Trả lời:"""

    answer = llm.invoke(gen_prompt).strip()
    result["answer"] = answer

    # Bước 4: Self-evaluation + retry
    if use_self_eval:
        evaluation = evaluate_answer(question, answer, context, llm)
        result["evaluation"] = evaluation

        if evaluation["should_retry"] and max_retries > 0:
            # Retry với câu hỏi gốc (không rewrite lần 2)
            result["retried"] = True
            result["num_retries"] = 1
            docs2 = retriever.get_relevant_documents(question)
            context2 = "\n\n".join([d.page_content for d in docs2])
            retry_prompt = f"""Câu trả lời trước chưa đủ tốt. Hãy thử lại với thông tin đầy đủ hơn.
{lang_instruction}

Ngữ cảnh bổ sung:
{context2}

Câu hỏi: {question}

Trả lời (cố gắng chi tiết hơn lần trước):"""
            result["answer"] = llm.invoke(retry_prompt).strip()

    return result
