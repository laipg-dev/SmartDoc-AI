from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from rag_engine import RagEngine

class ConversationalRAG:
    def __init__(self, model_name="qwen2.5:3b"):
        print(f"⏳ Đang kết nối với LLM ({model_name})...")
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.rag_engine = RagEngine()
        
        # SỬ DỤNG CHATPROMPTTEMPLATE CHO AI PHÂN VAI
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI thông minh chuyên trích xuất thông tin. Hãy đọc kỹ phần 'Tài liệu' và trả lời 'Câu hỏi' của người dùng. Viết ngắn gọn và đi thẳng vào vấn đề. Nếu tài liệu không chứa đáp án, hãy nói 'Không có thông tin'."),
            ("human", "Tài liệu:\n{context}\n\nCâu hỏi: {question}")
        ])

    def ask(self, query):
        # 1. Tìm kiếm nội dung liên quan
        results = self.rag_engine.search(query, top_k=3)
        
        # 2. Gom và dọn dẹp nội dung
        raw_context = "\n".join([doc.page_content for doc in results])
        context_text = " ".join(raw_context.split())
        
        # 3. Tạo luồng tin nhắn hội thoại
        messages = self.prompt_template.format_messages(context=context_text, question=query)
        
        # 4. Gửi cho Qwen xử lý và nhận luồng Stream
        print("\n🤖 Qwen đang tổng hợp câu trả lời...\n")
        stream_response = self.llm.stream(messages)
        
        return stream_response, results

# --- ĐOẠN CODE TEST NHANH ---
if __name__ == "__main__":
    qa_system = ConversationalRAG()
    
    print("\n" + "="*50)
    query = "Không được phép sử dụng loại bài báo nào?"
    print(f"🧑 Bạn: {query}")
    
    answer_stream, sources = qa_system.ask(query)
    
    print("🤖 Qwen: ", end="", flush=True)
    # Hứng từng token được sinh ra và in lên màn hình ngay lập tức
    for chunk in answer_stream:
        print(chunk.content, end="", flush=True)
        
    print("\n" + "="*50)