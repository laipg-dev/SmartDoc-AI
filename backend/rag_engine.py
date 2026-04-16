import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RagEngine:
    def __init__(self, vector_dir=None):
        # 1. FIX ĐƯỜNG DẪN: Lấy đường dẫn tuyệt đối của thư mục backend/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Trỏ ra ngoài 1 cấp để vào vector_store
        self.vector_dir = vector_dir or os.path.join(base_dir, "../vector_store")
        
        print("⏳ Đang tải mô hình Embedding (Multilingual MPNet)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.vector_store = None

    def create_vector_store(self, chunks):
        """Biến các đoạn văn bản thành Vector và lưu vào FAISS"""
        print(f"🧠 Đang chuyển đổi {len(chunks)} đoạn văn bản thành Vector...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Lưu vào ổ cứng để không phải tính toán lại mỗi khi khởi động app
        os.makedirs(self.vector_dir, exist_ok=True)
        self.vector_store.save_local(self.vector_dir)
        print(f"✅ Đã lưu Vector Database tại thư mục: {self.vector_dir}")

    def load_vector_store(self):
        """Tải Vector Database từ ổ cứng"""
        if os.path.exists(self.vector_dir):
            self.vector_store = FAISS.load_local(
                self.vector_dir, 
                self.embeddings, 
                allow_dangerous_deserialization=True # Bắt buộc ở LangChain bản mới khi load file local
            )
            print("✅ Đã tải Vector Database thành công!")
        else:
            print("⚠️ Chưa có Vector Database. Vui lòng tạo trước.")

    def search(self, query, top_k=3):
        """Tìm kiếm các đoạn văn bản liên quan đến câu hỏi"""
        if not self.vector_store:
            self.load_vector_store()
            
        print(f"🔍 Đang tìm kiếm thông tin cho: '{query}'...")
        # Tìm ra top_k đoạn văn bản có Vector gần giống với Vector của câu hỏi nhất
        results = self.vector_store.similarity_search(query, k=top_k)
        return results

# --- ĐOẠN CODE TEST NHANH ---
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    # 1. Đọc và cắt file lại (Dùng chính file tailieu.pdf của bạn)
    test_file = "../data/tailieu.pdf"
    if os.path.exists(test_file):
        processor = DocumentProcessor()
        chunks = processor.process_and_chunk(test_file)
        
        # 2. Khởi tạo RAG Engine và tạo Vector Store
        engine = RagEngine()
        engine.create_vector_store(chunks)
        
        # 3. Test thử tính năng tìm kiếm
        print("\n" + "="*50)
        query = "Không được phép sử dụng loại bài báo nào?"
        results = engine.search(query, top_k=1)
        
        print("\n🎯 KẾT QUẢ TÌM KIẾM TỐT NHẤT CHO CÂU HỎI:")
        print(f"Hỏi: {query}")
        print("-" * 30)
        print(results[0].page_content)
        print("="*50)
    else:
        print("Không tìm thấy file test.")