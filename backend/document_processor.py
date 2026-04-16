import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Khởi tạo bộ xử lý tài liệu với cấu hình cắt văn bản (Chunking Strategy)
        - chunk_size: Ký tự tối đa cho mỗi đoạn văn bản.
        - chunk_overlap: Số ký tự trùng lặp giữa 2 đoạn liên tiếp để giữ ngữ cảnh.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Ưu tiên cắt theo đoạn văn, sau đó mới đến câu, rồi đến từ
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def load_document(self, file_path):
        """Kiểm tra định dạng và đọc nội dung file thô"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        
        # Xử lý theo từng loại file (Yêu cầu 1)
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Định dạng {ext} chưa được hỗ trợ. Vui lòng dùng định dạng PDF hoặc DOCX.")

        # Trả về danh sách các trang/đoạn (Documents)
        return loader.load()

    def process_and_chunk(self, file_path):
        """Quy trình đọc file và cắt nhỏ thành các chunks (Yêu cầu 4)"""
        print(f"📄 Đang đọc file: {os.path.basename(file_path)}...")
        documents = self.load_document(file_path)
        
        print("✂️ Đang áp dụng chiến lược phân tách (Chunking)...")
        # Phân tách nội dung thành các chunks có kích thước đều nhau và giữ nguyên ngữ cảnh
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✅ Hoàn tất! Đã chia tài liệu thành {len(chunks)} đoạn văn bản (chunks).")
        return chunks

# --- ĐOẠN CODE TEST NHANH ---
# Nếu bạn chạy trực tiếp file này, đoạn code dưới đây sẽ thực thi để kiểm tra lỗi
if __name__ == "__main__":
    # Tạo sẵn thư mục data nếu chưa có
    os.makedirs("../data", exist_ok=True)
    
    print("Vui lòng copy 1 file PDF hoặc DOCX bất kỳ vào thư mục 'data/'")
    test_file = input("Nhập tên file trong thư mục data (VD: tailieu.pdf): ")
    test_path = os.path.join("../data", test_file)
    
    if os.path.exists(test_path):
        processor = DocumentProcessor()
        chunks = processor.process_and_chunk(test_path)
        
        # In thử nội dung của chunk đầu tiên để kiểm tra
        print("\n--- NỘI DUNG CHUNK ĐẦU TIÊN ---")
        print(chunks[0].page_content)
        print("--------------------------------")
    else:
        print("Không tìm thấy file test. Hãy chắc chắn bạn đã đưa file vào đúng thư mục data/")