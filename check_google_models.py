import os
from dotenv import load_dotenv
import google.generativeai as genai

def check_available_models():
    # Load API Key từ file .env
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Lỗi: Không tìm thấy GOOGLE_API_KEY trong file .env")
        return

    print("="*60)
    print("🔍 ĐANG KIỂM TRA QUYỀN TRUY CẬP MODEL CỦA BẠN TRÊN GOOGLE")
    print("="*60)
    
    try:
        # Cấu hình thư viện với API Key của bạn
        genai.configure(api_key=api_key)
        
        # Lấy danh sách tất cả các model mà API Key này được phép thấy
        models = genai.list_models()
        
        found_embedders = False
        print("Danh sách các model hỗ trợ tạo Vector (Embedding):")
        print("-" * 60)
        
        for m in models:
            # Chỉ lọc ra các model có phương thức 'embedContent'
            if 'embedContent' in m.supported_generation_methods:
                print(f"✅ Tên model: {m.name}")
                print(f"   - Mô tả: {m.description}")
                print(f"   - Tên hiển thị: {m.display_name}")
                found_embedders = True
                print("-" * 40)
                
        if not found_embedders:
            print("❌ CẢNH BÁO: Tài khoản/Project của bạn KHÔNG HỖ TRỢ bất kỳ model Embedding nào.")
            
    except Exception as e:
        print(f"❌ Lỗi khi kết nối với Google API: {e}")

if __name__ == "__main__":
    check_available_models()