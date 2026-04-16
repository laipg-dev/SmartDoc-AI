
=======
# SmartDoc AI - Intelligent Document Q&A System 🧠📄

Dự án xây dựng hệ thống RAG (Retrieval-Augmented Generation) thông minh, cho phép người dùng tải lên các tài liệu (PDF, DOCX) và đặt câu hỏi tương tác trực tiếp với dữ liệu bằng ngôn ngữ tự nhiên. 

Dự án được phát triển trong khuôn khổ môn học Open Source Software Development (Spring 2026).

## 🚀 Công nghệ sử dụng
- **Giao diện (UI):** Streamlit
- **AI Framework:** LangChain
- **Vector Database:** FAISS
- **Embedding Model:** sentence-transformers (Multilingual MPNet)
- **Local LLM:** Ollama (Qwen2.5:7b)

## 🛠 Yêu cầu hệ thống
- Môi trường Linux hoặc WSL (Windows Subsystem for Linux)
- Python 3.8+
- [Ollama](https://ollama.com/) đã được cài đặt trên hệ thống.

## ⚙️ Hướng dẫn Cài đặt & Chạy dự án

**Bước 1: Clone kho lưu trữ về máy**
```bash
git clone [https://github.com/thanhdinh2005/SmartDoc-AI.git](https://github.com/thanhdinh2005/SmartDoc-AI.git)
cd SmartDoc-AI

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:7b

streamlit run app.py
