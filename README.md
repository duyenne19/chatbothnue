HỆ THỐNG TƯ VẤN TUYỂN SINH ĐẠI HỌC SỬ DỤNG RAG & GEMINI
1. Giới thiệu

Đề tài xây dựng hệ thống chatbot tư vấn tuyển sinh đại học dựa trên kiến trúc Retrieval-Augmented Generation (RAG).
Hệ thống sử dụng dữ liệu tuyển sinh được thu thập từ website chính thức, xử lý và lưu trữ dưới dạng Markdown (.md), sau đó tìm kiếm và sinh câu trả lời bằng Gemini API.

Hệ thống không huấn luyện mô hình, không sử dụng kiến thức bên ngoài, đảm bảo câu trả lời chỉ dựa trên dữ liệu tuyển sinh đã thu thập.

2. Mục tiêu của hệ thống

Thu thập dữ liệu tuyển sinh từ website (crawler)

Chuẩn hóa dữ liệu thành Markdown

Xây dựng hệ thống truy hồi tri thức bằng vector (FAISS)

Sinh câu trả lời tự nhiên bằng mô hình ngôn ngữ lớn (Gemini)

Giảm hiện tượng hallucination của chatbot

3. Kiến trúc tổng thể

Hệ thống gồm 3 thành phần chính:

Web Crawler
    ↓
Markdown (.md) Data
    ↓
RAG Retrieval (Embedding + FAISS)
    ↓
Generation (Gemini API)
    ↓
Câu trả lời cho người dùng

Vai trò từng thành phần:

Crawler: thu thập và chuẩn hóa dữ liệu tuyển sinh

RAG Retrieval: tìm các đoạn văn liên quan đến câu hỏi

Generation: sinh câu trả lời dựa trên ngữ cảnh đã truy xuất

4. Cấu trúc project
rag_admissions/
│
├── crawler/                  # Thu thập dữ liệu
│   ├── __init__.py
│   ├── config.py
│   └── web_crawler.py
│
├── rag/                      # RAG core
│   ├── __init__.py
│   ├── config.py
│   ├── markdown_loader.py
│   ├── text_chunker.py
│   ├── vector_store.py
│   └── rag_chatbot.py
│
├── cli/                      # Giao diện dòng lệnh
│   ├── __init__.py
│   └── main.py
│
├── data/                     # Dữ liệu Markdown sau khi crawl
├── requirements.txt
└── README.md

5. Công nghệ sử dụng

Python 3.10+

Requests, BeautifulSoup: crawl dữ liệu web

Sentence-Transformers: tạo embedding văn bản

FAISS: tìm kiếm vector

Gemini API: sinh câu trả lời

Markdown (.md): định dạng lưu trữ tri thức

6. Nguyên lý hoạt động (RAG)

Load các file Markdown từ thư mục data/

Làm sạch và chia nhỏ văn bản (chunking)

Chuyển mỗi chunk thành vector embedding

Lưu embedding vào FAISS

Khi có câu hỏi:

Tìm top-k đoạn văn liên quan

Ghép thành context

Gửi context + câu hỏi cho Gemini để sinh câu trả lời

⚠️ Nếu thông tin không tồn tại trong dữ liệu, hệ thống trả lời:

“Tôi không tìm thấy thông tin này trong dữ liệu tuyển sinh.”

7. Cài đặt & chạy hệ thống
7.1 Cài thư viện
pip install -r requirements.txt

7.2 Thiết lập API key Gemini

Windows

setx GEMINI_API_KEY "YOUR_API_KEY"


Linux / macOS

export GEMINI_API_KEY="YOUR_API_KEY"

8. Sử dụng hệ thống
8.1 Crawl dữ liệu tuyển sinh
python -m cli.main


Chọn:

1. Crawl data


Nhập danh sách URL cần thu thập.

8.2 Chạy chatbot tư vấn tuyển sinh
python -m cli.main


Chọn:

2. Chatbot


Nhập câu hỏi, ví dụ:

Điểm chuẩn ngành Công nghệ thông tin năm 2024 là bao nhiêu?

9. Ưu điểm của hệ thống

Không cần huấn luyện mô hình

Giảm hallucination so với chatbot thông thường

Dữ liệu rõ nguồn gốc, có thể kiểm chứng

Dễ mở rộng, dễ cập nhật dữ liệu

10. Hạn chế

Chưa lưu FAISS index (cần rebuild khi khởi động)

Chưa hỗ trợ hội thoại nhiều lượt

Chất lượng phụ thuộc vào dữ liệu crawl

11. Hướng phát triển

Lưu và tải lại FAISS index

Cải thiện chunking theo semantic

Thêm reranking

Xây dựng giao diện web

12. Kết luận

Hệ thống đã triển khai thành công mô hình RAG cho bài toán tư vấn tuyển sinh, đảm bảo câu trả lời chính xác, minh bạch và có nguồn dữ liệu rõ ràng.
Đây là nền tảng tốt để mở rộng thành hệ thống tư vấn tuyển sinh quy mô lớn.