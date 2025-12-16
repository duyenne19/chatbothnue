# rag/rag_chatbot.py
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

from rag.config import RAGConfig
from rag.markdown_loader import MarkdownLoader
from rag.text_chunker import TextChunker
from rag.vector_store import VectorStore


class RAGChatbot:
    """
    RAG Chatbot tuyển sinh
    - Chỉ CHAT
    - KHÔNG crawl
    - Chỉ dùng dữ liệu Markdown đã chuẩn bị sẵn
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        # ===== Load Gemini API =====
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "❌ Thiếu GEMINI_API_KEY. "
                "Hãy cấu hình trong file .env"
            )

        genai.configure(api_key=api_key)

        # Cấu hình generation để câu trả lời tự nhiên hơn
        generation_config = genai.GenerationConfig(
            temperature=0.7,  # Vừa đủ creative nhưng vẫn chính xác
            top_p=0.9,
            top_k=40,
            max_output_tokens=1024,
        )

        # Safety settings để tránh bị block
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        self.model = genai.GenerativeModel(
            config.gemini_model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # ===== RAG components =====
        self.loader = MarkdownLoader(config.markdown_dir)
        self.chunker = TextChunker(
            size=config.chunk_size,
            overlap=config.overlap
        )
        self.store = VectorStore(config.embedding_model)

        self.ready = False

    # -------------------------------------------------
    def initialize(self):
        """
        Khởi tạo RAG:
        - Load Markdown
        - Chunk
        - Build vector store
        """
        print("📄 Đang load dữ liệu Markdown...")

        documents = self.loader.load()
        if not documents:
            raise RuntimeError(
                "❌ Không tìm thấy file content.md.\n"
                "👉 Hãy chạy crawler trước để thu thập dữ liệu."
            )

        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk(doc)
            chunks.extend(doc_chunks)

        if not chunks:
            raise RuntimeError(
                "❌ Dữ liệu không đủ để tạo chunk."
            )

        print(f"🧩 Tổng số chunk: {len(chunks)}")
        self.store.build(chunks)

        self.ready = True
        print("🤖 Chatbot sẵn sàng!")

    # -------------------------------------------------
    def ask(self, question: str) -> str:
        """
        Trả lời câu hỏi người dùng
        """
        if not self.ready:
            return "Hệ thống chưa sẵn sàng."

        if not question.strip():
            return "Vui lòng nhập câu hỏi."

        # ===== Retrieve =====
        contexts = self.store.search(
            question,
            self.config.top_k
        )

        if not contexts:
            return "Tôi không tìm thấy thông tin này trong dữ liệu tuyển sinh."

        context_text = "\n\n".join(contexts)

        # ===== Prompt =====
        prompt = f"""Bạn là trợ lý tư vấn tuyển sinh thông minh của Trường Đại học Sư phạm Hà Nội.

NHIỆM VỤ:
- Đọc kỹ ngữ cảnh được cung cấp
- Tổng hợp thông tin thành câu trả lời TỰ NHIÊN như người thật đang tư vấn
- Trả lời thân thiện, chuyên nghiệp
- CHỈ dùng thông tin từ ngữ cảnh, KHÔNG bịa đặt

QUY TẮC BẮT BUỘC:
✗ KHÔNG copy/paste nguyên văn từ ngữ cảnh
✗ KHÔNG dump thông tin dạng bullet points
✗ KHÔNG trả lời chung chung
✓ PHẢI viết thành câu văn tự nhiên, mạch lạc
✓ Nếu hỏi về ngành: giới thiệu tổng quan + cơ hội nghề nghiệp
✓ Nếu hỏi về điểm/chỉ tiêu: nêu con số cụ thể (nếu có)
✓ Nếu không có thông tin: "Hiện tôi chưa có dữ liệu về... Bạn có thể liên hệ phòng tuyển sinh để biết thêm chi tiết."

VÍ DỤ TRẢ LỜI TỐT:
Câu hỏi: "Ngành CNTT học những gì?"
Trả lời: "Ngành Công nghệ thông tin tại trường đào tạo các kiến thức nền tảng về lập trình, cơ sở dữ liệu, mạng máy tính và phát triển phần mềm. Sinh viên sẽ được học cả lý thuyết và thực hành qua các dự án thực tế. Sau khi tốt nghiệp, bạn có thể làm việc tại các công ty công nghệ, ngân hàng, hoặc trở thành giáo viên Tin học."

NGỮ CẢNH THAM KHẢO:
{context_text}

CÂU HỎI:
{question}

TRẢ LỜI (ngắn gọn, tự nhiên):"""

        # ===== Generate =====
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            return f"❌ Lỗi khi sinh câu trả lời: {e}"
