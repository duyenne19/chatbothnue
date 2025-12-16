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
        self.model = genai.GenerativeModel(config.gemini_model)

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
        prompt = f"""
        Bạn là chatbot tư vấn tuyển sinh của Trường Đại học Sư phạm Hà Nội.

        NHIỆM VỤ:
        - Trả lời thân thiện, rõ ràng, đúng vai trò tư vấn tuyển sinh
        - Sử dụng thông tin trong NGỮ CẢNH
        - Ưu tiên giải thích ngành học, cơ hội nghề nghiệp nếu câu hỏi chung
        - Chỉ trả lời điểm chuẩn khi người dùng hỏi cụ thể về điểm/năm

        KHÔNG ĐƯỢC:
        - Bịa thông tin
        - Trả lời ngoài dữ liệu

        NGỮ CẢNH:
        {context_text}

        CÂU HỎI:
        {question}

        TRẢ LỜI (giọng tư vấn, tiếng Việt, tự nhiên):
        """

        # ===== Generate =====
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            return f"❌ Lỗi khi sinh câu trả lời: {e}"
