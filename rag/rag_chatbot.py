# rag/rag_chatbot.py
from rag.config import RAGConfig
from rag.llm_gemini import GeminiLLM


class RAGChatbot:
    def __init__(self, config: RAGConfig, vector_store):
        self.config = config
        self.store = vector_store
        self.llm = GeminiLLM(config)

    def answer(self, question: str) -> str:
        docs = self.store.search(question, top_k=self.config.top_k)

        if not docs:
            return "❌ Không tìm thấy thông tin phù hợp trong dữ liệu."

        context = "\n\n".join(docs)

        prompt = f"""
Bạn là chatbot tư vấn tuyển sinh.
Chỉ trả lời dựa trên dữ liệu dưới đây, KHÔNG bịa.

DỮ LIỆU:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""
        return self.llm.generate(prompt)
