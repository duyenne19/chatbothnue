import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM:
    def __init__(self, model_name="models/gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("❌ Thiếu GEMINI_API_KEY trong .env")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_context_chars = 8000

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return "❌ Không tìm thấy thông tin phù hợp."

        context = context[: self.max_context_chars]

        prompt = f"""
        Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Sư phạm Hà Nội.

        NHIỆM VỤ:
        - Chỉ trả lời dựa trên thông tin có trong TÀI LIỆU.
        - Không được suy đoán, không bổ sung kiến thức bên ngoài.
        - Nếu TÀI LIỆU không chứa thông tin cần thiết, trả lời đúng một câu:
          "Hiện tại chưa có dữ liệu phù hợp để trả lời câu hỏi này."

        YÊU CẦU TRẢ LỜI:
        - Trả lời ngắn gọn, đúng trọng tâm.
        - Ưu tiên mô tả ngành/chương trình đào tạo nếu câu hỏi mang tính tra cứu.
        - Không lặp lại câu hỏi.
        - Không đưa ra nhận xét chủ quan.

        === TÀI LIỆU ===
        {context}

        === CÂU HỎI ===
        {query}

        === TRẢ LỜI (tối đa 5 câu) ===
        """

        try:
            response = self.model.generate_content(prompt)
        except Exception as e:
            return f"⚠️ Lỗi Gemini API: {e}"

        if not response or not getattr(response, "text", None):
            return "⚠️ Gemini không trả về nội dung."

        return response.text.strip()
