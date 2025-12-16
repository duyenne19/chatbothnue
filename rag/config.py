from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class RAGConfig:
    markdown_dir: str = os.path.join(BASE_DIR, "data")

    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3

    chunk_size: int = 300
    chunk_overlap: int = 50

    gemini_model: str = "models/gemini-2.5-flash"

    def validate(self):
        if not os.path.exists(self.markdown_dir):
            raise RuntimeError(f"❌ markdown_dir không tồn tại: {self.markdown_dir}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("❌ chunk_overlap phải nhỏ hơn chunk_size")
        if self.top_k <= 0:
            raise ValueError("❌ top_k phải > 0")
