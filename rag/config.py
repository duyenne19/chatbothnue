# rag/config.py
from dataclasses import dataclass


@dataclass
class RAGConfig:
    # ===== DATA =====
    markdown_dir: str = "data"

    # ===== EMBEDDING =====
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500  # Tăng từ 300 để mỗi chunk có nhiều thông tin hơn
    overlap: int = 100     # Tăng overlap để giữ context liên tục
    top_k: int = 5         # Tăng từ 3 để Gemini có nhiều ngữ cảnh hơn

    # ===== GEMINI =====
    # 🔥 MODEL ĐÚNG, KHÔNG 404
    gemini_model: str = "models/gemini-2.5-flash"
