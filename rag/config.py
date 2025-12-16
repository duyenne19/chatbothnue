# rag/config.py
from dataclasses import dataclass


@dataclass
class RAGConfig:
    # ===== DATA =====
    markdown_dir: str = "data"

    # ===== EMBEDDING =====
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 300
    overlap: int = 50
    top_k: int = 3

    # ===== GEMINI =====
    # 🔥 MODEL ĐÚNG, KHÔNG 404
    gemini_model: str = "models/gemini-2.5-flash"
