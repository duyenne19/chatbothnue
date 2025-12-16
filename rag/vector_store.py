# rag/vector_store.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build(self, texts: list[str]):
        print("🔎 Đang tạo embedding...")

        self.texts = texts
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"✅ FAISS index: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 3) -> list[str]:
        if not self.index:
            return []

        q_emb = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results
