import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, embedding_model: str):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        if not chunks:
            raise RuntimeError("❌ Không có chunk để build vector store")

        texts = [c["content"] for c in chunks]
        print("🔎 Đang tạo embedding...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks

        print(f"✅ FAISS index: {len(chunks)} vectors")

    def search(self, query: str, top_k: int = 3):
        if self.index is None:
            raise RuntimeError("❌ Vector store chưa được build")

        top_k = min(top_k, len(self.chunks))
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(q_emb, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "content": self.chunks[idx]["content"],
                "source": self.chunks[idx]["source"],
                "score": float(score)
            })

        return results
