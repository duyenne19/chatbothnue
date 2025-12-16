# rag/simple_vector_store.py
"""
Simple Vector Store sử dụng TF-IDF
KHÔNG cần download model từ HuggingFace
Hoạt động hoàn toàn OFFLINE
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleVectorStore:
    """Vector store đơn giản dùng TF-IDF thay vì neural embeddings"""

    def __init__(self, model_name: str = None):
        # model_name không dùng, chỉ giữ để tương thích interface
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Giới hạn số features
            ngram_range=(1, 2),  # Unigrams và bigrams
            min_df=1,  # Tần suất tối thiểu
            sublinear_tf=True  # Scale logarithmic
        )
        self.vectors = None
        self.texts = []

    def build(self, texts: list[str]):
        """Build TF-IDF vectors từ danh sách texts"""
        print("🔎 Đang tạo TF-IDF vectors (OFFLINE mode)...")

        self.texts = texts
        # Fit và transform texts thành TF-IDF vectors
        self.vectors = self.vectorizer.fit_transform(texts)

        print(f"✅ TF-IDF index: {len(texts)} documents, {self.vectors.shape[1]} features")

    def search(self, query: str, k: int = 3) -> list[str]:
        """Tìm kiếm top-k documents giống nhất với query"""
        if self.vectors is None:
            return []

        # Transform query thành TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Tính cosine similarity
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        # Lấy top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        # Trả về texts tương ứng
        results = []
        for idx in top_indices:
            if idx < len(self.texts) and similarities[idx] > 0:
                results.append(self.texts[idx])

        return results
