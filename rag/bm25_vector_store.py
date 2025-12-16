"""
BM25 Vector Store - Thuật toán ranking tốt hơn TF-IDF
BM25 (Best Matching 25) tính đến document length normalization
"""
import numpy as np
from typing import List
import math


class BM25VectorStore:
    """
    BM25 (Okapi BM25) - thuật toán ranking văn bản
    Thường cho kết quả tốt hơn TF-IDF cho information retrieval
    """

    def __init__(self, model_name: str = None, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            model_name: Không dùng, chỉ để tương thích interface
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.texts = []
        self.tokenized_corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.idf = {}
        self.N = 0  # Số documents

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split by whitespace và lowercase"""
        return text.lower().split()

    def _calculate_idf(self):
        """Tính IDF (Inverse Document Frequency) cho mỗi term"""
        self.idf = {}

        for term, df in self.doc_freqs.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            # +1 để tránh log(0)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def build(self, texts: List[str]):
        """
        Build BM25 index từ corpus

        Args:
            texts: Danh sách documents
        """
        print("🔎 Đang tạo BM25 index (OFFLINE mode)...")

        self.texts = texts
        self.N = len(texts)

        # Tokenize tất cả documents
        self.tokenized_corpus = [self._tokenize(text) for text in texts]

        # Tính document lengths
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Tính document frequency cho mỗi term
        self.doc_freqs = {}
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Tính IDF
        self._calculate_idf()

        print(f"✅ BM25 index: {self.N} documents, "
              f"{len(self.doc_freqs)} unique terms, "
              f"avg_length={self.avg_doc_length:.1f}")

    def _score_document(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Tính BM25 score cho một document

        Args:
            query_terms: Query đã tokenize
            doc_idx: Index của document

        Returns:
            BM25 score
        """
        score = 0.0
        doc = self.tokenized_corpus[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        # Term frequencies trong document này
        term_freqs = {}
        for term in doc:
            term_freqs[term] = term_freqs.get(term, 0) + 1

        for term in query_terms:
            if term not in self.idf:
                continue  # Term không có trong corpus

            # TF của term trong document
            tf = term_freqs.get(term, 0)

            # BM25 score component cho term này
            # score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            score += self.idf[term] * (numerator / denominator)

        return score

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Tìm kiếm top-K documents cho query

        Args:
            query: Query string
            k: Số lượng kết quả

        Returns:
            List của top-K documents
        """
        if self.N == 0:
            return []

        query_terms = self._tokenize(query)

        # Tính score cho tất cả documents
        scores = []
        for doc_idx in range(self.N):
            score = self._score_document(query_terms, doc_idx)
            scores.append((score, doc_idx))

        # Sort theo score giảm dần
        scores.sort(reverse=True, key=lambda x: x[0])

        # Lấy top-K
        top_k = scores[:k]

        # Trả về documents (chỉ những docs có score > 0)
        results = []
        for score, idx in top_k:
            if score > 0:
                results.append(self.texts[idx])

        return results

    def get_config(self) -> dict:
        """Trả về config của BM25"""
        return {
            'algorithm': 'BM25',
            'k1': self.k1,
            'b': self.b,
            'n_documents': self.N,
            'avg_doc_length': self.avg_doc_length,
            'vocab_size': len(self.doc_freqs)
        }
