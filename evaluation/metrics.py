"""
Evaluation Metrics cho RAG System
Bao gồm: Precision, Recall, F1, MRR, NDCG, Hit Rate
"""
import numpy as np
from typing import List, Dict, Tuple


class RAGMetrics:
    """Metrics để đánh giá chất lượng Retrieval-Augmented Generation"""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        Precision@K: Tỷ lệ documents liên quan trong top-K kết quả

        Args:
            retrieved: Danh sách documents được retrieve
            relevant: Danh sách documents thực sự liên quan
            k: Số lượng top results để đánh giá

        Returns:
            Precision score (0-1)
        """
        if not retrieved or k == 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        hits = sum(1 for doc in top_k if doc in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        Recall@K: Tỷ lệ documents liên quan được tìm thấy trong top-K

        Returns:
            Recall score (0-1)
        """
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        hits = sum(1 for doc in top_k if doc in relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        F1@K: Harmonic mean của Precision và Recall

        Returns:
            F1 score (0-1)
        """
        precision = RAGMetrics.precision_at_k(retrieved, relevant, k)
        recall = RAGMetrics.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """
        MRR (Mean Reciprocal Rank): Vị trí của document liên quan đầu tiên

        MRR = 1/rank của kết quả đúng đầu tiên

        Returns:
            MRR score (0-1)
        """
        relevant_set = set(relevant)

        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        NDCG@K (Normalized Discounted Cumulative Gain)
        Đánh giá chất lượng ranking, documents liên quan ở vị trí cao hơn được thưởng nhiều hơn

        Returns:
            NDCG score (0-1)
        """
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        # DCG: Tính gain với discount theo vị trí
        dcg = 0.0
        for rank, doc in enumerate(top_k, start=1):
            if doc in relevant_set:
                # Gain = 1 nếu relevant, 0 nếu không
                gain = 1.0
                # Discount theo log2(rank + 1)
                dcg += gain / np.log2(rank + 1)

        # IDCG: DCG lý tưởng (tất cả relevant docs ở đầu)
        ideal_length = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_length + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        Hit Rate@K: Có ít nhất 1 document liên quan trong top-K không?

        Returns:
            1.0 nếu có hit, 0.0 nếu không
        """
        top_k = retrieved[:k]
        relevant_set = set(relevant)

        for doc in top_k:
            if doc in relevant_set:
                return 1.0

        return 0.0

    @staticmethod
    def evaluate_retrieval(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Dict[int, float]]:
        """
        Đánh giá toàn diện retrieval system với nhiều queries

        Args:
            retrieved_docs: List các retrieved documents cho mỗi query
            relevant_docs: List các relevant documents cho mỗi query
            k_values: Các giá trị K để đánh giá

        Returns:
            Dictionary chứa tất cả metrics
        """
        results = {
            'precision': {},
            'recall': {},
            'f1': {},
            'ndcg': {},
            'hit_rate': {},
            'mrr': 0.0
        }

        n_queries = len(retrieved_docs)
        mrr_sum = 0.0

        for k in k_values:
            precision_sum = 0.0
            recall_sum = 0.0
            f1_sum = 0.0
            ndcg_sum = 0.0
            hit_sum = 0.0

            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                precision_sum += RAGMetrics.precision_at_k(retrieved, relevant, k)
                recall_sum += RAGMetrics.recall_at_k(retrieved, relevant, k)
                f1_sum += RAGMetrics.f1_at_k(retrieved, relevant, k)
                ndcg_sum += RAGMetrics.ndcg_at_k(retrieved, relevant, k)
                hit_sum += RAGMetrics.hit_rate_at_k(retrieved, relevant, k)

            results['precision'][k] = precision_sum / n_queries
            results['recall'][k] = recall_sum / n_queries
            results['f1'][k] = f1_sum / n_queries
            results['ndcg'][k] = ndcg_sum / n_queries
            results['hit_rate'][k] = hit_sum / n_queries

        # MRR tính riêng (không phụ thuộc vào K)
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            mrr_sum += RAGMetrics.mean_reciprocal_rank(retrieved, relevant)

        results['mrr'] = mrr_sum / n_queries

        return results


def print_evaluation_results(results: Dict, model_name: str = "Model"):
    """In kết quả đánh giá dễ đọc"""
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ: {model_name}")
    print(f"{'='*60}")

    print(f"\n🎯 MRR (Mean Reciprocal Rank): {results['mrr']:.4f}")

    print(f"\n📈 Metrics theo K:")
    print(f"{'Metric':<15} {'K=1':<10} {'K=3':<10} {'K=5':<10} {'K=10':<10}")
    print("-" * 60)

    for metric in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
        if metric in results and isinstance(results[metric], dict):
            values = results[metric]
            row = f"{metric.upper():<15}"
            for k in [1, 3, 5, 10]:
                if k in values:
                    row += f"{values[k]:<10.4f}"
                else:
                    row += f"{'N/A':<10}"
            print(row)

    print("=" * 60)
