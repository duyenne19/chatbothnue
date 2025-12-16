"""
Test Dataset cho RAG Chatbot Tuyển Sinh
Bao gồm câu hỏi, câu trả lời mong đợi, và relevant chunks
"""
from typing import List, Dict


class TestDataset:
    """Dataset để đánh giá chatbot"""

    @staticmethod
    def get_test_queries() -> List[Dict[str, any]]:
        """
        Trả về danh sách test queries với relevant information

        Returns:
            List of test cases, mỗi case gồm:
            - query: Câu hỏi
            - relevant_keywords: Keywords để identify relevant chunks
            - expected_topics: Các topics mong đợi trong câu trả lời
            - category: Loại câu hỏi
        """
        return [
            {
                "query": "Điểm chuẩn ngành Công nghệ thông tin là bao nhiêu?",
                "relevant_keywords": ["công nghệ thông tin", "CNTT", "điểm chuẩn", "7480201"],
                "expected_topics": ["điểm chuẩn", "ngành CNTT", "mã ngành"],
                "category": "admission_score"
            },
            {
                "query": "Trường có những ngành nào?",
                "relevant_keywords": ["ngành", "chương trình đào tạo", "mã xét tuyển"],
                "expected_topics": ["danh sách ngành", "các ngành đào tạo"],
                "category": "programs"
            },
            {
                "query": "Khung chương trình đào tạo ngành CNTT như thế nào?",
                "relevant_keywords": ["chương trình đào tạo", "công nghệ thông tin", "K69"],
                "expected_topics": ["môn học", "tín chỉ", "chương trình đào tạo"],
                "category": "curriculum"
            },
            {
                "query": "Làm sao để nhập học?",
                "relevant_keywords": ["nhập học", "hồ sơ", "nhaphoc.hnue.edu.vn"],
                "expected_topics": ["thủ tục nhập học", "hồ sơ", "thời gian"],
                "category": "enrollment"
            },
            {
                "query": "Điểm chuẩn ngành Sư phạm Toán học là bao nhiêu?",
                "relevant_keywords": ["sư phạm", "toán", "điểm chuẩn"],
                "expected_topics": ["điểm chuẩn", "sư phạm"],
                "category": "admission_score"
            },
            {
                "query": "Ngành Quản trị dịch vụ du lịch điểm chuẩn bao nhiêu?",
                "relevant_keywords": ["quản trị", "du lịch", "7810103", "điểm chuẩn", "20.25"],
                "expected_topics": ["điểm chuẩn", "du lịch"],
                "category": "admission_score"
            },
            {
                "query": "Thời gian nộp hồ sơ nhập học là khi nào?",
                "relevant_keywords": ["nhập học", "27/8", "28/8", "2025", "thời gian"],
                "expected_topics": ["thời gian", "deadline", "nhập học"],
                "category": "enrollment"
            },
            {
                "query": "Website nhập học là gì?",
                "relevant_keywords": ["nhaphoc.hnue.edu.vn", "trực tuyến"],
                "expected_topics": ["website", "online", "nhập học"],
                "category": "enrollment"
            },
        ]

    @staticmethod
    def get_hyperparameter_grid() -> Dict[str, List]:
        """
        Grid search parameters cho tối ưu hóa RAG

        Returns:
            Dictionary chứa các giá trị tham số cần thử
        """
        return {
            'chunk_size': [300, 400, 500, 600],
            'overlap': [50, 75, 100, 150],
            'top_k': [3, 5, 7, 10],
            'vectorizer_params': [
                {'max_features': 3000, 'ngram_range': (1, 1)},  # Unigrams only
                {'max_features': 5000, 'ngram_range': (1, 2)},  # Unigrams + Bigrams
                {'max_features': 7000, 'ngram_range': (1, 3)},  # Up to Trigrams
                {'max_features': 5000, 'ngram_range': (2, 2)},  # Bigrams only
            ]
        }

    @staticmethod
    def create_cv_splits(n_queries: int, n_folds: int = 5) -> List[Dict[str, List[int]]]:
        """
        Tạo cross-validation splits

        Args:
            n_queries: Số lượng queries
            n_folds: Số folds cho CV

        Returns:
            List of splits, mỗi split có train_indices và val_indices
        """
        import numpy as np

        indices = np.arange(n_queries)
        np.random.shuffle(indices)

        fold_size = n_queries // n_folds
        splits = []

        for i in range(n_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_folds - 1 else n_queries

            val_indices = indices[val_start:val_end].tolist()
            train_indices = np.concatenate([
                indices[:val_start],
                indices[val_end:]
            ]).tolist()

            splits.append({
                'train': train_indices,
                'val': val_indices,
                'fold': i + 1
            })

        return splits
