"""
Hyperparameter Optimization và Cross-Validation cho RAG System
"""
import itertools
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm

from rag.config import RAGConfig
from rag.markdown_loader import MarkdownLoader
from rag.text_chunker import TextChunker
from rag.simple_vector_store import SimpleVectorStore
from rag.bm25_vector_store import BM25VectorStore
from evaluation.metrics import RAGMetrics
from evaluation.test_data import TestDataset


class RAGOptimizer:
    """Tối ưu hóa hyperparameters cho RAG system"""

    def __init__(self, markdown_dir: str):
        self.markdown_dir = markdown_dir
        self.loader = MarkdownLoader(markdown_dir)
        self.documents = None

    def load_data(self):
        """Load markdown documents"""
        if self.documents is None:
            print("📄 Loading documents...")
            self.documents = self.loader.load()
            print(f"✅ Loaded {len(self.documents)} documents")

    def build_rag_system(
        self,
        chunk_size: int,
        overlap: int,
        vectorizer_type: str = 'tfidf',
        vectorizer_params: Dict = None
    ) -> Tuple[Any, List[str]]:
        """
        Build RAG system với params cụ thể

        Args:
            chunk_size: Kích thước chunk
            overlap: Overlap giữa các chunks
            vectorizer_type: 'tfidf' hoặc 'bm25'
            vectorizer_params: Parameters cho vectorizer

        Returns:
            (vector_store, chunks)
        """
        # Chunking
        chunker = TextChunker(size=chunk_size, overlap=overlap)
        chunks = []
        for doc in self.documents:
            chunks.extend(chunker.chunk(doc))

        # Vector store
        if vectorizer_type == 'bm25':
            # BM25 có params k1 và b
            k1 = vectorizer_params.get('k1', 1.5) if vectorizer_params else 1.5
            b = vectorizer_params.get('b', 0.75) if vectorizer_params else 0.75
            store = BM25VectorStore(k1=k1, b=b)
        else:  # tfidf
            # SimpleVectorStore với TF-IDF
            store = SimpleVectorStore()
            if vectorizer_params:
                # Update vectorizer params
                store.vectorizer.max_features = vectorizer_params.get('max_features', 5000)
                store.vectorizer.ngram_range = vectorizer_params.get('ngram_range', (1, 2))

        store.build(chunks)

        return store, chunks

    def evaluate_on_queries(
        self,
        vector_store: Any,
        chunks: List[str],
        queries: List[Dict],
        top_k: int
    ) -> Dict:
        """
        Đánh giá vector store trên test queries

        Returns:
            Metrics dictionary
        """
        retrieved_docs_list = []
        relevant_docs_list = []

        for query_data in queries:
            query = query_data['query']
            keywords = query_data['relevant_keywords']

            # Retrieve
            retrieved = vector_store.search(query, top_k)

            # Tìm relevant chunks (chunks chứa keywords)
            relevant = []
            for chunk in chunks:
                chunk_lower = chunk.lower()
                # Chunk relevant nếu chứa ít nhất 1 keyword
                if any(kw.lower() in chunk_lower for kw in keywords):
                    relevant.append(chunk)

            retrieved_docs_list.append(retrieved)
            relevant_docs_list.append(relevant)

        # Tính metrics
        results = RAGMetrics.evaluate_retrieval(
            retrieved_docs_list,
            relevant_docs_list,
            k_values=[1, 3, 5, top_k] if top_k not in [1, 3, 5] else [1, 3, 5]
        )

        return results

    def cross_validate(
        self,
        params: Dict,
        n_folds: int = 3,
        verbose: bool = False
    ) -> Dict:
        """
        Cross-validation cho RAG system

        Args:
            params: Hyperparameters
            n_folds: Số folds
            verbose: In chi tiết

        Returns:
            Average metrics across folds
        """
        self.load_data()

        test_queries = TestDataset.get_test_queries()
        cv_splits = TestDataset.create_cv_splits(len(test_queries), n_folds)

        fold_results = []

        for split in cv_splits:
            fold_num = split['fold']
            val_indices = split['val']

            # Queries cho fold này
            val_queries = [test_queries[i] for i in val_indices]

            # Build RAG với params
            store, chunks = self.build_rag_system(
                chunk_size=params['chunk_size'],
                overlap=params['overlap'],
                vectorizer_type=params.get('vectorizer_type', 'tfidf'),
                vectorizer_params=params.get('vectorizer_params', None)
            )

            # Evaluate
            results = self.evaluate_on_queries(
                store, chunks, val_queries, params['top_k']
            )

            fold_results.append(results)

            if verbose:
                print(f"\n📊 Fold {fold_num}/{n_folds}:")
                print(f"  MRR: {results['mrr']:.4f}")
                print(f"  NDCG@5: {results['ndcg'].get(5, 0):.4f}")

        # Average across folds
        avg_results = self._average_fold_results(fold_results)

        return avg_results

    def _average_fold_results(self, fold_results: List[Dict]) -> Dict:
        """Tính trung bình metrics qua các folds"""
        n_folds = len(fold_results)

        avg = {
            'precision': {},
            'recall': {},
            'f1': {},
            'ndcg': {},
            'hit_rate': {},
            'mrr': 0.0
        }

        # Average MRR
        avg['mrr'] = sum(r['mrr'] for r in fold_results) / n_folds

        # Average metrics theo K
        # Lấy tất cả K values từ fold đầu tiên
        k_values = list(fold_results[0]['precision'].keys())

        for k in k_values:
            for metric in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                values = [r[metric].get(k, 0) for r in fold_results]
                avg[metric][k] = sum(values) / n_folds

        return avg

    def grid_search(
        self,
        param_grid: Dict[str, List],
        n_folds: int = 3,
        scoring: str = 'ndcg@5'
    ) -> Tuple[Dict, List[Dict]]:
        """
        Grid Search để tìm hyperparameters tốt nhất

        Args:
            param_grid: Dictionary chứa lists of values cho mỗi param
            n_folds: Số folds cho CV
            scoring: Metric để optimize ('ndcg@5', 'mrr', 'f1@5', etc.)

        Returns:
            (best_params, all_results)
        """
        self.load_data()

        print(f"\n🔍 GRID SEARCH - Tối ưu hóa hyperparameters")
        print(f"Scoring metric: {scoring}")
        print(f"Cross-validation: {n_folds}-fold")

        # Generate tất cả combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        print(f"Tổng số combinations: {len(combinations)}")

        all_results = []
        best_score = -1
        best_params = None

        # Progress bar
        for combo in tqdm(combinations, desc="Grid Search"):
            params = dict(zip(param_names, combo))

            # Cross-validate
            cv_results = self.cross_validate(params, n_folds=n_folds, verbose=False)

            # Extract score
            score = self._extract_score(cv_results, scoring)

            # Save results
            result = {
                'params': params.copy(),
                'cv_results': cv_results,
                'score': score
            }
            all_results.append(result)

            # Update best
            if score > best_score:
                best_score = score
                best_params = params.copy()

        print(f"\n✅ Grid Search hoàn tất!")
        print(f"🏆 Best {scoring}: {best_score:.4f}")
        print(f"🎯 Best params: {best_params}")

        return best_params, all_results

    def _extract_score(self, results: Dict, scoring: str) -> float:
        """Extract score từ results dict theo scoring metric"""
        if scoring == 'mrr':
            return results['mrr']

        # Parse metric@k format
        if '@' in scoring:
            metric, k_str = scoring.split('@')
            k = int(k_str)
            return results.get(metric, {}).get(k, 0.0)

        # Default: MRR
        return results['mrr']

    def compare_models(
        self,
        models_config: List[Dict],
        test_queries: List[Dict] = None
    ) -> Dict[str, Dict]:
        """
        So sánh nhiều models/configurations

        Args:
            models_config: List of model configs
            test_queries: Test queries (nếu None sẽ dùng default)

        Returns:
            Dictionary with results for each model
        """
        self.load_data()

        if test_queries is None:
            test_queries = TestDataset.get_test_queries()

        results = {}

        print(f"\n🔬 SO SÁNH {len(models_config)} MODELS")
        print("=" * 60)

        for config in models_config:
            model_name = config.get('name', 'Unknown')
            print(f"\n📊 Đánh giá: {model_name}")

            # Build system
            store, chunks = self.build_rag_system(
                chunk_size=config.get('chunk_size', 500),
                overlap=config.get('overlap', 100),
                vectorizer_type=config.get('vectorizer_type', 'tfidf'),
                vectorizer_params=config.get('vectorizer_params', None)
            )

            # Evaluate
            metrics = self.evaluate_on_queries(
                store,
                chunks,
                test_queries,
                config.get('top_k', 5)
            )

            results[model_name] = {
                'config': config,
                'metrics': metrics
            }

            print(f"  MRR: {metrics['mrr']:.4f}")
            print(f"  NDCG@5: {metrics['ndcg'].get(5, 0):.4f}")
            print(f"  F1@5: {metrics['f1'].get(5, 0):.4f}")

        print("\n" + "=" * 60)

        return results
