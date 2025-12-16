"""
Script để chạy đánh giá và tối ưu hóa RAG system
Bao gồm:
- Hyperparameter optimization
- Cross-validation
- Model comparison
- Detailed reporting
"""
import json
import os
from datetime import datetime
from evaluation.optimizer import RAGOptimizer
from evaluation.test_data import TestDataset
from evaluation.metrics import print_evaluation_results


def run_model_comparison(data_dir: str = "data"):
    """
    So sánh các models/approaches khác nhau
    """
    print("\n" + "=" * 70)
    print("🔬 BƯỚC 1: SO SÁNH CÁC MODELS")
    print("=" * 70)

    optimizer = RAGOptimizer(data_dir)

    # Định nghĩa các models để so sánh
    models_config = [
        {
            'name': 'TF-IDF (Unigrams)',
            'chunk_size': 500,
            'overlap': 100,
            'top_k': 5,
            'vectorizer_type': 'tfidf',
            'vectorizer_params': {
                'max_features': 5000,
                'ngram_range': (1, 1)
            }
        },
        {
            'name': 'TF-IDF (Unigrams + Bigrams)',
            'chunk_size': 500,
            'overlap': 100,
            'top_k': 5,
            'vectorizer_type': 'tfidf',
            'vectorizer_params': {
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        },
        {
            'name': 'TF-IDF (Up to Trigrams)',
            'chunk_size': 500,
            'overlap': 100,
            'top_k': 5,
            'vectorizer_type': 'tfidf',
            'vectorizer_params': {
                'max_features': 7000,
                'ngram_range': (1, 3)
            }
        },
        {
            'name': 'BM25 (k1=1.5, b=0.75)',
            'chunk_size': 500,
            'overlap': 100,
            'top_k': 5,
            'vectorizer_type': 'bm25',
            'vectorizer_params': {
                'k1': 1.5,
                'b': 0.75
            }
        },
        {
            'name': 'BM25 (k1=2.0, b=0.5)',
            'chunk_size': 500,
            'overlap': 100,
            'top_k': 5,
            'vectorizer_type': 'bm25',
            'vectorizer_params': {
                'k1': 2.0,
                'b': 0.5
            }
        }
    ]

    results = optimizer.compare_models(models_config)

    # Print comparison table
    print("\n" + "=" * 70)
    print("📊 BẢNG SO SÁNH KẾT QUẢ")
    print("=" * 70)

    print(f"\n{'Model':<35} {'MRR':<10} {'NDCG@5':<10} {'F1@5':<10} {'P@5':<10}")
    print("-" * 70)

    for model_name, result in results.items():
        metrics = result['metrics']
        mrr = metrics['mrr']
        ndcg5 = metrics['ndcg'].get(5, 0)
        f1_5 = metrics['f1'].get(5, 0)
        p5 = metrics['precision'].get(5, 0)

        print(f"{model_name:<35} {mrr:<10.4f} {ndcg5:<10.4f} {f1_5:<10.4f} {p5:<10.4f}")

    # Tìm model tốt nhất
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['mrr'])
    print(f"\n🏆 MODEL TỐT NHẤT: {best_model[0]}")
    print(f"   MRR: {best_model[1]['metrics']['mrr']:.4f}")

    return results, best_model


def run_hyperparameter_optimization(data_dir: str = "data"):
    """
    Chạy hyperparameter optimization với grid search
    """
    print("\n" + "=" * 70)
    print("🎯 BƯỚC 2: TỐI ƯU HÓA HYPERPARAMETERS")
    print("=" * 70)

    optimizer = RAGOptimizer(data_dir)

    # Định nghĩa grid search space (reduced để chạy nhanh hơn)
    param_grid = {
        'chunk_size': [400, 500, 600],
        'overlap': [75, 100, 150],
        'top_k': [5],  # Fixed để đơn giản
        'vectorizer_type': ['tfidf', 'bm25'],
        'vectorizer_params': [
            {'max_features': 5000, 'ngram_range': (1, 2)},
            {'max_features': 7000, 'ngram_range': (1, 3)},
        ]
    }

    print("\n📋 Grid Search Parameters:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Run grid search
    best_params, all_results = optimizer.grid_search(
        param_grid,
        n_folds=3,
        scoring='ndcg@5'
    )

    # Print top 5 configurations
    print("\n🏅 TOP 5 CONFIGURATIONS:")
    print("-" * 70)

    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)

    for i, result in enumerate(sorted_results[:5], 1):
        params = result['params']
        score = result['score']

        print(f"\n#{i} - NDCG@5: {score:.4f}")
        print(f"  Chunk size: {params['chunk_size']}")
        print(f"  Overlap: {params['overlap']}")
        print(f"  Vectorizer: {params['vectorizer_type']}")
        print(f"  Params: {params['vectorizer_params']}")

    return best_params, all_results


def run_cross_validation(data_dir: str = "data", params: dict = None):
    """
    Chạy cross-validation với config cụ thể
    """
    print("\n" + "=" * 70)
    print("✅ BƯỚC 3: CROSS-VALIDATION ĐÁNH GIÁ")
    print("=" * 70)

    optimizer = RAGOptimizer(data_dir)

    if params is None:
        # Default params
        params = {
            'chunk_size': 500,
            'overlap': 100,
            'top_k': 5,
            'vectorizer_type': 'tfidf',
            'vectorizer_params': {
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        }

    print(f"\n📋 Configuration:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Run CV
    cv_results = optimizer.cross_validate(params, n_folds=5, verbose=True)

    # Print results
    print_evaluation_results(cv_results, "Cross-Validation (5-fold)")

    return cv_results


def generate_report(
    comparison_results,
    best_model,
    optimization_results,
    cv_results,
    output_file: str = "evaluation/results/evaluation_report.json"
):
    """
    Tạo báo cáo đánh giá chi tiết dạng JSON
    """
    print("\n" + "=" * 70)
    print("📝 TẠO BÁO CÁO ĐÁNH GIÁ")
    print("=" * 70)

    report = {
        'timestamp': datetime.now().isoformat(),
        'model_comparison': {},
        'best_model': {
            'name': best_model[0],
            'config': best_model[1]['config'],
            'metrics': best_model[1]['metrics']
        },
        'hyperparameter_optimization': {
            'best_params': optimization_results[0],
            'best_score': optimization_results[1][0]['score'] if optimization_results[1] else 0
        },
        'cross_validation': cv_results
    }

    # Add comparison results
    for model_name, result in comparison_results.items():
        report['model_comparison'][model_name] = {
            'config': result['config'],
            'metrics': result['metrics']
        }

    # Create directory nếu chưa có
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Báo cáo đã được lưu: {output_file}")

    return report


def main():
    """
    Main function - chạy toàn bộ evaluation pipeline
    """
    print("\n" + "=" * 70)
    print("🚀 RAG SYSTEM EVALUATION & OPTIMIZATION")
    print("=" * 70)

    data_dir = "data"

    try:
        # 1. Model comparison
        comparison_results, best_model = run_model_comparison(data_dir)

        # 2. Hyperparameter optimization
        best_params, optimization_results = run_hyperparameter_optimization(data_dir)

        # 3. Cross-validation với best params
        cv_results = run_cross_validation(data_dir, best_params)

        # 4. Generate report
        report = generate_report(
            comparison_results,
            best_model,
            (best_params, optimization_results),
            cv_results
        )

        print("\n" + "=" * 70)
        print("✅ ĐÁNH GIÁ HOÀN TẤT!")
        print("=" * 70)

        print(f"\n🎯 KẾT LUẬN:")
        print(f"  • Model tốt nhất: {best_model[0]}")
        print(f"  • MRR: {best_model[1]['metrics']['mrr']:.4f}")
        print(f"  • NDCG@5: {best_model[1]['metrics']['ndcg'].get(5, 0):.4f}")
        print(f"  • Best hyperparameters: {best_params}")

        return report

    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
