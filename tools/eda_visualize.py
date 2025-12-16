from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import statistics

from rag.markdown_loader import MarkdownLoader
from rag.text_chunker import TextChunker


# ================== PATH CHUẨN ==================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ================== STYLE ==================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "axes.grid": True,
    "grid.alpha": 0.3
})


def run_eda(data_dir):
    print("===== EDA TUYỂN SINH =====")

    loader = MarkdownLoader(data_dir)
    docs = loader.load()

    if not docs:
        print("❌ Không có dữ liệu Markdown")
        return

    chunker = TextChunker()
    chunks = chunker.split_documents(docs)

    if not chunks:
        print("❌ Không tạo được chunk")
        return

    lengths = [len(c["content"].split()) for c in chunks]
    domains = [c["metadata"]["domain"] for c in chunks]

    print(f"Số tài liệu gốc: {len(docs)}")
    print(f"Số chunk: {len(chunks)}")
    print(f"Độ dài trung bình: {int(statistics.mean(lengths))} từ")
    print(f"Độ dài trung vị: {int(statistics.median(lengths))} từ")

    # ======================================================
    # 1️⃣ HISTOGRAM ĐỘ DÀI CHUNK
    # ======================================================
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=30)
    plt.axvline(statistics.mean(lengths), linestyle="--", label="Trung bình")
    plt.axvline(statistics.median(lengths), linestyle=":", label="Trung vị")

    plt.title("Phân bố độ dài các đoạn văn bản (chunk)")
    plt.xlabel("Số từ trong mỗi chunk")
    plt.ylabel("Số lượng chunk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hist_chunk_length.png", dpi=300)
    plt.show()

    # ======================================================
    # 2️⃣ BAR CHART – NGUỒN DỮ LIỆU
    # ======================================================
    domain_counts = Counter(domains).most_common(10)
    labels, values = zip(*domain_counts)

    plt.figure(figsize=(10, 5))
    bars = plt.barh(labels, values)
    plt.xlabel("Số lượng chunk")
    plt.title("Top 10 nguồn dữ liệu tuyển sinh")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                 str(width), va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_domains.png", dpi=300)
    plt.show()

    # ======================================================
    # 3️⃣ BOXPLOT – PHÂN TÁN ĐỘ DÀI
    # ======================================================
    plt.figure(figsize=(6, 5))
    plt.boxplot(lengths, vert=True)
    plt.title("Phân tán độ dài các đoạn văn bản")
    plt.ylabel("Số từ")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_chunk_length.png", dpi=300)
    plt.show()

    print(f"📊 Đã lưu hình tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda(DATA_DIR)
