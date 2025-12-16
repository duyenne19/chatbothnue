from rag.markdown_loader import MarkdownLoader
from rag.text_chunker import TextChunker


def run_eda(data_dir: str):
    print(f"📂 Đang phân tích dữ liệu tại: {data_dir}")

    loader = MarkdownLoader(data_dir)
    docs = loader.load()

    print("\n===== THỐNG KÊ EDA =====")
    print(f"Số tài liệu gốc: {len(docs)}")

    if not docs:
        print("⚠️ Không có tài liệu Markdown để phân tích")
        return

    chunker = TextChunker()
    chunks = chunker.split_documents(docs)

    print(f"Số đoạn văn bản (chunks): {len(chunks)}")

    if not chunks:
        print("⚠️ Không tạo được chunk nào")
        return

    lengths = [len(c["text"].split()) for c in chunks]

    print(f"Độ dài trung bình mỗi chunk: {sum(lengths) // len(lengths)} từ")
    print(f"Chunk ngắn nhất: {min(lengths)} từ")
    print(f"Chunk dài nhất: {max(lengths)} từ")


if __name__ == "__main__":
    run_eda("../data")
