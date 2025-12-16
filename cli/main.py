# cli/main.py
from pathlib import Path

from rag.rag_chatbot import RAGChatbot
from rag.config import RAGConfig


def main():
    print("\n🤖 RAG TUYỂN SINH CHATBOT")
    print("Gõ 'exit' để thoát\n")

    # 🔥 FIX QUAN TRỌNG: lấy thư mục gốc project
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"

    bot = RAGChatbot(
        RAGConfig(
            markdown_dir=str(DATA_DIR)
        )
    )

    bot.initialize()

    while True:
        q = input("❓ ").strip()
        if q.lower() == "exit":
            break
        print("👉", bot.ask(q))


if __name__ == "__main__":
    main()
