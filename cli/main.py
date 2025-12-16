from rag.config import RAGConfig
from rag.bootstrap import build_knowledge_base
from rag.chatbot import RAGChatbot
from rag.llm_gemini import GeminiLLM

def main():
    print("🤖 RAG TUYỂN SINH CHATBOT")
    print("Gõ 'exit' để thoát\n")

    config = RAGConfig()
    config.validate()

    vector_store = build_knowledge_base(config)
    llm = GeminiLLM(config.gemini_model)
    bot = RAGChatbot(config, vector_store, llm)

    while True:
        q = input("❓ ").strip()
        if q.lower() == "exit":
            print("👋 Tạm biệt!")
            break
        if not q:
            continue
        print("👉", bot.ask(q))
        print("-" * 60)

if __name__ == "__main__":
    main()
