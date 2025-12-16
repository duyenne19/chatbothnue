class RAGChatbot:
    def __init__(self, config, vector_store, llm):
        self.config = config
        self.store = vector_store
        self.llm = llm
        self.min_score = 0.2
        self.max_context_chunks = 5

    def normalize_query(self, query: str) -> str:
        q = query.lower()
        mappings = {
            "cntt": "ngành công nghệ thông tin",
            "it": "ngành công nghệ thông tin",
            "công nghệ thông tin": "ngành công nghệ thông tin",
            "sư phạm toán": "ngành sư phạm toán",
        }
        for k, v in mappings.items():
            if k in q:
                return v
        return query

    def ask(self, query: str) -> str:
        query = self.normalize_query(query)
        results = self.store.search(query, self.config.top_k)

        filtered = [r for r in results if r["score"] >= self.min_score]
        if not filtered:
            return "❌ Không có dữ liệu đủ liên quan."

        filtered = filtered[: self.max_context_chunks]
        context = "\n\n".join(r["content"] for r in filtered)
        return self.llm.generate(query, context)
