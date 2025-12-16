class TextChunker:
    def __init__(self, chunk_size=300, overlap=50):
        if overlap >= chunk_size:
            raise ValueError("overlap phải nhỏ hơn chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents):
        chunks = []

        for doc in documents:
            text = doc.get("content", "")
            source = doc.get("source", "unknown")

            if not text.strip():
                continue

            words = text.split()
            start = 0

            while start < len(words):
                end = start + self.chunk_size
                chunk_text = " ".join(words[start:end]).strip()
                if chunk_text:
                    chunks.append({
                        "content": chunk_text,
                        "source": source
                    })
                start += self.chunk_size - self.overlap

        return chunks
