# rag/text_chunker.py
class TextChunker:
    def __init__(self, size: int = 300, overlap: int = 50):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []

        step = self.size - self.overlap

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.size]
            chunk = " ".join(chunk_words).strip()
            if chunk:
                chunks.append(chunk)

        return chunks
