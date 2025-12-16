from rag.markdown_loader import MarkdownLoader
from rag.text_chunker import TextChunker
from rag.vector_store import VectorStore

def build_knowledge_base(config):
    config.validate()

    loader = MarkdownLoader(config.markdown_dir)
    documents = loader.load()

    chunker = TextChunker(
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap
    )
    chunks = chunker.chunk_documents(documents)

    print(f"🧩 Tổng số chunk: {len(chunks)}")

    store = VectorStore(config.embedding_model)
    store.build(chunks)

    return store
