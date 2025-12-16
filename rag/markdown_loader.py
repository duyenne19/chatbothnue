# rag/markdown_loader.py
from pathlib import Path


class MarkdownLoader:
    def __init__(self, base_dir: str):
        # 🔥 FIX QUAN TRỌNG: resolve path tuyệt đối
        self.base_dir = Path(base_dir).resolve()

    def load(self) -> list[str]:
        texts = []

        print(f"📂 Đang tìm Markdown trong: {self.base_dir}")

        for md_file in self.base_dir.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
                texts.append(text)
                print(f"✅ Load: {md_file}")
            except Exception as e:
                print(f"❌ Lỗi đọc {md_file}: {e}")

        print(f"📄 Đã load {len(texts)} file Markdown")
        return texts
