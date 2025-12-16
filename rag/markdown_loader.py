import os

class MarkdownLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self):
        documents = []

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"❌ Thư mục không tồn tại: {self.data_dir}")

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.lower().endswith(".md"):
                    continue
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                except Exception as e:
                    print(f"⚠️ Không đọc được file: {path} | {e}")
                    continue

                if not content:
                    continue

                documents.append({
                    "content": content,
                    "source": os.path.normpath(path)
                })
                print(f"✅ Load: {path}")

        if not documents:
            raise RuntimeError("❌ Không có file Markdown hợp lệ")

        return documents
