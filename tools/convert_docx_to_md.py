from pathlib import Path
import pypandoc

# 🔥 tự download pandoc nếu chưa có
pypandoc.download_pandoc()

BASE_DIR = Path(__file__).resolve().parents[1]

def convert_all():
    docx_files = list(BASE_DIR.rglob("*.docx"))

    if not docx_files:
        print("❌ Không tìm thấy file .docx")
        print(f"👉 Đã tìm trong: {BASE_DIR}")
        return

    for docx in docx_files:
        md_path = docx.with_suffix("")  # bỏ .docx
        md_path = md_path.with_suffix(".md")

        print(f"🔄 Chuyển: {docx} → {md_path}")

        pypandoc.convert_file(
            source_file=str(docx),
            to="md",
            outputfile=str(md_path),
            extra_args=["--standalone"]
        )

        # xoá file docx sau khi chuyển
        docx.unlink()

    print("✅ Đã chuyển toàn bộ .docx → .md")

if __name__ == "__main__":
    convert_all()
