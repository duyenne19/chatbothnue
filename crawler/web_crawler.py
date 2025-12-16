import time
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from crawler.config import CrawlerConfig


class WebCrawler:
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": config.user_agent}
        )

    # --------------------------------------------------
    def crawl(self, urls: list[str]):
        for url in urls:
            try:
                print(f"🌐 Crawling: {url}")
                self._crawl_one(url)
                time.sleep(self.config.sleep)
            except Exception as e:
                print(f"❌ Lỗi crawl {url}: {e}")

    # --------------------------------------------------
    def _crawl_one(self, url: str):
        html = self._fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        # Xoá thẻ rác
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Tiêu đề
        title = (
            soup.title.text.strip()
            if soup.title else "Không có tiêu đề"
        )

        # Ưu tiên nội dung chính
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="detail-content")
            or soup.find("div", class_="content")
            or soup.body
        )

        if not main:
            print("⚠️ Không tìm thấy nội dung chính")
            return

        body = main.get_text("\n", strip=True)

        if len(body) < 300:
            print("⚠️ Nội dung quá ngắn, bỏ qua")
            return

        md = f"""# {title}

Nguồn: {url}

{body}
"""
        self._save(url, md)

    # --------------------------------------------------
    def _fetch(self, url: str) -> str:
        for _ in range(self.config.max_retries):
            r = self.session.get(
                url,
                timeout=self.config.timeout
            )
            if r.ok:
                r.encoding = "utf-8"
                return r.text
            time.sleep(1)
        raise RuntimeError("Fetch failed")

    # --------------------------------------------------
    def _save(self, url: str, md: str):
        domain = urlparse(url).netloc
        hid = hashlib.md5(url.encode()).hexdigest()[:8]

        base = (
            Path(self.config.output_dir)
            / domain
            / f"page_{hid}"
        )
        base.mkdir(parents=True, exist_ok=True)

        (base / "content.md").write_text(
            md, encoding="utf-8"
        )
        (base / "metadata.json").write_text(
            json.dumps(
                {"url": url},
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print(f"✅ Đã lưu: {base / 'content.md'}")


# ================= RUN RIÊNG =================
if __name__ == "__main__":
    urls = [
        "https://tuyensinh.hnue.edu.vn/tuyensinh2025/545",
        "https://tuyensinh.hnue.edu.vn/khungchuongtrinh/320",
        "https://tuyensinh.hnue.edu.vn/tuyensinh2025/544",
        "https://tuyensinh.hnue.edu.vn/tuyensinh2025/533",
        "https://tuyensinh.hnue.edu.vn/khungchuongtrinh/319",
        "https://hnue.edu.vn/tin-tuc/10678",
        "https://hnue.edu.vn/tin-tuc/10732"

        # thêm link ở đây
    ]

    crawler = WebCrawler(CrawlerConfig())
    crawler.crawl(urls)
    print("🎉 Crawl xong dữ liệu")
