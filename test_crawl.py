from crawler.web_crawler import WebCrawler
from crawler.config import CrawlerConfig

urls = [
    "https://tuyensinh.hnue.edu.vn/tuyensinh2025/545"
]

crawler = WebCrawler(CrawlerConfig())
crawler.crawl(urls)
