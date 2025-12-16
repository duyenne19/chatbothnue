# crawler/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrawlerConfig:
    # 🔥 FIX QUAN TRỌNG: data luôn ở ROOT/data
    output_dir: str = str(
        Path(__file__).resolve().parents[1] / "data"
    )

    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    timeout: int = 15
    sleep: float = 1.0
    max_retries: int = 3
