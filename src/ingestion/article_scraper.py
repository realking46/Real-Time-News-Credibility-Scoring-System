from pathlib import Path
from datetime import datetime, timezone
import hashlib
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SCRAPED_DIR = PROJECT_ROOT / "data" / "raw" / "scraped"
RAW_SCRAPED_DIR.mkdir(parents=True, exist_ok=True)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NewsCredibilityBot/0.1; "
        "+https://github.com/realking46/Real-Time-News-Credibility-Scoring-System)"
    )
}


def make_article_id(url: str, title: str) -> str:
    raw = f"{url}_{title}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def extract_article_text(url: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    paragraphs = soup.find_all("p")
    paragraph_texts = [
        p.get_text(" ", strip=True)
        for p in paragraphs
        if len(p.get_text(strip=True)) > 30
    ]

    text = " ".join(paragraph_texts)

    return {
        "article_id": make_article_id(url, title),
        "title": title,
        "text": text,
        "summary": text[:300],
        "url": url,
        "source_name": url.split("/")[2] if "://" in url else "",
        "published_at": "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "beautifulsoup_scraper",
        "label": "",
        "label_id": None,
    }


def scrape_urls(urls: list[str]) -> pd.DataFrame:
    rows = []

    for url in urls:
        try:
            print(f"Scraping: {url}")
            article = extract_article_text(url)

            if len(article["text"]) > 100:
                rows.append(article)
            else:
                print(f"[WARNING] Too little text extracted from: {url}")

        except Exception as e:
            print(f"[WARNING] Failed to scrape {url}: {e}")

    return pd.DataFrame(rows)


def main() -> None:
    urls = sys.argv[1:]

    if not urls:
        print("Usage:")
        print("python -m src.ingestion.article_scraper <url1> <url2>")
        return

    df = scrape_urls(urls)

    if df.empty:
        print("No articles scraped.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = RAW_SCRAPED_DIR / f"scraped_articles_{timestamp}.csv"
    parquet_path = RAW_SCRAPED_DIR / f"scraped_articles_{timestamp}.parquet"

    latest_csv = RAW_SCRAPED_DIR / "scraped_articles_latest.csv"
    latest_parquet = RAW_SCRAPED_DIR / "scraped_articles_latest.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    df.to_csv(latest_csv, index=False)
    df.to_parquet(latest_parquet, index=False)

    print("BeautifulSoup scraping completed.")
    print(f"Rows scraped: {len(df)}")
    print(f"Saved: {latest_parquet}")


if __name__ == "__main__":
    main()