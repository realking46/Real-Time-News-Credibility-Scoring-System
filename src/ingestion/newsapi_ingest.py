from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_NEWSAPI_DIR = PROJECT_ROOT / "data" / "raw" / "newsapi"
RAW_NEWSAPI_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"


def make_article_id(url: str, title: str) -> str:
    raw = f"{url}_{title}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def fetch_newsapi_articles(
    country: str = "us",
    category: str = "general",
    page_size: int = 50,
) -> pd.DataFrame:
    if not NEWSAPI_KEY:
        raise RuntimeError(
            "NEWSAPI_KEY not found. Add it to a .env file as NEWSAPI_KEY=your_key"
        )

    params = {
        "country": country,
        "category": category,
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }

    response = requests.get(NEWSAPI_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    articles = data.get("articles", [])

    rows = []

    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        content = article.get("content") or ""
        url = article.get("url") or ""
        source_name = article.get("source", {}).get("name", "")
        published_at = article.get("publishedAt", "")

        text = f"{title}. {description}. {content}".strip()

        rows.append(
            {
                "article_id": make_article_id(url, title),
                "title": title,
                "text": text,
                "summary": description,
                "url": url,
                "source_name": source_name,
                "published_at": published_at,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source_dataset": "newsapi",
                "label": "",
                "label_id": None,
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["article_id"])
        df = df[df["text"].str.len() > 20]
        df = df.reset_index(drop=True)

    return df


def main() -> None:
    df = fetch_newsapi_articles()

    if df.empty:
        print("No NewsAPI articles fetched.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = RAW_NEWSAPI_DIR / f"newsapi_{timestamp}.csv"
    parquet_path = RAW_NEWSAPI_DIR / f"newsapi_{timestamp}.parquet"

    latest_csv = RAW_NEWSAPI_DIR / "newsapi_latest.csv"
    latest_parquet = RAW_NEWSAPI_DIR / "newsapi_latest.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    df.to_csv(latest_csv, index=False)
    df.to_parquet(latest_parquet, index=False)

    print("NewsAPI ingestion completed.")
    print(f"Rows fetched: {len(df)}")
    print(f"Saved: {latest_parquet}")


if __name__ == "__main__":
    main()