from pathlib import Path
from datetime import datetime, timezone
import hashlib

import feedparser
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_LIVE_DIR = PROJECT_ROOT / "data" / "raw" / "live"
RAW_LIVE_DIR.mkdir(parents=True, exist_ok=True)


RSS_FEEDS = {
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "reuters_world": "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
    "the_guardian_world": "https://www.theguardian.com/world/rss",
}


def make_article_id(url: str, title: str) -> str:
    raw = f"{url}_{title}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def parse_entry(entry, source_name: str) -> dict:
    title = getattr(entry, "title", "")
    link = getattr(entry, "link", "")
    summary = getattr(entry, "summary", "")

    published = getattr(entry, "published", "")
    fetched_at = datetime.now(timezone.utc).isoformat()

    text = f"{title}. {summary}".strip()

    return {
        "article_id": make_article_id(link, title),
        "title": title,
        "text": text,
        "summary": summary,
        "url": link,
        "source_name": source_name,
        "published_at": published,
        "fetched_at": fetched_at,
        "source_dataset": "live_rss",
        "label": "",
        "label_id": None,
    }


def fetch_rss_feed(source_name: str, feed_url: str) -> pd.DataFrame:
    print(f"Fetching {source_name}: {feed_url}")

    feed = feedparser.parse(feed_url)

    rows = []
    for entry in feed.entries:
        rows.append(parse_entry(entry, source_name))

    return pd.DataFrame(rows)


def fetch_all_feeds() -> pd.DataFrame:
    all_frames = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            df = fetch_rss_feed(source_name, feed_url)
            if not df.empty:
                all_frames.append(df)
        except Exception as e:
            print(f"[WARNING] Failed to fetch {source_name}: {e}")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["article_id"])
    combined = combined[combined["text"].str.len() > 20]
    combined = combined.reset_index(drop=True)

    return combined


def main() -> None:
    live_df = fetch_all_feeds()

    if live_df.empty:
        print("No live articles fetched.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = RAW_LIVE_DIR / f"live_news_{timestamp}.csv"
    parquet_path = RAW_LIVE_DIR / f"live_news_{timestamp}.parquet"

    latest_csv = RAW_LIVE_DIR / "live_news_latest.csv"
    latest_parquet = RAW_LIVE_DIR / "live_news_latest.parquet"

    live_df.to_csv(csv_path, index=False)
    live_df.to_parquet(parquet_path, index=False)

    live_df.to_csv(latest_csv, index=False)
    live_df.to_parquet(latest_parquet, index=False)

    print("Live RSS ingestion completed.")
    print(f"Rows fetched: {len(live_df)}")
    print(f"Saved: {latest_parquet}")


if __name__ == "__main__":
    main()