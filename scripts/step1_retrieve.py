"""Step 1: Retrieve news articles via AskNews for each question × active window.

For each question and each active window, searches AskNews for relevant articles
and stores both the prompt-ready string and structured article dicts.

Usage:
    python scripts/step1_retrieve.py --input ultimates.json --output ultimates_w_news.json

Reads:  data/ultimates.json
Writes: data/ultimates_w_news.json
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from asknews_sdk import AskNewsSDK
from tqdm import tqdm
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from config import DATA_DIR


def resolve_path(path_str):
    """Resolve a path: absolute stays absolute, relative is relative to DATA_DIR."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return DATA_DIR / p


def fetch_articles_for_window(ask, query, ph_start_dt, window, max_range_days, n_articles):
    """Fetch articles for a single window, returning (as_string, as_dicts)."""
    win_start_dt = pd.to_datetime(window["start"])

    # If window starts at question open date (period 0), search before open
    if win_start_dt == ph_start_dt:
        search_start = ph_start_dt - pd.Timedelta(days=max_range_days)
        search_end = ph_start_dt
    else:
        search_start = ph_start_dt
        if (win_start_dt - search_start).days > max_range_days:
            search_start = win_start_dt - pd.Timedelta(days=max_range_days)
        search_end = win_start_dt

    start_ts = int(search_start.timestamp())
    end_ts = int(search_end.timestamp())

    response = ask.news.search_news(
        query=query,
        n_articles=n_articles,
        return_type="both",
        method="kw",
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        time_filter="pub_date",
        languages=["en"],
        try_cache="7d",
    )

    articles = []
    for art in response.as_dicts:
        articles.append({
            "title": art.eng_title or art.title or "",
            "url": str(art.article_url),
            "article_id": str(art.article_id) if art.article_id else "",
            "source": art.source_id or "",
            "domain_url": art.domain_url or "",
            "pub_date": art.pub_date.isoformat() if art.pub_date else "",
            "summary": art.summary or "",
            "key_points": art.key_points or [],
            "keywords": art.keywords or [],
            "sentiment": art.sentiment,
            "classification": art.classification or "",
            "country": art.country or "",
            "language": art.language or "",
            "image_url": str(art.image_url) if art.image_url else "",
            "markdown_citation": art.markdown_citation or "",
        })

    return response.as_string, articles, search_start, search_end


def main():
    parser = argparse.ArgumentParser(description="Retrieve AskNews articles per question window")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file (relative to DATA_DIR or absolute)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file (relative to DATA_DIR or absolute)")
    parser.add_argument("--n-articles", type=int, default=10,
                        help="Number of articles to retrieve per window (default: 10)")
    parser.add_argument("--max-range", type=int, default=160,
                        help="Max search range in days before window start (default: 160)")
    args = parser.parse_args()

    api_key = os.environ.get("ASKNEWS_API_KEY", "")
    if not api_key:
        raise ValueError("ASKNEWS_API_KEY not set in environment. Add it to your .env file.")

    ask = AskNewsSDK(
        api_key=api_key,
        scopes=["chat", "news", "stories"],
    )

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    data = pd.read_json(input_path)
    print(f"Loaded {len(data)} questions from {input_path}")

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        q = f"{row.question} {row.get('description', '')}"
        ph_start_dt = pd.to_datetime(row.ph_start, unit="ms", utc=True)

        for window in row.active_windows:
            as_string, articles, search_start, search_end = fetch_articles_for_window(
                ask, q, ph_start_dt, window, args.max_range, args.n_articles,
            )

            window["as_string"] = as_string
            window["as_dicts"] = articles

            print(f"Q: {row.question} | W{window['period_idx']} | "
                  f"{search_start.date()} to {search_end.date()} | {len(articles)} articles")

    data.to_json(output_path, orient="records", indent=2)
    print(f"\nSaved {len(data)} questions to {output_path}")


if __name__ == "__main__":
    main()
