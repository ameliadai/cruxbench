"""Fetch Polymarket markets with daily price history and tags.

Settings are in data_collection/data_config.py.

Usage:
    python data_collection/fetch_polymarket.py

Writes: data/polymarket_0325.json
"""

import ast
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Put the repo root first on sys.path so `data_collection` resolves to this
# repo's package, not a same-named package installed elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.data_config import (
    TARGET_TOTAL,
    START_DATE_MIN,
    MIN_VOLUME,
    CLOSED,
    WORKERS,
    PAGE_SIZE,
    PRICE_FIDELITY,
    PROCESSED_OUTPUT_PATH,
)

GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
CLOB_PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"

SESSION = requests.Session()


# ── 1. Fetch markets ─────────────────────────────────────────────────────────

def fetch_markets(
    target_total: int = TARGET_TOTAL,
    start_date_min: str = START_DATE_MIN,
    volume_num_min: int = MIN_VOLUME,
) -> pd.DataFrame:
    """Page through the Gamma API and return a DataFrame of markets."""
    base_params = {
        "order": "volume24hr",
        "ascending": "false",
        "volume_num_min": volume_num_min,
    }
    if CLOSED is None:
        # Fetch active (live) markets — active=true is required per API best practices
        base_params["active"] = "true"
    else:
        base_params["closed"] = str(CLOSED).lower()
    if start_date_min is not None:
        base_params["start_date_min"] = start_date_min

    # NOTE: The paper snapshot (2026-03-25) was fetched by paging with `offset`
    # through the full result set. The Gamma API has since started rejecting large
    # offsets (HTTP 422 beyond a few thousand), so this loop may stop early today
    # and cannot reproduce the full 108,101-market dump.
    all_markets = []
    offset = 0

    while len(all_markets) < target_total:
        params = dict(base_params)
        params["limit"] = min(PAGE_SIZE, target_total - len(all_markets))
        params["offset"] = offset

        r = requests.get(GAMMA_API_URL, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()

        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] offset={offset} got={len(batch)} total={len(all_markets) + len(batch)}")

        if not batch:
            break

        all_markets.extend(batch)
        offset += PAGE_SIZE

    print(f"Fetched {len(all_markets)} markets")
    df = pd.DataFrame(all_markets)
    df = df.dropna(subset=["endDate"])
    df = df.drop_duplicates(subset=["question", "description"])
    print(f"After dropping duplicates: {len(df)} markets")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_bracket_str(x):
    """Convert string representations of lists (e.g. "['Yes','No']") to actual lists."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            return x
    return x


# ── 2. Price history ──────────────────────────────────────────────────────────

def get_price_history(row, retries: int = 6) -> dict:
    """Fetch daily price history for each outcome of a market with retry on 429."""
    outcomes = _parse_bracket_str(row["outcomes"])
    if not isinstance(outcomes, list):
        outcomes = [outcomes]
    token_ids = _parse_bracket_str(row["clobTokenIds"])
    if not isinstance(token_ids, list):
        token_ids = [token_ids]

    if len(outcomes) != len(token_ids):
        raise ValueError(f"outcomes/token_ids length mismatch: {len(outcomes)} vs {len(token_ids)}")

    url = CLOB_PRICE_HISTORY_URL
    histories = {}

    for outcome, token_id in zip(outcomes, token_ids):
        params = {
            "market": str(token_id),
            "interval": "max",
            "fidelity": str(PRICE_FIDELITY),
        }

        for attempt in range(retries + 1):
            r = SESSION.get(url, params=params, timeout=30)

            if r.status_code != 429:
                r.raise_for_status()
                break

            ra = r.headers.get("Retry-After")
            sleep_s = float(ra) if ra is not None else min(60.0, 2 ** attempt) + random.random()
            time.sleep(sleep_s)
        else:
            raise requests.HTTPError(f"429 after {retries} retries for token {token_id}")

        raw = r.json().get("history", [])
        ts_price = {
            pd.to_datetime(pt["t"], unit="s", utc=True).isoformat(): pt["p"]
            for pt in raw
        }
        histories[str(outcome)] = ts_price

    return histories


def bulk_price_history(df: pd.DataFrame, max_workers: int = WORKERS) -> pd.DataFrame:
    """Fetch price history for all rows in parallel."""
    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_price_history, df.iloc[i].to_dict()): i
            for i in range(len(df))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Price history"):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                results[i] = {"error": str(e)}

    df = df.copy()
    df["price_history"] = results
    return df


# ── 3. Tags ───────────────────────────────────────────────────────────────────

def get_tags(market_id: str, retries: int = 6) -> list[str]:
    """Fetch tags for a market with exponential backoff on 429."""
    url = f"{GAMMA_API_URL}/{market_id}/tags"

    for attempt in range(retries + 1):
        r = SESSION.get(url, timeout=30)

        if r.status_code != 429:
            r.raise_for_status()
            return [t["label"] for t in r.json()]

        ra = r.headers.get("Retry-After")
        sleep_s = float(ra) if ra is not None else min(60.0, 2 ** attempt) + random.random()
        time.sleep(sleep_s)

    return []


def bulk_tags(df: pd.DataFrame, max_workers: int = WORKERS) -> pd.DataFrame:
    """Fetch tags for all rows in parallel."""
    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_tags, str(df.iloc[i]["id"])): i
            for i in range(len(df))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tags"):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error row {i}: {e}")
                results[i] = []

    df = df.copy()
    df["tags"] = results
    return df


# ── 4. Process ────────────────────────────────────────────────────────────────

def _get_resolution(row, hi=0.99):
    """Determine resolution from outcome prices. Returns winning outcome or None."""
    outcomes = row["outcomes"]
    prices = row["outcomePrices"]
    best_i = max(range(len(prices)), key=lambda i: float(prices[i]))
    best_price = float(prices[best_i])
    if best_price >= hi:
        return outcomes[best_i]
    return None


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Parse bracket strings and add resolution column."""
    df = df.apply(lambda col: col.map(_parse_bracket_str))
    df["resolution"] = [_get_resolution(row) for _, row in df.iterrows()]
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df = fetch_markets()
    df = bulk_price_history(df)
    df = bulk_tags(df)
    df = process(df)

    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(PROCESSED_OUTPUT_PATH, orient="records", lines=True)
    print(f"Saved {len(df)} markets to {PROCESSED_OUTPUT_PATH}")


if __name__ == "__main__":
    run()
