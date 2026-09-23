"""Fetch raw hourly Yes/No price histories from the Polymarket CLOB API.

Usage:
    python data_collection/fetch_hourly_prices.py --markets data/ultimates_pre_volume.json

Writes: data/hourly_prices/<market_id>.json
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


REPO = Path(__file__).resolve().parent.parent
DEFAULT_MARKETS = REPO / "data/ultimates_pre_volume.json"
DEFAULT_OUTPUT = REPO / "data/hourly_prices"

CLOB_URL = "https://clob.polymarket.com/prices-history"
FIDELITY_MINUTES = 60
CHUNK_SECONDS = 7 * 24 * 3600


def parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [value]
    return [value]


def fetch_chunk(session, token_id, start_ts, end_ts, retries=6):
    params = {
        "market": str(token_id),
        "startTs": str(start_ts),
        "endTs": str(end_ts),
        "fidelity": str(FIDELITY_MINUTES),
    }
    for attempt in range(retries + 1):
        response = session.get(CLOB_URL, params=params, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json().get("history", [])
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else min(60.0, 2**attempt) + random.random()
        time.sleep(delay)
    raise requests.HTTPError(f"429 after {retries} retries for token {token_id}")


def fetch_market(market):
    outcomes = parse_list(market["outcomes"])
    token_ids = parse_list(market["clobTokenIds"])
    start_ts = int(market["ph_start"]) // 1000
    end_ts = int(market["ph_end"]) // 1000
    histories = {}
    with requests.Session() as session:
        for outcome, token_id in zip(outcomes, token_ids):
            points = []
            cursor = start_ts
            while cursor < end_ts:
                next_cursor = min(cursor + CHUNK_SECONDS, end_ts)
                points.extend(fetch_chunk(session, token_id, cursor, next_cursor))
                cursor = next_cursor
            by_timestamp = {int(point["t"]): float(point["p"]) for point in points}
            histories[str(outcome)] = {
                datetime.fromtimestamp(ts, timezone.utc).isoformat(): price
                for ts, price in sorted(by_timestamp.items())
            }
    return histories


def fetch_and_write(market, output_dir, overwrite=False):
    market_id = int(market["id"])
    path = output_dir / f"{market_id}.json"
    if path.exists() and not overwrite:
        return market_id, "cached"
    path.write_text(json.dumps(fetch_market(market)))
    return market_id, "fetched"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    markets = json.loads(args.markets.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_and_write, market, args.output, args.overwrite): market
            for market in markets
        }
        for index, future in enumerate(as_completed(futures), 1):
            market = futures[future]
            try:
                market_id, status = future.result()
                print(f"[{index}/{len(markets)}] {market_id}: {status}")
            except Exception as error:
                failures.append(int(market["id"]))
                print(f"[{index}/{len(markets)}] {market['id']}: failed: {error}")

    if failures:
        raise SystemExit(f"Failed markets: {failures}")
    print(f"Wrote {len(markets)} raw hourly price histories to {args.output}")


if __name__ == "__main__":
    main()
