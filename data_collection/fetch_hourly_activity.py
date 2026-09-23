"""Fetch hourly on-chain trading volume for each market's selected window.

Reads Polymarket OrderFilled events from Polygon for the window plus 96 hours on
each side. RPC responses are cached under data/onchain_chunks.

Usage:
    POLYGON_RPC_URL=<url> python data_collection/fetch_hourly_activity.py --markets data/ultimates_pre_volume.json

Writes: data/hourly_activity/<market_id>.json
"""

import argparse
import datetime as dt
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv


REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
DEFAULT_MARKETS = REPO / "data/ultimates_pre_volume.json"
DEFAULT_PRICES = REPO / "data/hourly_prices"
DEFAULT_CACHE = REPO / "data/onchain_chunks"
DEFAULT_OUTPUT = REPO / "data/hourly_activity"

CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEGRISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"

CONTEXT_HOURS = 96
CHUNK_BLOCKS = 1000
POLYGON_BLOCK_SECONDS = 2.1
RPC_TIMEOUT_SECONDS = 60
RPC_PAUSE_SECONDS = 0.04


class ResponseTooLarge(Exception):
    pass


def parse_datetime(value):
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def floor_hour(value):
    return value.replace(minute=0, second=0, microsecond=0)


def rpc_call(rpc_url, method, params, retries=6):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                rpc_url,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "cruxbench/1.0"},
            )
            with urllib.request.urlopen(request, timeout=RPC_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode())
            if "error" in payload:
                message = str(payload["error"])
                if any(term in message.lower() for term in ("response size", "exceed", "more than")):
                    raise ResponseTooLarge(message)
                if attempt < retries:
                    time.sleep(min(30.0, 2**attempt))
                    continue
                raise RuntimeError(f"RPC error: {message}")
            return payload["result"]
        except urllib.error.HTTPError as error:
            text = ""
            try:
                text = error.read().decode()[:400]
            except Exception:
                pass
            if error.code == 400 and any(
                term in text.lower() for term in ("response size", "exceed", "more than", "range")
            ):
                raise ResponseTooLarge(text)
            if attempt < retries:
                time.sleep(min(30.0, 2**attempt))
                continue
            raise RuntimeError(f"HTTP {error.code}: {text}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < retries:
                time.sleep(min(30.0, 2**attempt))
                continue
            raise


def block_timestamp(rpc_url, block_number):
    block = rpc_call(rpc_url, "eth_getBlockByNumber", [hex(block_number), False])
    return int(block["timestamp"], 16)


def find_block_for_timestamp(rpc_url, target, low_block, high_block):
    low_ts = block_timestamp(rpc_url, low_block)
    high_ts = block_timestamp(rpc_url, high_block)
    if target <= low_ts:
        return low_block
    if target >= high_ts:
        return high_block
    while high_block - low_block > 1:
        fraction = (target - low_ts) / max(1, high_ts - low_ts)
        guess = max(
            low_block + 1,
            min(high_block - 1, low_block + int(fraction * (high_block - low_block))),
        )
        guess_ts = block_timestamp(rpc_url, guess)
        if guess_ts < target:
            low_block, low_ts = guess, guess_ts
        else:
            high_block, high_ts = guess, guess_ts
    return high_block


def get_logs(rpc_url, address, low_block, high_block):
    try:
        return rpc_call(
            rpc_url,
            "eth_getLogs",
            [{
                "fromBlock": hex(low_block),
                "toBlock": hex(high_block),
                "address": address,
                "topics": [ORDER_FILLED_TOPIC],
            }],
        )
    except ResponseTooLarge:
        if low_block == high_block:
            raise
        midpoint = (low_block + high_block) // 2
        return (
            get_logs(rpc_url, address, low_block, midpoint)
            + get_logs(rpc_url, address, midpoint + 1, high_block)
        )


def decode_order_filled(log):
    data = log["data"][2:]
    if len(data) < 64 * 5:
        return None

    def uint(index):
        return int(data[index * 64 : (index + 1) * 64], 16)

    maker_id, taker_id = uint(0), uint(1)
    maker_amount, taker_amount = uint(2), uint(3)
    if maker_id == 0 and taker_id != 0:
        return taker_id, maker_amount / 1_000_000, taker_amount / 1_000_000
    if taker_id == 0 and maker_id != 0:
        return maker_id, taker_amount / 1_000_000, maker_amount / 1_000_000
    return None


def fetch_chunk(rpc_url, cache_dir, address, low_block, high_block):
    path = cache_dir / f"{address.lower()}_{low_block}_{high_block}.json"
    if path.exists():
        return json.loads(path.read_text())

    anchor_ts = block_timestamp(rpc_url, low_block)
    rows = []
    for log in get_logs(rpc_url, address, low_block, high_block):
        decoded = decode_order_filled(log)
        if decoded is None:
            continue
        token_id, usdc, shares = decoded
        block_number = int(log["blockNumber"], 16)
        timestamp = int(anchor_ts + (block_number - low_block) * POLYGON_BLOCK_SECONDS)
        rows.append([str(token_id), usdc, shares, timestamp])
    result = {"anchor_ts": anchor_ts, "rows": rows}
    path.write_text(json.dumps(result))
    time.sleep(RPC_PAUSE_SECONDS)
    return result


def raw_price_bounds(path):
    yes = json.loads(path.read_text())["Yes"]
    timestamps = [floor_hour(parse_datetime(value)) for value in yes]
    if not timestamps:
        raise ValueError(f"No Yes-price observations in {path}")
    return min(timestamps), max(timestamps)


def market_groups(markets, prices_dir, context_hours):
    by_address = defaultdict(list)
    padding = dt.timedelta(hours=context_hours)
    for market in markets:
        market_id = int(market["id"])
        price_start, price_end = raw_price_bounds(prices_dir / f"{market_id}.json")
        window = market["active_windows"][0]
        start = max(price_start, floor_hour(parse_datetime(window["start"])) - padding)
        end = min(price_end, floor_hour(parse_datetime(window["end"])) + padding)
        if end < start:
            raise ValueError(f"No price coverage for selected window of market {market_id}")
        address = NEGRISK_EXCHANGE if market.get("negRisk") else CTF_EXCHANGE
        by_address[address].append({
            "id": market_id,
            "tokens": [str(value) for value in market["clobTokenIds"]],
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
        })
    return by_address


def sweep_exchange(rpc_url, cache_dir, address, groups, workers):
    latest = int(rpc_call(rpc_url, "eth_blockNumber", []), 16)
    low_timestamp = min(group["start"] for group in groups)
    high_timestamp = max(group["end"] for group in groups) + 3600
    low_block = find_block_for_timestamp(rpc_url, low_timestamp, 47_000_000, latest)
    high_block = find_block_for_timestamp(rpc_url, high_timestamp, low_block, latest)
    chunks = [
        (start, min(start + CHUNK_BLOCKS - 1, high_block))
        for start in range(low_block, high_block + 1, CHUNK_BLOCKS)
    ]
    print(f"{address}: blocks {low_block:,}..{high_block:,} ({len(chunks):,} chunks)")

    rows = []
    lock = threading.Lock()
    completed = 0

    def work(bounds):
        nonlocal completed
        chunk = fetch_chunk(rpc_url, cache_dir, address, *bounds)
        with lock:
            rows.extend(chunk["rows"])
            completed += 1
            if completed % 100 == 0 or completed == len(chunks):
                print(f"  {completed:,}/{len(chunks):,} chunks", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(work, chunks))
    return rows


def write_activity(groups, rows, output_dir):
    token_to_group = {
        token: group
        for group in groups
        for token in group["tokens"]
    }
    hourly = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
    for token, usdc, shares, timestamp in rows:
        group = token_to_group.get(str(token))
        if group is None or timestamp < group["start"] or timestamp > group["end"] + 3599:
            continue
        hour = (timestamp // 3600) * 3600
        cell = hourly[group["id"]][hour]
        cell[0] += float(usdc)
        cell[1] += 1
        cell[2] += float(shares)

    for group in groups:
        length = (group["end"] - group["start"]) // 3600 + 1
        volume = [0.0] * length
        trades = [0] * length
        shares = [0.0] * length
        for hour, (usdc, count, share_count) in hourly[group["id"]].items():
            index = (hour - group["start"]) // 3600
            if 0 <= index < length:
                volume[index] = round(usdc, 2)
                trades[index] = count
                shares[index] = round(share_count, 4)
        payload = {
            "t0": dt.datetime.fromtimestamp(group["start"], dt.timezone.utc).isoformat(),
            "vol": volume,
            "trades": trades,
            "shares": shares,
        }
        (output_dir / f"{group['id']}.json").write_text(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--raw-prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-hours", type=int, default=CONTEXT_HOURS)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    rpc_url = os.environ.get("POLYGON_RPC_URL")
    if not rpc_url:
        raise SystemExit("POLYGON_RPC_URL is not set")

    markets = json.loads(args.markets.read_text())
    groups_by_address = market_groups(markets, args.raw_prices, args.context_hours)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    for address, groups in groups_by_address.items():
        rows = sweep_exchange(rpc_url, args.cache, address, groups, args.workers)
        write_activity(groups, rows, args.output)
    print(f"Wrote {len(markets)} hourly activity files to {args.output}")


if __name__ == "__main__":
    main()
