"""Build volume-trusted hourly prices from raw prices and trading activity.

With V_t the USDC volume within +/-12 hours of hour t:

    alpha_t   = min(1, V_t / 1000)
    trusted_t = alpha_t * raw_t + (1 - alpha_t) * trusted_(t-1)

Usage:
    python data_collection/build_trusted_prices.py

Reads:  data/ultimates.json, data/hourly_prices/, data/hourly_activity/
Writes: data/hourly_prices_trusted/<market_id>.json
"""

import argparse
import datetime as dt
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
UTC = dt.timezone.utc

DEFAULT_MARKETS = REPO / "data/ultimates.json"
DEFAULT_RAW_PRICES = REPO / "data/hourly_prices"
DEFAULT_ACTIVITY = REPO / "data/hourly_activity"
DEFAULT_OUTPUT = REPO / "data/hourly_prices_trusted"

VOLUME_THRESHOLD_USDC = 1000.0
VOLUME_HALF_WINDOW_HOURS = 12


def parse_datetime(value):
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def floor_hour(value):
    return value.replace(minute=0, second=0, microsecond=0)


def load_uniform_yes(path):
    raw = json.loads(path.read_text())["Yes"]
    by_hour = {}
    for timestamp, value in sorted(raw.items(), key=lambda item: parse_datetime(item[0])):
        by_hour[floor_hour(parse_datetime(timestamp))] = float(value)

    start = min(by_hour)
    end = max(by_hour)
    length = int((end - start).total_seconds() // 3600) + 1
    values = []
    last = None
    for index in range(length):
        hour = start + dt.timedelta(hours=index)
        last = by_hour.get(hour, last)
        values.append(last)
    return start, values


def activity_aligned_prices(raw_start, raw_prices, activity_start, length):
    offset = int((activity_start - raw_start).total_seconds() // 3600)
    if offset < 0 or offset + length > len(raw_prices):
        raise ValueError("Hourly activity lies outside the raw price history")
    # The public benchmark price grid is represented to three decimals.
    return [round(value * 1000) / 1000 for value in raw_prices[offset : offset + length]]


def rebuild_ema(prices, volume, threshold, half_window):
    cumulative = [0.0]
    for value in volume:
        cumulative.append(cumulative[-1] + float(value or 0))

    trusted = prices[0]
    rebuilt = []
    for index, raw_price in enumerate(prices):
        lower = max(0, index - half_window)
        upper = min(len(prices) - 1, index + half_window)
        nearby_volume = cumulative[upper + 1] - cumulative[lower]
        alpha = min(1.0, nearby_volume / threshold)
        trusted = alpha * raw_price + (1 - alpha) * trusted
        rebuilt.append(trusted)
    return rebuilt


def write_history(path, start, yes_prices):
    yes = {}
    no = {}
    for index, value in enumerate(yes_prices):
        timestamp = (start + dt.timedelta(hours=index)).isoformat()
        yes[timestamp] = value
        no[timestamp] = 1.0 - value
    path.write_text(json.dumps({"Yes": yes, "No": no}))


def run(markets_path, raw_dir, activity_dir, output_dir, threshold, half_window):
    markets = json.loads(markets_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    for market in markets:
        market_id = int(market["id"])
        activity = json.loads((activity_dir / f"{market_id}.json").read_text())
        activity_start = floor_hour(parse_datetime(activity["t0"]))
        volume = activity["vol"]

        raw_start, raw_prices = load_uniform_yes(raw_dir / f"{market_id}.json")
        prices = activity_aligned_prices(
            raw_start, raw_prices, activity_start, len(volume)
        )
        trusted = rebuild_ema(prices, volume, threshold, half_window)
        write_history(output_dir / f"{market_id}.json", activity_start, trusted)

    print(f"Wrote {len(markets)} trusted price histories to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--raw-prices", type=Path, default=DEFAULT_RAW_PRICES)
    parser.add_argument("--activity", type=Path, default=DEFAULT_ACTIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--volume-threshold", type=float, default=VOLUME_THRESHOLD_USDC)
    parser.add_argument("--half-window", type=int, default=VOLUME_HALF_WINDOW_HOURS)
    args = parser.parse_args()
    run(
        args.markets,
        args.raw_prices,
        args.activity,
        args.output,
        args.volume_threshold,
        args.half_window,
    )


if __name__ == "__main__":
    main()
