"""Filter the raw Polymarket dump into the evaluation set.

Keeps resolved Yes/No markets and picks one active 30-day window per market.
Run twice: first without the window-volume filter, to get the markets whose
hourly prices and activity must be fetched; then with it, for the final set.

Usage:
    python data_collection/filter.py --input data/polymarket_0325.json --output data/ultimates_pre_volume.json --skip-window-volume-filter
    python data_collection/filter.py --input data/polymarket_0325.json --output data/ultimates.json
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


# ---------- Filter parameters ---------------------------------------------------

MIN_END_DATE          = "2025-09-01"
MIN_HORIZON_DAYS      = 60
MIN_PRICE_HIST_LENGTH = 30
MIN_WINDOW_DAYS       = 7
MIN_WINDOW_VOLUME_USDC = 1000.0

PERIOD_DAYS           = 30
CERTAINTY_THRESHOLD   = 0.1
SPIKE_TAIL_DAYS       = 3
RANDOM_SEED           = 42

EXCLUDED_TAGS  = {"sports", "crypto", "weather"}
EXCLUDED_WORDS = ["tweet", "say"]

SPORTS_KW = [
    'NFL', 'NBA', 'MLB', 'WNBA', 'UFC', 'NHL', 'AFC', 'NFC', 'Super Bowl',
    'World Series', 'Playoff', 'Championship', 'Heisman', 'Rookie of the Year',
    'Coach of the Year', 'MVP', 'Cy Young', 'Fantasy', 'touchdown', 'rushing',
    'interception', 'Premier League', 'Champions League', 'UEFA', 'FIFA',
    'Serie A', 'LaLiga', 'Carabao Cup',
    'AL Central', 'AL East', 'AL West', 'NL Central', 'NL East', 'NL West',
    'SEC ', 'ACC ', 'Big Ten', 'College Football',
    'PDC', 'PGL', 'LEC', 'StarLadder', 'Major 2025', 'Major 2026',
    'Survivor', 'Big Brother', 'Beast Games',
    'searched on Google', 'searched person on Google', 'searched athlete on Google',
    'Year in Search', 'FIDE World Cup', 'searched',
]
CRYPTO_KW = [
    'Bitcoin', 'BTC', 'Ethereum', 'ETH', 'Solana', 'SOL', 'XRP', 'Dogecoin',
    'Chainlink', 'Hyperliquid', 'BNB', 'FDV', 'airdrop', 'token', 'stablecoin',
    'DeFi', 'Uniswap', 'Pump.fun', 'Monad', 'MegaETH',
    'Litecoin', 'LTC', 'Synthetix', 'all time high', 'all-time high',
]
AWARDS_KW = [
    'Golden Globe', 'Grammy', 'Oscar', 'Academy Award', 'BAFTA', 'Emmy',
    'SAG Award', 'PGA Award', 'Sanremo', 'Eurovision', 'Intervision',
    'win Best ', 'nominated for Best ', 'Year in Search',
]
DESC_PHRASE_BLOCKLIST = [
    'any of the listed',
    'Chatbot Arena',
    "Humanity's Last Exam",
    'FrontierMath Benchmark',
    'AI World Cup',
]

DEFAULT_ACTIVITY_DIR = Path(__file__).resolve().parent.parent / "data/hourly_activity"


# ---------- Price-history helpers ----------------------------------------------

def parse_ph(ph):
    if isinstance(ph, str):
        try:
            return json.loads(ph)
        except Exception:
            return {}
    return ph if isinstance(ph, dict) else {}


def get_ph_date_range(ph):
    ph = parse_ph(ph)
    if not ph:
        return pd.NaT, pd.NaT
    first = next(iter(ph.values()), {})
    if isinstance(first, str):
        try:
            first = json.loads(first)
        except Exception:
            return pd.NaT, pd.NaT
    if not first:
        return pd.NaT, pd.NaT
    ts = pd.to_datetime(list(first.keys()), utc=True)
    return ts.min(), ts.max()


def get_price_history_length(ph):
    ph = parse_ph(ph)
    if not ph:
        return 0
    yes = ph.get("Yes", {})
    if isinstance(yes, str):
        try:
            yes = json.loads(yes)
        except Exception:
            return 0
    return len(yes)


def get_daily_prices(price_history):
    ph = parse_ph(price_history)
    yes = ph.get("Yes", {})
    if isinstance(yes, str):
        yes = json.loads(yes)
    s = pd.Series(yes, dtype=float).sort_index()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("D").last().ffill()


# ---------- Trading-activity helpers ------------------------------------------

def get_window_volume_usdc(market_id, window, activity_dir):
    """Observed USDC volume in the selected window, including its final day."""
    path = activity_dir / f"{int(market_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing hourly activity for market {market_id}: {path}")

    activity = json.loads(path.read_text())
    t0 = pd.to_datetime(activity["t0"], utc=True).floor("1h")
    start = pd.to_datetime(window["start"], utc=True).floor("1h")
    end = pd.to_datetime(window["end"], utc=True).floor("1h") + pd.Timedelta(hours=23)
    volume = activity.get("vol", [])

    first = max(0, int((start - t0) / pd.Timedelta(hours=1)))
    last = min(len(volume) - 1, int((end - t0) / pd.Timedelta(hours=1)))
    if last < first:
        return 0.0
    return float(sum(float(value or 0) for value in volume[first:last + 1]))


# ---------- Window detection ---------------------------------------------------

def count_periods(start_dt, end_dt):
    days = (end_dt - start_dt).days
    if days <= 0:
        return 1
    return -(-days // PERIOD_DAYS)


def get_period_range(start_dt, end_dt, idx):
    period_start = start_dt + pd.Timedelta(days=PERIOD_DAYS * idx)
    period_end   = start_dt + pd.Timedelta(days=PERIOD_DAYS * (idx + 1))
    max_end      = end_dt - pd.Timedelta(days=SPIKE_TAIL_DAYS)
    if period_end > max_end:
        period_end = max_end
    return period_start, period_end


def get_active_periods(start_dt, end_dt, price_history):
    """Return indices of 30-day periods where prices are not pinned near 0 or 1."""
    daily = get_daily_prices(price_history)
    n_periods = count_periods(start_dt, end_dt)
    active = []
    for m in range(n_periods):
        win_start = start_dt + pd.Timedelta(days=PERIOD_DAYS * m)
        win_end   = start_dt + pd.Timedelta(days=PERIOD_DAYS * (m + 1))
        if win_end > end_dt:
            win_end = end_dt
        prices = daily[(daily.index >= win_start) & (daily.index <= win_end)]
        if prices.empty:
            continue
        if (prices <= CERTAINTY_THRESHOLD).all() or (prices >= 1 - CERTAINTY_THRESHOLD).all():
            continue
        active.append(m)
    return active


# ---------- Question normalisation --------------------------------------------

_MONTHS = r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'

def normalize_question(q):
    q = q.lower()
    q = re.sub(_MONTHS, '', q)
    q = re.sub(r'\b\d{4}\b', '', q)
    q = re.sub(r'\b\d+\b', '', q)
    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def matches_any(text, keywords):
    return any(kw.lower() in text.lower() for kw in keywords)


# ---------- Pipeline -----------------------------------------------------------

def run(input_path: Path, output_path: Path, activity_dir: Path,
        apply_window_volume_filter: bool = True):
    df = pd.read_json(input_path, lines=True)
    funnel = [("raw markets", len(df))]

    ph_ranges = df["price_history"].apply(
        lambda ph: pd.Series(get_ph_date_range(ph), index=["ph_start", "ph_end"])
    )
    ph_ranges["ph_start"] = pd.to_datetime(ph_ranges["ph_start"], utc=True)
    ph_ranges["ph_end"]   = pd.to_datetime(ph_ranges["ph_end"],   utc=True)
    df = pd.concat([df, ph_ranges], axis=1)

    df = df[df["ph_end"] >= MIN_END_DATE]
    funnel.append((f"ph_end >= {MIN_END_DATE}", len(df)))

    df = df[df["resolution"].notna()]
    funnel.append(("has resolution", len(df)))

    df = df.drop_duplicates(subset=["question"])
    funnel.append(("dedup by question", len(df)))

    df = df[df["umaResolutionStatus"] == "resolved"]
    funnel.append(("umaResolutionStatus == resolved", len(df)))

    df = df[["Yes" in o for o in df["outcomes"]]]
    funnel.append(("'Yes' in outcomes", len(df)))

    df = df[~df["tags"].apply(lambda x: any(t.lower() in EXCLUDED_TAGS for t in x))]
    funnel.append(("tag exclusion", len(df)))

    for w in EXCLUDED_WORDS:
        df = df[[w.lower() not in q.lower() for q in df["question"]]]
    funnel.append((f"exclude words {EXCLUDED_WORDS}", len(df)))

    df["_end"]     = pd.to_datetime(df["endDate"],   utc=True, errors="coerce")
    df["_start"]   = pd.to_datetime(df["startDate"], utc=True, errors="coerce")
    df["_horizon"] = (df["_end"] - df["_start"]).dt.days
    df = df[df["_horizon"] >= MIN_HORIZON_DAYS].drop(columns=["_end", "_start", "_horizon"])
    funnel.append((f"horizon >= {MIN_HORIZON_DAYS}d", len(df)))

    df = df[df["question"].apply(lambda q: not matches_any(q, SPORTS_KW + CRYPTO_KW + AWARDS_KW))]
    funnel.append(("sports/crypto/awards keyword filter", len(df)))

    df["_phlen"] = df["price_history"].apply(get_price_history_length)
    df = df[df["_phlen"] >= MIN_PRICE_HIST_LENGTH].drop(columns=["_phlen"])
    funnel.append((f"price-history length >= {MIN_PRICE_HIST_LENGTH}", len(df)))

    df["_event_id"] = df["events"].apply(
        lambda x: x[0]["id"] if isinstance(x, list) and x else None
    )
    has_event = (df[df["_event_id"].notna()]
                   .groupby("_event_id", group_keys=False)
                   .sample(n=1, random_state=RANDOM_SEED))
    no_event = df[df["_event_id"].isna()]
    df = (pd.concat([has_event, no_event])
            .sort_values("volumeNum", ascending=False)
            .drop(columns=["_event_id"]))
    funnel.append(("event-id dedup", len(df)))

    df["_normq"] = df["question"].apply(normalize_question)
    df = df.drop_duplicates(subset="_normq").drop(columns=["_normq"]).reset_index(drop=True)
    funnel.append(("normalized-question dedup", len(df)))

    def desc_match(row):
        text = (row.get("description") or "") + " " + (row.get("question") or "")
        return any(p in text for p in DESC_PHRASE_BLOCKLIST)
    df = df[~df.apply(desc_match, axis=1)].reset_index(drop=True)
    funnel.append(("description-phrase blocklist", len(df)))

    # --- Window detection ---
    min_window_start = pd.Timestamp(MIN_END_DATE, tz="UTC")
    df = df[df["ph_end"] >= min_window_start]
    df["_anchor"] = df["ph_start"].clip(lower=min_window_start)

    df["active_periods"] = df.apply(
        lambda r: get_active_periods(r["_anchor"], r["ph_end"], r["price_history"]),
        axis=1,
    )
    df = df[df["active_periods"].apply(len) > 0]
    funnel.append(("has at least one active period", len(df)))

    df["active_windows"] = df.apply(
        lambda r: [
            {
                "period_idx": m,
                "start": get_period_range(r["_anchor"], r["ph_end"], m)[0].isoformat(),
                "end":   get_period_range(r["_anchor"], r["ph_end"], m)[1].isoformat(),
            }
            for m in r["active_periods"]
            if (get_period_range(r["_anchor"], r["ph_end"], m)[1]
                - get_period_range(r["_anchor"], r["ph_end"], m)[0]).days >= MIN_WINDOW_DAYS
        ],
        axis=1,
    )
    df = df[df["active_windows"].apply(len) > 0].drop(columns=["_anchor"])
    funnel.append((f"active window >= {MIN_WINDOW_DAYS}d", len(df)))

    # --- One random active window per question ---
    df["volumeNum"] = pd.to_numeric(df["volumeNum"], errors="coerce")
    df = df.sort_values("volumeNum", ascending=False).reset_index(drop=True)
    rng = random.Random(RANDOM_SEED)
    df["active_windows"] = df["active_windows"].apply(lambda ws: [rng.choice(ws)])
    df["active_periods"] = df["active_windows"].apply(lambda ws: [ws[0]["period_idx"]])
    df["num_active"]     = 1

    # --- Selected-window trading-volume threshold ---
    if apply_window_volume_filter:
        # Keep the historical field name for compatibility with the released
        # JSON; Polymarket reports the underlying volume in USDC.
        df["window_volume_usd"] = df.apply(
            lambda row: get_window_volume_usdc(
                row["id"], row["active_windows"][0], activity_dir
            ),
            axis=1,
        )
        df = df[df["window_volume_usd"] > MIN_WINDOW_VOLUME_USDC].reset_index(drop=True)
        funnel.append((f"selected-window USDC volume > {MIN_WINDOW_VOLUME_USDC:,.0f}", len(df)))
    else:
        funnel.append(("selected-window USDC volume filter skipped", len(df)))

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)

    print("Funnel:")
    for name, n in funnel:
        print(f"  {name:<40s} {n:>7,}")
    print(f"\nSaved {len(df)} questions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--activity",
        type=Path,
        default=DEFAULT_ACTIVITY_DIR,
        help="Directory containing one hourly-activity JSON file per market",
    )
    parser.add_argument(
        "--skip-window-volume-filter",
        action="store_true",
        help=(
            "Write the deterministic pre-volume cohort without reading activity files. "
            "Use this once to obtain the markets needed by fetch_hourly_prices.py "
            "and fetch_hourly_activity.py."
        ),
    )
    args = parser.parse_args()
    run(
        args.input,
        args.output,
        args.activity,
        apply_window_volume_filter=not args.skip_window_volume_filter,
    )
