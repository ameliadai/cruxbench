"""Forecast P(U) at the window start without crux information (baseline).

The forecaster sees the ultimate question, its description, and the same news
articles given to the generators.

Usage:
    python scripts/forecast_baseline.py
    python scripts/forecast_baseline.py --model gemini-3-flash-preview

Reads:  data/ultimates_w_news.json
Writes: outputs_forecast/<forecaster>/baseline_pu.json
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from config import (
    DATA_DIR,
    DEFAULT_FORECAST_MODEL,
    DEFAULT_FORECAST_PROVIDER,
    INPUT_FILE,
    PROJECT_ROOT,
)
from src import llm as llm_module
from src.llm import call_llm, model_dir_name, parse_json, resolve_provider_model
from src.prompts import load_forecast_prompt

DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "outputs_forecast"


def to_utc_ts(val):
    if isinstance(val, (int, float)) and val > 1e10:
        return pd.Timestamp(val, unit="ms", tz="UTC")
    return pd.to_datetime(val, utc=True)


def run(model: str, provider: str, output_dir: Path, limit: int | None = None):
    provider, model = resolve_provider_model(model, provider)
    input_path = DATA_DIR / INPUT_FILE
    df = pd.read_json(input_path)
    print(f"Loaded {len(df)} questions from {input_path}")
    print(f"Forecaster: {provider}/{model}")

    forecaster_dir = output_dir / model_dir_name(provider, model)
    forecaster_dir.mkdir(parents=True, exist_ok=True)
    out_path = forecaster_dir / "baseline_pu.json"
    llm_module.PARSE_FAILURE_LOG = str(forecaster_dir / "parse_failures.log")

    done_keys = set()
    results = []
    if out_path.exists():
        existing = pd.read_json(out_path)
        results = existing.to_dict(orient="records")
        done_keys = {(r["ultimate_question"], r["period_idx"]) for r in results}
        print(f"Resuming: {len(results)} already forecasted")

    first = len(results) == 0
    n_done_this_run = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        description = row.get("description", "") or ""
        market_id = row.get("id")
        ph_end = row.get("ph_end", "")
        end_date_str = to_utc_ts(ph_end).isoformat() if ph_end else ""

        for window in row["active_windows"]:
            period_idx = window["period_idx"]
            if (question, period_idx) in done_keys:
                continue

            start_date = pd.to_datetime(window["start"]).strftime("%Y-%m-%d")
            articles = window.get("as_dicts", [])

            prompt = load_forecast_prompt(
                question=question,
                description=description,
                start_date=start_date,
                end_date=end_date_str,
                articles=articles,
                subquestions=None,
            )

            if first:
                print(f"\n{'=' * 70}\nSANITY CHECK — first prompt:\n{'=' * 70}\n{prompt}\n{'=' * 70}\n")
                first = False

            text = call_llm(
                prompt,
                system="",
                model=model,
                provider=provider,
                web_search=False,
            )
            parsed = parse_json(text)
            if parsed is None:
                print(f"  [SKIP no JSON] {question[:60]}... period {period_idx}")
                continue
            if isinstance(parsed, list):
                if len(parsed) == 1 and isinstance(parsed[0], dict):
                    parsed = parsed[0]
                else:
                    print(f"  [SKIP unexpected list] {question[:60]}... period {period_idx}")
                    continue

            p_u = parsed.get("p_u")
            if p_u is None and "probability" in parsed:
                p_u = parsed["probability"]

            rationale = (parsed.get("rationale", "") or "").replace("\n", " ")
            print(f"  → p_u={p_u}  | {question[:60]}...")
            print(f"      rationale: {rationale[:200]}")

            results.append({
                "ultimate_question": question,
                "market_id": market_id,
                "period_idx": period_idx,
                "window_start": window["start"],
                "window_end": window["end"],
                "p_u": p_u,
                "rationale": parsed.get("rationale", ""),
                "forecaster_provider": provider,
                "forecaster_model": model,
            })
            done_keys.add((question, period_idx))
            n_done_this_run += 1

            if n_done_this_run % 10 == 0:
                pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
                print(f"    [CHECKPOINT] {len(results)} → {out_path}")

            if limit is not None and n_done_this_run >= limit:
                pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
                print(f"\nLimit {limit} reached. Saved {len(results)} → {out_path}")
                return

    pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
    print(f"\nDone: {len(results)} baseline forecasts → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "LLM provider. Defaults to the paper provider for the default model; "
            "otherwise inferred from a provider-prefixed model name."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_FORECAST_MODEL,
                        help="Forecaster model (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_BASE,
                        help="Output dir (default: outputs_forecast)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N new forecasts (smoke-test).")
    args = parser.parse_args()

    provider = args.provider
    if provider is None and args.model == DEFAULT_FORECAST_MODEL:
        provider = DEFAULT_FORECAST_PROVIDER
    run(
        model=args.model,
        provider=provider,
        output_dir=args.output_dir,
        limit=args.limit,
    )
