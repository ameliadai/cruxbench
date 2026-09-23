"""Forecast P(U | crux) at the window start, given one generator's resolved crux.

Same setup as forecast_baseline.py, plus the crux and its realized resolution.
Unresolvable cruxes and cruxes resolving after the market's last price are
skipped; the notebook uses the baseline forecast for those windows.

Usage:
    python scripts/forecast_with_crux.py --generator gpt-5.4
    python scripts/forecast_with_crux.py --generator gpt-5.4 --model gemini-3-flash-preview

Reads:  data/ultimates_w_news.json
        outputs/<generator>/resolved.json
Writes: outputs_forecast/<forecaster>/subq_pu__<generator>.json
"""

import argparse
import json
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
    OUTPUT_BASE,
    PROJECT_ROOT,
    RESOLVED_FILE,
)
from src import llm as llm_module
from src.llm import call_llm, model_dir_name, parse_json, resolve_provider_model
from src.prompts import load_forecast_prompt

DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "outputs_forecast"


def to_utc_ts(val):
    if isinstance(val, (int, float)) and val > 1e10:
        return pd.Timestamp(val, unit="ms", tz="UTC")
    return pd.to_datetime(val, utc=True)


def load_generator_cruxes(generator_dir: Path) -> dict:
    """Index one generator's cruxes by (ultimate_question, period_idx).

    Uses the first crux in each window.
    """
    path = generator_dir / RESOLVED_FILE
    with open(path) as f:
        data = json.load(f)

    index = {}
    for row in data:
        uq = row["question"]
        for w in row["active_windows"]:
            subs = w.get("subquestions", [])
            if not subs:
                continue
            index[(uq, w["period_idx"])] = subs[0]
    return index


def run(generator: str, model: str, provider: str,
        output_dir: Path, limit: int | None = None):
    provider, model = resolve_provider_model(model, provider)
    input_path = DATA_DIR / INPUT_FILE
    df = pd.read_json(input_path)
    print(f"Loaded {len(df)} questions from {input_path}")

    generator_dir = OUTPUT_BASE / generator
    cruxes = load_generator_cruxes(generator_dir)
    print(f"Loaded {len(cruxes)} cruxes from {generator_dir}")
    print(f"Forecaster: {provider}/{model}  |  Generator: {generator}")

    forecaster_dir = output_dir / model_dir_name(provider, model)
    forecaster_dir.mkdir(parents=True, exist_ok=True)
    out_path = forecaster_dir / f"subq_pu__{generator.replace('/', '_')}.json"
    llm_module.PARSE_FAILURE_LOG = str(forecaster_dir / "parse_failures.log")

    done_keys = set()
    results = []
    if out_path.exists():
        existing = pd.read_json(out_path)
        results = existing.to_dict(orient="records")
        done_keys = {(r["ultimate_question"], r["period_idx"]) for r in results}
        print(f"Resuming: {len(results)} already forecasted")

    first = len(results) == 0
    n_done = 0
    n_skip_no_crux = 0
    n_skip_unresolvable = 0
    n_skip_late = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        description = row.get("description", "") or ""
        market_id = row.get("id")
        ph_end = row.get("ph_end", "")
        u_end_ts = to_utc_ts(ph_end) if ph_end else None
        end_date_str = u_end_ts.isoformat() if u_end_ts is not None else ""

        for window in row["active_windows"]:
            period_idx = window["period_idx"]
            if (question, period_idx) in done_keys:
                continue

            crux = cruxes.get((question, period_idx))
            if crux is None:
                n_skip_no_crux += 1
                continue
            if not crux.get("resolvable"):
                n_skip_unresolvable += 1
                continue

            crux_rdt = crux.get("resolution_datetime", "") or ""
            if not crux_rdt:
                n_skip_unresolvable += 1
                continue
            try:
                crux_rdt_ts = pd.to_datetime(crux_rdt, utc=True)
            except Exception:
                n_skip_unresolvable += 1
                continue
            if u_end_ts is not None and crux_rdt_ts > u_end_ts:
                n_skip_late += 1
                continue

            start_date = pd.to_datetime(window["start"]).strftime("%Y-%m-%d")
            articles = window.get("as_dicts", [])

            prompt = load_forecast_prompt(
                question=question,
                description=description,
                start_date=start_date,
                end_date=end_date_str,
                articles=articles,
                subquestions=[crux],
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
                "generator_model": generator,
                "subquestion": crux.get("question", ""),
                "subq_resolution": crux.get("resolution", ""),
                "subq_resolution_datetime": crux_rdt,
                "subq_end_date": crux.get("end_date", ""),
            })
            done_keys.add((question, period_idx))
            n_done += 1

            if n_done % 10 == 0:
                pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
                print(f"    [CHECKPOINT] {len(results)} → {out_path}")

            if limit is not None and n_done >= limit:
                pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
                print(f"\nLimit {limit} reached. Saved {len(results)} → {out_path}")
                _print_skip_summary(n_skip_no_crux, n_skip_unresolvable, n_skip_late)
                return

    pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
    print(f"\nDone: {len(results)} treatment forecasts → {out_path}")
    _print_skip_summary(n_skip_no_crux, n_skip_unresolvable, n_skip_late)


def _print_skip_summary(no_crux, unresolvable, late):
    print(f"  Skipped — no crux for window:        {no_crux}")
    print(f"  Skipped — crux unresolvable:         {unresolvable}")
    print(f"  Skipped — crux resolves after U end: {late}")
    print("  (analysis notebook imputes baseline P(U) for these.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True,
                        help="Generator model folder under outputs/ (e.g., gpt-5.4)")
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
        generator=args.generator,
        model=args.model,
        provider=provider,
        output_dir=args.output_dir,
        limit=args.limit,
    )
