"""Step 2: Generate subquestions for each ultimate question × active window.

Reads ultimate questions with embedded news articles (from step 1) and
generates subquestions for each active window.

Usage:
    python scripts/step2_generate.py
    python scripts/step2_generate.py --model gpt-5.4

Reads:  data/ultimates_w_news.json   (set via INPUT_FILE in config.py)
Writes: outputs/<model>/subquestions.json
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from config import (
    DEFAULT_GENERATION_PROVIDER, DEFAULT_GENERATION_MODEL,
    NUM_SUBQUESTIONS, OUTPUT_BASE, OUTPUT_DIR, DATA_DIR,
    SUBQUESTIONS_FILE, INPUT_FILE,
)
from src import llm as llm_module
from src.llm import call_llm, model_path, parse_json, resolve_provider_model
from src.prompts import load_question_gen_sys_prompt, load_generation_prompt

QUESTION_GEN_PROMPT = load_question_gen_sys_prompt()

# JSON schema for structured output (used by vLLM, Gemini, and OpenRouter; ignored by others).
SUBQUESTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "background_information": {"type": "string"},
            "resolution_criteria": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "question", "background_information", "resolution_criteria",
            "start_date", "end_date", "rationale",
        ],
    },
}


def generate_subquestions(
    ultimate_question: str,
    description: str,
    start_date: str,
    end_date: str,
    period_end: str,
    n: int = NUM_SUBQUESTIONS,
    model: str = DEFAULT_GENERATION_MODEL,
    provider: str = DEFAULT_GENERATION_PROVIDER,
    articles: list[dict] | None = None,
) -> list[dict]:
    """Generate binary forecasting subquestions for an ultimate question."""
    prompt = load_generation_prompt(
        ultimate_question, description, start_date, end_date, period_end, n, articles=articles,
    )
    text = call_llm(
        prompt, system=QUESTION_GEN_PROMPT, model=model,
        provider=provider,
        web_search=False, json_schema=SUBQUESTION_SCHEMA,
    )
    parsed = parse_json(text)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    if "subquestions" in parsed:
        return parsed["subquestions"]
    if "question" in parsed:
        return [parsed]  # single subquestion returned as a bare dict
    return []


def to_utc_ts(val):
    """Convert a value to a UTC Timestamp, handling ms-epoch ints and strings."""
    if isinstance(val, (int, float)) and val > 1e10:
        return pd.Timestamp(val, unit="ms", tz="UTC")
    return pd.to_datetime(val, utc=True)


def is_valid_subquestion(sub: dict, window_end) -> bool:
    """Check that the subquestion end_date falls on or before the window end date."""
    sub_end = sub.get("end_date", "")
    if not sub_end or not window_end:
        return True
    try:
        return pd.to_datetime(sub_end, utc=True).date() <= to_utc_ts(window_end).date()
    except Exception:
        return True


def _save(df, out_path):
    """Save DataFrame as JSON array of records (same format as input)."""
    df.to_json(out_path, orient="records", indent=2)


def run(generation_model: str = DEFAULT_GENERATION_MODEL,
        provider: str = DEFAULT_GENERATION_PROVIDER,
        n_subs: int = NUM_SUBQUESTIONS, output_dir: Path = None):
    provider, generation_model = resolve_provider_model(generation_model, provider)
    # Load filtered data with embedded articles
    input_path = DATA_DIR / INPUT_FILE
    df = pd.read_json(input_path)
    total_windows = sum(len(row["active_windows"]) for _, row in df.iterrows())
    print(f"Loaded {len(df)} questions, {total_windows} active windows from {input_path}")
    print(f"Provider/model: {provider}/{generation_model}")

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / SUBQUESTIONS_FILE
    llm_module.PARSE_FAILURE_LOG = str(out_dir / "parse_failures.log")

    # Save runtime config
    runtime_config = {
        "generation_model": generation_model,
        "provider": provider,
        "n_subs": n_subs,
        "input_file": str(input_path),
        "output_dir": str(out_dir),
        "web_search": False,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(runtime_config, f, indent=2)

    # Build set of already-generated keys for resuming
    done_keys = set()
    if out_path.exists():
        existing = pd.read_json(out_path)
        for _, row in existing.iterrows():
            for w in row.get("active_windows", []):
                if "subquestions" in w:
                    done_keys.add(f"{row['question']}||period_{w['period_idx']}")
        print(f"Resuming: {len(done_keys)} question-period pairs already generated, skipping those")
        # Use existing data as base (preserves prior subquestions)
        df = existing

    first = len(done_keys) == 0
    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        description = row.get("description", "")
        ph_end = row.get("ph_end", "")
        end_date_str = to_utc_ts(ph_end).isoformat() if ph_end else ""

        for wi, window in enumerate(row["active_windows"]):
            period_idx = window["period_idx"]
            period_start = window["start"]
            period_end = window["end"]
            articles = window.get("as_dicts", [])
            key = f"{question}||period_{period_idx}"

            if key in done_keys:
                continue

            start_date = pd.to_datetime(period_start).strftime("%Y-%m-%d")
            # Subquestion end_date = min(today + 30 days, ultimate_end - 3 days)
            today_plus_30 = pd.to_datetime(period_start) + pd.Timedelta(days=30)
            ultimate_end_minus_3 = to_utc_ts(ph_end) - pd.Timedelta(days=3)
            period_end_dt = min(today_plus_30, ultimate_end_minus_3)
            period_end_str = period_end_dt.strftime("%Y-%m-%d")

            print(f"\nGenerating for: {question[:60]}... (period {period_idx})")
            subs = generate_subquestions(
                question, description,
                end_date=end_date_str, start_date=start_date,
                period_end=period_end_str,
                n=n_subs, model=generation_model,
                provider=provider,
                articles=articles,
            )

            if first:
                prompt_preview = load_generation_prompt(
                    question, description, start_date, end_date_str, period_end_str, n_subs, articles=articles,
                )
                print(f"\n{'='*60}\nSystem prompt:\n{'='*60}\n{QUESTION_GEN_PROMPT}\n{'='*60}")
                print(f"\nUser prompt:\n{'='*60}\n{prompt_preview}\n{'='*60}\n")
                first = False

            for s in subs:
                s["valid_period"] = is_valid_subquestion(s, period_end)

            # Embed subquestions into the window
            df.at[idx, "active_windows"][wi]["subquestions"] = subs
            done_keys.add(key)

            for s in subs:
                flag = "" if s["valid_period"] else " [INVALID DATE]"
                print(f"  - {s['question']}{flag}")
            count += 1

            if count % 5 == 0:
                _save(df, out_path)
                print(f"    [CHECKPOINT] Saved to {out_path}")

    total = sum(
        len(s) for _, r in df.iterrows()
        for w in r["active_windows"] for s in [w.get("subquestions", [])]
    )
    print(f"\nDone: {total} subquestions for {len(df)} questions")

    _save(df, out_path)
    print(f"Saved to {out_path}")


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
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--n-subs", type=int, default=NUM_SUBQUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output dir. Defaults to outputs/<model>.")
    args = parser.parse_args()

    provider = args.provider
    if provider is None and args.model == DEFAULT_GENERATION_MODEL:
        provider = DEFAULT_GENERATION_PROVIDER
    provider, model = resolve_provider_model(args.model, provider)

    out_dir = args.output_dir or (OUTPUT_BASE / model_path(provider, model))
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        generation_model=model,
        provider=provider,
        n_subs=args.n_subs,
        output_dir=out_dir,
    )
