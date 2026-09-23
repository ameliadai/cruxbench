"""Step 3a: Resolve generated subquestions with an LLM using AskNews or web search.

`--output-dir` selects the generator's folder (default: outputs/claude-opus-4-6);
`--model` selects the resolver.

Usage:
    python scripts/step3a_resolve.py --output-dir outputs/gpt-5.4
    python scripts/step3a_resolve.py --source websearch
    python scripts/step3a_resolve.py --no-save-debug

Reads:  outputs/<generator>/subquestions.json
Writes: outputs/<generator>/resolved_raw.json

By default, the full model responses and retrieved AskNews articles are saved
in resolved_raw.json; Step 3b shows the articles to the verifier. The released
resolved_raw.json and resolved.json files omit these logs to keep the release
small.
"""

import json
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from config import (
    DEFAULT_RESOLVE_PROVIDER,
    DEFAULT_RESOLVE_MODEL,
    NUM_SUBQUESTIONS,
    OUTPUT_DIR,
    RAW_RESOLVED_FILE,
    SUBQUESTIONS_FILE,
)
from src.llm import (
    AskNewsRateLimitError,
    call_llm,
    call_llm_with_asknews,
    parse_json,
    resolve_provider_model,
)
from src.prompts import RESOLVE_SYSTEM_PROMPT, RESOLVE_ASKNEWS_SYSTEM_PROMPT, load_resolve_prompt


def _save(df, out_path, save_debug=True):
    if not save_debug:
        # Keep only the parsed resolution, compact evidence, and source URLs.
        for _, row in df.iterrows():
            for window in row.get("active_windows", []):
                for subquestion in window.get("subquestions", []):
                    subquestion.pop("_full_response", None)
                    subquestion.pop("_articles", None)
    df.to_json(out_path, orient="records", indent=2)


def run(model: str = DEFAULT_RESOLVE_MODEL, provider: str = DEFAULT_RESOLVE_PROVIDER,
        output_dir: Path = None, source: str = "asknews", n_subs: int = None,
        save_debug: bool = True):
    provider, model = resolve_provider_model(model, provider)
    use_asknews = source == "asknews"
    if use_asknews:
        sys_prompt = RESOLVE_ASKNEWS_SYSTEM_PROMPT
        print(f"Using AskNews with {provider}/{model} for resolution")
    else:
        sys_prompt = RESOLVE_SYSTEM_PROMPT
        print(f"Using web search with {provider}/{model} for resolution")

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    subs_path = out_dir / SUBQUESTIONS_FILE

    # If --n-subs is not provided, use n_subs from the run's config.json,
    # otherwise NUM_SUBQUESTIONS from config.py.
    if n_subs is None:
        cfg_path = out_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                n_subs = json.load(f).get("n_subs")
    if n_subs is None:
        n_subs = NUM_SUBQUESTIONS
    print(f"Limiting to first {n_subs} subquestion(s) per window")

    df = pd.read_json(subs_path)
    total = sum(
        len(s) for _, r in df.iterrows()
        for w in r["active_windows"] for s in [w.get("subquestions", [])]
    )
    print(f"Loaded {total} subquestions across {len(df)} questions from {subs_path}")

    out_path = out_dir / RAW_RESOLVED_FILE

    # Load existing results to resume
    done_keys = set()
    if out_path.exists():
        existing = pd.read_json(out_path)
        for _, row in existing.iterrows():
            for w in row.get("active_windows", []):
                for s in w.get("subquestions", []):
                    if "resolution" in s:
                        done_keys.add(f"{row['question']}||period_{w['period_idx']}||{s['question']}")
        print(f"Resuming: {len(done_keys)} subquestions already resolved, skipping those")
        df = existing

    first = len(done_keys) == 0
    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]

        for wi, window in enumerate(row["active_windows"]):
            period_idx = window["period_idx"]
            subs = window.get("subquestions", [])
            if not subs:
                continue
            subs = subs[:n_subs]

            valid_subs = [(si, s) for si, s in enumerate(subs) if s.get("valid_period", True)]
            if not valid_subs:
                continue

            print(f"\n{'=' * 80}")
            print(f"Question: {question[:60]}... (period {period_idx})")
            print(f"{'=' * 80}")

            for si, s in valid_subs:
                sub_key = f"{question}||period_{period_idx}||{s['question']}"
                if sub_key in done_keys:
                    continue

                prompt = load_resolve_prompt(
                    s["question"],
                    s.get("resolution_criteria", ""),
                    s.get("start_date", ""),
                    s.get("end_date", ""),
                )

                if first:
                    print(f"\n{'=' * 60}\nSANITY CHECK — system prompt:\n{'=' * 60}\n{sys_prompt}\n{'=' * 60}")
                    print(f"\nSANITY CHECK — user prompt:\n{'=' * 60}\n{prompt}\n{'=' * 60}\n")
                    first = False

                print(f"  Resolving: {s['question'][:80]}...")

                if use_asknews:
                    try:
                        response = call_llm_with_asknews(
                            prompt, system=sys_prompt, model=model,
                            provider=provider, return_debug=save_debug,
                        )
                        if save_debug:
                            text, debug = response
                        else:
                            text, debug = response, None
                    except AskNewsRateLimitError as e:
                        print(f"\n  [FATAL] AskNews rate limit reached: {e}")
                        print("  Saving progress and exiting.")
                        _save(df, out_path, save_debug)
                        return
                else:
                    text = call_llm(
                        prompt,
                        system=sys_prompt,
                        model=model,
                        provider=provider,
                        web_search=True,
                    )
                    debug = None
                if text is None:
                    print("    -> SKIPPED (API error)")
                    continue
                result = parse_json(text)
                if result is None:
                    print("    -> SKIPPED (no JSON parsed)")
                    continue

                # Normalise source_urls: accept both source_url (str) and source_urls (list)
                if "source_url" in result and "source_urls" not in result:
                    result["source_urls"] = [result.pop("source_url")] if result["source_url"] else []
                elif "source_urls" not in result:
                    result["source_urls"] = []

                if save_debug and debug:
                    result["_full_response"] = debug["full_response"]
                    result["_articles"] = debug["articles"]

                # Embed resolution into the subquestion
                df.at[idx, "active_windows"][wi]["subquestions"][si].update(result)
                done_keys.add(sub_key)

                urls = result.get("source_urls") or []
                print(
                    f"    -> resolvable={result.get('resolvable')} | "
                    f"{result.get('resolution')} | "
                    f"datetime={result.get('resolution_datetime', '')} | "
                    f"{str(result.get('evidence', ''))} | "
                    f"sources={len(urls)}"
                )
                count += 1

                if count % 5 == 0:
                    _save(df, out_path, save_debug)
                    print(f"    [CHECKPOINT] Saved to {out_path}")

    # Final save
    _save(df, out_path, save_debug)

    n_total = sum(
        len(w.get("subquestions", []))
        for _, r in df.iterrows() for w in r["active_windows"]
    )
    n_resolvable = sum(
        1 for _, r in df.iterrows()
        for w in r["active_windows"]
        for s in w.get("subquestions", [])
        if s.get("resolvable")
    )
    print(f"\nDone: {n_resolvable}/{n_total} subquestions resolvable across {len(df)} questions")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "LLM provider. Defaults to the paper provider for the default model; "
            "otherwise inferred from a provider-prefixed model name."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_RESOLVE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output dir (must contain subquestions.json)")
    parser.add_argument("--source", choices=["asknews", "websearch"], default="asknews",
                        help="Resolution source: asknews (default, paper setting) or websearch")
    parser.add_argument("--n-subs", type=int, default=None,
                        help="Resolve only the first N subquestions per window. "
                             "If omitted, read n_subs from <output-dir>/config.json, "
                             "else use NUM_SUBQUESTIONS from config.py.")
    parser.add_argument(
        "--no-save-debug",
        dest="save_debug",
        action="store_false",
        help="Do not store full model responses and retrieved AskNews articles.",
    )
    args = parser.parse_args()

    provider = args.provider
    if provider is None and args.model == DEFAULT_RESOLVE_MODEL:
        provider = DEFAULT_RESOLVE_PROVIDER

    run(
        model=args.model,
        provider=provider,
        output_dir=args.output_dir,
        source=args.source,
        n_subs=args.n_subs,
        save_debug=args.save_debug,
    )
