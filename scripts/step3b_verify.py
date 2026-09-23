"""Step 3b: Verify resolution timestamps with a timestamp-verification agent.

For each market, the verifier checks every generator's Step 3a timestamp against
the underlying public evidence and aligns cruxes across models that resolve on
the same event. Only timestamps change; Yes/No labels are kept.

Usage:
    python scripts/step3b_verify.py

Reads:  outputs/<gen>/resolved_raw.json
Writes: outputs/timestamp_verifications.json
        outputs/<gen>/resolved.json
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from config import (
    DEFAULT_VERIFY_PROVIDER,
    DEFAULT_VERIFY_MODEL,
    RAW_RESOLVED_FILE,
    RESOLVED_FILE,
)
from src.llm import call_llm, parse_json, resolve_provider_model

OUTPUTS = REPO / "outputs"
VERIFICATION_PATH = OUTPUTS / "timestamp_verifications.json"
GENERATORS = [
    ("claude-opus-4-6",        "claude-opus-4-6"),
    ("gpt-5.4",                "gpt-5.4"),
    ("gpt-3.5-turbo-1106",     "gpt-3.5-turbo-1106"),
    ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
    ("llama-3.3-70B",          "together/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("qwen3-235B",             "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"),
    ("deepseek-v3.1",          "together/deepseek-ai/DeepSeek-V3.1"),
    ("llama-3.1-8B",           "vllm/meta-llama/Llama-3.1-8B-Instruct"),
]

SYSTEM_PROMPT = """You are a resolution-time adjudicator for a forecasting benchmark. Several language models each proposed one crux (subquestion) for the same ultimate market question, and a resolver assigned each crux a Yes/No resolution and a timestamp. Resolver timestamps are noisy: the same real-world event sometimes received different timestamps for different models' cruxes, and some 'No' resolutions were stamped at an evidence time even though the answer was not yet determined.

Your job is to adjudicate TIMESTAMPS ONLY. Never change a Yes/No label.

RULES
1. Event grouping: identify which cruxes are resolved by the same underlying real-world event (across models). All cruxes in one event group must receive the IDENTICAL timestamp: the earliest time the resolving evidence plausibly became public, based on the evidence provided (event dates, article dates, official announcement times). Cruxes resolved by genuinely different events (e.g., an arrest vs. a later formal charging, when the crux wording requires the latter) belong to different groups even if close in time.
2. Yes resolutions: timestamp = the canonical public time of the resolving event (per its event group).
3. No resolutions: timestamp = the crux's OWN deadline, at end_date T23:59:59Z (rule "no_deadline") — UNLESS an event strictly before the deadline made the No logically or institutionally irreversible, e.g. the subject was eliminated from the process, an authority issued a binding and final negative decision, or the only possible path to Yes was closed (rule "no_irreversible", timestamp = that event's canonical time). Announcements of delay, postponement, low likelihood, or intent are NOT irreversible: the answer remains open until the deadline. Worked example: a crux asks "will NASA complete a rehearsal before Feb 16?"; on Feb 3 the first attempt is scrubbed and NASA announces a second rehearsal is needed and the launch is moving to March. This is rule "no_deadline" (stamp Feb 16): schedules can change again, so completion before Feb 16 remained possible — a delay announcement is evidence of No, not determination of No. Contrast: if that crux's deadline were Feb 5 and completing another attempt within 2 days were physically/procedurally impossible, rule "no_irreversible" (stamp the scrub announcement time) would apply. Apply "no_irreversible" only when you can state WHAT made Yes impossible before the deadline, not merely unlikely. Two hard requirements for "no_irreversible":
   (a) NO HINDSIGHT: you must justify impossibility using only information available AT the candidate event's time. That the Yes-event in fact never happened before the deadline (known from later evidence) is NOT an argument — every No crux trivially satisfies that. If your justification cites what happened afterwards, it is invalid; use "no_deadline".
   (b) BINDING CONSTRAINT: the impossibility must follow from a binding physical, legal, or procedural constraint with fixed dates (e.g., candidate eliminated from a completed vote; the only scheduled session already past; a law's effective date). Repair timelines, agency scheduling projections, "requires X first" chains, and stated targets are expectations, not constraints — they can be revised, so they are NOT irreversible.
4. If the resolver's existing timestamp already satisfies these rules, keep it.
5. Use UTC ISO format like 2026-02-03T17:00:00Z. Timestamps must lie within the crux's [start_date, end_date 23:59:59] window.
6. TIMEZONE DISCIPLINE AND TIME PRECISION: source times are often stated in local timezones (ET, PT, CET, ...) or as bare dates. Assign the event time by this ladder, and report which level you used in "time_precision":
   - "stated": an article or official source explicitly states the event time. Rationale must cite WHICH source (title or URL), QUOTE the stated time with its timezone, and SHOW the conversion to UTC (e.g., "NASA blog: 'scrubbed at 12:00 p.m. EST Feb 3' -> 17:00 UTC").
   - "convention": no explicit time, but the event has a documented standard release time (e.g., "BLS Employment Situation releases at 8:30 a.m. ET -> 12:30 UTC"). Name the convention in the rationale.
   - "date_only": only a date is determinable. Do NOT invent an hour: use T00:00:00Z of the earliest date the evidence was public, and say so in the rationale.
   Never present a guessed hour as stated.

Return JSON only — an array with EXACTLY one entry per crux listed in the input:
[
  {"model": "<model name as given>",
   "event_group": "<short label, e.g. 'wdr-scrub' or 'deadline'>",
   "rule": "yes_event" | "no_deadline" | "no_irreversible",
   "adjudicated_datetime": "YYYY-MM-DDTHH:MM:SSZ",
   "time_precision": "stated" | "convention" | "date_only",
   "changed": true|false,
   "rationale": "1-3 sentences; for event-timestamped rules, cite the source and show the timezone conversion to UTC"}
]"""


def _crux_block(model, s):
    ev = s.get("evidence")
    ev_str = ev if isinstance(ev, str) else json.dumps(ev, ensure_ascii=False)
    # Tail of the evidence: the resolver's final event detection / judgement
    # lives at the end of the step-wise dict.
    ev_tail = ("..." + ev_str[-2200:]) if len(ev_str) > 2200 else ev_str
    arts = s.get("_articles") or []
    art_lines = []
    for a in arts[:3]:
        r = (a.get("result") or "") if isinstance(a, dict) else str(a)
        if r.strip():
            art_lines.append(r[:700])
    art_str = ("\n".join(art_lines)) if art_lines else "(none available)"
    return (
        f"--- CRUX [{model}] ---\n"
        f"Question: {s.get('question')}\n"
        f"Resolution criteria: {str(s.get('resolution_criteria'))[:600]}\n"
        f"Crux window: {s.get('start_date')} to {s.get('end_date')}\n"
        f"Resolver label: {s.get('resolution')}   Resolver timestamp: {s.get('resolution_datetime')}\n"
        f"Resolver evidence (final steps): {ev_tail}\n"
        f"Sources: {str(s.get('source_urls'))[:400]}\n"
        f"Retrieved articles (title | source | published date | summary):\n{art_str}\n"
    )


def resolution_input_path(subdir):
    """Path to the Step 3a output for one generator."""
    raw_path = OUTPUTS / subdir / RAW_RESOLVED_FILE
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found; run step3a_resolve.py first.")
    return raw_path


def load_markets():
    """market_id -> {question, cruxes: {model: subq_dict}}"""
    markets = {}
    for name, sub in GENERATORS:
        df = pd.read_json(resolution_input_path(sub))
        for _, r in df.iterrows():
            mid = int(r["id"])
            m = markets.setdefault(mid, {"question": r["question"], "cruxes": {}})
            for w in r["active_windows"]:
                subs = w.get("subquestions") or []
                if subs:
                    m["cruxes"][name] = subs[0]
    return markets


def verify_market(provider, model, mid, market):
    cruxes = {g: s for g, s in market["cruxes"].items()
              if s.get("resolvable") and s.get("resolution_datetime")}
    if not cruxes:
        return {"market_id": mid, "entries": [], "note": "no resolvable cruxes"}

    user = (f"ULTIMATE MARKET QUESTION: {market['question']}\n\n"
            + "\n".join(_crux_block(g, s) for g, s in cruxes.items())
            + f"\nAdjudicate the {len(cruxes)} cruxes above now. Return the JSON array only.")

    entries, raw = None, None
    t0 = time.time()
    stats = {"attempts": 0}
    # Request the maximum output length first so long answers are not cut off.
    # If the call fails or the reply is not a parseable JSON list, retry once
    # with a smaller cap; if both fail, the market is recorded as unparseable.
    for max_tok in (64000, 32000):
        raw = call_llm(
            user,
            system=SYSTEM_PROMPT,
            provider=provider,
            model=model,
            max_tokens=max_tok,
            temperature=0,
        )
        stats["attempts"] += 1
        entries = parse_json(raw)
        if isinstance(entries, list):
            break
    stats["elapsed_s"] = round(time.time() - t0, 1)
    if not isinstance(entries, list):
        return {"market_id": mid, "entries": [], "error": "unparseable", "raw": raw, "stats": stats}

    # Mechanical enforcement + bookkeeping
    out = []
    for e in entries:
        g = e.get("model")
        s = cruxes.get(g)
        if s is None:
            continue
        old = s.get("resolution_datetime")
        rule = e.get("rule")
        verified = e.get("adjudicated_datetime")
        prec = e.get("time_precision")
        if rule == "no_deadline":
            # The stated deadline can fall anywhere in the crux window and may be
            # timezone-qualified (e.g. 'by Feb 10, 11:59 PM ET'). Accept the LLM's
            # timestamp if it lies in [start_date, end_date + 36h]; otherwise fall
            # back to end_date (as given if it includes a time, else T23:59:59Z).
            end = str(s.get("end_date") or "")
            end_day = end[:10]
            fallback = end if "T" in end else f"{end_day}T23:59:59Z"
            try:
                verified_ts = pd.to_datetime(verified, utc=True)
                lo = pd.to_datetime(str(s.get("start_date"))[:10], utc=True)
                hi = pd.to_datetime(end_day, utc=True) + pd.Timedelta(hours=36)
                ok = lo <= verified_ts <= hi
            except (ValueError, TypeError):
                ok = False
            if not ok:
                verified = fallback
            prec = "deadline"
        out.append({
            "model": g, "resolution": s.get("resolution"),
            "old_datetime": old, "adjudicated_datetime": verified,
            "rule": rule, "event_group": e.get("event_group"),
            "time_precision": prec,
            "changed": (verified or "")[:16] != (old or "")[:16],
            "rationale": e.get("rationale"),
        })
    missing = set(cruxes) - {e["model"] for e in out}
    rec = {"market_id": mid, "question": market["question"], "entries": out, "stats": stats}
    if missing:
        rec["error"] = f"missing entries for: {sorted(missing)}"
    return rec


def apply_verifications(verifications):
    """Write the canonical per-generator resolution files."""
    new_time = {(record["market_id"], entry["model"]): entry["adjudicated_datetime"]
                for record in verifications for entry in record.get("entries", [])}
    for name, sub in GENERATORS:
        df = pd.read_json(resolution_input_path(sub))
        n = 0
        for _, r in df.iterrows():
            for w in r["active_windows"]:
                for s in (w.get("subquestions") or []):
                    # Drop the search logs from resolved.json; they remain in
                    # resolved_raw.json when Step 3a saved them.
                    s.pop("_full_response", None)
                    s.pop("_articles", None)
                    key = (int(r["id"]), name)
                    if key in new_time and s.get("resolution_datetime"):
                        if new_time[key] and new_time[key] != s["resolution_datetime"]:
                            s["resolution_datetime"] = new_time[key]
                            n += 1
        out_path = OUTPUTS / sub / RESOLVED_FILE
        df.to_json(out_path, orient="records", indent=2)
        print(f"  {name}: {n} timestamps changed -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--provider",
        default=None,
        help=(
            "LLM provider. Defaults to the paper provider for the default model; "
            "otherwise inferred from a provider-prefixed model name."
        ),
    )
    ap.add_argument("--model", default=DEFAULT_VERIFY_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="verify only the first N pending markets")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    ap.add_argument("--no-write", action="store_true", help="do not write canonical resolved.json files")
    args = ap.parse_args()
    if args.provider is None and args.model == DEFAULT_VERIFY_MODEL:
        args.provider = DEFAULT_VERIFY_PROVIDER
    args.provider, args.model = resolve_provider_model(args.model, args.provider)

    print("loading resolution inputs for all generators ...")
    markets = load_markets()
    print(f"{len(markets)} markets loaded")

    verifications = (json.loads(VERIFICATION_PATH.read_text())
                     if VERIFICATION_PATH.exists() else [])
    done = {record["market_id"] for record in verifications}
    todo = [mid for mid in sorted(markets) if mid not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"{len(done)} markets already verified, running {len(todo)} "
        f"(provider/model={args.provider}/{args.model})"
    )

    lock = threading.Lock()

    def _run(mid):
        try:
            return verify_market(args.provider, args.model, mid, markets[mid])
        except Exception as e:
            return {"market_id": mid, "entries": [], "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, mid): mid for mid in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            with lock:
                verifications.append(rec)
                VERIFICATION_PATH.write_text(
                    json.dumps(verifications, indent=2, ensure_ascii=False)
                )

    entries = [entry for record in verifications for entry in record.get("entries", [])]
    errs = [record for record in verifications if record.get("error")]
    print(f"\n{len(verifications)} markets verified, {len(entries)} crux entries, "
          f"{len(errs)} with errors")
    stats = [record["stats"] for record in verifications if record.get("stats")]
    if stats:
        tot_time = sum(s.get("elapsed_s", 0) for s in stats)
        n_retry = sum(1 for s in stats if s.get("attempts", 1) > 1)
        print(
            f"time: {tot_time/60:.1f} min total, "
            f"{tot_time/max(1,len(stats)):.0f}s/market avg | retries: {n_retry}"
        )
    if entries:
        s = pd.DataFrame(entries)
        print("\nby rule:")
        print(s.groupby("rule").agg(n=("model", "size"), changed=("changed", "sum")).to_string())
        print("\nby time precision:")
        print(s.groupby("time_precision", dropna=False).size().to_string())
        print(f"\ntotal changed timestamps: {s.changed.sum()} / {len(s)}")

    if not args.no_write:
        print("\nwriting canonical resolution files ...")
        apply_verifications(verifications)


if __name__ == "__main__":
    main()
