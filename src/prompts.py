def load_question_gen_sys_prompt():
    return """You are a helpful assistant to forecasting research scientists. The goal is to generate subquestions / cruxes for a given ultimate question.

TASK BACKGROUND:
In forecasting, we often care about answering questions that are either too complex, too broad or too far in the future to answer directly. If the outcome of a question will only be known in 20 years, we would like to have indicators for which track the world is on, with respect to the outcome of the question. For example, if the ultimate question regards nuclear safety in 2050, we would like to have indicators in the near term which could be used as indicators for the ultimate question - such as the number of nuclear power plants in the world, the number of nuclear weapons in the world, the number of nuclear accidents in the world, etc.

These indicators are called subquestions / cruxes. They are indicators that can be used to track the world's progress towards the ultimate question. We want the subquestions to be grounded in current world events that pertain to the ultimate question. Ideal subquestions are such that they are cruxes for ultimate question disagreements; i.e if 2 forecasters disagree on the outcome of the ultimate question, they are more likely to agree on the outcome of the final question once the subquestion is resolved.

TASK:
- Generate subquestions that are meaningful early indicators of the ultimate question's outcome — not restatements of the ultimate question with an earlier end date.

GUIDELINES:

Questions:
- The question should be binary. This means that the outcome of the question should be either yes or no. The question should be formulated in such a way that forecasters answer with a probability. Hence, a question such as "will X increase or decrease" is not a good question. A better question is "Will X increase by at least 10 units before [specific date]?". A higher probability answer implies more likely 'yes', a lower probability implies more likely 'no'.
- Informative: Is this crux / subquestion likely to provide evidence for the ultimate outcome question that the user cares about? The question should have a high value of information - meaning that knowing the outcome of this question would update forecasters on their forecast of the ultimate outcome.
- Good questions usually do not have an expected forecast very near 0 or 1, because then the forecasts will not be informative. This is a general guideline, not a rule: there may be exceptions to this.
- Consider potential loopholes in the resolution that make the forecasts uninformative. I.e. if we are curious about the difference between 2 competing LLMs on a benchmark, and the benchmark becomes saturated, the question will be resolvable, but not informative (both competitors will have the same score of 100%).
- The question should be a clear and concise question that is easy to understand.
- The question should be grounded in current world events. For whatever question you want to ask, ensure you know the current state of that area to choose appropriate thresholds for the question.
- All terms in the question and resolution criteria must be clearly defined, eliminating ambiguity.
- Time-bounded: The subquestion end_date MUST be before the end date of the ultimate question. The subquestion end_date must also be AFTER the start date of the ultimate question (i.e. today's date). Both the ultimate question end date and start date will be provided to you.

Background information:
- The background information should provide enough context to fully understand the question and why it is relevant.
- The background information should include the current state of the area, including relevant thresholds for the question. For example if you are forecasting the number of nuclear weapons in the world, you should include the current number of nuclear weapons in the world. If you are forecasting the number of AI jobs, you should include the current number of AI jobs etc. These should influence the thresholds for the question.
- The background information must provide citations.
- Citations MUST include actual, clickable URLs (http/https). Do NOT output placeholder citation tokens like "citeturn5search5...". Inline the source URL next to claims or include them in a short Sources list.
- At the end of Background information, add a short "Sources:" list with each item on its own line containing the source title (or site) and full URL.

Resolution criteria:
- Aim for tight resolution criteria. The resolution criteria should leave little room for discretion in deciding the resolution. As best you can, try to limit the scope for ex-post quarrels about what really happened, and who was right.
- Define your terms. Questions of the sort "will X occur?" often hinge on how X is defined. It is therefore important to spell out your definitions with extra care. Don't worry about being too pedantic here!
- Be concrete. Try to specify precisely and in detail which steps should or shouldn't be followed when resolving the question. Examples are helpful for making these instructions concrete.
- Use authoritative sources, when possible. Good options are numerical data regularly published by a reliable publicly available source. Note that you should be sure that the sources will be available at the time of resolution, or otherwise you might want to specify alternative sources of information.
- Consider and account for edge-cases. Try to imagine scenarios for which the resolution conditions fail to cleanly apply, or cases that are just on the edge of counting towards resolution. If such scenarios or edge-cases are plausible, you should clarify how the question should resolve when such events rear their head.
- Consider fall-back criteria. When you have a resolution that should be easy to check assuming all goes well, try to handle also the case where all doesn't go well. What if the data source you specified stops being published? Is there anything else odd that might happen to make the outcome unclear?
- Consider delayed events: if the event is postponed beyond the resolution window, the question should be considered unresolvable — specify this explicitly in the resolution criteria.
- Try to account for unknown unknowns. Think about how the resolution criteria behave when something you don't expect happens anyway.
- Well-researched: The question should be well-researched and grounded in current world events. It must provide high quality sources for the resolution criteria. The background information must provide enough context to fully understand the question and why it is relevant. Both the resolution criteria and the background information must provide citations.
- For the Resolution criteria, include the specific data source(s) you propose to use and provide the full URLs. Avoid any placeholder tokens; always include explicit links.
- Time-bounded: The question should specify when it will be resolved. MUST explicitly state the start date and end date of the subquestion, clearly defining the resolution window. For example: "This question resolves Yes if [condition] occurs between [start_date] and [end_date] (inclusive)." This ensures resolvers know the exact time window during which qualifying events must occur.
"""


def format_articles(articles: list[dict], max_articles: int = 10) -> str:
    """Format AskNews articles (with summaries) into a text block for the prompt."""
    if not articles:
        return "No relevant news articles found."
    lines = []
    for i, art in enumerate(articles[:max_articles], 1):
        pub_date = art.get("pub_date", "")[:10]
        source = art.get("source", "")
        title = art.get("title", "")
        summary = art.get("summary", "")
        lines.append(f"[{i}] {title} ({pub_date}, {source})")
        if summary:
            lines.append(f"    {summary}")
    return "\n".join(lines)


def load_generation_prompt(
    ultimate_question: str,
    description: str,
    start_date: str,
    end_date: str,
    period_end: str,
    n: int,
    articles: list[dict] | None = None,
) -> str:
    articles_block = ""
    if articles:
        articles_block = (
            f"\n--- Possibly relevant news articles (published before {start_date}) ---\n"
            f"{format_articles(articles)}\n"
            f"--- End of articles ---\n"
        )

    return f"""Ultimate question: {ultimate_question}
Description: {description}
Today's date: {start_date}
Ultimate question end date: {end_date}

Generate {n} subquestion(s) whose start_date is today ({start_date}) and end_date is no later than {period_end}.
IMPORTANT:
- Every subquestion end_date MUST be between {start_date} and {period_end}.
- The resolution_criteria for each subquestion MUST explicitly mention the subquestion's start_date ({start_date}) and end_date, clearly stating the window during which qualifying events count toward resolution. For example: "This question resolves Yes if [condition] occurs between {start_date} and [end_date]."
- DO NOT generate a subquestion that is just the ultimate question with an earlier deadline or tighter resolution criteria. A good subquestion asks about a different observable event that could be an indicator of the ultimate outcome.

{articles_block}
RETURN FORMAT:
Return a JSON array of dictionaries. Each dictionary MUST contain:
- question: the subquestion / crux
- background_information: background information on the question. Include citations with explicit, clickable URLs and add a short Sources list at the end.
- resolution_criteria: the resolution criteria. Must explicitly state the start_date and end_date defining the resolution window. You must provide links (full URLs) to the sources which you propose to use for the resolution criteria. Do not include placeholder citation tokens.
- start_date: "{start_date}" (today's date, MUST be exactly this value)
- end_date: the date when the question will be resolved (in UTC). MUST be between {start_date} and {period_end}.
- rationale: a rationale for why this question is relevant to the ultimate question

Return ONLY the JSON array, no other text."""


RESOLVE_SYSTEM_PROMPT = (
    "You are a research assistant that determines whether forecasting questions "
    "have resolved Yes or No based on current information from the web. "
    "Search the web for the latest information and determine the resolution."
)

RESOLVE_ASKNEWS_SYSTEM_PROMPT = (
    "You are a research assistant that determines whether forecasting questions "
    "have resolved Yes or No based on news articles. "
    "Use the asknews_search tool to find relevant news articles and determine the resolution."
)


def load_resolve_prompt(question: str, resolution_criteria: str, start_date: str, end_date: str) -> str:
    return f"""Determine whether the following binary forecasting question has resolved Yes or No.

Question: {question}
Resolution criteria: {resolution_criteria}
Start date: {start_date}
End date: {end_date}

You are a careful, high-precision resolution assistant. Your goal is to determine whether the question resolved Yes or No using explicit, verifiable evidence.

IMPORTANT — resolution rules:
- Only events that occurred BETWEEN {start_date}T00:00:00Z and {end_date}T23:59:59Z (inclusive) count.
- Prefer precision over recall. If uncertain, return resolvable=false.
- Do NOT guess, infer, or assume. Only use explicit evidence.

resolvable=false if ANY of the following:
- The underlying event or data is missing, unavailable, or not reported
- The resolution criteria are ambiguous, contradictory, or unclear
- The evidence is indirect, inferred, or does not exactly match the criteria
- A reliable first-occurrence timestamp cannot be established
- Conflicting evidence cannot be resolved
- Any reason prevents a definitive Yes/No answer

resolution=Yes:
- The resolving event explicitly occurred within the window
- If multiple qualifying events exist, use the FIRST occurrence
- You MUST verify that no earlier qualifying occurrence exists in the window

resolution=No:
- If the condition became definitively impossible before end_date → resolution_datetime = the exact datetime it became impossible. Always use the earliest date at which a definitive No conclusion is supported by explicit evidence. "Definitively impossible" means there is explicit, public evidence of a discrete event that forecloses the Yes outcome (e.g., a deadline passed, a vote failed, a candidate withdrew) — not merely that the probability appeared low.
- If the window closed with no earlier definitive No moment → resolution_datetime = {end_date}

EVIDENCE REQUIREMENTS (ALL must be satisfied for resolvable=true):
1. Evidence directly matches the resolution criteria (no interpretation required)
2. Event is explicitly stated (not inferred)
3. Event is clearly within the window
4. A reliable timestamp exists (not approximate)
5. Source is high-quality OR multiple sources agree

If any condition fails → resolvable=false

STRICT RULES FOR resolution_datetime:
- Must be an exact ISO 8601 UTC datetime
- No vague language ("around", "approximately", etc.)
- If timestamp is not explicitly available or cannot be reliably determined → resolvable=false

Follow this procedure:

STEP 1 — DECOMPOSE
- Identify the exact condition required for Yes
- Identify what observable evidence would confirm it
- Identify key entities, thresholds, and timing constraints

STEP 2 — TARGETED SEARCH
- Generate multiple precise search queries including:
  * entity + metric + date
  * variations to find earliest occurrence
  * queries to disprove the event

STEP 3 — EVENT DETECTION
- Identify candidate events
- Reject any that:
  * require inference
  * are indirect or approximate
  * do not clearly satisfy criteria

STEP 4 — CRITERIA MATCHING
- Confirm the event EXACTLY satisfies the resolution criteria
- Reject partial or approximate matches

STEP 5 — TIMESTAMP EXTRACTION
- Extract earliest exact timestamp of occurrence
- Reject if timestamp is vague or inferred

STEP 6 — FIRST OCCURRENCE CHECK
- Search specifically for earlier qualifying events
- If you cannot confirm this is the first → resolvable=false

STEP 7 — VERIFICATION (CRITICAL)
- Attempt to disprove your conclusion:
  * look for contradictory evidence
  * check similar events outside window
  * confirm correct interpretation
- If contradiction exists → resolvable=false

STEP 8 — FINAL DECISION
- Yes: event occurred within window (first occurrence)
- No: condition never met within window
- Otherwise: resolvable=false

Double check before answering:
1. Evidence is direct, explicit, and within the window
2. No inference or assumptions were used
3. Timestamp is exact and reliable
4. FIRST occurrence condition is satisfied
5. No contradictory evidence exists
6. If any doubt remains → resolvable=false

Return STRICT JSON:
{{   
    "evidence": {{
        "step1_decompose": "<exact condition for Yes, observable evidence required, key entities/thresholds/timing>",
        "step2_search_queries": "<search queries used and what each was intended to find>",
        "step3_event_detection": "<candidate events identified and any rejected with reasons>",
        "step4_criteria_matching": "<confirmation or rejection of exact criteria match — include inline citations [source](url)>",
        "step5_timestamp_extraction": "<exact timestamp found, or why it could not be extracted>",
        "step6_first_occurrence_check": <whether an earlier qualifying occurrence was found>,
        "step7_verification": "<contradictory evidence checked, alternative interpretations considered>",
        "step8_final_decision": "<final reasoning tying all steps to the resolution>"
    }},
    "resolvable": true or false,
    "resolution": "Yes" or "No" or "",
    "resolution_datetime": "<exact UTC datetime in ISO 8601, or empty if resolvable=false>",
    "source_urls": ["url1", "url2", ...]
}}
"""


def load_forecast_prompt(
    question: str,
    description: str,
    start_date: str,
    end_date: str | None = None,
    articles: list[dict] | None = None,
    subquestions: list[dict] | None = None,
) -> str:
    """Forecast P(U) at t=window.start.

    Baseline: pass `articles` (same set fed to the generator) and leave `subquestions=None`.
    Treatment (oracle): also pass `subquestions`, each as a dict with keys
        question, background_information, resolution_criteria, start_date, end_date,
        resolution (Yes/No/""), resolution_datetime (ISO).
    Resolutions are appended as known facts about future events; this is intentional
    information leakage — the forecaster acts as a measurement device for how
    informative the crux set is in retrospect.
    """
    articles_block = ""
    if articles:
        articles_block = (
            f"\n--- Possibly relevant news articles (published before {start_date}) ---\n"
            f"{format_articles(articles)}\n"
            f"--- End of articles ---\n"
        )

    cruxes_block = ""
    if subquestions:
        blocks = []
        for i, s in enumerate(subquestions, 1):
            res = s.get("resolution", "") or ""
            rdt = s.get("resolution_datetime", "") or ""
            sub_start = s.get("start_date", "") or ""
            sub_end = s.get("end_date", "") or ""
            bg = (s.get("background_information", "") or "").strip()
            rc = (s.get("resolution_criteria", "") or "").strip()
            window_str = f"{sub_start} to {sub_end}" if sub_start or sub_end else ""
            outcome = f"{res}" + (f" at {rdt}" if rdt else "")

            parts = [f"[{i}] {s['question']}"]
            if window_str:
                parts.append(f"    Window: {window_str}")
            if rc:
                parts.append(f"    Resolution criteria: {rc}")
            if bg:
                parts.append(f"    Background: {bg}")
            parts.append(f"    Resolved: {outcome}")
            blocks.append("\n".join(parts))

        cruxes_block = (
            f"\n--- Resolved subquestions (these resolutions are known facts about events that occurred AFTER {start_date}; treat them as ground truth) ---\n"
            f"{chr(10).join(blocks)}\n"
            f"--- End of subquestions ---\n"
        )

    end_block = f"Ultimate question end date: {end_date}\n" if end_date else ""

    return f"""You are an expert forecaster. Estimate the probability that the ultimate question resolves YES.

Today's date: {start_date}
{end_block}Ultimate question: {question}
Description: {description}
{articles_block}{cruxes_block}
Return STRICT JSON:
{{
  "p_u": <0-1>,
  "rationale": "<brief explanation>"
}}
"""