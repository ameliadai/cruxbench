import json
import os
import time

import pandas as pd
# Provider SDKs are optional: install only the ones you call (see requirements.txt).
try:
    from anthropic import Anthropic, APIStatusError
except ImportError:
    Anthropic = None

    class APIStatusError(Exception):
        """Placeholder so `except APIStatusError` works without the anthropic SDK."""

try:
    from asknews_sdk import AskNewsSDK
except ImportError:
    AskNewsSDK = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _require(sdk, package: str):
    if sdk is None:
        raise ImportError(f"This provider needs the `{package}` package: pip install {package}")
    return sdk

from config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    TOGETHER_API_KEY,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    DEFAULT_CALL_PROVIDER,
    DEFAULT_CALL_MODEL,
)

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_PROVIDERS = {"anthropic", "openai", "gemini", "together", "openrouter", "vllm"}

# Where vLLM/HuggingFace caches model weights. Override with HF_HOME or
# VLLM_CACHE_DIR env vars (useful when $HOME is on a quota'd filesystem).
VLLM_CACHE_DIR = os.environ.get("VLLM_CACHE_DIR") or os.environ.get("HF_HOME", "")

# Where parse_json writes failing outputs. Callers override this per-run (e.g. to the
# pipeline's output dir) before invoking call_llm.
PARSE_FAILURE_LOG = "parse_failures.log"


class AskNewsRateLimitError(Exception):
    """Raised when AskNews returns 403014 (free-tier usage limit)
    or 403015 (billable overage limit)."""

_anthropic_client = None
_openai_client = None
_together_client = None
_openrouter_client = None
_gemini_client = None
_asknews_client = None
_vllm_engine = None
_vllm_model_name = None


def _get_asknews():
    global _asknews_client
    if _asknews_client is None:
        api_key = os.environ.get("ASKNEWS_API_KEY", "")
        if not api_key:
            raise ValueError("ASKNEWS_API_KEY not set in environment.")
        _asknews_client = _require(AskNewsSDK, "asknews")(
            api_key=api_key,
            scopes=["chat", "news", "stories"],
        )
    return _asknews_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = _require(Anthropic, "anthropic")(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = _require(OpenAI, "openai")(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_together():
    global _together_client
    if _together_client is None:
        _together_client = _require(OpenAI, "openai")(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)
    return _together_client


def get_openrouter_client():
    """Return the shared OpenRouter OpenAI-compatible client."""
    global _openrouter_client
    if _openrouter_client is None:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in environment.")
        _openrouter_client = _require(OpenAI, "openai")(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    return _openrouter_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment.")
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _get_vllm(model_name: str):
    """Lazily load a vLLM engine. Only one model can be loaded per process."""
    global _vllm_engine, _vllm_model_name
    if _vllm_engine is not None:
        if _vllm_model_name != model_name:
            raise ValueError(
                f"vLLM already loaded with {_vllm_model_name!r}; cannot switch to {model_name!r} "
                "in the same process."
            )
        return _vllm_engine

    if VLLM_CACHE_DIR:
        os.makedirs(VLLM_CACHE_DIR, exist_ok=True)
        os.environ.setdefault("HF_HOME", VLLM_CACHE_DIR)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", VLLM_CACHE_DIR)

    from vllm import LLM
    llm_kwargs = dict(model=model_name, dtype="auto", gpu_memory_utilization=0.9)
    if VLLM_CACHE_DIR:
        llm_kwargs["download_dir"] = VLLM_CACHE_DIR
    _vllm_engine = LLM(**llm_kwargs)
    _vllm_model_name = model_name
    return _vllm_engine


def resolve_provider_model(model: str, provider: str | None = None) -> tuple[str, str]:
    """Resolve an explicit provider and provider-native model name.

    The preferred interface passes provider and model separately. For backward
    compatibility, provider-prefixed model names such as ``together/meta-llama/...``
    and legacy unprefixed Claude/Gemini/OpenAI names are also recognized.
    """
    if provider is not None:
        provider = provider.lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}. Choose from: "
                f"{', '.join(sorted(SUPPORTED_PROVIDERS))}."
            )
        prefix = f"{provider}/"
        if model.startswith(prefix):
            model = model.removeprefix(prefix)
        return provider, model

    for candidate in ("openrouter", "together", "vllm", "anthropic", "openai", "gemini"):
        prefix = f"{candidate}/"
        if model.startswith(prefix):
            return candidate, model.removeprefix(prefix)

    # Preserve the original model-name inference used by the benchmark scripts.
    if model.startswith("claude"):
        return "anthropic", model
    if model.startswith("gemini"):
        return "gemini", model
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai", model
    return DEFAULT_CALL_PROVIDER, model


def model_path(provider: str, model: str) -> str:
    """Output path for a provider/model pair under outputs/.

    Models served through Together, OpenRouter, or local vLLM keep the provider
    prefix (``together/meta-llama/...``); Anthropic, OpenAI, and Gemini models use
    the bare model name.
    """
    return f"{provider}/{model}" if provider in {"together", "vllm", "openrouter"} else model


def model_dir_name(provider: str, model: str) -> str:
    """Single-level folder name for outputs_forecast/ (``together_meta-llama_...``)."""
    return model_path(provider, model).replace("/", "_")


def call_llm(prompt: str, system: str, model: str = DEFAULT_CALL_MODEL,
             provider: str | None = None, web_search: bool = False,
             json_schema: dict | None = None, max_tokens: int = 16000,
             temperature: float | None = None) -> str:
    """Call a supported provider/model pair and return the raw text response."""
    provider, model = resolve_provider_model(model, provider)

    if provider == "anthropic":
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if web_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
        if temperature is not None:
            kwargs["temperature"] = temperature

        for attempt in range(3):
            try:
                msg = _get_anthropic().messages.create(**kwargs)
                break
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 2:
                    wait = 2 ** attempt * 5
                    print(f"  [RETRY] Anthropic {e.status_code}, retrying in {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] Anthropic {e.status_code} after {attempt+1} attempts, skipping: {e.message}")
                    return None
        else:
            return None

        text = ""
        for block in msg.content:
            if block.type == "text":
                text += block.text
        return text

    elif provider == "vllm":
        if web_search:
            raise ValueError("Provider 'vllm' does not support web_search in this pipeline.")
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        engine = _get_vllm(model)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        sp_kwargs = dict(
            temperature=0.7 if temperature is None else temperature,
            max_tokens=max_tokens,
        )
        if json_schema is not None:
            sp_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(**sp_kwargs)
        outputs = engine.chat(messages, sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    elif provider == "together":
        if web_search:
            raise ValueError("Provider 'together' does not support web_search in this pipeline.")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        request_max_tokens = max_tokens
        for attempt in range(3):
            try:
                resp = _get_together().chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=request_max_tokens,
                    **({"temperature": temperature} if temperature is not None else {}),
                )
                return resp.choices[0].message.content
            except Exception as e:
                err_msg = str(e)
                if "422" in err_msg and "max_new_tokens" in err_msg:
                    # Reduce max_tokens to fit context window
                    request_max_tokens = max(512, request_max_tokens // 2)
                    print(f"  [RETRY] Context too long, reducing max_tokens to {request_max_tokens}")
                    continue
                if attempt < 2:
                    wait = 2 ** attempt * 5
                    print(f"  [RETRY] Together {e}, retrying in {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] Together failed after 3 attempts: {e}")
                    return None
        return None

    elif provider == "openrouter":
        if web_search:
            raise ValueError(
                "OpenRouter web search is not enabled by this pipeline. "
                "Use --source asknews or choose anthropic, openai, or gemini."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_response", "schema": json_schema},
            }
        for attempt in range(3):
            try:
                resp = get_openrouter_client().chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                if "response_format" in kwargs:
                    kwargs.pop("response_format")
                    print(
                        "  [RETRY] OpenRouter model rejected structured output; "
                        "retrying with prompt-only JSON guidance"
                    )
                    continue
                if attempt < 2:
                    wait = 2 ** attempt * 5
                    print(f"  [RETRY] OpenRouter {e}, retrying in {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] OpenRouter failed after 3 attempts: {e}")
                    return None

    elif provider == "gemini":
        from google.genai import types as genai_types
        client = _get_gemini()
        config_kwargs = dict(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if web_search:
            config_kwargs["tools"] = [genai_types.Tool(google_search=genai_types.GoogleSearch())]
        if json_schema is not None and not web_search:
            # google_search and response_schema cannot be combined.
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema

        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(**config_kwargs),
                )
                return resp.text
            except Exception as e:
                status = getattr(e, "code", None) or getattr(e, "status_code", None)
                if status is not None and status >= 500 and attempt < 2:
                    wait = 2 ** attempt * 5
                    print(f"  [RETRY] Gemini {status}, retrying in {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] Gemini failed after {attempt+1} attempts: {e}")
                    return None
        return None

    elif provider == "openai":
        kwargs = dict(
            model=model,
            instructions=system,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        if web_search:
            kwargs["tools"] = [{"type": "web_search"}]
        if temperature is not None:
            kwargs["temperature"] = temperature

        for attempt in range(3):
            try:
                resp = _get_openai().responses.create(**kwargs)
                break
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status is not None and status >= 500 and attempt < 2:
                    wait = 2 ** attempt * 5
                    print(f"  [RETRY] OpenAI {status}, retrying in {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] OpenAI failed after {attempt+1} attempts: {e}")
                    return None
        else:
            return None

        text = ""
        for item in resp.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        text += block.text
        return text

    raise AssertionError(f"Unhandled provider: {provider}")


ASKNEWS_TOOL = {
    "name": "asknews_search",
    "description": (
        "Search for news articles via AskNews. Returns titles, summaries, "
        "publication dates, source URLs, and key points for matching articles. "
        "Use this to find evidence about whether an event occurred. "
        "IMPORTANT: Always provide start_date and end_date to narrow results. "
        "The date range must not exceed 150 days."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant news articles.",
            },
            "n_articles": {
                "type": "integer",
                "description": "Number of articles to return (default 10, max 20).",
                "default": 10,
            },
            "start_date": {
                "type": "string",
                "description": "Start date for article search in YYYY-MM-DD format (optional).",
            },
            "end_date": {
                "type": "string",
                "description": "End date for article search in YYYY-MM-DD format (optional).",
            },
        },
        "required": ["query"],
    },
}

OPENAI_ASKNEWS_TOOL = {
    "type": "function",
    "function": {
        "name": ASKNEWS_TOOL["name"],
        "description": ASKNEWS_TOOL["description"],
        "parameters": ASKNEWS_TOOL["input_schema"],
    },
}


def _execute_asknews_search(query: str, n_articles: int = 10,
                            start_date: str = None, end_date: str = None) -> str:
    """Execute an AskNews search and return formatted results."""
    ask = _get_asknews()

    MAX_RANGE_DAYS = 150  # AskNews limit is 160; use 150 for safety margin

    # Parse dates and clamp range to avoid AskNews 160-day limit
    end_dt = pd.to_datetime(end_date, utc=True) if end_date else pd.Timestamp.now("UTC")
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
    else:
        start_dt = end_dt - pd.Timedelta(days=MAX_RANGE_DAYS)

    if (end_dt - start_dt).days > MAX_RANGE_DAYS:
        start_dt = end_dt - pd.Timedelta(days=MAX_RANGE_DAYS)

    kwargs = dict(
        query=query,
        n_articles=min(n_articles, 20),
        return_type="both",
        method="kw",
        start_timestamp=int(start_dt.timestamp()),
        end_timestamp=int(end_dt.timestamp()),
        time_filter="pub_date",
        languages=["en"],
        try_cache="7d",
    )

    response = ask.news.search_news(**kwargs)

    # Format results for the model
    lines = []
    for i, art in enumerate(response.as_dicts, 1):
        title = art.eng_title or art.title or ""
        pub = art.pub_date.isoformat()[:10] if art.pub_date else ""
        source = art.source_id or ""
        url = str(art.article_url)
        summary = art.summary or ""
        key_points = art.key_points or []

        lines.append(f"[{i}] {title}")
        lines.append(f"    Source: {source} | Published: {pub}")
        lines.append(f"    URL: {url}")
        if summary:
            lines.append(f"    Summary: {summary}")
        lines.append("")

    if not lines:
        return "No articles found for this query."
    return "\n".join(lines)


def _openai_compatible_client(provider: str):
    if provider == "openai":
        return _get_openai()
    if provider == "openrouter":
        return get_openrouter_client()
    if provider == "together":
        return _get_together()
    raise ValueError(f"Provider {provider!r} is not OpenAI-compatible.")


def _call_openai_compatible_with_asknews(prompt: str, system: str, model: str,
                                         provider: str, max_tool_calls: int,
                                         return_debug: bool):
    """Run the AskNews tool loop through OpenAI-compatible chat completions."""
    client = _openai_compatible_client(provider)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    articles = []
    calls = 0

    while True:
        kwargs = dict(model=model, messages=messages, max_tokens=16000)
        if calls < max_tool_calls:
            kwargs["tools"] = [OPENAI_ASKNEWS_TOOL]

        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt * 5
                    print(
                        f"  [RETRY] {provider} {e}, retrying in {wait}s "
                        f"(attempt {attempt+1}/3)"
                    )
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] {provider} failed after 3 attempts: {e}")
                    return (None, None) if return_debug else None

        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])
        if not tool_calls:
            text = message.content or ""
            if return_debug:
                return text, {"articles": articles, "full_response": text}
            return text

        assistant_tool_calls = []
        for tool_call in tool_calls:
            assistant_tool_calls.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            })
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": assistant_tool_calls,
        })

        for tool_call in tool_calls:
            if tool_call.function.name != ASKNEWS_TOOL["name"]:
                result_text = f"Unknown tool: {tool_call.function.name}"
            elif calls >= max_tool_calls:
                result_text = "Search limit reached. No more searches available."
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                    print(f"    [AskNews] query={arguments.get('query', '')[:60]}...")
                    result_text = _execute_asknews_search(
                        query=arguments["query"],
                        n_articles=arguments.get("n_articles", 10),
                        start_date=arguments.get("start_date"),
                        end_date=arguments.get("end_date"),
                    )
                    articles.append({
                        "query": arguments.get("query", ""),
                        "result": result_text,
                    })
                    calls += 1
                except Exception as e:
                    if "403014" in str(e) or "403015" in str(e):
                        raise AskNewsRateLimitError(str(e))
                    result_text = f"Error searching AskNews: {e}"
                    print(f"    [AskNews ERROR] {e}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

        if calls >= max_tool_calls:
            messages.append({
                "role": "user",
                "content": (
                    "You have used all available searches. Based only on the "
                    "evidence gathered so far, output your final JSON answer now."
                ),
            })


def call_llm_with_asknews(prompt: str, system: str, model: str = DEFAULT_CALL_MODEL,
                          provider: str | None = None, max_tool_calls: int = 10,
                          return_debug: bool = False):
    """Call an LLM with asknews_search as a tool, handling the tool-use loop.

    If return_debug=True, returns (text, debug) where debug is a dict with:
      - 'articles': list of {'query': str, 'result': str} for each AskNews call
      - 'full_response': the raw final text from the model
    Otherwise returns just the text string.
    """
    provider, model = resolve_provider_model(model, provider)
    if provider in {"openai", "openrouter", "together"}:
        return _call_openai_compatible_with_asknews(
            prompt=prompt,
            system=system,
            model=model,
            provider=provider,
            max_tool_calls=max_tool_calls,
            return_debug=return_debug,
        )
    if provider != "anthropic":
        raise ValueError(
            "The agentic AskNews resolver supports anthropic, openai, openrouter, "
            "and together. Use --source websearch with gemini, or choose one of "
            "the supported AskNews providers."
        )

    client = _get_anthropic()
    messages = [{"role": "user", "content": prompt}]
    articles = []

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                messages=messages,
                tools=[ASKNEWS_TOOL],
            )
            break
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < 2:
                wait = 2 ** attempt * 5
                print(f"  [RETRY] Anthropic {e.status_code}, retrying in {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                print(f"  [ERROR] Anthropic {e.status_code} after {attempt+1} attempts: {e.message}")
                return (None, None) if return_debug else None
    else:
        return (None, None) if return_debug else None

    calls = 0
    while msg.stop_reason == "tool_use" and calls < max_tool_calls:
        # Process tool calls from the response
        tool_results = []
        for block in msg.content:
            if block.type == "tool_use" and block.name == "asknews_search":
                inp = block.input
                print(f"    [AskNews] query={inp.get('query', '')[:60]}...")
                try:
                    result_text = _execute_asknews_search(
                        query=inp["query"],
                        n_articles=inp.get("n_articles", 10),
                        start_date=inp.get("start_date"),
                        end_date=inp.get("end_date"),
                    )
                except Exception as e:
                    if "403014" in str(e) or "403015" in str(e):
                        raise AskNewsRateLimitError(str(e))
                    result_text = f"Error searching AskNews: {e}"
                    print(f"    [AskNews ERROR] {e}")

                articles.append({"query": inp.get("query", ""), "result": result_text})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
                calls += 1

        if not tool_results:
            break

        messages.append({"role": "assistant", "content": msg.content})
        messages.append({"role": "user", "content": tool_results})

        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=system,
                    messages=messages,
                    tools=[ASKNEWS_TOOL],
                )
                break
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 2:
                    wait = 2 ** attempt * 5
                    print(f"  [RETRY] Anthropic {e.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] Anthropic {e.status_code}: {e.message}")
                    return (None, None) if return_debug else None
        else:
            return (None, None) if return_debug else None

    # If the loop exited because we hit max_tool_calls but the model still wants
    # to use tools, force a final concluding call without tools so it outputs JSON.
    if msg.stop_reason == "tool_use" and calls >= max_tool_calls:
        print(f"  [WARN] Reached max_tool_calls={max_tool_calls}, forcing final answer.")
        # Must provide a tool_result for every unanswered tool_use block, otherwise
        # Anthropic returns 400 ("tool_use ids found without tool_result blocks").
        dummy_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "Search limit reached. No more searches available.",
            }
            for block in msg.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "assistant", "content": msg.content})
        messages.append({
            "role": "user",
            "content": dummy_results + [{
                "type": "text",
                "text": (
                    "You have used all available searches. "
                    "Based only on the evidence gathered so far, output your final JSON answer now."
                ),
            }],
        })
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=system,
                    messages=messages,
                )
                break
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 2:
                    wait = 2 ** attempt * 5
                    print(f"  [RETRY] Anthropic {e.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] Anthropic {e.status_code}: {e.message}")
                    return (None, None) if return_debug else None
        else:
            return (None, None) if return_debug else None

    # Extract final text
    text = ""
    for block in msg.content:
        if block.type == "text":
            text += block.text

    if return_debug:
        return text, {"articles": articles, "full_response": text}
    return text


def parse_json(text: str) -> list | dict:
    """Parse JSON from LLM output, handling markdown fences or extra text."""
    if text is None:
        return None
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = len(text)
    end = -1
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        s = text.find(open_char)
        e = text.rfind(close_char)
        if s != -1 and e != -1 and s < start:
            start = s
            end = e
    if start < len(text) and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            # Multiple JSON objects — decode only the first one
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text, start)
                return obj
            except json.JSONDecodeError:
                pass

    try:
        with open(PARSE_FAILURE_LOG, "a") as _f:
            _f.write(f"\n===== len={len(text)} =====\n{text}\n")
    except Exception:
        pass
    print(
        f"[WARNING] Could not parse JSON (len={len(text)}) "
        f"START={text[:200]!r} ... END={text[-200:]!r}"
    )
    return None
