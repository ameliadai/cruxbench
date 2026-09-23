import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
TOGETHER_API_KEY  = os.environ.get("TOGETHER_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Paper defaults.
DEFAULT_GENERATION_PROVIDER = "anthropic"
DEFAULT_GENERATION_MODEL = "claude-opus-4-6"
DEFAULT_RESOLVE_PROVIDER = "anthropic"
DEFAULT_RESOLVE_MODEL    = "claude-sonnet-4-6"
DEFAULT_VERIFY_PROVIDER  = "openrouter"
DEFAULT_VERIFY_MODEL     = "anthropic/claude-sonnet-5"
DEFAULT_FORECAST_PROVIDER = "anthropic"
DEFAULT_FORECAST_MODEL    = "claude-haiku-4-5-20251001"

# General fallback used by src.llm.call_llm().
DEFAULT_CALL_PROVIDER = "anthropic"
DEFAULT_CALL_MODEL       = "claude-sonnet-4-6"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_BASE  = PROJECT_ROOT / "outputs"
OUTPUT_DIR   = OUTPUT_BASE / DEFAULT_GENERATION_MODEL

# Pipeline config
NUM_SUBQUESTIONS      = 1

# File names
INPUT_FILE        = "ultimates_w_news.json"
SUBQUESTIONS_FILE = "subquestions.json"
RAW_RESOLVED_FILE = "resolved_raw.json"
RESOLVED_FILE     = "resolved.json"
