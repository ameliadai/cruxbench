from pathlib import Path


TARGET_TOTAL = 1000000
START_DATE_MIN = None  # e.g. "2025-01-01T00:00:00Z", or None for no restriction
MIN_VOLUME = 10000
CLOSED = True  # Paper dataset uses resolved historical markets.
PRICE_FIDELITY = 1440  # minutes between price points (1440 = daily)

WORKERS = 16
PAGE_SIZE = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_OUTPUT_PATH = DATA_DIR / "polymarket_0325.json"
