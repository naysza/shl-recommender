"""
Centralized configuration. All values can be overridden via environment
variables so the same code runs locally and on Render without edits.
"""
import os


class Settings:
    # --- LLM ---
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    LLM_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "18"))

    # --- Catalog ---
    CATALOG_URL: str = os.environ.get(
        "CATALOG_URL",
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json",
    )
    # Where we cache the fetched catalog on disk so we don't refetch on every
    # cold start (Render free tier spins down and back up frequently).
    CATALOG_CACHE_PATH: str = os.environ.get(
        "CATALOG_CACHE_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "catalog_cache.json")
    )
    # Fallback fixture used only if the live fetch fails AND no cache exists
    # (e.g. running fully offline during development).
    CATALOG_FALLBACK_PATH: str = os.environ.get(
        "CATALOG_FALLBACK_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "catalog_sample.json")
    )
    CATALOG_FETCH_TIMEOUT_SECONDS: float = float(os.environ.get("CATALOG_FETCH_TIMEOUT_SECONDS", "20"))
    # Refetch the live catalog if the cache is older than this many hours.
    CATALOG_MAX_CACHE_AGE_HOURS: float = float(os.environ.get("CATALOG_MAX_CACHE_AGE_HOURS", "24"))

    # --- Agent behavior ---
    MAX_TURNS: int = int(os.environ.get("MAX_TURNS", "8"))
    MAX_RECOMMENDATIONS: int = int(os.environ.get("MAX_RECOMMENDATIONS", "10"))
    MIN_RECOMMENDATIONS: int = int(os.environ.get("MIN_RECOMMENDATIONS", "1"))
    RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", "25"))


settings = Settings()
