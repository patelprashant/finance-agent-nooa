"""On-disk categorization cache.

Categorizing a row costs an LLM call. Re-running the app over the same data
shouldn't pay that cost again, so we keep a small JSON cache on disk keyed on
the row's identifying content (description + amount). Change either and the key
changes, so stale results aren't reused.

Dependency-free: stdlib `json` + `hashlib`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def default_cache_path() -> Path:
    """Return the default cache file path (`.cache/categorization.json`)."""
    return Path(__file__).resolve().parents[2] / ".cache" / "categorization.json"


class CategorizationCache:
    """A write-through JSON cache of categorization results.

    The stored value for each key is the categorizer's output as a plain dict
    (category, sub_category, is_business, tax_deductible, note). The file is
    loaded once on init and written out on every `set`.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else default_cache_path()
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if self._path.exists():
            try:
                with self._path.open(encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # A corrupt or unreadable cache is a cache miss, not a crash.
                return {}
        return {}

    def key(self, description: str, amount: float) -> str:
        """Return the sha256 hex key for a row's description + amount."""
        raw = f"{description}|{amount:.2f}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        """Return the cached value for `key`, or None on a miss."""
        return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        """Store `value` under `key` and write the cache through to disk."""
        self._data[key] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
