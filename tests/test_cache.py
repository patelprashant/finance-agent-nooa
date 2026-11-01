"""Tests for the on-disk categorization cache.

Covers a set/get round-trip, that a changed amount yields a different key and a
miss, and that entries persist across a reopened cache pointing at the same
temp file.
"""

from __future__ import annotations

from finance_agent.cache import CategorizationCache

VALUE = {
    "category": "dining",
    "sub_category": "coffee",
    "is_business": False,
    "tax_deductible": False,
    "note": "a latte",
}


def test_set_then_get_round_trips(cache):
    key = cache.key("Cafe latte", 12.00)
    assert cache.get(key) is None  # miss before set

    cache.set(key, VALUE)
    assert cache.get(key) == VALUE


def test_changed_amount_yields_different_key_and_miss(cache):
    key = cache.key("Cafe latte", 12.00)
    cache.set(key, VALUE)

    other_key = cache.key("Cafe latte", 12.50)
    assert other_key != key
    assert cache.get(other_key) is None


def test_changed_description_yields_different_key(cache):
    key = cache.key("Cafe latte", 12.00)
    other_key = cache.key("Bakery", 12.00)
    assert other_key != key


def test_persists_across_reopen(cache_path):
    first = CategorizationCache(cache_path)
    key = first.key("Cafe latte", 12.00)
    first.set(key, VALUE)

    # A brand-new cache pointed at the same file should load the stored entry.
    reopened = CategorizationCache(cache_path)
    assert reopened.get(key) == VALUE
