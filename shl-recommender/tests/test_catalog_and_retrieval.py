import json
import os

import pytest

from app.catalog import _normalize, _is_packaged_solution, KEY_TO_CODE
from app.retrieval import CatalogIndex

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog_sample.json")


@pytest.fixture(scope="module")
def sample_items():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = [i for i in (_normalize(r) for r in raw) if i is not None]
    assert len(items) > 0
    return items


@pytest.fixture(scope="module")
def index(sample_items):
    return CatalogIndex(sample_items)


def test_packaged_solutions_are_excluded():
    assert _is_packaged_solution("Entry Level Cashier Solution")
    assert _is_packaged_solution("Customer Service Phone Solution")
    assert not _is_packaged_solution("Core Java (Advanced Level) (New)")
    assert not _is_packaged_solution("Automata - SQL (New)")


def test_test_type_code_mapping():
    assert KEY_TO_CODE["Knowledge & Skills"] == "K"
    assert KEY_TO_CODE["Personality & Behavior"] == "P"
    assert KEY_TO_CODE["Simulations"] == "S"


def test_search_returns_relevant_java_results(index):
    results = index.search("Java developer", top_k=5)
    assert len(results) > 0
    names = [r.item.name for r in results]
    assert any("Java" in n for n in names)


def test_search_respects_duration_filter(index):
    results = index.search("programming", top_k=25, max_duration_minutes=10)
    for r in results:
        digits = "".join(ch for ch in r.item.duration if ch.isdigit())
        if digits:
            assert int(digits) <= 10


def test_find_by_name_exact_and_fuzzy(index):
    exact = index.find_by_name("Docker (New)")
    assert exact is not None
    assert exact.name == "Docker (New)"

    fuzzy = index.find_by_name("Docker")
    assert fuzzy is not None
    assert "Docker" in fuzzy.name


def test_is_known_url(index, sample_items):
    real_url = sample_items[0].url
    assert index.is_known_url(real_url)
    assert not index.is_known_url("https://www.shl.com/products/product-catalog/view/totally-fake/")


def test_empty_query_returns_no_results(index):
    assert index.search("", top_k=5) == []
