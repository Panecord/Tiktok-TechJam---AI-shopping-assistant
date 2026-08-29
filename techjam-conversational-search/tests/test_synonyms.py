"""Tests for static synonym-aware color/material matching (v2.9.0 / Task 3).

The fixed COLORS/MATERIALS vocab misses common catalog phrasings ("emerald",
"elastane", "wine", ...). These tests pin the deterministic synonym table:

  * extraction normalizes a synonym to its canonical vocab value;
  * `_slot_match_score` resolves a canonical slot value against a product whose text
    contains only a synonym (e.g. slot `green` matches text "emerald dress").
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from starter.agent import (  # noqa: E402
    Agent,
    _attr_value_from_text,
    _slot_value_in_text,
)


class _Stub:
    def __init__(self, text: str) -> None:
        self.products = {"A": {"parent_asin": "A", "title": text}}
        self._searchable_lc = {"A": text}


# -- Extraction-side normalization -------------------------------------------

@pytest.mark.parametrize(
    "text,attr,expected",
    [
        ("emerald dress", "color", "green"),
        ("sage blouse", "color", "green"),
        ("wine jacket", "color", "red"),
        ("cobalt sweater", "color", "blue"),
        ("elastane leggings", "material", "spandex"),
        ("viscose blouse", "material", "rayon"),
        ("faux leather bag", "material", "polyurethane"),
        ("rose gold necklace", "color", "pink"),
        ("off white tee", "color", "white"),
        # canonical vocab still wins when present
        ("green dress", "color", "green"),
        ("cotton shirt", "material", "cotton"),
        ("leather bag", "material", "leather"),
    ],
)
def test_synonym_extraction_normalizes(text, attr, expected):
    assert _attr_value_from_text(text, attr) == expected


# -- Match-side resolution ----------------------------------------------------

@pytest.mark.parametrize(
    "value,text,attr,expected",
    [
        ("green", "emerald dress", "color", True),
        ("green", "blue dress", "color", False),
        ("red", "wine shoes", "color", True),
        ("spandex", "elastane leggings", "material", True),
        ("rayon", "viscose blouse", "material", True),
        ("wool", "tweed blazer", "material", True),
        ("polyester", "silk scarf", "material", False),
        ("casual", "casual shirt", "style", True),
    ],
)
def test_slot_value_in_text(value, text, attr, expected):
    assert _slot_value_in_text(value, text, attr) is expected


def test_slot_match_score_resolves_synonym():
    agent = _Stub("emerald dress")
    assert Agent._slot_match_score(agent, "A", {"color": "green"}) == 1.0
    assert Agent._slot_match_score(agent, "A", {"color": "blue"}) == 0.0
