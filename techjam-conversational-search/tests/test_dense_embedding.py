"""Tests for replacing TF-IDF dense retrieval with sentence embeddings (Task 3).

The dense retrieval now uses a pretrained sentence-embedding model (inference only, no
fine-tuning) when `sentence_transformers` + `numpy` are available. When they are not, it
falls back to the existing TF-IDF path so the agent stays runnable.

Tests:
  * `test_dense_mode_falls_back_to_tfidf` — runs everywhere, proves the fallback works.
  * `test_embed_dense_scores_ranks_by_similarity` — skipped unless numpy/transformers are
    installed; proves the embedding cosine scoring orders docs by similarity.
  * `test_browsing_hr_embed_vs_tfidf` — skipped unless the libs are installed; runs the
    browsing-scenario A/B that the task asks for (Hit Rate@10: TF-IDF vs embeddings).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from starter.agent import Agent  # noqa: E402


def _has_libs() -> bool:
    try:
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class FakeAgent(Agent):
    """Lightweight agent that skips the expensive catalog build."""

    def __init__(self) -> None:
        self.products = {
            "A": {"parent_asin": "A", "title": "cotton shirt", "categories": ["clothing"],
                  "features": [], "details": {}, "store": "s", "description": "cotton running shirt"},
            "B": {"parent_asin": "B", "title": "wool jacket", "categories": ["clothing"],
                  "features": [], "details": {}, "store": "s", "description": "wool winter jacket"},
        }
        self.order = ["A", "B"]


def test_dense_mode_falls_back_to_tfidf(monkeypatch):
    """When the optional libs are missing the agent must use the TF-IDF fallback."""
    real_import = __import__

    def fake_import(name, *a, **k):
        if name in ("numpy", "sentence_transformers"):
            raise ModuleNotFoundError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    agent = FakeAgent()
    agent._build_dense_index()
    assert agent.dense_mode == "tfidf"
    # Use a term shared by both docs so it survives the TF-IDF min_df filter.
    scores = agent._dense_scores("clothing")
    assert isinstance(scores, list)
    assert scores and scores[0][1] > 0.0
    assert agent.order[scores[0][0]] == "A"


@pytest.mark.skipif(not _has_libs(), reason="numpy / sentence-transformers not installed")
def test_embed_dense_scores_ranks_by_similarity():
    """Embedding cosine scoring should order docs by descending similarity."""
    import numpy as np

    class FakeEmbed:
        def encode(self, texts, **kwargs):
            # Query vector aligned to doc A -> A scores highest.
            return np.array([[1.0, 0.0]])

    agent = FakeAgent()
    agent.dense_mode = "embed"
    agent._embed_model = FakeEmbed()
    # Docs: A=[1,0], B=[0,1] (normalized). Query=[1,0].
    agent._emb_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    res = agent._dense_scores("cotton shirt")
    assert res[0] == (0, 1.0)
    assert res[1] == (1, 0.0)


@pytest.mark.skipif(not _has_libs(), reason="numpy / sentence-transformers not installed")
def test_browsing_hr_embed_vs_tfidf():
    """Hit Rate@10 for browsing sessions: TF-IDF vs sentence embeddings.

    This is the A/B the task asks for. It is expensive (it embeds the full catalog at
    startup) so it is gated behind the libs check.
    """
    root = str(REPO)
    sys.path.insert(0, root)
    from evaluator.local_evaluator import (  # noqa: E402
        MAX_TURNS, TOP_K, catalog_index, customer_reply, coarse_category,
        evaluate, initial_message, load_jsonl, materialize_hidden_fields,
    )

    agg = {"tfidf": None, "embed": None}
    for mode in ("tfidf", "embed"):
        import os
        if mode == "tfidf":
            os.environ["COPILOT_EMBED_MODEL"] = "__nonexistent__"  # forces fallback to TF-IDF
        else:
            os.environ.pop("COPILOT_EMBED_MODEL", None)
        samples = [s for s in load_jsonl(REPO / "data" / "public_set.jsonl") if s["scenario_type"] == "browsing"]
        catalog_ids, categories, products = catalog_index(REPO / "data" / "catalog.jsonl")
        agent = Agent(REPO / "data" / "catalog.jsonl")
        res = evaluate(agent, samples, catalog_ids, categories, products)
        agg[mode] = res["hit_rate_at_10"]

    assert agg["embed"] >= agg["tfidf"], f"embeddings should not regress browsing HR ({agg})"
