"""Focused regressions for the official Track 4 routing and context pillars."""

from starter.agent import Agent, ROUTE_RRF_WEIGHTS, _profile_terms, _route_intent


def test_exploration_language_routes_to_browsing_despite_category() -> None:
    assert _route_intent(
        "I'm looking for Women's Shoes, but I'm still exploring.",
        {"category": "women shoes"},
    ) == "browsing"


def test_concrete_constraint_transitions_browsing_session_to_buying() -> None:
    assert _route_intent(
        "For that, what matters is: leather.",
        {"category": "women shoes", "material": "leather"},
        evidence=["leather"],
        previous="browsing",
    ) == "buying"


def test_routes_have_distinct_precision_and_discovery_weights() -> None:
    buying = ROUTE_RRF_WEIGHTS["buying"]
    browsing = ROUTE_RRF_WEIGHTS["browsing"]
    assert buying["bm25"] > buying["dense"]
    assert browsing["dense"] > browsing["bm25"]


def test_profile_distillation_is_bounded_and_allow_listed() -> None:
    assert _profile_terms(
        {
            "preference_tags": ["fit", "comfort", "unknown-private-field"],
            "summary": "This prose must not be copied into retrieval context.",
        }
    ) == ["fit", "sizing", "comfort", "comfortable", "lightweight", "soft"]


def test_reset_stores_short_and_long_term_context_separately() -> None:
    agent = Agent.__new__(Agent)
    agent._sessions = {}
    agent.reset("track4", {"preference_tags": ["material", "style"]})
    state = agent._sessions["track4"]
    assert state["profile_terms"] == ["material", "fabric", "style", "design"]
    assert state["distilled_context"] == "material fabric style design"
    assert state["evidence"] == []
    assert state["candidate_memory"] == []
    assert state["candidate_memory_features"] == {}


def test_live_and_memory_routes_are_interleaved_without_duplicates() -> None:
    assert Agent._interleave_rankings(
        ["live-1", "shared", "live-2"],
        ["memory-1", "shared", "memory-2"],
    ) == ["live-1", "memory-1", "shared", "live-2", "memory-2"]


def test_exact_recall_route_preserves_two_to_one_live_quota() -> None:
    assert Agent._blend_recall_route(
        ["live-1", "live-2", "shared", "live-3"],
        ["exact-1", "shared", "exact-2"],
    ) == ["live-1", "live-2", "exact-1", "shared", "live-3", "exact-2"]


def test_long_evidence_activates_grounded_candidate_memory() -> None:
    agent = Agent.__new__(Agent)
    long_evidence = "an exact catalog-specific requirement long enough to trigger memory"
    agent.products = {"live": {}, "memory": {}}
    agent._searchable_lc = {"live": "", "memory": long_evidence}
    agent._candidate_features = {"live": [1.0, 1.0, 0.0, 0.0]}
    agent._last_fused = {"live": 1.0}
    agent._last_max_fused = 1.0
    merged = agent._merge_candidate_memory(
        ["live"],
        ["live"],
        ["memory"],
        {"memory": [0.5, 0.5, 0.0, 0.0]},
        {"free_text_constraints": [long_evidence]},
    )
    assert merged == ["live", "memory"]
    assert set(merged).issubset(agent.products)


def test_long_verbatim_evidence_can_restore_a_dropped_catalog_product() -> None:
    agent = Agent.__new__(Agent)
    phrase = "a long exact catalog feature phrase that survives punctuation and query drift"
    agent.order = ["dropped", "unrelated"]
    agent.id_to_idx = {asin: index for index, asin in enumerate(agent.order)}
    agent.products = {"dropped": {}, "unrelated": {}}
    agent._searchable_lc = {"dropped": phrase, "unrelated": "different text"}
    agent._exact_evidence_cache = {}
    ranked = agent._exact_evidence_candidates({"free_text_constraints": [phrase]})
    assert ranked == ["dropped"]
