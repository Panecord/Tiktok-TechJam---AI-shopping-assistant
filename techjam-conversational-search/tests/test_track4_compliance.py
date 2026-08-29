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
