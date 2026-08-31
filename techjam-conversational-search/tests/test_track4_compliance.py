"""Focused regressions for the official Track 4 routing and context pillars."""

from starter.agent import (
    Agent,
    ROUTE_RRF_WEIGHTS,
    _initial_category_context,
    _profile_terms,
    _route_intent,
)


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


def test_constraint_card_prefers_the_source_of_observed_answers() -> None:
    agent = Agent.__new__(Agent)
    agent._constraint_card_cache = {}
    agent.products = {
        "source": {
            "title": "Red cotton shirt",
            "features": ["Machine washable", "Imported", "Pull On closure"],
            "details": {},
        },
        "incidental": {
            "title": "Red cotton shirt",
            "features": ["Different leading feature", "Another feature", "Third feature"],
            "details": {},
            "description": "Machine washable",
        },
    }
    requirements = {"free_text_constraints": ["cotton", "Machine washable"]}
    source = agent._constraint_card_score("source", requirements)
    incidental = agent._constraint_card_score("incidental", requirements)
    # Leading-position matches are scored just under 1.0 (CARD_POSITION_DECAY) so that
    # ties among equally card-consistent candidates break toward the product whose
    # *first* constraints explain the answers. The ordering is the invariant.
    assert source > incidental
    assert source >= 0.9
    assert incidental < 0.75


def test_original_category_phrase_survives_later_clarification_turns() -> None:
    assert _initial_category_context(
        "I'm looking for Women Shoes Boots. A key requirement is: leather."
    ) == "women shoes boots"


def test_precision_slate_expands_and_restarts_after_an_intent_pivot() -> None:
    buying = {"initial_route": "buying", "pivot_seen": False}
    browsing = {"initial_route": "browsing", "pivot_seen": False}
    pivot = {"initial_route": "buying", "pivot_seen": True, "precision_epoch_turn": 3}
    # Inside the precision epoch a single hero result is shown; past it the slate
    # opens to full top_k for recall. A pivot restarts a fresh epoch, so the limit
    # follows the epoch turn rather than the absolute turn number.
    assert Agent._precision_slate_limit(buying, 1, 10) == 1
    assert Agent._precision_slate_limit(buying, 5, 10) == 1
    assert Agent._precision_slate_limit(buying, 6, 10) == 10
    assert Agent._precision_slate_limit(browsing, 4, 10) == 1
    assert Agent._precision_slate_limit(pivot, 8, 10) == 1


def test_explicit_slate_rejection_immediately_restores_top_k_recall() -> None:
    rejected = {
        "initial_route": "browsing",
        "pivot_seen": False,
        "slate_rejected": True,
    }
    assert Agent._precision_slate_limit(rejected, 2, 10) == 10
