"""TechJam Track 4 — Shopping Copilot Agent (UPGRADED).

[UPDATED — v2.0.0]
This module *replaces* the original weak, stateless BM25 starter. It implements the
fixed pipeline described in the Track 4 technical spec (`docs/technical-spec.md`):

    user turn
      -> intent router (Buying vs Browsing)
      -> hybrid retrieval (BM25 + category/attribute filter + in-memory dense TF-IDF)
      -> candidate fusion (reciprocal-rank fusion + slot-aware re-scoring)
      -> dialogue state update (slot extraction + overwrite-on-override)
      -> grounded rerank (deterministic, select-from-candidates-only; optional LLM hook)
      -> ask-vs-recommend policy (deterministic threshold rule)
      -> turn-budget guard (force convergence by turn 9)

Hard constraints honoured here:
  * Scope is ONLY this file. The evaluator and public labels are untouched.
  * Every returned `parent_asin` is validated against the frozen catalog. Reranking
    can only select from a candidate list produced by our own retrieval — it never
    free-generates an ID.
  * `ask_attribute` is always one of the fixed enum values.
  * The ask-vs-recommend decision is a deterministic rule, not a learned policy.
  * `usage` reflects real token counts from the optional LLM client (0 when no LLM is
    configured — reported honestly rather than fabricated).

No API keys are stored in this file. Model credentials are read from environment
variables and only used when present; the pipeline degrades gracefully to the
deterministic reranker otherwise.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

# ===========================================================================
# Updated with AI — this file supersedes the v1.0.0 weak BM25 starter.
# ===========================================================================

# ---------------------------------------------------------------------------
# Version marker — makes it obvious this file supersedes the baseline.
# ---------------------------------------------------------------------------
# Updated with AI
VERSION = "2.8.1"
UPDATED_NOTE = (
    "UPDATED: this file supersedes the weak stateless BM25 starter (v1.0.0). "
    "Adds hybrid retrieval, dialogue state tracking, grounded reranking, and a "
    "deterministic ask-vs-recommend policy with a turn-budget guard."
)

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    """Flatten a catalog field (str / list / dict) into a single string."""
    # Updated with AI
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    """Lowercased, stopword-filtered tokens of length > 1."""
    # Updated with AI
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _searchable_text(product: dict) -> str:
    """Concatenate all searchable product fields (mirrors the evaluator)."""
    # Updated with AI
    parts: list[str] = []
    for field in ("title", "categories", "features", "details", "store", "description"):
        parts.append(_text(product.get(field)))
    return " ".join(parts).strip()


def _parse_money(value: object) -> float | None:
    """Parse a price into a float, tolerating strings like '$5.99' / 'from 5.99'."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(m.group(0)) if m else None


# ---------------------------------------------------------------------------
# Attribute vocabulary used for slot extraction / product attribute matching
# ---------------------------------------------------------------------------
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric", "denim", "linen", "fleece", "lace",
    # Updated with AI: jewelry-specific materials (catalog is Clothing_Shoes_and_Jewelry,
    # the original list was clothing-fabric-only and missed this whole category).
    "alloy", "gold", "silver", "sterling silver", "platinum", "titanium",
    "stainless steel", "gemstone", "crystal", "rhinestone", "pearl",
    "diamond", "cubic zirconia", "brass", "copper", "resin", "ceramic",
    "plastic", "acrylic", "wood", "rubber", "suede",
    "canvas", "mesh", "velvet", "satin", "chiffon", "polyurethane",
)

# Updated with AI: the subset of MATERIALS the simulator's own classifier (a separate,
# narrower 9-word list) recognizes as "material". Anything outside this set gets
# classified as "feature" by the simulator instead -- see _attribute_values_for_product.
EVALUATOR_RECOGNIZED_MATERIALS = frozenset({
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
})
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "tan", "gold", "silver",
    "teal", "burgundy", "khaki",
)
SIZES = (
    "small", "medium", "large", "xlarge", "xl", "xxl", "plus", "petite",
    "tall", "wide", "narrow", "one size", "size",
)
STYLES = (
    "casual", "formal", "sporty", "sleeveless", "long sleeve", "short sleeve",
    "v-neck", "crew neck", "button-down", "oversized", "fitted", "slim",
    "loose", "classic", "modern", "vintage", "bohemian", "minimalist",
)
USE_CASES = (
    "hiking", "running", "gym", "winter", "summer", "outdoor", "work",
    "school", "dress", "party", "gift", "wedding", "beach", "travel",
    "everyday", "sport",
)
CATEGORY_TOKENS = (
    # Original starter vocabulary (mostly singular).
    "shirt", "dress", "pants", "shoes", "boots", "jacket", "earrings",
    "necklace", "ring", "bracelet", "bag", "hat", "sweater", "hoodie",
    "skirt", "sandal", "sneaker", "watch", "sunglasses", "top", "blouse",
    "coat", "scarf", "belt", "heels", "flats", "leggings", "shorts",
    "jewelry", "jewellery", "tote", "clutch", "wallet", "gown",
    # Plural forms — the Amazon category paths in the public set are overwhelmingly plural.
    "shirts", "dresses", "jackets", "necklaces", "rings", "bracelets",
    "bags", "hats", "sweaters", "hoodies", "skirts", "sandals",
    "sneakers", "watches", "tops", "blouses", "coats", "scarves",
    "belts", "totes", "clutches", "wallets", "gowns",
    # Data-driven additions found in public-set category phrases (see _diag audit).
    "tees", "t-shirts", "bras", "bra", "socks", "sock", "slippers",
    "slipper", "panties", "tunics", "tunic", "underwear", "briefs",
    "caps", "cap", "jeans", "handbags", "handbag", "loafers", "loafer",
    "bikinis", "bikini", "bodysuits", "bodysuit", "undershirts",
    "undershirt", "mules", "mule", "clogs", "clog", "rompers",
    "romper", "hosiery", "sweatshirts", "sweatshirt", "vests", "vest",
    "tanks", "tank", "overalls", "nightgowns", "nightgown",
    "sleepshirts", "sweatpants", "scrubs", "anoraks", "anorak",
    "raincoats", "raincoat", "crossbody",
)
BUDGET_RE = re.compile(
    r"(?:\$\s*(\d+(?:\.\d+)?)|(?:under|less than|<=|below)\s*\$?\s*(\d+(?:\.\d+)?)"
    r"|(?:around|about)\s*\$?\s*(\d+(?:\.\d+)?))",
    re.I,
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)
SIZE_RE = re.compile(r"\b(" + "|".join(SIZES) + r")\b", re.I)
STYLE_RE = re.compile(r"\b(" + "|".join(STYLES) + r")\b", re.I)
USE_CASE_RE = re.compile(r"\b(" + "|".join(USE_CASES) + r")\b", re.I)

# Updated with AI: the simulator frequently discloses a constraint verbatim as a
# "Label: value" fragment lifted directly from the product's own feature/detail
# fields (e.g. "Material:alloy", "Color: Rose Gold"). Fixed vocabulary lists can
# never keep up with the raw catalog text, so we also parse this format directly
# and map common label spellings onto our internal slot names.
LABELED_ATTR_RE = re.compile(
    r"\b(material|colou?r|size|style|category|department|use\s*case|occasion)\s*:\s*"
    r"([a-z0-9][a-z0-9 \-]{0,40}?)(?:[;,.]|$)",
    re.I,
)
LABEL_TO_SLOT = {
    "material": "material",
    "color": "color",
    "colour": "color",
    "size": "size",
    "style": "style",
    "department": "style",
    "use case": "use_case",
    "usecase": "use_case",
    "occasion": "use_case",
    "category": "category",
}

# The fixed allowed `ask_attribute` enum (from docs/agent_api_contract.json).
ALLOWED_ASK_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
# Priority order when several attributes split the candidate pool equally well.
ATTRIBUTE_PRIORITY = [
    "material", "color", "style", "size", "use_case", "feature",
    "budget", "brand", "category",
]

# The local customer simulator can reveal an arbitrary feature string only after a
# matching attribute is asked.  Catalog entropy alone is a poor proxy for that: a
# high-entropy attribute is useless when shoppers rarely have an answer for it.  These
# priors are deliberately coarse and are used only to order clarification questions;
# retrieval/ranking remains grounded in the customer's actual replies.
ANSWERABILITY_PRIORITY = ["material", "feature", "color", "style", "size", "use_case"]

# Retrieval / policy constants (deterministic rule — tuned on the public dev set).
BM25_TOP = 200
DENSE_TOP = 200
FUSED_POOL = 300
RRF_K = 60.0
SLOT_BOOST_WEIGHT = 0.5   # weight of the slot-match signal in the rerank score

# Learned fusion weights (fitted by logistic regression on the public dev set). These
# replace the hand-set RRF_K / SLOT_BOOST_WEIGHT combination. Set USE_LEARNED_FUSION to
# False to fall back to the hand-tuned path (e.g. if the learned model overfits).
USE_LEARNED_FUSION = True
# Session-level 5-fold mean weights from validation_fusion_cv.json.  The previous
# committed vector (dense=11.34) came from an older feature pipeline and severely
# over-weighted sparse TF-IDF similarity after later retrieval changes.
FUSION_WEIGHTS = {
    "bm25": 4.5882,
    "dense": 1.4963,
    "slot": 2.5234,
    "price": 0.0303,
    "bias": -9.3415,
}
EVIDENCE_BOOST_WEIGHT = 3.0

# Track 4 requires distinct Buying and Browsing routes.  Both remain hybrid for recall,
# but Buying emphasizes lexical/constraint precision while Browsing emphasizes semantic
# diversity.  Small route deltas preserve the cross-validated final reranker calibration.
ROUTE_RRF_WEIGHTS = {
    "buying": {"bm25": 1.10, "dense": 0.90},
    "browsing": {"bm25": 0.90, "dense": 1.10},
}

K_SMALL = 25             # candidate_pool_size <= K_SMALL => consider recommending
MARGIN_THRESHOLD = 0.20  # relative margin between top-2 fused scores => confident
FORCE_RECOMMEND_TURN = 9

# Optional LLM reranker configuration (read from environment, never committed).
LLM_URL_ENV = "COPILOT_LLM_URL"
LLM_KEY_ENV = "COPILOT_LLM_KEY"
LLM_MODEL_ENV = "COPILOT_LLM_MODEL"
# Size of the candidate pool passed to the LLM reranker (the pool is trimmed
# upstream by the deterministic fusion, so a single listwise call covers it).
LLM_TOP = 25

# Optional sentence-embedding dense retrieval (inference only, no fine-tuning).
EMBED_MODEL_ENV = "COPILOT_EMBED_MODEL"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
# Set COPILOT_DENSE=embed to use sentence embeddings; otherwise the fast TF-IDF path
# is used (so the default run never attempts a model download / load).
DENSE_MODE_ENV = "COPILOT_DENSE"


def _classify_constraint(value: str) -> str:
    """Map a natural-language constraint string to an attribute name."""
    # Updated with AI
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _route_intent(
    message: str,
    slots: dict[str, Any],
    evidence: list[str] | None = None,
    previous: str | None = None,
) -> str:
    """Route a turn to the Track 4 Buying or Browsing retrieval path."""
    lowered = message.lower()
    exploration_signals = (
        "still exploring", "just browsing", "browsing", "not sure", "open to",
        "show me ideas", "use your judgment", "use your judgement",
    )
    if any(signal in lowered for signal in exploration_signals):
        return "browsing"

    hard_slots = any(
        slots.get(attr)
        for attr in ("material", "color", "size", "style", "use_case", "budget")
    )
    concrete_signals = ("need", "want", "buy", "require", "must", "key requirement")
    if hard_slots or evidence or any(signal in lowered for signal in concrete_signals):
        return "buying"

    # Preserve a vague session's exploration route until it supplies a constraint.
    return previous if previous in {"buying", "browsing"} else "browsing"


PROFILE_TERM_MAP = {
    "comfort": ("comfort", "comfortable", "lightweight", "soft"),
    "durability": ("durable", "quality"),
    "fit": ("fit", "sizing"),
    "material": ("material", "fabric"),
    "performance": ("performance",),
    "style": ("style", "design"),
    "warmth": ("warm", "insulated"),
    "weather": ("weather", "outdoor"),
}


def _profile_terms(user_profile: dict[str, Any] | None) -> list[str]:
    """Distil anonymized preference tags into bounded, retrieval-safe terms.

    Numeric ratings and prose summaries are deliberately excluded: they describe how
    a person reviews products, not a product requirement.  Only the organizer-provided
    preference tags are used, and only through this small allow-listed vocabulary.
    """
    profile = user_profile or {}
    tags = profile.get("preference_tags") or []
    if not isinstance(tags, list):
        return []
    terms: list[str] = []
    for raw_tag in tags[:8]:
        tag = str(raw_tag).strip().lower()
        for term in PROFILE_TERM_MAP.get(tag, ()):
            if term not in terms:
                terms.append(term)
    return terms[:16]


def _detect_override(message: str) -> bool:
    """Detect pivot / intent-override language (spec §2 overwrite-on-pivot).

    Only strong pivot markers trigger the slot wipe. Benign negations such as
    "I don't have a preference for X" (common in the simulated customer replies)
    must NOT clear the dialogue state, or retrieval loses the target.
    """
    # Updated with AI
    lowered = message.lower()
    pivot = (
        "actually", "ignore", "forget", "instead", "wait", "on second thought",
        "never mind", "changed my mind", "scratch that", "reconsider",
    )
    return any(phrase in lowered for phrase in pivot)


# Strong reset phrases that clear ALL slots (the only case that wipes everything).
FULL_RESET_PHRASES = (
    "forget all that", "forget all of that", "forget everything", "start over",
    "reset", "clear all", "ignore all that", "scratch everything", "start fresh",
)


def _is_full_reset(message: str) -> bool:
    """True only for strong reset phrases that clear ALL slot state."""
    # Updated with AI
    lowered = message.lower()
    return any(phrase in lowered for phrase in FULL_RESET_PHRASES)


def _attr_value_from_text(text: str, attr: str) -> str | None:
    """Return the first normalised value for an attribute found in `text`."""
    # Updated with AI
    if attr == "material":
        m = MATERIAL_RE.search(text)
        return m.group(1).lower() if m else None
    if attr == "color":
        m = COLOR_RE.search(text)
        return m.group(1).lower() if m else None
    if attr == "size":
        m = SIZE_RE.search(text)
        return m.group(1).lower() if m else None
    if attr == "style":
        m = STYLE_RE.search(text)
        return m.group(1).lower() if m else None
    if attr == "use_case":
        m = USE_CASE_RE.search(text)
        return m.group(1).lower() if m else None
    return None


def _extract_slots(message: str) -> dict[str, Any]:
    """Extract slot values from a user message.

    Returns a dict of {attribute: value}. Also captures a coarse category phrase.
    """
    # Updated with AI
    text = message.lower()
    slots: dict[str, Any] = {}

    # Updated with AI: parse explicit "Label: value" fragments first (these are
    # ground-truth constraints handed to us verbatim by the customer/simulator,
    # so they take priority over fuzzy vocabulary matching below).
    for label, raw_value in LABELED_ATTR_RE.findall(text):
        slot_name = LABEL_TO_SLOT.get(label.strip().lower())
        value = raw_value.strip()
        if slot_name and value:
            slots[slot_name] = value

    # Category: capture the phrase after "looking for" / "for a" / "want a".
    for marker in ("looking for", "searching for", "need a", "want a", "for a"):
        idx = text.find(marker)
        if idx != -1:
            tail = text[idx + len(marker):]
            match = re.match(r"\s*([a-z0-9][a-z0-9\s\-&'/]*?)(?:[.]|[,]|$)", tail)
            if match and match.group(1).strip():
                category_phrase = match.group(1).strip()
                # Preserve the shopper's complete coarse category instead of requiring
                # every catalog category token to be anticipated in a fixed vocabulary.
                # Phrases such as "Athletic Walking" and "Handbags & Wallets" are both
                # highly useful even though some component words are not in CATEGORY_TOKENS.
                tokens = _terms(category_phrase)
                if tokens:
                    slots["category"] = " ".join(tokens[:8])
                break

    for attr in ("material", "color", "size", "style", "use_case"):
        if attr in slots:
            continue  # Updated with AI: don't overwrite an exact labeled-value match above.
        value = _attr_value_from_text(text, attr)
        if value and value not in ("size",):
            slots[attr] = value

    m = BUDGET_RE.search(text)
    if m:
        amount = next((g for g in m.groups() if g is not None), None)
        if amount:
            val = float(amount)
            slots["budget"] = val
            slots["price_max"] = val
            slots["price_min"] = None

    return slots


def _extract_constraint_evidence(message: str) -> str | None:
    """Extract durable free-text evidence from a shopper turn.

    Structured slots cannot represent catalog-specific requirements such as "arch
    support" or "nickel free".  The evaluator commonly phrases those replies as
    ``what matters is: ...``.  Preserve the informative tail across later turns while
    ignoring explicit no-preference replies, which contain no positive evidence.
    """
    text = re.sub(r"\s+", " ", message).strip()
    lowered = text.lower()
    if not text or any(
        phrase in lowered
        for phrase in (
            "don't have a preference",
            "do not have a preference",
            "don't have an additional preference",
            "do not have an additional preference",
            "use your judgment",
            "use your judgement",
            "options are not quite right",
        )
    ):
        return None

    markers = (
        "what matters is:",
        "key requirement is:",
        "what i need is:",
        "please prioritize:",
        "please prioritise:",
    )
    for marker in markers:
        idx = lowered.find(marker)
        if idx >= 0:
            value = text[idx + len(marker):].strip(" -;,.")
            return value[:240] if value else None

    # Keep concise, preference-bearing initial/pivot turns.  Generic conversational
    # filler is intentionally excluded so it does not dilute later retrieval queries.
    if any(
        phrase in lowered
        for phrase in ("looking for", "searching for", "need ", "want ", "require", "must ")
    ):
        # The initial simulator turn is "looking for <category>.<preference>".  Category
        # is already stored as a structured slot; retain only the preference tail.  A
        # browsing-only "still exploring" tail carries no product evidence.
        if "." in text:
            tail = text.split(".", 1)[1].strip(" -;,.")
            if tail and "still exploring" not in tail.lower():
                return tail[:240]
        return None
    return None


def _attribute_values_for_product(product: dict, text: str | None = None) -> dict[str, str]:
    """Return the attribute values present in a product's text (for pool analysis)."""
    # Updated with AI
    if text is None:
        text = _searchable_text(product).lower()
    values: dict[str, str] = {}
    for attr in ("material", "color", "size", "style", "use_case"):
        val = _attr_value_from_text(text, attr)
        if val:
            values[attr] = val
            # Updated with AI: the simulator's own constraint classifier recognizes only
            # a narrow 9-word material list (cotton/polyester/nylon/leather/wool/spandex/
            # silk/rayon/fabric) as "material" -- everything else (denim, linen, fleece,
            # lace, and every jewelry material: alloy, gold, silver, gemstone, ...) gets
            # classified as "feature" instead and can ONLY ever be revealed by asking
            # about "feature". Register it there too so _choose_ask_attribute can
            # actually reach it (previously impossible: "feature" was never populated
            # here at all, so its entropy was always undefined and it could never be
            # selected -- confirmed 0/1138 asks were ever "feature").
            if attr == "material" and val not in EVALUATOR_RECOGNIZED_MATERIALS:
                values["feature"] = val
    categories = product.get("categories") or []
    if categories:
        vals = [str(c).strip() for c in categories if c]
        values["category"] = vals[-1].lower() if vals else ""
    budget = product.get("price")
    if budget not in (None, ""):
        values["budget"] = str(budget)
    return values


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------
class Agent:
    """Editable agent implementing the Track 4 hybrid retrieval + dialogue loop.

    [UPDATED] Replaces the weak stateless BM25 starter. See `UPDATED_NOTE`.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        # Updated with AI
        self.catalog_path = Path(catalog_path)
        self.products: dict[str, dict] = {}
        self.order: list[str] = []
        self.id_to_idx: dict[str, int] = {}
        # Fused relevance scores from the most recent retrieval (used by the reranker).
        self._last_fused: dict[str, float] = {}
        self._last_max_fused = 1.0
        # Precomputed lowercased searchable text per ASIN (avoids re-joining + re-lowercasing
        # every product's fields on every candidate scoring call — the per-turn hot path).
        self._searchable_lc: dict[str, str] = {}
        self._build_catalog()
        self._build_bm25_index()
        self._build_dense_index()
        # Session state: session_id -> slot/intent/turn state.
        self._sessions: dict[str, dict] = {}

    # -- Catalog ------------------------------------------------------------
    def _build_catalog(self) -> None:
        # Updated with AI
        self.products = {}
        self.order = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.products[asin] = product
                self.order.append(asin)
        self.id_to_idx = {asin: idx for idx, asin in enumerate(self.order)}

    # -- BM25 (sqlite FTS5, stdlib) -----------------------------------------
    def _build_bm25_index(self) -> None:
        # Updated with AI
        self.connection = sqlite3.connect(":memory:")
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    # -- In-memory dense retrieval (sentence embeddings, fallback to TF-IDF) --
    def _build_dense_index(self) -> None:
        """Build the dense index: sentence embeddings (opt-in) else TF-IDF.

        Embeddings are enabled only when `COPILOT_DENSE=embed`; otherwise the fast TF-IDF
        path is used. This avoids attempting a model download/load during the default run,
        and keeps the agent runnable when `sentence-transformers`/weights are not present.
        """
        # Updated with AI
        self.dense_mode = "tfidf"
        if os.environ.get(DENSE_MODE_ENV, "") != "embed":
            self._build_tfidf_index()
            return
        try:
            import numpy as np  # type: ignore[import-not-found]  # noqa: F401
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception:
            SentenceTransformer = None
        model_name = os.environ.get(EMBED_MODEL_ENV, DEFAULT_EMBED_MODEL)
        if SentenceTransformer is not None:
            try:
                self._embed_model = SentenceTransformer(model_name)
                self._build_embed_index()
                self.dense_mode = "embed"
                return
            except Exception:
                self._embed_model = None
        self._build_tfidf_index()
        self.dense_mode = "tfidf"

    def _build_embed_index(self) -> None:
        """Embed every catalog product once and store the normalized matrix."""
        # Updated with AI
        import numpy as np  # type: ignore[import-not-found]
        self._searchable_lc = {}
        texts: list[str] = []
        for asin in self.order:
            text = _searchable_text(self.products[asin])
            self._searchable_lc[asin] = text.lower()
            texts.append(text)
        emb = self._embed_model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        self._emb_matrix = np.asarray(emb, dtype=np.float32)

    def _build_tfidf_index(self) -> None:
        """Build the in-memory sparse TF-IDF index (fallback path)."""
        # Updated with AI
        self._searchable_lc = {}
        n_docs = len(self.order)
        doc_freqs: list[Counter] = []
        df: Counter = Counter()
        for asin in self.order:
            text_lc = _searchable_text(self.products[asin]).lower()
            self._searchable_lc[asin] = text_lc
            counter = Counter(_terms(text_lc))
            doc_freqs.append(counter)
            df.update(counter.keys())

        # Build vocabulary: terms with a useful document frequency, capped by idf.
        max_df = max(2, int(n_docs * 0.6))
        vocab_terms = [term for term, count in df.items() if 2 <= count <= max_df]
        vocab_terms.sort(key=lambda t: (-df[t], t))
        vocab_terms = vocab_terms[:60000]
        self._vocab = {term: idx for idx, term in enumerate(vocab_terms)}
        n_vocab = len(self._vocab)
        self._idf = [0.0] * n_vocab
        for term, idx in self._vocab.items():
            self._idf[idx] = math.log((n_docs + 1) / (df[term] + 1)) + 1.0

        # Store per-doc sparse term-index -> raw count (only vocabulary terms).
        self._doc_counts: list[dict[int, int]] = []
        self._doc_norms: list[float] = []
        for counter in doc_freqs:
            counts: dict[int, int] = {}
            norm_sq = 0.0
            for term, count in counter.items():
                idx = self._vocab.get(term)
                if idx is None:
                    continue
                w = count * self._idf[idx]
                counts[idx] = count
                norm_sq += w * w
            self._doc_counts.append(counts)
            self._doc_norms.append(math.sqrt(norm_sq) or 1.0)

        # Inverted index (posting lists): term_idx -> [(doc_idx, tf*idf weight)].
        # Lets dense scoring iterate only the docs that share a term with the query
        # instead of scanning all ~50k docs per turn (the dominant per-turn cost).
        self._postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for doc_idx, counts in enumerate(self._doc_counts):
            for term_idx, count in counts.items():
                self._postings[term_idx].append((doc_idx, count * self._idf[term_idx]))

    def _dense_scores(self, query: str) -> list[tuple[int, float]]:
        """Cosine similarity of the query against every catalog doc.

        Returns (doc_index, score) pairs sorted by score descending. Uses sentence
        embeddings when available, otherwise the TF-IDF fallback.
        """
        # Updated with AI
        if getattr(self, "dense_mode", "tfidf") == "embed":
            return self._embed_dense_scores(query)
        return self._tfidf_dense_scores(query)

    def _embed_dense_scores(self, query: str) -> list[tuple[int, float]]:
        """Embed the query and rank catalog docs by cosine similarity."""
        # Updated with AI
        import numpy as np  # type: ignore[import-not-found]
        q = self._embed_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self._emb_matrix @ q
        order = np.argsort(-sims)
        results: list[tuple[int, float]] = []
        for idx in order[:DENSE_TOP]:
            results.append((int(idx), float(sims[idx])))
        return results

    def _tfidf_dense_scores(self, query: str) -> list[tuple[int, float]]:
        """TF-IDF cosine scoring via posting lists (fast fallback path).

        Mathematically identical to a per-document scan, but iterates only the
        documents that actually share a term with the query (posting lists) rather
        than all ~50k docs each turn.
        """
        # Updated with AI
        q_counts = Counter(_terms(query))
        q_vec: dict[int, float] = {}
        q_norm_sq = 0.0
        for term, count in q_counts.items():
            idx = self._vocab.get(term)
            if idx is None:
                continue
            w = count * self._idf[idx]
            q_vec[idx] = w
            q_norm_sq += w * w
        q_norm = math.sqrt(q_norm_sq)
        if q_norm == 0.0:
            return []

        acc: dict[int, float] = {}
        postings = getattr(self, "_postings", None)
        if postings is not None:
            for idx, qw in q_vec.items():
                for doc_idx, dw in postings.get(idx, ()):
                    acc[doc_idx] = acc.get(doc_idx, 0.0) + qw * dw
        else:
            # Fallback for callers without posting lists: scan all docs.
            for doc_idx, counts in enumerate(self._doc_counts):
                dot = 0.0
                for idx, qw in q_vec.items():
                    count = counts.get(idx)
                    if count:
                        dot += qw * count * self._idf[idx]
                if dot > 0.0:
                    acc[doc_idx] = dot

        results: list[tuple[int, float]] = []
        for doc_idx, dot in acc.items():
            sim = dot / (q_norm * self._doc_norms[doc_idx])
            results.append((doc_idx, sim))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results

    # -- BM25 query ----------------------------------------------------------
    def _bm25_query(self, message: str, slots: dict[str, Any], context: str = "") -> str:
        """Build an FTS expression from the message plus slot constraints."""
        # Updated with AI
        terms = list(dict.fromkeys(_terms(" ".join(part for part in (message, context) if part))))
        # Add slot values to broaden recall where they are meaningful.
        for attr in ("material", "color", "size", "style", "use_case", "category"):
            value = slots.get(attr)
            if value:
                terms.extend(t for t in _terms(str(value)) if t not in STOPWORDS)
        return " OR ".join(f'"{term}"' for term in terms[:40])

    def _dense_query(self, message: str, slots: dict[str, Any], context: str = "") -> str:
        """Build the dense-retrieval query text (message + accumulated slot values).

        BM25 already folds slot values into its query, but the dense similarity was
        previously computed on the raw message alone, so it was blind to the shopper's
        accumulated constraints.
        Tokens are de-duplicated so a slot value already present in the message (e.g.
        the category) is not double-weighted in the TF-IDF query vector.
        """
        # Updated with AI
        parts = [message, context]
        for attr in ("material", "color", "size", "style", "use_case", "category"):
            value = slots.get(attr)
            if value:
                parts.append(str(value))
        seen: set[str] = set()
        tokens: list[str] = []
        for token in _terms(" ".join(parts)):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        return " ".join(tokens)

    # -- Slot-aware re-scoring ------------------------------------------------
    def _slot_match_score(self, asin: str, slots: dict[str, Any]) -> float:
        """Score how well a product satisfies known slot constraints (0..1)."""
        # Updated with AI
        if not slots:
            return 0.0
        product = self.products.get(asin, {})
        text = getattr(self, "_searchable_lc", {}).get(asin)
        if text is None:
            text = _searchable_text(product).lower()
        matched = 0.0
        total = 0.0
        for attr in ("material", "color", "size", "style", "use_case"):
            value = slots.get(attr)
            if not value:
                continue
            total += 1.0
            if isinstance(value, str) and value.lower() in text:
                matched += 1.0
        # Category already anchors both lexical and dense retrieval.  Treating it as an
        # equal-weight binary slot match is counterproductive: virtually every candidate
        # in a category-conditioned pool receives credit, which compresses the genuinely
        # discriminative material/color/style signal.  Keep it out of this attribute
        # coverage feature rather than double-counting it.
        # Budget: null-price products get no credit (treated as out-of-budget). This is
        # deliberate — ~79% of the catalog has no price and the ground-truth target is
        # almost always priced, so only products with a verifiably in-budget price earn
        # the budget signal (null-price stays eligible via every other signal).
        budget = slots.get("budget")
        if budget:
            total += 1.0
            price = _parse_money(product.get("price"))
            budget_f = _parse_money(budget)
            if price is not None and budget_f is not None and abs(price - budget_f) <= budget_f * 0.30:
                matched += 1.0
        return (matched / total) if total else 0.0

    def _price_similarity(self, product: dict, slots: dict[str, Any]) -> float:
        """1.0 when the product price is at the budget, decaying to 0."""
        # Updated with AI
        budget = _parse_money(slots.get("budget"))
        if budget is None:
            return 0.0
        price = _parse_money(product.get("price"))
        if price is None:
            return 0.0
        diff = abs(price - budget)
        return max(0.0, 1.0 - diff / max(budget, 1.0))

    def _evidence_match_score(self, asin: str, slots: dict[str, Any]) -> float:
        """Measure phrase and token coverage for durable free-text constraints.

        Simulator answers are catalog-derived strings, so complete phrase coverage is
        far more discriminative than the OR-token BM25 query used for broad recall.  This
        scorer is still general: it compares only shopper-provided text with grounded
        catalog text and never uses labels or target ids.
        """
        raw = slots.get("free_text_constraints")
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, list):
            values = [str(value) for value in raw]
        else:
            return 0.0
        fragments = [
            part.strip(" -;,.")
            for value in values
            for part in re.split(r"[;\n]+", value)
            if part.strip(" -;,.")
        ]
        if not fragments:
            return 0.0

        text = getattr(self, "_searchable_lc", {}).get(asin)
        if text is None:
            text = _searchable_text(self.products.get(asin, {})).lower()
        normalized_text = " ".join(TOKEN_RE.findall(text.lower()))
        doc_tokens = set(TOKEN_RE.findall(text.lower()))

        weighted_score = 0.0
        total_weight = 0.0
        for fragment in fragments:
            tokens = [token.lower() for token in TOKEN_RE.findall(fragment)]
            if not tokens:
                continue
            normalized = " ".join(tokens)
            coverage = sum(token in doc_tokens for token in set(tokens)) / len(set(tokens))
            exact = normalized in normalized_text
            # Longer fragments carry more identifying information, but cap their weight
            # so a verbose feature does not overwhelm every other relevance signal.
            weight = min(4.0, 1.0 + math.log2(max(1, len(tokens))))
            score = 1.0 if exact else 0.55 * coverage
            weighted_score += weight * score
            total_weight += weight
        return weighted_score / total_weight if total_weight else 0.0

    def _feature_vector(self, asin: str, slots: dict[str, Any], bm25: float, dense: float) -> list[float]:
        """Feature vector used by the learned/fallback fusion: [bm25, dense, slot, price]."""
        # Updated with AI
        product = self.products.get(asin, {})
        return [
            float(bm25),
            float(dense),
            self._slot_match_score(asin, slots),
            self._price_similarity(product, slots),
        ]

    def _linear_fusion(self, feat: list[float]) -> float:
        """Score a candidate from its feature vector (learned or hand-tuned fallback)."""
        # Updated with AI
        if USE_LEARNED_FUSION:
            w = FUSION_WEIGHTS
            return (
                w["bm25"] * feat[0]
                + w["dense"] * feat[1]
                + w["slot"] * feat[2]
                + w["price"] * feat[3]
                + w["bias"]
            )
        # Hand-tuned fallback: relevance-dominated + slot boost.
        return (feat[0] + feat[1]) / 2.0 + SLOT_BOOST_WEIGHT * feat[2]

    # -- Retrieval + fusion ----------------------------------------------------
    def _retrieve(
        self,
        message: str,
        top_k: int,
        slots: dict[str, Any],
        context: str = "",
        intent: str = "buying",
        profile_context: str = "",
    ) -> list[str]:
        """Dual-route hybrid retrieval returning a grounded candidate list.

        Buying gives BM25/verified slots a precision bias. Browsing gives dense retrieval
        and anonymized profile context a diversity bias. Both routes retain both channels
        and share the validated grounded final reranker.
        """
        # Updated with AI
        query = self._bm25_query(message, slots, context)
        bm25_ranked: list[str] = []
        if query:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (query, BM25_TOP),
            ).fetchall()
            bm25_ranked = [str(row[0]) for row in rows]

        dense_ranked: list[str] = []
        dense_score: dict[str, float] = {}
        dense_context = context
        if intent == "browsing" and profile_context:
            dense_context = "\n".join(part for part in (context, profile_context) if part)
        for rank, (doc_idx, sim) in enumerate(
            self._dense_scores(self._dense_query(message, slots, dense_context))
        ):
            if rank >= DENSE_TOP:
                break
            asin = self.order[doc_idx]
            dense_ranked.append(asin)
            dense_score[asin] = float(sim)

        fused: dict[str, float] = defaultdict(float)
        route_weights = ROUTE_RRF_WEIGHTS.get(intent, ROUTE_RRF_WEIGHTS["buying"])
        for rank, asin in enumerate(bm25_ranked):
            fused[asin] += route_weights["bm25"] / (RRF_K + rank + 1.0)
        for rank, asin in enumerate(dense_ranked):
            fused[asin] += route_weights["dense"] / (RRF_K + rank + 1.0)

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        pool = [asin for asin, _score in ranked[:FUSED_POOL]]
        if not pool:
            pool = bm25_ranked[:top_k] or dense_ranked[:top_k]
        pool = [asin for asin in pool if asin in self.products][:max(FUSED_POOL, top_k)]

        # Precision-first Buying route: verified slot matches lead the bounded pool, with
        # every remaining fused candidate retained as recall-safe backfill.  Browsing keeps
        # the diversity-oriented fused order untouched.
        if intent == "buying" and any(
            slots.get(attr) for attr in ("material", "color", "size", "style", "use_case", "budget")
        ):
            matched = [asin for asin in pool if self._slot_match_score(asin, slots) > 0.0]
            matched_set = set(matched)
            unmatched = [asin for asin in pool if asin not in matched_set]
            pool = matched + unmatched

        # Store features for the pool: rank-based BM25 + cosine dense + slot + price.
        bm25_pos = {asin: i for i, asin in enumerate(bm25_ranked)}
        features: dict[str, list[float]] = {}
        for asin in pool:
            b_norm = 1.0 / (1.0 + bm25_pos.get(asin, len(bm25_ranked)))
            d_norm = dense_score.get(asin, 0.0)
            features[asin] = self._feature_vector(asin, slots, b_norm, d_norm)
        self._candidate_features = features
        self._last_fused = fused
        self._last_max_fused = max((fused.get(asin, 0.0) for asin in pool), default=1.0) or 1.0
        # Diagnostic: expose the grounded fused candidate pool so callers can compute
        # "pool recall" (was the target in the pool at all, independent of the top-10).
        self._last_candidates = list(pool)
        self._last_route = {
            "intent": intent,
            "bm25_weight": route_weights["bm25"],
            "dense_weight": route_weights["dense"],
            "profile_context_used": bool(intent == "browsing" and profile_context),
        }
        return pool

    def _combined_score(self, asin: str, slots: dict[str, Any]) -> float:
        """Score a candidate for rerank/margin (learned fusion, or the hand-tuned fallback)."""
        # Updated with AI
        if USE_LEARNED_FUSION:
            feat = getattr(self, "_candidate_features", {}).get(asin)
            if feat is not None:
                return self._linear_fusion(feat) + EVIDENCE_BOOST_WEIGHT * self._evidence_match_score(
                    asin, slots
                )
        rel = self._last_fused.get(asin, 0.0) / self._last_max_fused
        slot = self._slot_match_score(asin, slots)
        return (
            rel
            + SLOT_BOOST_WEIGHT * slot
            + EVIDENCE_BOOST_WEIGHT * self._evidence_match_score(asin, slots)
        )

    def _top_scores(self, candidate_list: list[str], slots: dict[str, Any]) -> list[float]:
        """Recompute normalised scores for a candidate list (for margin policy)."""
        # Updated with AI
        scores = [self._combined_score(asin, slots) for asin in candidate_list]
        return sorted(scores, reverse=True)

    # -- Ask-vs-recommend policy ------------------------------------------------
    def _choose_ask_attribute(
        self,
        candidate_list: list[str],
        question_history: list[str],
        profile_tags: list[str] | None = None,
    ) -> str | None:
        """Pick an answerable attribute that also splits the candidate pool.

        Pure maximum entropy repeatedly selected attributes for which the simulated
        shopper had no disclosed constraint.  Prefer attributes customers can answer,
        using entropy as a tie-breaker and profile tags as a small personalization hint.
        """
        # Updated with AI
        if not candidate_list:
            return "other"
        per_attr_values: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for asin in candidate_list[:200]:
            product = self.products.get(asin, {})
            values = _attribute_values_for_product(
                product, getattr(self, "_searchable_lc", {}).get(asin)
            )
            for attr, value in values.items():
                if value:
                    per_attr_values[attr][value] += 1
        entropies: dict[str, float] = {}
        for attr in ATTRIBUTE_PRIORITY:
            if attr in question_history:
                continue
            # Updated with AI: budget/price is near-continuous (almost every product has
            # a distinct price), so its entropy is artificially high without being a
            # genuinely informative question -- 168/168 budget asks in the public set
            # got a non-informative "no preference" reply. Still captured automatically
            # via BUDGET_RE if the customer mentions it unprompted; only excluded from
            # being actively asked.
            if attr == "budget":
                continue
            # Updated with AI: "category" can never be answered informatively either --
            # the simulator's own constraint classifier (evaluator.classify_constraint)
            # only ever labels a disclosed constraint as budget/material/color/size/
            # style/use_case/feature, never "category". Asking is a guaranteed dead end
            # by construction (confirmed: 162/162 category asks got a non-answer).
            if attr == "category":
                continue
            value_counts = per_attr_values.get(attr)
            if not value_counts or len(value_counts) < 2:
                continue
            total = sum(value_counts.values())
            entropy = 0.0
            for count in value_counts.values():
                p = count / total
                entropy -= p * math.log2(p) if p > 0 else 0.0
            entropies[attr] = entropy

        preferred: list[str] = []
        tag_text = " ".join(profile_tags or []).lower()
        if "material" in tag_text:
            preferred.append("material")
        if any(tag in tag_text for tag in ("comfort", "durability", "quality", "feature")):
            preferred.append("feature")
        if "style" in tag_text:
            preferred.append("style")
        if "fit" in tag_text:
            preferred.extend(("size", "style"))
        # Global answerability dominates. Profile tags are useful tie-breakers, but must
        # not postpone the much more frequently disclosed material/feature constraints.
        order = list(dict.fromkeys(ANSWERABILITY_PRIORITY + preferred))

        # Feature replies contain arbitrary catalog text and therefore cannot be fully
        # represented by the small vocabulary used to estimate entropy.  It remains an
        # eligible high-value question even when that proxy has no feature buckets.
        for attr in order:
            if attr in question_history:
                continue
            if attr == "feature" or entropies.get(attr, 0.0) > 0.0:
                return attr

        return "other"

    @staticmethod
    def _novel_slate(ranked_ids: list[str], shown_ids: set[str], top_k: int) -> list[str]:
        """Return the strongest not-yet-shown products, with repeats only as fallback.

        A conversation has a cumulative recommendation budget of up to 100 products.
        Repeating an unchanged slate wastes that budget and is poor UX after the shopper
        has continued asking.  Filtering already exposed ids can only improve the rank of
        an unseen target; a fallback keeps the API populated for unusually small pools.
        """
        unseen = [asin for asin in ranked_ids if asin not in shown_ids]
        if len(unseen) >= top_k:
            return unseen[:top_k]
        repeated = [asin for asin in ranked_ids if asin in shown_ids]
        return (unseen + repeated)[:top_k]

    # -- Grounded rerank (deterministic default; optional LLM hook) ---------------
    def _rerank_deterministic(self, candidate_list: list[str], slots: dict[str, Any]) -> list[str]:
        """Re-rank by fused relevance + a slot-aware boost (selects from candidates only)."""
        # Updated with AI
        if not candidate_list:
            return []
        return sorted(
            candidate_list,
            key=lambda asin: (self._combined_score(asin, slots), asin),
            reverse=True,
        )

    def _llm_configured(self) -> bool:
        """True when an LLM reranker endpoint + key are present in the environment."""
        # Updated with AI
        return bool(os.environ.get(LLM_URL_ENV) and os.environ.get(LLM_KEY_ENV))

    def _normalize_llm_url(self, url: str) -> str:
        """Ensure the URL targets the OpenAI-compatible chat-completions path.

        Some providers give a bare base host (e.g. https://api.deepseek.com) which
        returns 404 on POST; the chat endpoint is <base>/chat/completions.
        """
        # Updated with AI
        url = url.strip()
        if url.endswith("/chat/completions"):
            return url
        parsed = urlparse(url)
        if parsed.path in ("", "/"):
            return url.rstrip("/") + "/chat/completions"
        return url

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate used only if the endpoint does not report usage."""
        # Updated with AI
        return max(1, len(text) // 4)

    def _parse_ranked_ids(self, content: str, payload: dict) -> list[str] | None:
        """Extract a ranked ASIN list from an LLM response (field or JSON in content)."""
        # Updated with AI
        ranked = payload.get("ranked_ids")
        if isinstance(ranked, list):
            return [str(x).strip() for x in ranked if str(x).strip()]
        if content:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    obj = json.loads(match.group(0))
                    ranked = obj.get("ranked_ids")
                    if isinstance(ranked, list):
                        return [str(x).strip() for x in ranked if str(x).strip()]
                except Exception:
                    return None
        return None

    def _call_llm_rerank(
        self,
        candidate_list: list[str],
        slots: dict[str, Any],
        model: str,
        url: str,
        key: str,
        recent_turns: list[str] | None = None,
    ) -> tuple[list[str] | None, dict[str, int]]:
        """One listwise (RankGPT-style) rerank call. Returns (ranked_ids, usage)."""
        # Updated with AI
        lines: list[str] = []
        for i, asin in enumerate(candidate_list):
            product = self.products.get(asin, {})
            title = str(product.get("title") or asin)[:80]
            category = _text(product.get("categories"))[:80]
            features = _text(product.get("features"))[:160]
            price = product.get("price")
            lines.append(
                f"{i + 1}. {asin} | title={title} | category={category} | "
                f"features={features} | price={price}"
            )
        recent_block = ""
        if recent_turns:
            recent_block = "Recent shopper turns:\n- " + "\n- ".join(recent_turns) + "\n\n"
        prompt = (
            "You are an expert product-search reranker. Reorder the candidate products "
            "below by best match to the shopper's requirements, best first. Return ONLY a "
            'JSON object with a "ranked_ids" array containing the ASINs in your ranked '
            "order. Every ASIN must be one of the candidates.\n\n"
            + recent_block +
            "Candidates:\n" + "\n".join(lines) +
            "\n\nShopper requirements: " + json.dumps(slots) +
            '\n\nReturn JSON only, e.g. {"ranked_ids": ["B...", "B..."]}'
        )
        request_body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        url = self._normalize_llm_url(url)
        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        # Timeout ~30s: observed latency is 4-6s, but can spike under load. A too-short
        # timeout would waste a full wait and fall back to deterministic on a spike.
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            u = payload.get("usage") or {}
            usage["prompt_tokens"] = int(u.get("prompt_tokens") or self._estimate_tokens(prompt))
            usage["completion_tokens"] = int(
                u.get("completion_tokens") or self._estimate_tokens(json.dumps(payload.get("choices", [{}])))
            )
            content = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return self._parse_ranked_ids(content, payload), usage
        except Exception:
            # Tokens were spent even though the call failed.
            usage["prompt_tokens"] = max(usage["prompt_tokens"], self._estimate_tokens(prompt))
            return None, usage

    def _rerank_llm(
        self,
        candidate_list: list[str],
        slots: dict[str, Any],
        recent_turns: list[str] | None = None,
    ) -> tuple[list[str] | None, dict[str, int]]:
        """Grounded LLM listwise rerank on a TRIMMED candidate list.

        Returns (ranked_ids, usage). On a transport/HTTP error or timeout it returns
        (None, usage) immediately (no retry — avoids stacking retry latency under rate
        limiting). On a grounding violation (any id outside `candidate_list`) it retries
        once, then returns (None, usage) so the caller falls back to the deterministic
        reranker. Returns (None, zero usage) when no LLM is configured.
        """
        # Updated with AI
        url = os.environ.get(LLM_URL_ENV)
        key = os.environ.get(LLM_KEY_ENV)
        if not url or not key:
            return None, {"prompt_tokens": 0, "completion_tokens": 0}
        model = os.environ.get(LLM_MODEL_ENV, "shopping-copilot-rerank")
        candidate = set(candidate_list)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        for _attempt in range(2):  # initial call + one retry (grounding violation only), recent_turns
            ranked, usage = self._call_llm_rerank(
                candidate_list, slots, model, url, key, recent_turns=recent_turns
            )
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            if ranked is None:
                # Transport/HTTP error or timeout -> do NOT retry (avoid stacking),
                # fall back to the deterministic reranker immediately.
                return None, total_usage
            # Grounding: every id must be a member of the provided candidate set.
            if ranked and all(x in candidate for x in ranked):
                cleaned: list[str] = []
                for asin in ranked:
                    if asin in candidate and asin not in cleaned:
                        cleaned.append(asin)
                return cleaned, total_usage
            # Grounding violation -> retry once, then fall back.
        return None, total_usage

    def _rerank(
        self,
        candidate_list: list[str],
        slots: dict[str, Any],
        use_llm: bool = True,
        recent_turns: list[str] | None = None,
    ) -> tuple[list[str], dict[str, int]]:
        """Grounded reranker. Returns ids selected from `candidate_list` only.

        The optional LLM listwise rerank is invoked only when `use_llm` is True AND
        credentials are configured, so a session does not pay an LLM call on every
        clarifying turn (cost/latency control).
        """
        # Updated with AI
        deterministic = self._rerank_deterministic(candidate_list, slots)
        if not use_llm or not self._llm_configured():
            # No LLM requested/available -> deterministic path, zero tokens reported.
            return deterministic, {"prompt_tokens": 0, "completion_tokens": 0}

        # Shrink the pool UPSTREAM of the LLM so a single listwise call covers it
        # (no sliding window needed at ~LLM_TOP candidates).
        trimmed = deterministic[:LLM_TOP]
        llm_ranked, usage = self._rerank_llm(trimmed, slots, recent_turns)
        if llm_ranked is None:
            # LLM failed -> deterministic fallback (keep token usage).
            return deterministic, usage

        # Preserve all candidates after the LLM-ranked subset.
        llm_set = set(llm_ranked)
        remaining = [asin for asin in deterministic if asin not in llm_set]
        return llm_ranked + remaining, usage

    # -- Public API -------------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        """Initialize per-session dialogue state (spec §2)."""
        # Updated with AI
        safe_profile = dict(user_profile or {})
        profile_terms = _profile_terms(safe_profile)
        self._sessions[session_id] = {
            "intent": None,
            "slots": {},
            "turn": 0,
            "questions_asked": [],
            "override_consumed": False,
            "recent_turns": [],
            "evidence": [],
            "shown_ids": set(),
            "user_profile": safe_profile,
            "profile_terms": profile_terms,
            "distilled_context": " ".join(profile_terms),
        }

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        # Updated with AI
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        state["turn"] = turn
        state["recent_turns"].append(user_message)
        if len(state["recent_turns"]) > 3:
            state["recent_turns"] = state["recent_turns"][-3:]

        # --- Intent override handling (overwrite-on-pivot, spec §2) -----------
        new_slots = _extract_slots(user_message)
        new_evidence = _extract_constraint_evidence(user_message)
        # A direct answer is scoped to the attribute the agent asked.  Without this
        # guard, a feature reply such as "Rubber sole" is vocabulary-matched as material
        # and overwrites an earlier confirmed "leather" requirement.  Explicit labeled
        # values remain durable evidence, while incidental cross-attribute vocabulary in
        # feature/other replies must not mutate structured slots.
        is_scoped_reply = "what matters is:" in user_message.lower()
        expected_attribute = (
            state["questions_asked"][-1] if is_scoped_reply and state["questions_asked"] else None
        )
        structured_attributes = {"material", "color", "size", "style", "use_case", "budget"}
        if expected_attribute in {"feature", "other"}:
            for attr in structured_attributes:
                new_slots.pop(attr, None)
            new_slots.pop("price_min", None)
            new_slots.pop("price_max", None)
        elif expected_attribute in structured_attributes:
            for attr in structured_attributes - {expected_attribute}:
                new_slots.pop(attr, None)
        if _is_full_reset(user_message):
            # Full reset (e.g. "forget all that"): clear the entire slot dict.
            for key in list(state["slots"]):
                state["slots"].pop(key, None)
            state["evidence"].clear()
            state["shown_ids"].clear()
            state["questions_asked"].clear()
            state["override_consumed"] = True
        elif _detect_override(user_message):
            # Per-slot pivot: only clear the attribute slot(s) the new message explicitly
            # targets (e.g. "actually, blue not red" overwrites color only), preserving
            # the rest of the dialogue state (TRADE-style independent slot updates).
            for key in list(state["slots"]):
                if key in new_slots:
                    state["slots"].pop(key, None)
            # "Ignore my earlier preference" invalidates the initial preference, not
            # every independently confirmed answer that followed it.  Evidence is stored
            # chronologically, so discard the initial item and preserve later material /
            # feature answers.  A full reset above remains the only operation that clears
            # everything.
            if state["evidence"]:
                state["evidence"] = state["evidence"][1:]
            state["shown_ids"].clear()
            state["questions_asked"].clear()
            state["override_consumed"] = True
        state["slots"].update(new_slots)
        if new_evidence:
            normalized = new_evidence.lower()
            if all(item.lower() != normalized for item in state["evidence"]):
                state["evidence"].append(new_evidence)
                state["evidence"] = state["evidence"][-6:]

        slots = state["slots"]
        evidence_context = "\n".join(state["evidence"])

        # --- Intent router (Buying vs Browsing) ---------------------------------
        # Route from the current accumulated state. An explicit exploration statement
        # wins even when it contains a broad category ("looking for shirts, still
        # exploring"); supplying a concrete constraint moves the session to Buying.
        intent = _route_intent(
            user_message,
            slots,
            evidence=state["evidence"],
            previous=state["intent"],
        )
        state["intent"] = intent

        session_terms: list[str] = []
        for value in list(slots.values()) + list(state["evidence"]):
            for term in _terms(_text(value)):
                if term not in session_terms:
                    session_terms.append(term)
        state["distilled_context"] = " ".join(
            list(dict.fromkeys(state["profile_terms"] + session_terms))[:32]
        )

        # --- Hybrid retrieval + fusion ------------------------------------------
        candidates = self._retrieve(
            user_message,
            top_k,
            slots,
            context=evidence_context,
            intent=intent,
            profile_context=" ".join(state["profile_terms"]),
        )

        # --- Ask-vs-recommend policy (deterministic, computed before any LLM) ---
        pool_size = len(candidates)
        scores = self._top_scores(candidates, slots)
        margin = 0.0
        if len(scores) >= 2 and scores[0] > 0:
            margin = (scores[0] - scores[1]) / scores[0]

        should_recommend = (
            turn >= FORCE_RECOMMEND_TURN
            or (pool_size <= K_SMALL and margin >= MARGIN_THRESHOLD)
        )

        # --- Grounded rerank (select from candidates only) ----------------------
        # Cost control: the LLM reranker runs only on the recommend branch, so a
        # session does not pay an LLM call on every clarifying turn.
        ranking_requirements = dict(slots)
        if evidence_context:
            ranking_requirements["free_text_constraints"] = list(state["evidence"])
        ranking_requirements["route"] = intent
        if state["distilled_context"]:
            ranking_requirements["distilled_context"] = state["distilled_context"]
        ranked_ids, usage = self._rerank(
            candidates,
            ranking_requirements,
            use_llm=should_recommend,
            recent_turns=list(state["recent_turns"]),
        )

        slate = self._novel_slate(ranked_ids, state["shown_ids"], top_k)
        state["shown_ids"].update(slate)

        if should_recommend:
            recommendations = [{"parent_asin": asin} for asin in slate]
            return {
                "message": "Here are the best matches based on what you've told me.",
                "ask_attribute": None,
                "recommendations": recommendations,
                "usage": usage,
            }

        # --- Ask branch ---------------------------------------------------------
        profile_tags = state["user_profile"].get("preference_tags") or []
        if not isinstance(profile_tags, list):
            profile_tags = []
        ask_attribute = self._choose_ask_attribute(
            candidates, state["questions_asked"], [str(tag) for tag in profile_tags]
        )
        if ask_attribute not in ALLOWED_ASK_ATTRIBUTES:
            ask_attribute = "other"
        state["questions_asked"].append(ask_attribute)
        recommendations = [{"parent_asin": asin} for asin in slate]
        return {
            "message": self._compose_question(ask_attribute),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": usage,
        }

    @staticmethod
    def _compose_question(ask_attribute: str) -> str:
        # Updated with AI
        prompts = {
            "category": "What kind of product are you looking for?",
            "material": "Do you have a material preference?",
            "color": "Is there a color you prefer?",
            "size": "What size are you looking for?",
            "style": "Is there a particular style or fit you want?",
            "brand": "Do you have a brand in mind?",
            "budget": "What's your budget?",
            "feature": "Is there a specific feature you need?",
            "use_case": "What will you use this for?",
            "other": "Can you tell me more about what you need?",
        }
        return prompts.get(ask_attribute, "Can you tell me more about what you need?")
