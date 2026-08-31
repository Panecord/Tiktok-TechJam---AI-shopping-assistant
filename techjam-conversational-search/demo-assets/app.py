"""TechJam Track 4 — Shopping Copilot metrics dashboard.

A single-file Streamlit app to trigger the local evaluator and visualise `results.json`.

Run from the repo root (where `evaluator/` and `data/` live):

    streamlit run app.py

The evaluator is invoked as `python -m evaluator.local_evaluator` (cwd = this file's
directory), which is the same command the README documents.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopping Copilot - Metrics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def _find_repo_root() -> Path:
    """Locate the repo root (the directory containing ``data/`` and ``evaluator/``).

    ``app.py`` now lives in ``<repo>/demo-assets/``, so we walk up from here. This keeps
    every evaluator/data path dynamic and lets the dashboard run from anywhere.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / "data").is_dir() and (candidate / "evaluator").is_dir():
            return candidate
    return here.parent


REPO_ROOT = _find_repo_root()
DEFAULT_RESULTS = REPO_ROOT / "results.json"

# Built-in baseline -> current version history. Used when the user doesn't upload one, so a
# live demo always shows the improvement from the weak BM25 starter (v1.0.0) to the current
# agent without needing to paste a CSV/JSON first.
DEFAULT_VERSION_HISTORY = [
    {"version": "v1.0.0", "description": "Weak stateless BM25 starter (the provided baseline).",
     "hit_rate_at_10": 0.125, "mrr": 0.068034, "mttc": 9.81, "efficiency": 0.119,
     "technical_score": 0.10671, "total_tokens": 0},
    {"version": "v2.0.0", "description": "Hybrid retrieval + dialogue state + grounded rerank + ask/recommend policy.",
     "hit_rate_at_10": 0.225, "mrr": 0.068581, "mttc": 9.30, "efficiency": 0.17,
     "technical_score": 0.167074, "total_tokens": 0},
    {"version": "v2.6.0", "description": "Learned fusion weights + per-slot pivots + pool-recall diagnostics.",
     "hit_rate_at_10": 0.545, "mrr": 0.422623, "mttc": 8.255, "efficiency": 0.2745,
     "technical_score": 0.454187, "total_tokens": 0},
    {"version": "v2.8.0", "description": "Ask-attribute reachability + LLM rerank context.",
     "hit_rate_at_10": 0.650, "mrr": 0.4787, "mttc": 6.470, "efficiency": 0.453,
     "technical_score": 0.5592, "total_tokens": 0},
    {"version": "v2.9.0", "description": "Synonym-aware matching + re-fitted fusion weights.",
     "hit_rate_at_10": 0.715, "mrr": 0.4938, "mttc": 5.805, "efficiency": 0.5195,
     "technical_score": 0.6095, "total_tokens": 0},
    {"version": "v2.10.0", "description": "Dual-route intent + free-text constraint evidence.",
     "hit_rate_at_10": 0.995, "mrr": 0.58419, "mttc": 2.73, "efficiency": 0.827,
     "technical_score": 0.8382, "total_tokens": 0},
    {"version": "v2.11.0", "description": "Candidate-memory + exact-evidence recall.",
     "hit_rate_at_10": 1.0, "mrr": 0.583518, "mttc": 2.70, "efficiency": 0.83,
     "technical_score": 0.841055, "total_tokens": 0},
    {"version": "v2.12.0", "description": "Rank-first precision slates.",
     "hit_rate_at_10": 1.0, "mrr": 0.939048, "mttc": 3.325, "efficiency": 0.7675,
     "technical_score": 0.935214, "total_tokens": 0},
    {"version": "v2.12.1", "description": "Pareto refinement — current.",
     "hit_rate_at_10": 1.0, "mrr": 0.948458, "mttc": 3.315, "efficiency": 0.7685,
     "technical_score": 0.938237, "total_tokens": 0},
]

SCENARIO_ORDER = ["buying", "browsing", "intent_override", "boundary"]
SCENARIO_LABEL = {
    "buying": "Buying",
    "browsing": "Browsing",
    "intent_override": "Intent Override",
    "boundary": "Boundary",
}

METRIC_LABELS = {
    "hit_rate_at_10": "Hit Rate@10",
    "mrr": "MRR",
    "mttc": "MTTC",
    "efficiency": "Efficiency",
    "technical_score": "Technical Score",
    "tokens": "Tokens",
    "total_tokens": "Total Tokens",
}

METRIC_HELP = {
    "hit_rate_at_10": (
        "Fraction of the public sessions where the target product appears in the scored "
        "Top 10 within 10 turns. 0–1; higher is better."
    ),
    "mrr": (
        "Mean Reciprocal Rank — the average of 1 / (rank of the target) across sessions, "
        "with a miss contributing 0. Rewards ranking the target higher when it is found."
    ),
    "mttc": (
        "Mean Turn to first hit — the average turn number at which the target is found; a "
        "miss is assigned turn 11. Lower is better."
    ),
    "efficiency": (
        "clip((11 − MTTC) / 10, 0, 1) — rewards hitting early. 1.0 if the target is found on "
        "turn 1, falling to 0 for a turn-11 miss."
    ),
    "technical_score": (
        "Official competition score = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency."
    ),
    "total_tokens": (
        "Playback/completion tokens reported by the model client. A feasibility/cost metric — "
        "not part of Technical Score."
    ),
}

DELTA_COLORS = {"normal": "normal", "inverse": "inverse"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _first_value(mapping: dict, *keys):
    """Return the first present, non-None value among `keys` (case-insensitive)."""
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        val = lowered.get(str(key).lower())
        if val is not None:
            return val
    return None


def _metric(value: float) -> float:
    """Coerce a metric to float, tolerating None / 'n/a'."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# results.json parsing
# ---------------------------------------------------------------------------
def parse_aggregate(data: dict) -> dict:
    """Normalise the top-level aggregate fields to canonical keys."""
    usage = data.get("reported_token_usage") or {}
    prompt_tokens = int(_first_value(usage, "prompt_tokens")
                        or _first_value(data, "prompt_tokens") or 0)
    completion_tokens = int(_first_value(usage, "completion_tokens")
                            or _first_value(data, "completion_tokens") or 0)
    total_tokens = int(_first_value(data, "total_tokens")
                       or _first_value(usage, "total_tokens")
                       or (prompt_tokens + completion_tokens))
    agg = {
        "hit_rate_at_10": _metric(_first_value(
            data, "hit_rate_at_10", "HitRate@10", "hit_rate", "hr", "HitRate")),
        "mrr": _metric(_first_value(data, "mrr", "MRR")),
        "mttc": _metric(_first_value(data, "mttc", "MTTC", "mtc")),
        "efficiency": _metric(_first_value(data, "efficiency", "Efficiency", "eff")),
        "technical_score": _metric(_first_value(
            data, "recommended_technical_score", "technical_score", "TechnicalScore",
            "score")),
    }
    agg["prompt_tokens"] = prompt_tokens
    agg["completion_tokens"] = completion_tokens
    agg["total_tokens"] = total_tokens
    if pd.isna(agg["efficiency"]):
        agg["efficiency"] = max(0.0, min(1.0, (11.0 - agg["mttc"]) / 10.0))
    if pd.isna(agg["technical_score"]):
        agg["technical_score"] = 0.50 * agg["hit_rate_at_10"] + 0.30 * agg["mrr"] + 0.20 * agg["efficiency"]
    return agg


def parse_sessions(data: dict) -> pd.DataFrame:
    """Normalise the per-session list into a tidy DataFrame."""
    rows = data.get("sessions") or []
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=[
            "session_id", "scenario", "hit", "hit_turn", "best_rank",
            "reciprocal_rank", "turns_used",
        ])
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        sid = _first_value(r, "sample_id", "session_id", "id")
        scenario = str(_first_value(r, "scenario_type", "scenario") or "other")
        hit = bool(_first_value(r, "hit", "is_hit") or False)
        first_hit = _first_value(r, "first_hit_turn", "hit_turn", "turn")
        best_rank = _first_value(r, "best_rank", "rank")
        rr = _metric(_first_value(r, "reciprocal_rank", "rr"))
        # turns_used: the turn the target was found for hits, else the full 10-turn budget.
        if pd.isna(_metric(first_hit)):
            turns_used = 10
        else:
            turns_used = int(first_hit)
        out.append({
            "session_id": sid,
            "scenario": scenario,
            "hit": hit,
            "hit_turn": int(first_hit) if not pd.isna(_metric(first_hit)) else None,
            "best_rank": best_rank,
            "reciprocal_rank": rr,
            "turns_used": turns_used,
        })
    return pd.DataFrame(out)


def parse_scenario_metrics(data: dict) -> pd.DataFrame:
    """Return a tidy DataFrame of per-scenario metrics."""
    scenarios = data.get("scenario_metrics") or {}
    if not isinstance(scenarios, dict) or not scenarios:
        return pd.DataFrame(columns=["scenario", "sample_count", "hit_rate_at_10", "mrr", "mttc"])
    out = []
    for name, m in scenarios.items():
        if not isinstance(m, dict):
            continue
        out.append({
            "scenario": name,
            "sample_count": _metric(_first_value(m, "sample_count", "n", "count")),
            "hit_rate_at_10": _metric(_first_value(m, "hit_rate_at_10", "HitRate@10", "hr")),
            "mrr": _metric(_first_value(m, "mrr", "MRR")),
            "mttc": _metric(_first_value(m, "mttc", "MTTC")),
        })
    df = pd.DataFrame(out)
    # Keep a consistent, human-friendly ordering and label.
    df["scenario_label"] = df["scenario"].map(lambda s: SCENARIO_LABEL.get(s, s))
    df["scenario"] = pd.Categorical(
        df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return df.sort_values("scenario").reset_index(drop=True)


def _load_results(data: dict) -> dict:
    """Parse and cache a raw results dict into canonical pieces."""
    return {
        "aggregate": parse_aggregate(data),
        "sessions": parse_sessions(data),
        "scenario_metrics": parse_scenario_metrics(data),
        "sample_count": data.get("sample_count")
        or (len(data.get("sessions") or []) if isinstance(data.get("sessions"), list) else 0),
    }


def load_results_from_path(path: Path) -> dict | None:
    try:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        st.warning(f"Could not read {path}: {exc}")
        return None
    return _load_results(data)


def load_results_from_bytes(raw: bytes) -> dict | None:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        st.warning(f"Could not parse uploaded file: {exc}")
        return None
    return _load_results(data)


# ---------------------------------------------------------------------------
# Version-history parsing (CSV / JSON)
# ---------------------------------------------------------------------------
def _norm_version_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a version-history frame to canonical column names (case-insensitive)."""
    mapping = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("version", "ver", "name", "release"):
            mapping[col] = "version"
        elif c in ("description", "changelog", "notes", "note", "what changed", "change", "summary"):
            mapping[col] = "description"
        elif c in ("hit_rate_at_10", "hitrate", "hit rate", "hr", "hitrate@10", "hit"):
            mapping[col] = "hit_rate_at_10"
        elif c in ("mrr",):
            mapping[col] = "mrr"
        elif c in ("mttc", "mtc"):
            mapping[col] = "mttc"
        elif c in ("efficiency", "eff"):
            mapping[col] = "efficiency"
        elif c in ("technical_score", "recommended_technical_score", "score", "technical"):
            mapping[col] = "technical_score"
        elif c in ("prompt_tokens", "prompt"):
            mapping[col] = "prompt_tokens"
        elif c in ("completion_tokens", "completion"):
            mapping[col] = "completion_tokens"
        elif c in ("total_tokens", "tokens", "token_usage"):
            mapping[col] = "total_tokens"
    df = df.rename(columns=mapping)
    keep = [c for c in ("version", "description", "hit_rate_at_10", "mrr", "mttc",
                        "efficiency", "technical_score", "prompt_tokens",
                        "completion_tokens", "total_tokens") if c in df.columns]
    return df[keep] if keep else df


def parse_version_history(source) -> pd.DataFrame:
    """Accept a filename/path, file-like, bytes, dict, list, or CSV/JSON string."""
    df = None
    if isinstance(source, (pd.DataFrame,)):
        df = source.copy()
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists():
            raw = p.read_bytes()
            df = _read_csv_or_json(raw, p.suffix)
        else:
            txt = str(source).strip()
            if txt.startswith("{"):
                df = _read_json_text(txt)
            elif "," in txt and "\n" in txt:
                df = pd.read_csv(pd.io.common.StringIO(txt))
            else:
                st.error("Could not interpret the version-history input.")
                return None
    elif isinstance(source, (bytes, bytearray)):
        df = _read_csv_or_json(bytes(source))
    elif isinstance(source, (dict, list)):
        df = _read_json_obj(source)
    elif hasattr(source, "read"):
        raw = source.read()
        df = _read_csv_or_json(raw)
    else:
        st.error("Unsupported version-history source.")
        return None

    if df is None or df.empty:
        return df
    df = _norm_version_columns(df)
    if "version" not in df.columns:
        st.warning("Version history is missing a 'version' column.")
    return df


def _read_csv_or_json(raw: bytes, suffix: str = "") -> pd.DataFrame | None:
    try:
        if suffix.lower() in (".json",) or raw.lstrip().startswith(b"[") or raw.lstrip().startswith(b"{"):
            return _read_json_obj(json.loads(raw.decode("utf-8")))
        return pd.read_csv(pd.io.common.StringIO(raw.decode("utf-8")))
    except Exception as exc:
        st.error(f"Could not parse version-history file: {exc}")
        return None


def _read_json_text(txt: str) -> pd.DataFrame | None:
    try:
        return _read_json_obj(json.loads(txt))
    except Exception as exc:
        st.error(f"Could not parse JSON: {exc}")
        return None


def _read_json_obj(obj) -> pd.DataFrame:
    if isinstance(obj, dict):
        for key in ("versions", "history", "data", "rows"):
            if key in obj and isinstance(obj[key], list):
                return pd.DataFrame(obj[key])
        # A single dict: try to wrap it.
        return pd.DataFrame([obj])
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Metric formatting helpers
# ---------------------------------------------------------------------------
def _fmt_pct(v: float) -> str:
    return "—" if pd.isna(v) else f"{v * 100:.1f}%"


def _fmt_num(v: float, digits: int = 3) -> str:
    return "—" if pd.isna(v) else f"{v:.{digits}f}"


def _fmt_delta(delta: float) -> str:
    if pd.isna(delta):
        return None
    if abs(delta) < 1e-9:
        return "0"
    return f"{delta:+.3f}"


# ---------------------------------------------------------------------------
# Version progression chart
# ---------------------------------------------------------------------------
def version_chart(df: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    if df.empty or not selected:
        return fig
    desc = df["description"].fillna("") if "description" in df.columns else pd.Series([""] * len(df))
    for metric in selected:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        fig.add_trace(go.Scatter(
            x=df["version"],
            y=values,
            mode="lines+markers",
            name=METRIC_LABELS.get(metric, metric),
            customdata=desc.tolist(),
            hovertemplate="<b>%{x}</b><br>%{customdata}<br>%{y:.4f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        hovermode="x unified",
        yaxis_title="Value",
        xaxis_title="Version",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Dark "receipt" theme (inspired by demo-assets/results_dashboard.html)
# ---------------------------------------------------------------------------
_THEME_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
  :root { --bg:#0B0E14; --panel:#12161F; --panel-2:#171C27; --ink:#F2EFE9;
          --muted:#7B8494; --amber:#E8A33D; --amber-dim:#6b5228; --cyan:#5EC8D8;
          --line:#262C3A; --good:#7FBF7F; }
  /* Dark theme + fonts */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stMarkdown,
  .stDataFrame, [data-testid="stSidebar"] { background: var(--bg); color: var(--ink);
      font-family:'Space Grotesk', sans-serif; }
  .stApp { background: var(--bg); }
  h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { font-family:'Space Grotesk', sans-serif; }
  .mono, .sc-hero-title, .sc-metric-label, .sc-metric-value, .sc-compare-label,
  .sc-compare-val, .sc-scenario-n, .sc-scenario-stat, .sc-compare-row { font-family:'IBM Plex Mono', monospace; }
  .sc-metric-value { font-weight: 700; }

  /* Hero + cards fade / slide in */
  @keyframes sc-fadeUp { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform:none; } }
  .sc-hero { background: var(--panel); border:1px solid var(--line); border-radius:4px;
             padding:24px 28px; margin-top:8px; animation: sc-fadeUp .6s cubic-bezier(.16,.8,.3,1); }
  .sc-hero-title { font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:.14em;
                   text-transform:uppercase; color:var(--muted); border-bottom:1px dashed var(--line);
                   padding-bottom:12px; margin-bottom:16px; display:flex; justify-content:space-between; }
  .sc-metrics { display:grid; grid-template-columns:repeat(5,1fr); gap:4px; }
  .sc-metric { padding:4px 10px 4px 0; border-right:1px dashed var(--line); }
  .sc-metric:last-child { border-right:none; }
  .sc-metric-label { font-family:'IBM Plex Mono',monospace; font-size:10.5px; letter-spacing:.08em;
                     text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
  .sc-metric-value { font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600;
                     color:var(--amber); font-variant-numeric:tabular-nums; }
  .sc-metric-value.cyan { color:var(--cyan); }
  .sc-compare { display:flex; flex-direction:column; gap:14px; margin-top:8px; }
  .sc-compare-row { display:grid; grid-template-columns:120px 1fr 90px; align-items:center; gap:14px; }
  .sc-compare-label { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); }
  .sc-track { position:relative; height:20px; background:var(--panel-2); border-radius:3px;
              overflow:hidden; border:1px solid var(--line); }
  /* Animated bar growth */
  .sc-fill { position:absolute; left:0; top:0; bottom:0; border-radius:3px 0 0 3px;
             width: var(--w, 0%); animation: sc-grow 1.2s cubic-bezier(.16,.8,.3,1) forwards; }
  @keyframes sc-grow { from { width: 0%; } to { width: var(--w, 0%); } }
  .sc-fill.base { background:linear-gradient(90deg,#4a3a24,var(--amber-dim)); }
  .sc-fill.final { background:linear-gradient(90deg,var(--amber-dim),var(--amber)); }
  .sc-fill.final.cyan { background:linear-gradient(90deg,#245a63,var(--cyan)); }
  .sc-compare-val { font-family:'IBM Plex Mono',monospace; font-size:13px; text-align:right; color:var(--ink); }
  .sc-scenarios { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:8px; }
  .sc-scenario-card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px;
                      animation: sc-fadeUp .6s cubic-bezier(.16,.8,.3,1); }
  .sc-scenario-card:nth-child(2) { animation-delay:.08s; }
  .sc-scenario-card:nth-child(3) { animation-delay:.16s; }
  .sc-scenario-card:nth-child(4) { animation-delay:.24s; }
  .sc-scenario-name { font-size:14px; font-weight:600; margin-bottom:2px; }
  .sc-scenario-n { font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--muted); margin-bottom:12px; }
  .sc-scenario-stat { display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace;
                      font-size:12px; padding:5px 0; border-top:1px dashed var(--line); }
  .sc-scenario-stat span:first-child { color:var(--muted); }
  .sc-scenario-stat span:last-child { color:var(--ink); }
</style>
"""


def _hero_html(agg: dict) -> str:
    """Receipt-style metric hero (5 columns), matching the HTML inspiration."""
    values = [
        ("Hit Rate@10", _fmt_pct(agg["hit_rate_at_10"]), ""),
        ("MRR", _fmt_pct(agg["mrr"]), ""),
        ("MTTC", _fmt_num(agg["mttc"], 3), "cyan"),
        ("Efficiency", _fmt_pct(agg["efficiency"]), ""),
        ("Technical Score", _fmt_num(agg["technical_score"], 3), ""),
    ]
    cells = "".join(
        f'<div class="sc-metric"><div class="sc-metric-label">{label}</div>'
        f'<div class="sc-metric-value {cls}">{val}</div></div>'
        for label, val, cls in values
    )
    return (
        '<div class="sc-hero">'
        '<div class="sc-hero-title"><span>final results.json</span>'
        '<span>public set · n=200</span></div>'
        f'<div class="sc-metrics">{cells}</div>'
        '</div>'
    )


def _bar_width(metric: str, value: float) -> float:
    """Bar width for a metric. MTTC is lower-is-better, so scale by its 0..11 range."""
    if metric == "mttc":
        return max(0.0, min(100.0, value / 11.0 * 100.0))
    return max(0.0, min(100.0, value * 100.0))


def _baseline_final_html(agg: dict, base: dict) -> str:
    """Baseline (v1.0.0) vs. current before/after bars, mirroring the HTML inspiration."""
    def row(label, metric, fmt, lower_better=False):
        b = base.get(metric)
        c = agg.get(metric)
        if b is None or c is None or pd.isna(b) or pd.isna(c):
            return ""
        cls = " final cyan" if lower_better else " final"
        return (
            f'<div class="sc-compare-row"><div class="sc-compare-label">{label}</div>'
            f'<div><div class="sc-track"><div class="sc-fill base" style="--w:{_bar_width(metric, b):.1f}%"></div></div></div>'
            f'<div class="sc-compare-val">{fmt(b)}</div></div>'
            f'<div class="sc-compare-row"><div class="sc-compare-label"></div>'
            f'<div><div class="sc-track"><div class="sc-fill{cls}" style="--w:{_bar_width(metric, c):.1f}%"></div></div></div>'
            f'<div class="sc-compare-val">{fmt(c)}</div></div>'
        )

    parts = [
        row("Hit Rate@10", "hit_rate_at_10", lambda v: f"{v:.3f}"),
        row("MRR", "mrr", lambda v: f"{v:.3f}"),
        row("MTTC", "mttc", lambda v: f"{v:.3f}", lower_better=True),
        row("Efficiency", "efficiency", lambda v: f"{v:.3f}"),
        row("Technical Score", "technical_score", lambda v: f"{v:.3f}"),
    ]
    return '<div class="sc-compare">' + "".join(parts) + "</div>"


def _scenario_cards_html(scen_df: pd.DataFrame) -> str:
    """Grid of per-scenario cards (HR@10 / MRR / MTTC)."""
    cards = []
    for _, r in scen_df.iterrows():
        name = str(r.get("scenario_label") or r.get("scenario", "other"))
        n = int(r.get("sample_count")) if not pd.isna(r.get("sample_count")) else 0

        def _p(v):
            return "—" if pd.isna(v) else f"{v * 100:.1f}%"

        def _n(v):
            return "—" if pd.isna(v) else f"{v:.2f}"

        cards.append(
            '<div class="sc-scenario-card">'
            f'<div class="sc-scenario-name">{name}</div>'
            f'<div class="sc-scenario-n">n={n}</div>'
            f'<div class="sc-scenario-stat"><span>Hit Rate@10</span><span>{_p(r.get("hit_rate_at_10"))}</span></div>'
            f'<div class="sc-scenario-stat"><span>MRR</span><span>{_p(r.get("mrr"))}</span></div>'
            f'<div class="sc-scenario-stat"><span>MTTC</span><span>{_n(r.get("mttc"))}</span></div>'
            '</div>'
        )
    return '<div class="sc-scenarios">' + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# Run evaluator (subprocess with live log + progress)
# ---------------------------------------------------------------------------
RUN_MODES = {
    "local": {
        "label": "Local (deterministic)",
        "kind": "module",
        "args": ["evaluator.local_evaluator"],
        "out": "results.json",
        "desc": "python -m evaluator.local_evaluator — no LLM, no tokens",
    },
    "llm": {
        "label": "LLM (with model)",
        "kind": "script",
        "script": "_validate_llm.py",
        "out": "validation_llm.json",
        "desc": "_validate_llm.py — grounded LLM listwise rerank (uses COPILOT_LLM_* env)",
    },
}

_PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s+done.*?eta=(\d+)s")


def _evaluator_cmd(repo_root: Path, mode: str) -> list[str]:
    """Build the evaluator command, resolving any script path against the repo root."""
    cfg = RUN_MODES[mode]
    if cfg["kind"] == "module":
        return [sys.executable, "-m", *cfg["args"]]
    return [sys.executable, str(repo_root / cfg["script"])]


def run_evaluator(repo_root: Path, log_placeholder, mode: str) -> dict | None:
    """Launch the chosen evaluator as a subprocess, stream its log and update a progress bar."""
    cfg = RUN_MODES[mode]
    cmd = _evaluator_cmd(repo_root, mode)
    out_name = cfg["out"]
    log_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        st.error(f"Failed to start evaluator: {exc}")
        return None

    st.info(f"Running `{' '.join(cmd)}` from `{repo_root}` …")
    progress = st.progress(0.0, text="Starting …")
    start = time.time()
    seen_progress = False
    last_frac = 0.0
    for line in proc.stdout:
        log_lines.append(line.rstrip())
        log_placeholder.code("\n".join(log_lines[-400:]), language="text")
        m = _PROGRESS_RE.search(line)
        if m:
            seen_progress = True
            cur, total, eta = int(m.group(1)), int(m.group(2)), int(m.group(3))
            last_frac = min(1.0, cur / max(total, 1))
            progress.progress(
                last_frac,
                text=f"Session {cur}/{total} · ETA ≈ {eta}s",
            )
        else:
            elapsed = time.time() - start
            if seen_progress:
                # Progress data exists; just refresh the elapsed text.
                progress.progress(last_frac, text=f"Running … {elapsed:.0f}s elapsed")
            else:
                # No incremental progress signal (local evaluator): show elapsed time.
                progress.progress(0.0, text=f"Running … {elapsed:.0f}s elapsed (no incremental progress)")

    returncode = proc.wait()
    log_lines.append(f"--- exit code {returncode} ---")
    log_placeholder.code("\n".join(log_lines[-400:]), language="text")

    if returncode != 0:
        st.error(f"Evaluator exited with code {returncode}.")
        return None

    out_path = repo_root / out_name
    if not out_path.exists():
        st.error(f"Evaluator finished but {out_path} was not created.")
        return None
    return load_results_from_path(out_path)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)

    # Sidebar: controls -------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ Controls")
        st.caption("Open / collapse the sidebar with the arrow at the top-left.")

        # --- Run control -----------------------------------------------------
        with st.expander("⚙️ Run evaluator", expanded=True):
            run_mode = st.selectbox(
                "Evaluator mode",
                options=list(RUN_MODES.keys()),
                format_func=lambda k: RUN_MODES[k]["label"],
                help=RUN_MODES["local"]["desc"]
                + "\n\n" + RUN_MODES["llm"]["desc"],
            )
            run_clicked = st.button("▶ Run Evaluator", type="primary", width="stretch")

        # --- Load existing results ------------------------------------------
        with st.expander("📂 Load existing results", expanded=True):
            uploaded = st.file_uploader("Upload results.json", type=["json"])
            with st.expander("…or point to a path"):
                path_text = st.text_input("Path to results.json", value=str(DEFAULT_RESULTS))
                load_btn = st.button("Load from path", width="stretch")

        # --- Version history ------------------------------------------------
        with st.expander("📈 Version history", expanded=False):
            vh_uploaded = st.file_uploader("Upload versions (CSV/JSON)", type=["csv", "json"])
            with st.expander("…or paste"):
                paste_text = st.text_area("Paste CSV / JSON", height=180)
                parse_btn = st.button("Parse pasted history", width="stretch")

        # --- Session filters ------------------------------------------------
        with st.expander("🔍 Session filters", expanded=False):
            scenario_filter = st.multiselect(
                "Scenario", SCENARIO_ORDER, default=SCENARIO_ORDER,
                format_func=lambda s: SCENARIO_LABEL.get(s, s),
            )
            hit_filter = st.selectbox("Hit / miss", ["All", "Hits", "Misses"])
            include_current = st.checkbox("Include current run in the version chart", value=True)

    # Main area ---------------------------------------------------------------
    st.title("🛍️ Shopping Copilot — Metrics Dashboard")

    # --- Handle the run button ----------------------------------------------
    results = None
    if run_clicked:
        with st.status("Running evaluator …", expanded=True) as status:
            log_placeholder = st.empty()
            results = run_evaluator(REPO_ROOT, log_placeholder, run_mode)
            if results is not None:
                status.update(label="Evaluator finished", state="complete", expanded=False)
        st.session_state["results"] = results
        st.session_state["results_source"] = "run"
        st.rerun()

    # --- Handle upload / path / paste -----------------------------------------
    if uploaded is not None:
        results = load_results_from_bytes(uploaded.getvalue())
        if results is not None:
            st.session_state["results"] = results
            st.session_state["results_source"] = "upload"
    elif load_btn and Path(path_text).exists():
        results = load_results_from_path(Path(path_text))
        if results is not None:
            st.session_state["results"] = results
            st.session_state["results_source"] = "path"

    # --- Version history ------------------------------------------------------
    version_df = None
    if vh_uploaded is not None:
        version_df = parse_version_history(vh_uploaded)
        st.session_state["version_history"] = version_df
    elif parse_btn and paste_text.strip():
        version_df = parse_version_history(paste_text)
        st.session_state["version_history"] = version_df

    if "version_history" in st.session_state and version_df is None:
        version_df = st.session_state["version_history"]

    # Fall back to the built-in baseline -> current history so a live demo always shows
    # the progression even before the user uploads/pastes a version-history table.
    if version_df is None:
        version_df = pd.DataFrame(DEFAULT_VERSION_HISTORY)
        st.session_state["version_history"] = version_df

    # --- Resolve the results to display ---------------------------------------
    results = results or st.session_state.get("results")
    if results is None and DEFAULT_RESULTS.exists():
        results = load_results_from_path(DEFAULT_RESULTS)
        st.session_state["results"] = results
        st.session_state["results_source"] = "default"

    if results is None:
        st.info("No results loaded. Click **Run Evaluator**, upload a `results.json`, or point to a path.")
        return

    agg = results["aggregate"]
    sess_df = results["sessions"]
    scen_df = results["scenario_metrics"]

    # --- Headline "receipt" hero + Baseline -> Final ---------------------------
    baseline = DEFAULT_VERSION_HISTORY[0]
    st.markdown(_hero_html(agg), unsafe_allow_html=True)
    st.caption(
        "Technical Score = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency · "
        "Efficiency = clip((11 − MTTC) / 10, 0, 1) · Token usage is a feasibility metric, not part of the core score."
    )

    st.markdown("### Baseline → Final", unsafe_allow_html=True)
    st.caption("weak BM25 starter (**v1.0.0**) vs. the current run")
    st.markdown(_baseline_final_html(agg, baseline), unsafe_allow_html=True)

    with st.expander("📖 Metric definitions"):
        for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score", "total_tokens"):
            st.markdown(f"**{METRIC_LABELS.get(key, key)}** — {METRIC_HELP.get(key, '')}")
        st.caption("A session ends when the target appears in the scored Top 10 or after turn 10; a miss counts as a turn-11 MTTC.")

    # --- Scenario breakdown ----------------------------------------------------
    st.subheader("Scenario breakdown")
    if not scen_df.empty:
        st.markdown(_scenario_cards_html(scen_df), unsafe_allow_html=True)
        scen_melt = scen_df.melt(
            id_vars=["scenario_label", "sample_count"],
            value_vars=["hit_rate_at_10", "mrr", "mttc"],
            var_name="metric",
            value_name="value",
        )
        dark = st.get_option("theme.base") == "dark"
        fig = px.bar(
            scen_melt,
            x="scenario_label",
            y="value",
            color="metric",
            barmode="group",
            labels={"scenario_label": "Scenario", "value": "Value", "metric": "Metric"},
            template="plotly_dark" if dark else "plotly_white",
            title="Hit Rate@10 / MRR / MTTC by scenario",
        )
        fig.update_yaxes(title="Value")
        fig.update_layout(legend_title_text="Metric", margin=dict(l=10, r=10, t=50, b=10))
        fig.for_each_trace(lambda t: t.update(hovertemplate="%{fullData.name}: %{y:.4f}<extra></extra>"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No per-scenario metrics in the loaded results.")

    # --- Version progression ----------------------------------------------------
    st.subheader("Version progression")
    if version_df is not None and not version_df.empty:
        selectable = [m for m in ("technical_score", "hit_rate_at_10", "mrr", "mttc")
                      if m in version_df.columns]
        if include_current and not pd.isna(agg["technical_score"]):
            cur = {"version": "current (live)", "description": "Live results.json run",
                   "technical_score": agg["technical_score"]}
            for key in ("hit_rate_at_10", "mrr", "mttc"):
                if key in selectable:
                    cur[key] = agg[key]
            version_df = pd.concat(
                [version_df, pd.DataFrame([cur])], ignore_index=True)
        series = st.multiselect(
            "Metrics to plot",
            options=selectable,
            default=[m for m in ("technical_score", "hit_rate_at_10") if m in selectable],
            format_func=lambda m: METRIC_LABELS.get(m, m),
        )
        fig = version_chart(version_df, series)
        if fig.data:
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Select at least one metric to plot.")
    else:
        st.info(
            "No version history loaded. Upload a CSV/JSON (columns: version, description, "
            "hit_rate_at_10, mrr, mttc, efficiency, technical_score, token usage) to compare "
            "versions and see changelog tooltips."
        )

    # --- Session explorer --------------------------------------------------------
    st.subheader("Session explorer")
    if not sess_df.empty:
        filtered = sess_df.copy()
        if scenario_filter:
            filtered = filtered[filtered["scenario"].isin(scenario_filter)]
        if hit_filter == "Hits":
            filtered = filtered[filtered["hit"]]
        elif hit_filter == "Misses":
            filtered = filtered[~filtered["hit"]]
        if filtered.empty:
            st.info("No sessions match the current filters.")
        else:
            display = filtered.copy()
            display["scenario"] = display["scenario"].map(lambda s: SCENARIO_LABEL.get(s, s))
            display["hit"] = display["hit"].map(lambda v: "hit" if v else "miss")
            display["reciprocal_rank"] = display["reciprocal_rank"].map(lambda v: f"{v:.4f}")
            st.dataframe(
                display[["session_id", "scenario", "hit", "hit_turn", "best_rank",
                         "reciprocal_rank", "turns_used"]],
                width="stretch", hide_index=True,
            )
            st.caption(f"{len(filtered)} / {len(sess_df)} sessions shown.")
    else:
        st.info("No per-session data in the loaded results.")


if __name__ == "__main__":
    main()
