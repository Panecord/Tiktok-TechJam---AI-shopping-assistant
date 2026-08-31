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

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = REPO_ROOT / "results.json"

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

    # --- Headline metric row ---------------------------------------------------
    prev = None
    if version_df is not None and not version_df.empty:
        prev = version_df.iloc[-1]  # most recent previous version

    def _delta(metric_key: str) -> float | None:
        if prev is None or metric_key not in prev.columns:
            return None
        pv = _metric(_first_value(prev.to_dict(), metric_key,
                                  {"hit_rate_at_10": "HitRate@10"}.get(metric_key, metric_key)))
        cur = agg[metric_key]
        if pd.isna(pv) or pd.isna(cur):
            return None
        return float(cur - pv)

    cols = st.columns(6)
    cards = [
        ("hit_rate_at_10", _fmt_pct(agg["hit_rate_at_10"]), "normal"),
        ("mrr", _fmt_pct(agg["mrr"]), "normal"),
        ("mttc", _fmt_num(agg["mttc"], 2), "inverse"),
        ("efficiency", _fmt_pct(agg["efficiency"]), "normal"),
        ("technical_score", _fmt_num(agg["technical_score"], 4), "normal"),
        ("total_tokens", f"{int(agg['total_tokens']):,}", "normal"),
    ]
    for col, (key, value, color) in zip(cols, cards):
        delta = _delta(key)
        tk_delta = delta if not pd.isna(delta) else None
        col.metric(
            label=METRIC_LABELS.get(key, key),
            value=value,
            delta=_fmt_delta(tk_delta) if tk_delta is not None else None,
            delta_color=color,
            help=METRIC_HELP.get(key),
        )
    st.caption(
        "Technical Score = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency · "
        "Efficiency = clip((11 − MTTC) / 10, 0, 1) · Token usage is a feasibility metric, not part of the core score."
    )
    with st.expander("📖 Metric definitions"):
        for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score", "total_tokens"):
            st.markdown(f"**{METRIC_LABELS.get(key, key)}** — {METRIC_HELP.get(key, '')}")
        st.caption("A session ends when the target appears in the scored Top 10 or after turn 10; a miss counts as a turn-11 MTTC.")

    # --- Scenario breakdown ----------------------------------------------------
    st.subheader("Scenario breakdown")
    if not scen_df.empty:
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

        show = scen_df.copy()
        show["hit_rate_at_10"] = show["hit_rate_at_10"].map(lambda v: f"{v*100:.1f}%")
        show["mrr"] = show["mrr"].map(lambda v: f"{v*100:.1f}%")
        show["mttc"] = show["mttc"].map(lambda v: f"{v:.2f}")
        show["n"] = show["sample_count"].map(lambda v: int(v) if not pd.isna(v) else v)
        st.dataframe(
            show[["scenario_label", "n", "hit_rate_at_10", "mrr", "mttc"]],
            width="stretch", hide_index=True,
        )
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
