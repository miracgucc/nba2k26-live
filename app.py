import importlib
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from period_predictor import (
    analyze_archive_4500_symmetry,
    analyze_barem_symmetry,
    analyze_matchup_live_summary,
    analyze_sequential_barem_wave,
    analyze_smart_35m_decision,
    analyze_team_pair_history,
    build_q3_q4_streak_signal,
    build_team_pair_table,
    get_cluster_from_score,
    get_csv_headers,
    get_league_config,
    get_range_from_score,
    predict_live_inplay,
    predict_period_scores,
    resolve_effective_barem_odds,
    simulate_35m_strategies,
    validate_symmetry_log,
)
from pull_today_results import discover_active_domain, pull_today_results
from run_live_auto_scraper import (
    CACHE_FILE,
    _get_scraper_paths,
    init_csv,
    merge_cloud_backups,
    run_auto_scraper,
)
from update_all_checkpoints import auto_settle_checkpoints

TR_TZ = timezone(timedelta(hours=3))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NBA 2K26 CyberLeague & Euro Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "soft_light"

if "active_league" not in st.session_state:
    st.session_state.active_league = "nba"


def toggle_ui_theme():
    st.session_state.ui_theme = "matte_dark" if st.session_state.ui_theme == "soft_light" else "soft_light"


def set_active_league(league_key: str):
    lk = str(league_key or "nba").lower()
    st.session_state.active_league = lk if lk in ("nba", "nba2k27", "euro") else "nba"


IS_DARK = st.session_state.ui_theme == "matte_dark"
ACTIVE_LEAGUE = st.session_state.active_league
IS_EURO = ACTIVE_LEAGUE == "euro"
IS_NBA2K27 = ACTIVE_LEAGUE == "nba2k27"
LEAGUE_CFG = get_league_config(ACTIVE_LEAGUE)
BAREM_RANGES = LEAGUE_CFG["barem_ranges"]
CLUSTER_RANGES = LEAGUE_CFG["cluster_ranges"]
BAREM_TO_ODD_COL = LEAGUE_CFG["barem_odd_cols"]
BAREM_TO_PROB_COL = LEAGUE_CFG["barem_prob_cols"]
CSV_HEADERS = get_csv_headers(ACTIVE_LEAGUE)

_LEAGUE_PATHS = _get_scraper_paths(ACTIVE_LEAGUE)
CSV_PATH = _LEAGUE_PATHS["csv_path"]
MATCHES_CSV_PATH = _LEAGUE_PATHS["matches_path"]
LIVE_STATE_FILE = _LEAGUE_PATHS["state_file"]
TS_FILE = _LEAGUE_PATHS["ts_file"]

LEAGUE_BAREM_N = len(BAREM_RANGES)
Q_LEN = float(LEAGUE_CFG["quarter_length"])
CLOSE_MIN = float(LEAGUE_CFG["close_min"])
CLOSE_MIN_INT = int(LEAGUE_CFG["close_min_int"])
DEC_MIN = int(LEAGUE_CFG["decision_min"])
CLOSING_START = int(LEAGUE_CFG["closing_start_min"])
FINAL_WIN_START = int(LEAGUE_CFG["final_win_start"])
MID_Q3_MIN = int(LEAGUE_CFG["mid_q3_min"])
HALFTIME_MIN = int(LEAGUE_CFG["halftime_min"])
EARLY_Q2_MIN = int(LEAGUE_CFG["early_q2_min"])
EXP_REM_CLOSE = int(LEAGUE_CFG["exp_rem_close"])
EXP_REM_DEC = float(LEAGUE_CFG["exp_rem_decision"])

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON BACKGROUND DAEMON THREAD (24/7 ZERO-CRASH AUTO SCRAPER)
# ─────────────────────────────────────────────────────────────────────────────
SCRAPE_LOCK = threading.Lock()


ENGINE_VERSION = "v7.1-selective-calibrated"


@st.cache_resource
def _ensure_clean_git_tracked_files():
    """Keep git-tracked seed CSV/JSON files clean on Streamlit Cloud so git pull never blocks."""
    try:
        import subprocess
        home_dir = os.environ.get("HOME", "")
        if os.environ.get("GITHUB_ACTIONS") != "true" and (
            home_dir.startswith("/home/appuser") or home_dir.startswith("/home/adminuser")
        ):
            subprocess.run(
                [
                    "git",
                    "checkout",
                    "--",
                    "symmetry_analysis_log.csv",
                    "matches_data.csv",
                    "cloud_symmetry_backup.csv",
                    "cloud_matches_backup.csv",
                    "live_matches_state.json",
                    "nba2k27_symmetry_log.csv",
                    "nba2k27_matches_data.csv",
                    "cloud_nba2k27_symmetry_backup.csv",
                    "cloud_nba2k27_matches_backup.csv",
                    "nba2k27_live_matches_state.json",
                    "euro_symmetry_log.csv",
                    "euro_matches_data.csv",
                    "cloud_euro_symmetry_backup.csv",
                    "cloud_euro_matches_backup.csv",
                    "euro_live_matches_state.json",
                ],
                cwd=BASE_DIR,
                timeout=3,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


_ensure_clean_git_tracked_files()


class BackgroundEngineState:
    def __init__(self):
        self.version = ENGINE_VERSION
        self.is_running = False
        self.last_run_ts = 0.0
        self.last_run_str = "Hazırlanıyor"
        self.next_poll_sec = 60
        self.last_error = ""
        self.scan_count = 0


@st.cache_resource
def get_background_engine(version: str = ENGINE_VERSION) -> BackgroundEngineState:
    state = BackgroundEngineState()

    def _worker():
        while state.version == ENGINE_VERSION:
            try:
                with SCRAPE_LOCK:
                    state.is_running = True
                    poll_sec = run_auto_scraper(force_manual=False)
                    state.last_run_ts = time.time()
                    state.last_run_str = datetime.now(TR_TZ).strftime("%H:%M:%S")
                    state.next_poll_sec = max(10, min(300, int(poll_sec or 60)))
                    state.scan_count += 1
                    state.last_error = ""
            except Exception as e:
                state.last_error = str(e)
                state.next_poll_sec = 30
            finally:
                state.is_running = False

            sleep_target = time.time() + state.next_poll_sec
            while time.time() < sleep_target and state.version == ENGINE_VERSION:
                time.sleep(1.0)

    t = threading.Thread(target=_worker, daemon=True, name=f"NBA2K26_LiveEngine_{version}")
    t.start()
    return state


bg_engine = get_background_engine(ENGINE_VERSION)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS & ANALYTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def reset_all_data_clean_slate(league: str = ACTIVE_LEAGUE):
    paths = _get_scraper_paths(league)
    headers = get_csv_headers(league)
    with SCRAPE_LOCK:
        with open(paths["csv_path"], "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
        matches_headers = (
            "game_id,date,home_team,away_team,home_q1,away_q1,home_q2,away_q2,"
            "home_q3,away_q3,home_q4,away_q4,ot_home,ot_away,home_total,away_total,"
            "winner,total_score,margin,is_ot,top_scorer,top_scorer_points\n"
        )
        with open(paths["matches_path"], "w", encoding="utf-8") as f:
            f.write(matches_headers)
        for jf in [paths["ts_file"], paths["state_file"]]:
            with open(jf, "w", encoding="utf-8") as f:
                json.dump({}, f)


def load_live_state(league: str = ACTIVE_LEAGUE) -> dict:
    cfg = get_league_config(league)
    paths = _get_scraper_paths(league)
    for candidate in (paths["state_file"], os.path.join(BASE_DIR, cfg["files"]["state_file"])):
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=45, show_spinner=False)
def _fetch_github_csv(filename: str) -> pd.DataFrame:
    """Fetch latest cloud-bot CSV directly from GitHub Raw so Streamlit Cloud never freezes on stale deploys."""
    try:
        import io
        import urllib.request

        bucket = int(time.time() // 45)
        url = f"https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/{filename}?cb={bucket}"
        req = urllib.request.Request(
            url,
            headers={
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw_text = resp.read().decode("utf-8")
        if raw_text.strip():
            return pd.read_csv(io.StringIO(raw_text))
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def load_symmetry_log(league: str = ACTIVE_LEAGUE) -> pd.DataFrame:
    """
    Stateless loader: combines local runtime log, git seed log, local backup, AND live GitHub Raw cloud-bot backup
    for the selected league. COMPLETED rows always win over PENDING rows for the same (game_id, check_time).
    """
    cfg = get_league_config(league)
    paths = _get_scraper_paths(league)
    headers = get_csv_headers(league)
    files = cfg["files"]
    min_gid = int(cfg.get("min_clean_gid", 0)) if cfg["key"] == "nba" else 770000000

    init_csv(league)
    frames = [
        _read_csv_safe(paths["csv_path"]),
        _read_csv_safe(os.path.join(BASE_DIR, files["log_seed"])),
        _read_csv_safe(os.path.join(BASE_DIR, files["log_cloud"])),
        _fetch_github_csv(files["log_cloud"]),
    ]
    frames = [f for f in frames if not f.empty and "game_id" in f.columns and "check_time" in f.columns]
    if not frames:
        return pd.DataFrame(columns=headers)
    df = pd.concat(frames, ignore_index=True)
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["_gnum"] = df["game_id"].str.extract(r"(\d+)")[0].fillna(0).astype(int)
    df = df[df["_gnum"] >= min_gid].copy()
    if "game_name" in df.columns:
        df = df[df["game_name"].astype(str).str.contains(r"\(cyber\)", case=False, na=False)].copy()
    if df.empty:
        return pd.DataFrame(columns=headers)
    df["_done"] = (df["status"].astype(str) == "COMPLETED").astype(int)
    df = df.sort_values("_done", kind="stable").drop_duplicates(subset=["game_id", "check_time"], keep="last")
    df["_min"] = df["check_time"].astype(str).str.extract(r"Minute\s+(\d+)")[0].fillna(0).astype(int)
    df = df.sort_values(["_gnum", "_min"], kind="stable").drop(columns=["_gnum", "_done", "_min"]).reset_index(drop=True)

    _b_cols = [
        (b_name, cfg["barem_odd_cols"][b_name], cfg["barem_prob_cols"][b_name])
        for b_name, _, _ in cfg["barem_ranges"]
    ]
    # Fast path: only recompute analyze_barem_symmetry for rows missing highest_prob_range/value_range or recent 150 rows
    n_total = len(df)
    recent_cutoff = max(0, n_total - 150)
    smart_favs = df["highest_prob_range"].astype(str).tolist() if "highest_prob_range" in df.columns else [""] * n_total
    smart_vals = df["value_range"].astype(str).tolist() if "value_range" in df.columns else [""] * n_total
    for idx_r in range(recent_cutoff, n_total):
        r = df.iloc[idx_r]
        o_d = {}
        p_d = {}
        for b_lbl, oc, pc in _b_cols:
            try:
                o_d[b_lbl] = float(r.get(oc, 99.0) or 99.0)
            except Exception:
                o_d[b_lbl] = 99.0
            try:
                p_d[b_lbl] = float(r.get(pc, 0.0) or 0.0)
            except Exception:
                p_d[b_lbl] = 0.0
        sym_r = analyze_barem_symmetry(o_d, p_d, league=league)
        smart_favs[idx_r] = sym_r.get("highest_prob_range") or smart_favs[idx_r]
        smart_vals[idx_r] = sym_r.get("value_range") or smart_vals[idx_r]
    df["highest_prob_range"] = smart_favs
    df["value_range"] = smart_vals
    return df


@st.cache_data(ttl=25, show_spinner=False)
def load_completed_matches(league: str = ACTIVE_LEAGUE) -> pd.DataFrame:
    cfg = get_league_config(league)
    paths = _get_scraper_paths(league)
    files = cfg["files"]
    frames = [
        _read_csv_safe(paths["matches_path"]),
        _read_csv_safe(os.path.join(BASE_DIR, files["matches_seed"])),
        _read_csv_safe(os.path.join(BASE_DIR, files["matches_cloud"])),
        _fetch_github_csv(files["matches_cloud"]),
    ]
    frames = [f for f in frames if not f.empty and "game_id" in f.columns]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df = df[df["game_id"].str.startswith("CB-")].copy()
    num_cols = [
        "home_q1", "away_q1", "home_q2", "away_q2",
        "home_q3", "away_q3", "home_q4", "away_q4",
        "ot_home", "ot_away", "home_total", "away_total",
        "total_score", "margin", "is_ot", "top_scorer_points",
    ]
    for nc in num_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce")
    req_m_cols = [c for c in ["home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"] if c in df.columns]
    if req_m_cols:
        df = df.dropna(subset=req_m_cols).copy()
    if "total_score" in df.columns:
        df = df[df["total_score"] > 0].copy()
    return df.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)


def check_cluster_hit(cluster_name: str, actual_range: str, league: str = ACTIVE_LEAGUE) -> bool:
    if not cluster_name or not actual_range or pd.isna(cluster_name) or pd.isna(actual_range):
        return False
    c_str = str(cluster_name).strip()
    a_str = str(actual_range).strip()
    for name, _, _, member_tuple in get_league_config(league)["cluster_ranges"]:
        if name == c_str:
            return a_str in member_tuple
    return False


def compute_team_dna_table(df_matches: pd.DataFrame, league: str = ACTIVE_LEAGUE) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()
    cfg = get_league_config(league)
    over_thr = int(cfg.get("high_over_threshold", 211))
    teams = sorted(
        set(df_matches["home_team"].dropna().unique()).union(set(df_matches["away_team"].dropna().unique()))
    )
    rows = []
    for t in teams:
        sub = df_matches[(df_matches["home_team"] == t) | (df_matches["away_team"] == t)].copy()
        if sub.empty:
            continue
        n_games = len(sub)
        tot_scores = sub["total_score"].astype(float)
        q1_tot = (sub["home_q1"].astype(float) + sub["away_q1"].astype(float)).mean()
        q2_tot = (sub["home_q2"].astype(float) + sub["away_q2"].astype(float)).mean()
        q3_tot = (sub["home_q3"].astype(float) + sub["away_q3"].astype(float)).mean()
        q4_tot = (sub["home_q4"].astype(float) + sub["away_q4"].astype(float)).mean()

        barems = tot_scores.apply(lambda v: get_range_from_score(v, league=league))
        clusters = tot_scores.apply(lambda v: get_cluster_from_score(v, league=league))
        top_barem = barems.mode().iloc[0] if not barems.mode().empty else "-"
        top_barem_pct = (barems == top_barem).mean() * 100.0
        top_cluster = clusters.mode().iloc[0] if not clusters.mode().empty else "-"
        top_cluster_pct = (clusters == top_cluster).mean() * 100.0
        over_pct = (tot_scores >= over_thr).mean() * 100.0

        rows.append(
            {
                "Takım": t,
                "Maç Sayısı": n_games,
                "Ort. Toplam Sayı": round(tot_scores.mean(), 1),
                "Std. Sapma": round(tot_scores.std() if n_games > 1 else 0.0, 1),
                "Q1 Ort.": round(q1_tot, 1),
                "Q2 Ort.": round(q2_tot, 1),
                "Q3 Ort.": round(q3_tot, 1),
                "Q4 Ort.": round(q4_tot, 1),
                "En Çok Çıkan Barem": f"{top_barem} (%{top_barem_pct:.0f})",
                "En Güçlü 20'li Küme": f"{top_cluster} (%{top_cluster_pct:.0f})",
                f"{over_thr}+ Üst Oranı": f"%{over_pct:.1f}",
            }
        )
    df_dna = pd.DataFrame(rows)
    if not df_dna.empty:
        df_dna = df_dna.sort_values(by=["Maç Sayısı", "Ort. Toplam Sayı"], ascending=[False, False])
    return df_dna


# ─────────────────────────────────────────────────────────────────────────────
# EYE-FRIENDLY NORDIC SOFT SLATE / WARM MATTE THEME PALETTE
# ─────────────────────────────────────────────────────────────────────────────
if not IS_DARK:
    THEME_VARS = """
        --bg-app: #f3f5f8;
        --bg-card: #ffffff;
        --bg-subcard: #f8fafc;
        --border-subtle: #e2e8f0;
        --border-focus: #cbd5e1;
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --shadow-card: 0 2px 10px rgba(15, 23, 42, 0.04);
        --pill-bg: #ecfdf5;
        --pill-border: #a7f3d0;
        --pill-text: #047857;
        --score-bg: #eff6ff;
        --score-border: #bfdbfe;
        --score-text: #1d4ed8;
        --fav1-bg: #ecfdf5;
        --fav1-border: #10b981;
        --fav1-text: #065f46;
        --fav2-bg: #eff6ff;
        --fav2-border: #3b82f6;
        --fav2-text: #1e40af;
        --val-bg: #fffbeb;
        --val-border: #f59e0b;
        --val-text: #92400e;
        --wave-fav-bg: #dcfce7;
        --wave-fav-border: #10b981;
        --wave-fav-accent: #059669;
        --wave-fav-text: #064e3b;
        --wave-down1-bg: #e0f2fe;
        --wave-down1-border: #0284c7;
        --wave-down1-accent: #0369a1;
        --wave-down1-text: #0c4a6e;
        --wave-down2-bg: #e0e7ff;
        --wave-down2-border: #4f46e5;
        --wave-down2-accent: #4338ca;
        --wave-down2-text: #1e1b4b;
        --wave-up1-bg: #ffedd5;
        --wave-up1-border: #ea580c;
        --wave-up1-accent: #c2410c;
        --wave-up1-text: #7c2d12;
        --wave-up2-bg: #ffe4e6;
        --wave-up2-border: #e11d48;
        --wave-up2-accent: #be123c;
        --wave-up2-text: #881337;
    """
else:
    THEME_VARS = """
        --bg-app: #181c24;
        --bg-card: #212631;
        --bg-subcard: #282e3b;
        --border-subtle: #323949;
        --border-focus: #434c5e;
        --text-primary: #e5e9f0;
        --text-secondary: #d8dee9;
        --text-muted: #9aa5b8;
        --shadow-card: 0 4px 14px rgba(0, 0, 0, 0.18);
        --pill-bg: rgba(16, 185, 129, 0.12);
        --pill-border: rgba(16, 185, 129, 0.35);
        --pill-text: #6ee7b7;
        --score-bg: rgba(59, 130, 246, 0.15);
        --score-border: rgba(59, 130, 246, 0.35);
        --score-text: #93c5fd;
        --fav1-bg: rgba(16, 185, 129, 0.16);
        --fav1-border: #34d399;
        --fav1-text: #a7f3d0;
        --fav2-bg: rgba(59, 130, 246, 0.16);
        --fav2-border: #60a5fa;
        --fav2-text: #bfdbfe;
        --val-bg: rgba(245, 158, 11, 0.15);
        --val-border: #fbbf24;
        --val-text: #fde68a;
        --wave-fav-bg: rgba(16, 185, 129, 0.20);
        --wave-fav-border: #10b981;
        --wave-fav-accent: #059669;
        --wave-fav-text: #a7f3d0;
        --wave-down1-bg: rgba(14, 165, 233, 0.20);
        --wave-down1-border: #38bdf8;
        --wave-down1-accent: #0284c7;
        --wave-down1-text: #bae6fd;
        --wave-down2-bg: rgba(99, 102, 241, 0.24);
        --wave-down2-border: #818cf8;
        --wave-down2-accent: #4f46e5;
        --wave-down2-text: #c7d2fe;
        --wave-up1-bg: rgba(249, 115, 22, 0.20);
        --wave-up1-border: #fb923c;
        --wave-up1-accent: #ea580c;
        --wave-up1-text: #fed7aa;
        --wave-up2-bg: rgba(244, 63, 94, 0.24);
        --wave-up2-border: #fb7185;
        --wave-up2-accent: #e11d48;
        --wave-up2-text: #fecdd3;
    """

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {{
    {THEME_VARS}
}}

.stApp {{
    background-color: var(--bg-app) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.block-container,
.stMainBlockContainer,
[data-testid="stMainBlockContainer"] {{
    padding-top: 3.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1460px !important;
}}

/* Command Header */
.cmd-header {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.05rem 1.4rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    box-shadow: var(--shadow-card);
}}

.cmd-title {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
}}

.cmd-sub {{
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 0.18rem;
    font-weight: 500;
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--pill-bg);
    border: 1px solid var(--pill-border);
    color: var(--pill-text);
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 0.79rem;
    font-weight: 600;
}}

/* Executive KPI Cards */
.kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.95rem 1.15rem;
    box-shadow: var(--shadow-card);
}}

.kpi-label {{
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    font-weight: 700;
}}

.kpi-val {{
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    margin: 0.22rem 0;
}}

.kpi-sub {{
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 500;
}}

/* Live Match Card */
.match-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.2rem 1.35rem;
    margin-bottom: 1.1rem;
    box-shadow: var(--shadow-card);
}}

.match-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.8rem;
    margin-bottom: 0.9rem;
}}

.teams-title {{
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--text-primary);
}}

.score-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 800;
    background: var(--score-bg);
    border: 1px solid var(--score-border);
    color: var(--score-text);
    padding: 0.28rem 0.85rem;
    border-radius: 9px;
}}

.info-strip {{
    background: var(--bg-subcard);
    border: 1px solid var(--border-subtle);
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
}}

.info-strip-label {{
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

.info-strip-val {{
    font-size: 0.91rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 0.15rem;
}}

/* 9-Column / 6-Column Barem Grids */
.barem-grid {{
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 0.45rem;
    margin-top: 0.8rem;
}}

.barem-grid-6 {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.5rem;
    margin-top: 0.8rem;
}}

@media (max-width: 1100px) {{
    .barem-grid, .barem-grid-6 {{
        grid-template-columns: repeat(3, 1fr);
    }}
}}

.barem-box {{
    background: var(--bg-subcard);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 0.55rem 0.35rem;
    text-align: center;
    transition: all 0.15s ease;
}}

.barem-box.fav1 {{
    background: var(--fav1-bg);
    border: 1.5px solid var(--fav1-border);
}}

.barem-box.fav2 {{
    background: var(--fav2-bg);
    border: 1.5px solid var(--fav2-border);
}}

.barem-box.val-pick {{
    background: var(--val-bg);
    border: 1.5px solid var(--val-border);
}}

.b-name {{
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 0.2rem;
}}

.b-odd {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--text-primary);
}}

.b-prob {{
    font-size: 0.71rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
    font-weight: 600;
}}

.b-tag {{
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.12rem 0.42rem;
    border-radius: 5px;
    margin-top: 0.22rem;
}}

.section-hdr {{
    font-size: 1.08rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 1.35rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}}

/* Soft Button Styling */
div.stButton > button {{
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 0.85rem !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# TOP LEAGUE SWITCHER, COMMAND BAR & CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
live_state = load_live_state(ACTIVE_LEAGUE)
active_domain = live_state.get("domain") or "https://1xbet.global"
last_update_str = live_state.get("updated_at") or bg_engine.last_run_str
scan_dur = live_state.get("duration_sec", 0.0)
next_in = (
    max(0, int((bg_engine.last_run_ts + bg_engine.next_poll_sec) - time.time()))
    if bg_engine.last_run_ts > 0
    else 5
)

_title_icon = "🇪🇺" if IS_EURO else "🏀"
_barem_desc = f"{LEAGUE_BAREM_N}'{'lı' if LEAGUE_BAREM_N == 6 else 'lu'} Barem • 4x{int(Q_LEN)} Dk • Kapanış {CLOSE_MIN_INT}. Dk"

st.markdown(
    f"""
<div class="cmd-header">
    <div>
        <div class="cmd-title">{_title_icon} {LEAGUE_CFG['title']}</div>
        <div class="cmd-sub">Canlı Barem ({_barem_desc}), Kapanış Simetrisi, Takım Barem DNA'sı & Kasa/Dutching Terminali</div>
    </div>
    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;">
        <div class="status-pill">
            <span>●</span>
            <span>Canlı Akış Aktif ({active_domain.replace('https://', '')})</span>
            <span style="opacity:0.5;">|</span>
            <span>Son Tarama: {last_update_str} ({scan_dur}sn)</span>
            <span style="opacity:0.5;">|</span>
            <span>Sonraki: {next_in}sn</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

sw_c0, sw_c1, sw_c2, sw_c3 = st.columns([0.9, 1.55, 1.55, 1.55], vertical_alignment="center")
with sw_c0:
    st.markdown(
        '<div style="background:var(--bg-card); border:1px solid var(--border-subtle); '
        'border-radius:10px; padding:0.52rem 0.75rem; text-align:center; '
        'font-size:0.82rem; font-weight:800; color:var(--text-primary); '
        'box-shadow:var(--shadow-card);">🏆 LİG GEÇİŞİ</div>',
        unsafe_allow_html=True,
    )
with sw_c1:
    _nba_lbl = (
        "✅ 🏀 NBA 2K26 CyberLeague (Aktif • 9 Barem)"
        if ACTIVE_LEAGUE == "nba"
        else "🏀 NBA 2K26 CyberLeague (4x12 Dk • 9 Barem)"
    )
    st.button(
        _nba_lbl,
        key="btn_league_nba",
        on_click=set_active_league,
        args=("nba",),
        type="primary" if ACTIVE_LEAGUE == "nba" else "secondary",
        use_container_width=True,
    )
with sw_c2:
    _nba27_lbl = (
        "✅ 🔥 NBA 2K27 CyberLeague (Aktif • 9 Barem)"
        if ACTIVE_LEAGUE == "nba2k27"
        else "🔥 NBA 2K27 CyberLeague (4x12 Dk • 9 Barem)"
    )
    st.button(
        _nba27_lbl,
        key="btn_league_nba2k27",
        on_click=set_active_league,
        args=("nba2k27",),
        type="primary" if ACTIVE_LEAGUE == "nba2k27" else "secondary",
        use_container_width=True,
    )
with sw_c3:
    _euro_lbl = (
        "✅ 🇪🇺 NBA 2K26 Cyber Euro League (Aktif • 6 Barem)"
        if IS_EURO
        else "🇪🇺 NBA 2K26 Cyber Euro League (4x10 Dk • 6 Barem)"
    )
    st.button(
        _euro_lbl,
        key="btn_league_euro",
        on_click=set_active_league,
        args=("euro",),
        type="primary" if IS_EURO else "secondary",
        use_container_width=True,
    )

ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5, ctrl_c6 = st.columns([1.25, 1.25, 1.35, 1.0, 0.85, 0.9])

with ctrl_c1:
    if st.button("⚡ Anlık Canlı Barem Çek", use_container_width=True, type="primary"):
        with st.spinner(f"1xBet / Mirror üzerinden canlı {LEAGUE_CFG['short_name']} maçları çekiliyor (~2 sn)..."):
            with SCRAPE_LOCK:
                run_auto_scraper(force_manual=True, league=ACTIVE_LEAGUE)
                bg_engine.last_run_ts = time.time()
                bg_engine.last_run_str = datetime.now().strftime("%H:%M:%S")
        st.toast(f"✅ {LEAGUE_CFG['short_name']} canlı maçları ve barem oranları güncellendi!", icon="🏀")
        st.rerun()

with ctrl_c2:
    if st.button("🔄 Bitmiş Maçları Sonuçlandır", use_container_width=True):
        with st.spinner(f"Bitmiş {LEAGUE_CFG['short_name']} maç sonuçları kontrol ediliyor..."):
            pull_today_results(only_tracked=True, league=ACTIVE_LEAGUE)
            settled = auto_settle_checkpoints(league=ACTIVE_LEAGUE)
        st.toast(f"✅ {settled} barem kaydı sonuçlandırıldı!", icon="🎯")
        st.rerun()

with ctrl_c3:
    if st.button(LEAGUE_CFG["archive_btn_label"], use_container_width=True):
        _days = int(LEAGUE_CFG["archive_days"])
        with st.spinner(f"Son {_days} günlük tüm tamamlanmış {LEAGUE_CFG['short_name']} maçları çekiliyor..."):
            added = pull_today_results(only_tracked=False, days_back=_days + 5, league=ACTIVE_LEAGUE)
            auto_settle_checkpoints(league=ACTIVE_LEAGUE)
        st.toast(f"📥 {added} yeni tamamlanmış maç {LEAGUE_CFG['short_name']} arşivine eklendi!", icon="📊")
        st.rerun()

with ctrl_c4:
    if st.button("🗑️ Verileri Sıfırla", use_container_width=True):
        reset_all_data_clean_slate(ACTIVE_LEAGUE)
        st.toast(f"🧹 {LEAGUE_CFG['short_name']} kayıtları sıfırlandı (0 kayıt). Temiz sayfa hazır!", icon="✨")
        st.rerun()

with ctrl_c5:
    theme_btn_label = "🌙 Mat Gece" if not IS_DARK else "🌤️ Ferah Gün"
    st.button(theme_btn_label, on_click=toggle_ui_theme, use_container_width=True)

with ctrl_c6:
    auto_ui_refresh = st.toggle("⏱️ Oto-Yenile", value=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD FRAGMENT (AUTO-REFRESHING WITHOUT FULL PAGE FLICKER)
# ─────────────────────────────────────────────────────────────────────────────
@st.fragment(run_every="15s" if auto_ui_refresh else None)
def render_live_dashboard():
    state = load_live_state(ACTIVE_LEAGUE)
    df_log = load_symmetry_log(ACTIVE_LEAGUE)
    df_hist = load_completed_matches(ACTIVE_LEAGUE)
    live_matches = state.get("live_matches", [])

    active_count = len(live_matches)
    open_barem_count = sum(1 for m in live_matches if m.get("is_barem_open"))
    hist_count = len(df_hist)

    total_logs = len(df_log)
    df_comp = (
        df_log[df_log["status"] == "COMPLETED"].copy()
        if not df_log.empty and "status" in df_log.columns
        else pd.DataFrame()
    )
    df_pend = (
        df_log[df_log["status"] == "PENDING"].copy()
        if not df_log.empty and "status" in df_log.columns
        else pd.DataFrame()
    )

    comp_count = len(df_comp)
    pend_count = len(df_pend)

    if comp_count > 0:
        df_comp_kpi = df_comp.copy()
        df_comp_kpi["_min_num"] = (
            df_comp_kpi["check_time"]
            .astype(str)
            .str.extract(r"Minute\s+(\d+)", expand=False)
            .astype(float)
            .fillna(0)
        )
        df_comp_32 = df_comp_kpi[df_comp_kpi["_min_num"] >= CLOSING_START].copy()
        if not df_comp_32.empty and "game_id" in df_comp_32.columns:
            df_eval_kpi = df_comp_32.sort_values("_min_num").groupby("game_id", as_index=False).last()
        elif not df_comp_32.empty:
            df_eval_kpi = df_comp_32
        else:
            df_eval_kpi = df_comp_kpi
        eval_n = len(df_eval_kpi)
        fav_wins = sum(
            1 for _, r in df_eval_kpi.iterrows() if str(r.get("actual_range")) == str(r.get("highest_prob_range"))
        )
        val_wins = sum(
            1 for _, r in df_eval_kpi.iterrows() if str(r.get("actual_range")) == str(r.get("value_range"))
        )
        clu_wins = sum(
            1
            for _, r in df_eval_kpi.iterrows()
            if check_cluster_hit(r.get("best_cluster_name"), r.get("actual_range"), league=ACTIVE_LEAGUE)
        )
        fav_rate = (fav_wins / eval_n) * 100.0 if eval_n > 0 else 0.0
        val_rate = (val_wins / eval_n) * 100.0 if eval_n > 0 else 0.0
        clu_rate = (clu_wins / eval_n) * 100.0 if eval_n > 0 else 0.0
    else:
        eval_n = fav_wins = val_wins = clu_wins = 0
        fav_rate = val_rate = clu_rate = 0.0

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
        <div class="kpi-card" title="Şu an oynanan canlı {LEAGUE_CFG['short_name']} maç sayısı ve geçmiş arşivdeki toplam maç sayısı.">
            <div class="kpi-label">🔴 Canlı {LEAGUE_CFG['short_name']}</div>
            <div class="kpi-val">{active_count} Maç</div>
            <div class="kpi-sub">{open_barem_count} Maçta Barem Açık &bull; {hist_count} Geçmiş Arşiv</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
        <div class="kpi-card" title="{CLOSING_START}. dakika ve sonrasında kapanış oranı kaydedilmiş {eval_n} tamamlanmış maç ve toplam {total_logs} dakikalık kayıt.">
            <div class="kpi-label">📋 {CLOSING_START}+ Dk Kapanış Maçı</div>
            <div class="kpi-val">{eval_n} Maç</div>
            <div class="kpi-sub">{pend_count} Canlı Kayıt &bull; {comp_count} Toplam Dakika Kaydı</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
        <div class="kpi-card" title="Sadece {CLOSING_START}. dakika ve sonrasındaki kapanış kayıtlarında (her maça 1 kupon baz alınarak) 1xBet'in 1. Favori bareminin tutma oranı.">
            <div class="kpi-label">⭐ 1xBet Favori ({CLOSING_START}+ Dk)</div>
            <div class="kpi-val" style="color:#059669;">%{fav_rate:.1f}</div>
            <div class="kpi-sub">{fav_wins} / {eval_n} Maç İsabet ({CLOSING_START}+ Dk Kapanış)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
        <div class="kpi-card" title="Akıllı Değerli Barem (Smart Value): 1xBet piyasa olasılığı (%80), Monte Carlo model olasılığı (%20) ve simetri kurallarını birleştirerek en kârlı tek baremi seçer.">
            <div class="kpi-label">🔥 Akıllı Value ({CLOSING_START}+ Dk)</div>
            <div class="kpi-val" style="color:#d97706;">%{val_rate:.1f}</div>
            <div class="kpi-sub">{val_wins} / {eval_n} Maç İsabet ({CLOSING_START}+ Dk Kapanış)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
        <div class="kpi-card" title="{CLOSING_START}. dakika ve sonrasında yan yana en güçlü iki baremin (20-Sayı Kümesi) tutma oranı.">
            <div class="kpi-label">🛡️ 20-Sayı Küme ({CLOSING_START}+ Dk)</div>
            <div class="kpi-val" style="color:#2563eb;">%{clu_rate:.1f}</div>
            <div class="kpi-sub">{clu_wins} / {eval_n} Maç İsabet ({CLOSING_START}+ Dk Kapanış)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SEQUENTIAL BAREM WAVE STRIP (LAST MATCHES: 0, -1, +1 WAVE DETECTOR)
    # ─────────────────────────────────────────────────────────────────────────
    wave_global = analyze_sequential_barem_wave(df_hist, df_log, league=ACTIVE_LEAGUE)
    recent_chain = wave_global.get("recent_chain", [])
    if recent_chain:
        pills_html = ""
        cnt_fav_all = sum(1 for x in recent_chain if x["raw_shift"] == 0)
        cnt_down_all = sum(1 for x in recent_chain if x["raw_shift"] < 0)
        cnt_up_all = sum(1 for x in recent_chain if x["raw_shift"] > 0)
        tot_chain_n = max(1, len(recent_chain))

        for idx_c, item in reversed(list(enumerate(recent_chain, 1))):
            sh = item["raw_shift"]
            t_short = (
                f"{str(item['home_team']).replace(' (cyber)', '').split()[-1]} - "
                f"{str(item['away_team']).replace(' (cyber)', '').split()[-1]}"
            )
            dt_str = str(item.get("date", ""))
            time_only = dt_str.split(" ")[-1] if " " in dt_str else dt_str
            h_tot = item.get("home_total", 0)
            a_tot = item.get("away_total", 0)
            sc_pair = f"{h_tot}:{a_tot}" if (h_tot and a_tot) else f"{item['total_score']}"
            if sh == 0:
                p_bg = "var(--wave-fav-bg)"
                p_bd = "var(--wave-fav-border)"
                p_acc = "var(--wave-fav-accent)"
                p_tx = "var(--wave-fav-text)"
                tag_txt = "🎯 Tam Favori Geldi (0)"
            elif sh == -1:
                p_bg = "var(--wave-down1-bg)"
                p_bd = "var(--wave-down1-border)"
                p_acc = "var(--wave-down1-accent)"
                p_tx = "var(--wave-down1-text)"
                tag_txt = "📉 -1 Alt Bareme Düştü"
            elif sh <= -2:
                p_bg = "var(--wave-down2-bg)"
                p_bd = "var(--wave-down2-border)"
                p_acc = "var(--wave-down2-accent)"
                p_tx = "var(--wave-down2-text)"
                tag_txt = f"❄️ {sh} Alt Bareme Düştü"
            elif sh == 1:
                p_bg = "var(--wave-up1-bg)"
                p_bd = "var(--wave-up1-border)"
                p_acc = "var(--wave-up1-accent)"
                p_tx = "var(--wave-up1-text)"
                tag_txt = "📈 +1 Üst Bareme Çıktı"
            else:
                p_bg = "var(--wave-up2-bg)"
                p_bd = "var(--wave-up2-border)"
                p_acc = "var(--wave-up2-accent)"
                p_tx = "var(--wave-up2-text)"
                tag_txt = f"🔥 +{sh} Üst Bareme Çıktı"

            odd_36_val = float(item.get("exp_odd_36", 0.0) or 0.0)
            src_txt = str(item.get("source", ""))
            if "Banko Kilit" in src_txt:
                ref_line = f'📊 {CLOSE_MIN_INT}.Dk Favori: <b>{item["exp_barem_36"]} (🔒 Kilit)</b> ➔ Biten: <b>{item["act_barem"]}</b>'
            elif odd_36_val > 0:
                ref_line = f'📊 {CLOSE_MIN_INT}.Dk Favori: <b>{item["exp_barem_36"]} ({odd_36_val:.2f})</b> ➔ Biten: <b>{item["act_barem"]}</b>'
            else:
                ref_line = f'🧮 {CLOSE_MIN_INT}.Dk Tahmin: <b>{item["exp_barem_36"]}</b> ➔ Biten: <b>{item["act_barem"]}</b>'
            drift_line = item.get("drift_label", "")

            pills_html += (
                f'<div style="background:{p_bg}; border:1.5px solid {p_bd}; border-left:5px solid {p_acc}; color:{p_tx}; border-radius:10px; padding:0.5rem 0.68rem; min-width:160px; flex:1 1 185px;">'
                f'<div style="font-size:0.67rem; font-weight:700; opacity:0.88;">#{idx_c} &bull; 🕒 {time_only} | {t_short}</div>'
                f'<div style="font-size:0.71rem; font-weight:700; margin-top:0.08rem;">Skor: {sc_pair} ({item["total_score"]} Sayı)</div>'
                f'<div style="margin:0.18rem 0;"><span style="background:{p_acc}; color:#ffffff; font-size:0.75rem; font-weight:800; padding:0.14rem 0.5rem; border-radius:6px; display:inline-block;">{tag_txt}</span></div>'
                f'<div style="font-size:0.66rem; opacity:0.96;">{ref_line}</div>'
                f'<div style="font-size:0.63rem; opacity:0.86; margin-top:0.08rem;">{drift_line}</div>'
                f"</div>"
            )

        last5_badges = []
        for x5 in recent_chain[-5:]:
            sh5 = x5["raw_shift"]
            if sh5 == 0:
                last5_badges.append(
                    '<span style="background:var(--wave-fav-accent); color:#fff; padding:0.12rem 0.45rem; border-radius:6px; font-size:0.73rem; font-weight:800;">🎯 Favori (0)</span>'
                )
            elif sh5 < 0:
                c5 = "var(--wave-down1-accent)" if sh5 == -1 else "var(--wave-down2-accent)"
                last5_badges.append(
                    f'<span style="background:{c5}; color:#fff; padding:0.12rem 0.45rem; border-radius:6px; font-size:0.73rem; font-weight:800;">📉 {sh5} Alt</span>'
                )
            else:
                c5 = "var(--wave-up1-accent)" if sh5 == 1 else "var(--wave-up2-accent)"
                last5_badges.append(
                    f'<span style="background:{c5}; color:#fff; padding:0.12rem 0.45rem; border-radius:6px; font-size:0.73rem; font-weight:800;">📈 +{sh5} Üst</span>'
                )
        last5_chain_html = ' <span style="color:var(--text-muted); font-weight:800;">➔</span> '.join(last5_badges)

        last10_items = recent_chain[-10:]
        l10_n = len(last10_items)
        l10_fav_hits = sum(1 for x in last10_items if x["raw_shift"] == 0)
        l10_near_hits = sum(1 for x in last10_items if abs(x["raw_shift"]) <= 1)
        if l10_near_hits >= max(1, int(l10_n * 0.7)):
            form_badge_html = (
                f'<span style="font-size:0.78rem; font-weight:800; color:#059669; background:rgba(5,150,105,0.1); '
                f'border:1px solid rgba(5,150,105,0.3); padding:0.24rem 0.7rem; border-radius:999px;" '
                f'title="Son {l10_n} maçın {l10_near_hits} tanesinde {CLOSE_MIN_INT}. dakika favorisi veya hemen 1 komşu baremi geldi. Piyasa şu an düzenli ve öngörülebilir ilerliyor.">'
                f'🔥 Son {l10_n} Maç Sistem Formu: {l10_fav_hits}/{l10_n} Tam Favori &bull; {l10_near_hits}/{l10_n} Komşu İkili (🟢 Piyasa Düzenli)</span>'
            )
        else:
            form_badge_html = (
                f'<span style="font-size:0.78rem; font-weight:800; color:#d97706; background:rgba(217,119,6,0.1); '
                f'border:1px solid rgba(217,119,6,0.3); padding:0.24rem 0.7rem; border-radius:999px;" '
                f'title="Son {l10_n} maçta sürpriz sapmalar yaşandı. Tek barem yerine İki Barem Kapatma (Çifte Şans) tercih etmeniz önerilir.">'
                f'⚠️ Son {l10_n} Maç Sistem Formu: {l10_fav_hits}/{l10_n} Tam Favori &bull; {l10_near_hits}/{l10_n} Komşu İkili (🟠 Dalgalı — İkili Kapatın)</span>'
            )

        st.markdown(
            f"""
            <div style="background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:13px; padding:0.9rem 1.2rem; margin-top:0.9rem; box-shadow:var(--shadow-card);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.55rem;">
                    <div style="font-size:0.92rem; font-weight:800; color:var(--text-primary); display:flex; align-items:center; flex-wrap:wrap; gap:0.35rem;">
                        <span>🌊 Son Biten Maçlarda {CLOSE_MIN_INT}. Dakika Favorisi Tuttu mu, Kaydı mı? (Toplam {len(recent_chain)} Maç)</span>
                        <span style="margin-left:0.25rem;">Son 5 Maç: {last5_chain_html}</span>
                    </div>
                    <div style="display:flex; gap:0.45rem; flex-wrap:wrap; align-items:center;">
                        {form_badge_html}
                        <div style="font-size:0.78rem; font-weight:700; color:var(--fav1-text); background:var(--pill-bg); border:1px solid var(--pill-border); padding:0.24rem 0.7rem; border-radius:999px;">
                            🔮 Sıradaki Maç Beklentisi: 📉 1 Alt: %{wave_global['prob_minus_1']} &bull; 🎯 Tam Favori: %{wave_global['prob_0']} &bull; 📈 1 Üst: %{wave_global['prob_plus_1']}
                        </div>
                    </div>
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center; margin-bottom:0.65rem; font-size:0.75rem; font-weight:700;">
                    <span style="background:var(--wave-fav-bg); border:1.5px solid var(--wave-fav-border); color:var(--wave-fav-text); padding:0.2rem 0.65rem; border-radius:8px;">
                        🟢 🎯 Tam Favori Geldi (0): <b>{cnt_fav_all} Maç (%{cnt_fav_all/tot_chain_n*100:.1f})</b>
                    </span>
                    <span style="background:var(--wave-down1-bg); border:1.5px solid var(--wave-down1-border); color:var(--wave-down1-text); padding:0.2rem 0.65rem; border-radius:8px;">
                        🔵 📉 -1 / -2 Alt Bareme Düştü (Soğudu): <b>{cnt_down_all} Maç (%{cnt_down_all/tot_chain_n*100:.1f})</b>
                    </span>
                    <span style="background:var(--wave-up1-bg); border:1.5px solid var(--wave-up1-border); color:var(--wave-up1-text); padding:0.2rem 0.65rem; border-radius:8px;">
                        🟠 📈 +1 / +2 Üst Bareme Çıktı (Isındı): <b>{cnt_up_all} Maç (%{cnt_up_all/tot_chain_n*100:.1f})</b>
                    </span>
                </div>
                <div style="display:flex; gap:0.45rem; flex-wrap:wrap; max-height:340px; overflow-y:auto; padding-right:4px;">
                    {pills_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: LIVE MATCH CARDS & BAREM MATRIX
    # ─────────────────────────────────────────────────────────────────────────
    _sec1_lbl = "6'lı" if LEAGUE_BAREM_N == 6 else "9'lu"
    st.markdown(
        f'<div class="section-hdr">🔴 Canlı Maç Kartları & Anlık {_sec1_lbl} Barem Matrisi (0 - {CLOSE_MIN_INT}. Dakika)</div>',
        unsafe_allow_html=True,
    )

    if not live_matches:
        st.info(
            f"Şu an aktif canlı {LEAGUE_CFG['short_name']} maçı bekleniyor veya tarama yapılıyor. "
            "Yukarıdaki **'⚡ Anlık Canlı Barem Çek'** butonuna basarak anında kontrol edebilirsiniz."
        )
    else:
        # Closing Decision Alert Banner at the very top of Live Matches
        closing_alerts_html = ""
        for m_al in live_matches:
            el_al = float(m_al.get("elapsed_mins", 0.0) or 0.0)
            f1_al = m_al.get("bookie_fav_1") or "-"
            f1_odd_al = float(m_al.get("bookie_fav_1_odd", 99.0) or 99.0)
            if MID_Q3_MIN <= el_al <= CLOSE_MIN + 0.2 and m_al.get("is_barem_open", False) and f1_odd_al < 90.0:
                th_al = str(m_al.get("team_home", "")).replace(" (cyber)", "")
                ta_al = str(m_al.get("team_away", "")).replace(" (cyber)", "")
                hsc_al = int(m_al.get("home_score", 0) or 0)
                asc_al = int(m_al.get("away_score", 0) or 0)
                rem_lock = max(0.0, CLOSE_MIN - el_al)
                sm_al = analyze_smart_35m_decision(
                    fav1_barem=f1_al,
                    fav1_odd=f1_odd_al,
                    fav2_barem=m_al.get("bookie_fav_2") or "-",
                    fav2_odd=float(m_al.get("bookie_fav_2_odd", 99.0) or 99.0),
                    game_id=str(m_al.get("game_id", "")),
                    df_log=df_log,
                    recent_chain=recent_chain,
                    score_gap=abs(hsc_al - asc_al),
                    cur_tot=hsc_al + asc_al,
                    elapsed_mins=el_al,
                    odds_map=m_al.get("odds", {}),
                    league=ACTIVE_LEAGUE,
                    proj_tot=float(m_al.get("predicted_total", 0.0) or 0.0),
                )
                _al_is_pass = str(sm_al.get("zone_type", "")).startswith("PASS_")
                _al_border = "#64748b" if _al_is_pass else "#059669"
                _al_bg = "rgba(100,116,139,0.10)" if _al_is_pass else "rgba(5,150,105,0.12)"
                closing_alerts_html += (
                    f'<div style="background:{_al_bg}; border:2px solid {_al_border}; border-radius:12px; padding:0.75rem 1.1rem; margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem; box-shadow:var(--shadow-card);">'
                    f'<div>'
                    f'<span style="background:{_al_border}; color:#ffffff; padding:0.18rem 0.6rem; border-radius:6px; font-size:0.74rem; font-weight:800;">🚨 KARAR ANI (KAPANIŞA {rem_lock:.1f} DK KALDI)</span> '
                    f'<span style="font-size:0.98rem; font-weight:800; color:var(--text-primary); margin-left:0.4rem;">🏀 {th_al} vs {ta_al} ({el_al:.1f}. Dk | Skor: {hsc_al}-{asc_al})</span>'
                    f'<div style="font-size:0.84rem; color:var(--text-primary); margin-top:0.28rem;">'
                    f'<b>🎯 Tek Barem:</b> {sm_al["primary_pick"]} &nbsp;&bull;&nbsp; <b>🛡️ İkili Kapatma:</b> {sm_al["backup_pick"]} &nbsp;&bull;&nbsp; <b>💰 Önerilen:</b> {sm_al["stake_badge"]}'
                    f'</div>'
                    f'<div style="font-size:0.78rem; color:var(--text-secondary); margin-top:0.15rem;">{sm_al["dutch_split_badge"]}</div>'
                    f'</div>'
                    f'</div>'
                )
        if closing_alerts_html:
            st.markdown(closing_alerts_html, unsafe_allow_html=True)

        for m in live_matches:
            t_home = m.get("team_home", "")
            t_away = m.get("team_away", "")
            h_sc = m.get("home_score", 0)
            a_sc = m.get("away_score", 0)
            tot_sc = h_sc + a_sc
            q_num = m.get("quarter", 1)
            el_m = m.get("elapsed_mins", 0.0)
            rem_q = m.get("remaining_mins", Q_LEN)
            is_open = m.get("is_barem_open", False)
            status_code = m.get("status", "")
            h2h_n = m.get("h2h_sample_size", 0)

            comp_qs = m.get("completed_quarters", [])
            q_str_parts = [f"Q{idx+1}: {hq}-{aq} ({hq+aq})" for idx, (hq, aq) in enumerate(comp_qs)]
            q_str_parts.append(f"Q{q_num} (Canlı): {m.get('live_q_score', '0-0')}")
            quarters_summary = " &nbsp;&bull;&nbsp; ".join(q_str_parts)

            if status_code == "PRE_GAME":
                badge_html = (
                    '<span style="background:var(--bg-subcard); border:1px solid var(--border-subtle); '
                    'color:var(--text-secondary); padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    "⏳ MAÇ BAŞLAMADI</span>"
                )
            elif is_open:
                rem_to_close = max(0.0, CLOSE_MIN - el_m)
                badge_html = (
                    '<span style="background:var(--pill-bg); border:1px solid var(--pill-border); '
                    'color:var(--pill-text); padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    f"🟢 BAREM AÇIK (Kapanışa {rem_to_close:.1f} dk)</span>"
                )
            else:
                badge_html = (
                    '<span style="background:rgba(225,29,72,0.1); border:1px solid rgba(225,29,72,0.28); '
                    'color:#e11d48; padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    f"🔒 {CLOSE_MIN_INT}+ DK BAREM KAPANDI — Maç Sonucu Bekleniyor</span>"
                )

            pred_tot = m.get("predicted_total", round(LEAGUE_CFG["prior_q"] * 4.0, 1))
            pred_h = m.get("predicted_home_final", round(LEAGUE_CFG["prior_q"] * 2.0, 1))
            pred_a = m.get("predicted_away_final", round(LEAGUE_CFG["prior_q"] * 2.0, 1))
            fav1 = m.get("bookie_fav_1") or "-"
            fav1_odd = m.get("bookie_fav_1_odd", 99.0)
            fav2 = m.get("bookie_fav_2") or "-"
            _gap_now = abs(int(h_sc) - int(a_sc))
            sym_live = analyze_barem_symmetry(
                m.get("odds", {}),
                m.get("barem_probs", {}),
                league=ACTIVE_LEAGUE,
                proj_tot=pred_tot,
                score_gap=_gap_now,
            )
            val_rng = sym_live.get("value_range") or m.get("value_range") or "-"
            val_has_edge = bool(sym_live.get("has_value_edge", False))
            val_status_lbl = sym_live.get("value_status_label") or f"🔥 Value: {val_rng}"
            val_reason = sym_live.get("value_reason") or "Piyasa ve model simetrisine göre en yüksek beklenen değerli barem."
            wing_badge_live = sym_live.get("wing_badge") or ""
            sym_guidance_live = sym_live.get("sym_guidance") or ""
            clu_name = sym_live.get("best_cluster_name") or m.get("best_cluster_name") or "-"
            clu_prob = sym_live.get("best_cluster_prob") or m.get("best_cluster_prob", 0.0)
            dutch_odd = m.get("dutch_odd", 0.0)
            sym_gap = m.get("symmetry_gap", 99.0)

            q3_sig = m.get("q3_q4_signal") or {}
            q3_sig_label = q3_sig.get("signal", "Nötr Q4 Beklentisi")
            q4_burst_p = q3_sig.get("burst_prob", 35.0)
            q4_coll_p = q3_sig.get("collapse_prob", 25.0)

            match_wave = analyze_sequential_barem_wave(
                df_hist,
                df_log,
                current_fav_barem=fav1,
                league=ACTIVE_LEAGUE,
                odds_map=m.get("odds", {}),
                proj_tot=pred_tot,
            )
            w_b0 = match_wave["target_0_barem"]
            w_bm1 = match_wave["target_minus_1_barem"]
            w_bp1 = match_wave["target_plus_1_barem"]
            w_bm2 = match_wave["target_minus_2_barem"]
            w_bp2 = match_wave["target_plus_2_barem"]
            w_rec_pair = match_wave["recommended_pair"]
            w_rec_prob = match_wave["recommended_pair_prob"]

            smart = analyze_smart_35m_decision(
                fav1_barem=fav1,
                fav1_odd=float(fav1_odd or 0.0),
                fav2_barem=fav2,
                fav2_odd=float(m.get("bookie_fav_2_odd", 99.0) or 99.0),
                game_id=str(m.get("game_id", "")),
                df_log=df_log,
                recent_chain=recent_chain,
                score_gap=_gap_now,
                cur_tot=int(tot_sc),
                elapsed_mins=float(el_m),
                odds_map=m.get("odds", {}),
                league=ACTIVE_LEAGUE,
                proj_tot=pred_tot,
            )
            smart_color = {
                "SUPER_SIGNAL": "#059669",
                "GOLDEN": "#059669",
                "GOLDEN_PAIR": "#059669",
                "GOLDEN_REBOUND": "#059669",
                "CALC_VALUE": "#059669",
                "PASS_CLUTCH": "#dc2626",
                "PASS_WEAK": "#dc2626",
                "PASS_LOW_ODD": "#dc2626",
                "PASS_SELECTIVE": "#64748b",
                "TRAP_EXTREME": "#e11d48",
                "TRAP_LOW": "#d97706",
            }.get(smart["zone_type"], "var(--text-secondary)")
            h2h_live = analyze_matchup_live_summary(df_hist, t_home, t_away, league=ACTIVE_LEAGUE)
            pair_h2h = analyze_team_pair_history(df_hist, t_home, t_away, last_n=5, league=ACTIVE_LEAGUE)
            pair_live_line = (
                f'<div style="font-size:0.72rem; color:var(--fav1-text); margin-top:0.1rem;">🤝 Direkt Çift ({pair_h2h.get("n", 0)} Maç): Ort. <b>{pair_h2h.get("avg_total", 0.0)}</b> &bull; En Sık: <b>{pair_h2h.get("top_barem", "-")}</b></div>'
                if pair_h2h.get("n", 0) > 0
                else ""
            )
            main_reason_short = str(smart.get("reasoning", "")).split(" • ")[0]
            smart_html = (
                f'<div class="info-strip" style="grid-column: 1 / -1; border-left:4px solid {smart_color}; padding:0.75rem 0.95rem;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.55rem;">'
                f'<div class="info-strip-label" style="font-size:0.84rem; margin:0;">👑 {DEC_MIN}. Dakika Ne Oynamalı? (Net Karar Özeti)</div>'
                f'<span style="background:var(--bg-subtle); border:1px solid {smart_color}; color:{smart_color}; padding:0.18rem 0.65rem; border-radius:999px; font-size:0.76rem; font-weight:800;">{smart["zone_badge"]}</span>'
                f'</div>'
                f'<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.55rem; margin-bottom:0.5rem;">'
                f'<div style="background:var(--bg-subtle); border:1px solid var(--border-subtle); border-left:3px solid #059669; border-radius:8px; padding:0.5rem 0.7rem;" title="Tek bir barem oynayacaksanız sistemin geçmiş kapanış analizine göre önerdiği 1 numaralı barem.">'
                f'<div style="font-size:0.68rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">🎯 1. Tercih (Tek Barem Oyna)</div>'
                f'<div style="font-size:0.9rem; font-weight:800; color:var(--text-primary); margin-top:0.15rem;">{smart["primary_pick"]}</div>'
                f'</div>'
                f'<div style="background:var(--bg-subtle); border:1px solid var(--border-subtle); border-left:3px solid #2563eb; border-radius:8px; padding:0.5rem 0.7rem;" title="{smart["dutch_split_tip"]}">'
                f'<div style="font-size:0.68rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">🛡️ 2. Tercih (Garanti İkili Kapat)</div>'
                f'<div style="font-size:0.9rem; font-weight:800; color:var(--text-primary); margin-top:0.15rem;">{smart["backup_pick"]}</div>'
                f'<div style="font-size:0.74rem; color:var(--text-secondary); margin-top:0.2rem; line-height:1.35;">{smart["dutch_split_badge"]}</div>'
                f'</div>'
                f'<div style="background:var(--bg-subtle); border:1px solid var(--border-subtle); border-left:3px solid {smart_color}; border-radius:8px; padding:0.5rem 0.7rem;" title="Yakın maçta (0-5 fark) 0 TL Pas Geç, normal maçta 1000 TL, çok güçlü sinyalde 2000 TL önerilir.">'
                f'<div style="font-size:0.68rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">💰 Ne Kadar Oynanmalı?</div>'
                f'<div style="font-size:0.9rem; font-weight:800; color:{smart_color}; margin-top:0.15rem;">{smart["stake_badge"]}</div>'
                f'</div>'
                f'</div>'
                f'<div style="font-size:0.8rem; color:var(--text-primary); background:var(--bg-subtle); padding:0.42rem 0.7rem; border-radius:6px; line-height:1.45;">'
                f'💬 <b>Kısa Özet:</b> {main_reason_short}.'
                f'</div>'
                f'<details style="margin-top:0.45rem; font-size:0.76rem; color:var(--text-secondary); cursor:pointer;">'
                f'<summary style="font-weight:700; color:var(--text-muted); user-select:none;">🔍 Detaylı Simetri Kontrollerini Göster (İsteğe Bağlı — 8 Kriter)</summary>'
                f'<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:0.35rem; margin-top:0.45rem; padding:0.45rem; background:var(--bg-subtle); border-radius:8px;">'
                f'<div title="{smart["sec_dir_tip"]}">• {smart["sec_dir_badge"]}</div>'
                f'<div title="{smart["margin_tip"]}">• {smart["margin_badge"]}</div>'
                f'<div title="{smart["edge_tip"]}">• {smart["edge_badge"]}</div>'
                f'<div title="{smart["drift_tip"]}">• {smart["drift_badge"]}</div>'
                f'<div title="{smart["twin_tip"]}">• {smart["twin_badge"]}</div>'
                f'<div title="{smart["wing_tip"]}">• {smart["wing_badge"]}</div>'
                f'<div title="{smart["mod10_tip"]}">• {smart["mod10_badge"]}</div>'
                f'<div title="{smart["pendulum_tip"]}">• {smart["pendulum_badge"]}</div>'
                f'</div>'
                f'</details>'
                f'</div>'
            )

            odds_map = m.get("odds", {})
            probs_map = m.get("barem_probs", {})
            barem_names_card = [b[0] for b in BAREM_RANGES]
            fav1_idx_card = barem_names_card.index(fav1) if fav1 in barem_names_card else -1
            boxes_html = ""
            for idx_b, (b_name, _, _) in enumerate(BAREM_RANGES):
                odd_v = float(odds_map.get(b_name, 99.0))
                prob_v = float(probs_map.get(b_name, 0.0))
                odd_disp = f"{odd_v:.2f}" if odd_v < 90.0 else "🔒"

                if fav1_idx_card >= 0:
                    b_diff = idx_b - fav1_idx_card
                    if b_diff == 0:
                        rel_pos_lbl = "🎯 Favori (Merkez)"
                        rel_pos_tip = f"{b_name}: Şu an 1xBet'in en düşük oran verdiği 1. Favori barem."
                    elif b_diff == -1:
                        rel_pos_lbl = "📉 1 Alt Barem"
                        rel_pos_tip = f"{b_name}: Favori baremin ({fav1}) hemen 1 kademe altındaki barem."
                    elif b_diff == 1:
                        rel_pos_lbl = "📈 1 Üst Barem"
                        rel_pos_tip = f"{b_name}: Favori baremin ({fav1}) hemen 1 kademe üstündeki barem."
                    elif b_diff < -1:
                        rel_pos_lbl = f"❄️ {abs(b_diff)} Alt Barem"
                        rel_pos_tip = f"{b_name}: Favori baremin ({fav1}) {abs(b_diff)} kademe altındaki barem."
                    else:
                        rel_pos_lbl = f"🔥 {b_diff} Üst Barem"
                        rel_pos_tip = f"{b_name}: Favori baremin ({fav1}) {b_diff} kademe üstündeki barem."
                else:
                    rel_pos_lbl = ""
                    rel_pos_tip = f"{b_name} baremi (Model olasılığı: %{prob_v:.1f})"

                cls_list = ["barem-box"]
                tag_html = ""
                if b_name == fav1 and odd_v < 90.0:
                    cls_list.append("fav1")
                    tag_html = '<div class="b-tag" style="background:var(--fav1-border); color:#ffffff;">⭐ 1. FAVORİ</div>'
                elif b_name == fav2 and odd_v < 90.0:
                    cls_list.append("fav2")
                    tag_html = '<div class="b-tag" style="background:var(--fav2-border); color:#ffffff;">🥈 2. FAVORİ</div>'

                if val_has_edge and b_name == val_rng and odd_v < 90.0 and b_name != fav1:
                    cls_list.append("val-pick")
                    tag_html = '<div class="b-tag" style="background:var(--val-border); color:#ffffff;">🔥 DEĞERLİ ORAN</div>'
                elif val_has_edge and b_name == val_rng and b_name == fav1 and odd_v < 90.0:
                    tag_html = '<div class="b-tag" style="background:var(--fav1-border); color:#ffffff;">⭐ FAVORİ + DEĞERLİ</div>'

                rel_pos_html = (
                    f'<div style="font-size:0.62rem; font-weight:700; color:var(--text-muted); margin-top:0.1rem;">{rel_pos_lbl}</div>'
                    if rel_pos_lbl
                    else ""
                )

                boxes_html += (
                    f'<div class="{" ".join(cls_list)}" style="cursor:help;" title="{rel_pos_tip}">'
                    f'<div class="b-name">{b_name}</div>'
                    f'{rel_pos_html}'
                    f'<div class="b-odd">{odd_disp}</div>'
                    f'<div class="b-prob">Model: %{prob_v:.1f}</div>'
                    f"{tag_html}"
                    f"</div>"
                )

            sym_text = (
                f"1.Fav: {fav1} ({fav1_odd:.2f}) &bull; 2.Fav: {fav2} (Ortak: {dutch_odd:.2f})"
                if sym_gap < 2.0 and fav1_odd < 90
                else f"1. Favori: {fav1} ({fav1_odd if fav1_odd < 90 else '-'})"
            )
            if wing_badge_live:
                sym_text += f' &bull; <span style="font-size:0.75rem; color:var(--text-primary);">{wing_badge_live}</span>'
            barem_grid_cls = "barem-grid-6" if LEAGUE_BAREM_N == 6 else "barem-grid"

            st.markdown(
                f"""
            <div class="match-card">
                <div class="match-top">
                    <div>
                        <div class="teams-title">🏀 {t_home} &nbsp;<span style="color:var(--text-muted); font-weight:500;">vs</span>&nbsp; {t_away}</div>
                        <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.22rem; font-weight:500;">
                            ⏱️ <b>Q{q_num} — {el_m:.1f}. Dakika</b> (Çeyrekte kalan: {rem_q:.1f} dk) &nbsp;&bull;&nbsp; {quarters_summary} &nbsp;&bull;&nbsp; 📚 <b>Geçmiş Veri: {h2h_n} Maç</b>
                        </div>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;">
                        {badge_html}
                        <div class="score-badge">{h_sc} - {a_sc} <span style="font-size:0.85rem; opacity:0.8;">({tot_sc})</span></div>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.65rem; margin-bottom:0.4rem;">
                    <div class="info-strip" style="cursor:help;" title="İki takımın geçmiş maçlarındaki çeyrek tempolarını ve şu anki canlı skoru birleştirerek maçın kaç kaç biteceğini 10.000 kez simüle eder.">
                        <div class="info-strip-label">🎯 Monte Carlo Maç Sonu Tahmini [?]</div>
                        <div class="info-strip-val">
                            {pred_h:.0f} - {pred_a:.0f} &nbsp;<span style="color:var(--score-text);">(Toplam: {pred_tot:.1f} Sayı)</span>
                        </div>
                    </div>
                    <div class="info-strip" style="cursor:help;" title="{smart['odd_trend_tip']}">
                        <div class="info-strip-label">⚖️ Barem Simetri & Oran Yönü [?]</div>
                        <div class="info-strip-val" style="color:var(--fav1-text);">
                            {sym_text}
                            <div style="font-size:0.74rem; color:var(--text-secondary); margin-top:0.12rem;">{smart['odd_trend_badge']}</div>
                        </div>
                    </div>
                    <div class="info-strip" style="cursor:help;" title="20-Sayı Kümesi: Yan yana iki baremin (toplam 20 sayılık geniş aralığın) gelme ihtimalidir. Akıllı Değer Nedeni: {val_reason}">
                        <div class="info-strip-label">🛡️ En Güçlü 20-Sayı Kümesi & Akıllı Değer [?]</div>
                        <div class="info-strip-val" style="color:var(--val-text);">
                            Küme: {clu_name} (%{clu_prob:.1f}) &nbsp;&bull;&nbsp; {val_status_lbl}
                        </div>
                    </div>
                    <div class="info-strip" style="cursor:help;" title="{h2h_live['q4_pace_tip']}">
                        <div class="info-strip-label">🤝 İki Takım Geçmişi & 4. Çeyrek Karakteri [?]</div>
                        <div class="info-strip-val" style="font-size:0.8rem; line-height:1.4;">
                            Son 5 Maç: <b>{h2h_live['last5_str']}</b> &bull; En Çok: <b>{h2h_live['top_cluster_str']}</b>
                            {pair_live_line}
                            <div style="font-size:0.74rem; color:var(--text-secondary); margin-top:0.12rem;">{h2h_live['q4_pace_badge']}</div>
                        </div>
                    </div>
                    <div class="info-strip" style="cursor:help;" title="3. çeyrek temposuna bakarak 4. çeyrekte sayı patlaması mı (yüksek skor) yoksa tempo çöküşü mü (rölanti/düşük skor) yaşanacağını tahmin eder.">
                        <div class="info-strip-label">⚡ Q3 ➔ Q4 Patlama / Çöküş Dedektörü [?]</div>
                        <div class="info-strip-val">
                            {q3_sig_label} <span style="font-size:0.76rem; color:var(--text-muted);">(Patlama %{q4_burst_p:.0f} / Çöküş %{q4_coll_p:.0f})</span>
                        </div>
                    </div>
                    <div class="info-strip" style="cursor:help;" title="Şu anki 1. Favori baremi merkeze alır; maç biraz düşük biterse 1 veya 2 Alt Barem, biraz yüksek biterse 1 veya 2 Üst Barem hangi aralığa denk geliyor onu açıkça gösterir.">
                        <div class="info-strip-label">🌊 Komşu Baremler & Dalga Hedefi (Alt / Favori / Üst) [?]</div>
                        <div class="info-strip-val" style="color:var(--score-text); font-size:0.81rem; line-height:1.45;">
                            ❄️ 2 Alt: <b>{w_bm2}</b> &bull; 📉 1 Alt: <b>{w_bm1}</b> &bull; 🎯 <b>Favori: {w_b0}</b> &bull; 📈 1 Üst: <b>{w_bp1}</b> &bull; 🔥 2 Üst: <b>{w_bp2}</b>
                            <div style="font-size:0.74rem; color:var(--fav1-text); margin-top:0.12rem;">🛡️ En Güçlü İkili Barem: <b>{w_rec_pair}</b> (Tutma İhtimali: %{w_rec_prob})</div>
                        </div>
                    </div>
                    {smart_html}
                </div>
                <div class="{barem_grid_cls}">
                    {boxes_html}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: MATCH-ISOLATED BAREM FLOW & CLOSING SYMMETRY TABLES (0 - close)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-hdr">📊 Maç Bazlı Ayrıştırılmış Barem & Simetri Akış Tablosu (0 - {CLOSE_MIN_INT}. Dakika)</div>',
        unsafe_allow_html=True,
    )

    if df_log.empty:
        st.info(
            f"Henüz kaydedilmiş {LEAGUE_CFG['short_name']} barem akış verisi bulunmuyor. Canlı bir maçta barem açıkken (0-{CLOSE_MIN_INT}. dk) her maç kendi bağımsız kutusunda buraya eklenecektir."
        )
    else:
        all_gids_chrono = list(df_log["game_id"].dropna().unique())
        gid_to_seq = {gid: idx + 1 for idx, gid in enumerate(all_gids_chrono)}
        gid_label_map = {}
        for gid in reversed(all_gids_chrono):
            g_rows = df_log[df_log["game_id"] == gid]
            g_nm = str(g_rows.iloc[-1].get("game_name", gid)).replace(" (cyber)", "")
            g_st_icon = "⏳" if (g_rows["status"] == "PENDING").any() else "✅"
            g_sc = str(g_rows.iloc[-1].get("actual_score", "")).replace("nan", "").strip()
            sc_txt = f" ({g_sc})" if g_sc else ""
            seq_no = gid_to_seq.get(gid, 0)
            gid_label_map[f"#{seq_no} {g_st_icon} {g_nm}{sc_txt} [{gid}]"] = gid

        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([1.05, 1.65, 1.15, 1.15, 1.15])
        with f_col1:
            status_filter = st.selectbox(
                "Durum Filtresi",
                ["Tümü", "⏳ Canlı / Bekleyen (PENDING)", "✅ Sonuçlanan (COMPLETED)"],
                index=0,
                key=f"s2_st_{ACTIVE_LEAGUE}",
            )
        with f_col2:
            unique_game_options = [f"Tüm Maçlar ({len(all_gids_chrono)} Maç Ayrı Tabloda)"] + list(gid_label_map.keys())
            game_filter = st.selectbox("Maç Filtresi (Maç No & ID)", unique_game_options, index=0, key=f"s2_gm_{ACTIVE_LEAGUE}")
        with f_col3:
            _min_opt_all = f"Tüm Dakikalar (0-{CLOSE_MIN_INT}. dk)"
            _min_opt_close = f"🔥 Kapanış Bölgesi ({MID_Q3_MIN}-{CLOSE_MIN_INT}. dk)"
            _min_opt_early = f"⚡ Erken/Orta Periyot (0-{MID_Q3_MIN - 1}. dk)"
            min_filter = st.selectbox(
                "Dakika Aralığı",
                [_min_opt_all, _min_opt_close, _min_opt_early],
                index=0,
                key=f"s2_mn_{ACTIVE_LEAGUE}",
            )
        with f_col4:
            layout_mode = st.selectbox(
                "Görünüm Modu",
                ["📑 Her Maç Ayrı Sekmede (Tab)", "🗂️ Her Maç Ayrı Kutuda (Alt Alta)"],
                index=0,
                key=f"s2_ly_{ACTIVE_LEAGUE}",
            )
        with f_col5:
            page_options = ["Son 10 Maç (Hızlı)", "Son 25 Maç", "26 - 50. Maçlar", "51 - 75. Maçlar", "76 - 100. Maçlar", "101+ Maçlar", "Tüm Maçları Göster (Sınırsız)"]
            page_filter = st.selectbox("Sayfa / Grup Seçimi", page_options, index=0, key=f"s2_pg_{ACTIVE_LEAGUE}")

        df_view = df_log.copy()
        if "PENDING" in status_filter:
            df_view = df_view[df_view["status"] == "PENDING"]
        elif "COMPLETED" in status_filter:
            df_view = df_view[df_view["status"] == "COMPLETED"]

        if not game_filter.startswith("Tüm Maçlar"):
            selected_gid = gid_label_map.get(game_filter)
            if selected_gid:
                df_view = df_view[df_view["game_id"] == selected_gid]

        if min_filter == _min_opt_close:
            df_view = df_view[
                df_view["check_time"]
                .astype(str)
                .str.extract(r"Minute\s+(\d+)", expand=False)
                .astype(float)
                .fillna(0)
                >= MID_Q3_MIN
            ]
        elif min_filter == _min_opt_early:
            df_view = df_view[
                df_view["check_time"]
                .astype(str)
                .str.extract(r"Minute\s+(\d+)", expand=False)
                .astype(float)
                .fillna(0)
                < MID_Q3_MIN
            ]

        if df_view.empty:
            st.info("Seçilen filtreye uygun barem kaydı bulunamadı.")
        else:
            game_ids_ordered = []
            for gid in df_view["game_id"].dropna().unique()[::-1]:
                if gid not in game_ids_ordered:
                    game_ids_ordered.append(gid)

            pending_gids = [
                gid for gid in game_ids_ordered if (df_view[df_view["game_id"] == gid]["status"] == "PENDING").any()
            ]
            completed_gids = [gid for gid in game_ids_ordered if gid not in pending_gids]
            sorted_gids_all = pending_gids + completed_gids

            if page_filter == "Son 10 Maç (Hızlı)":
                sorted_gids = sorted_gids_all[:10]
            elif page_filter == "Son 25 Maç":
                sorted_gids = sorted_gids_all[:25]
            elif page_filter == "26 - 50. Maçlar":
                sorted_gids = sorted_gids_all[25:50]
            elif page_filter == "51 - 75. Maçlar":
                sorted_gids = sorted_gids_all[50:75]
            elif page_filter == "76 - 100. Maçlar":
                sorted_gids = sorted_gids_all[75:100]
            elif page_filter == "101+ Maçlar":
                sorted_gids = sorted_gids_all[100:]
            else:
                sorted_gids = sorted_gids_all

            st.caption(
                f"📋 **Gösterilen Maç Sayısı:** `{len(sorted_gids)}` / Toplam `{len(sorted_gids_all)}` Maç "
                f"(Sekme sınırı kaldırıldı — sağa kaydırarak veya üstteki **Maç Filtresi** / **Alt Alta Görünüm** ile tüm maçları inceleyebilirsiniz)."
            )

            def _build_single_match_df(sub_df: pd.DataFrame) -> pd.DataFrame:
                sub_copy = sub_df.copy()
                sub_copy["_min_num"] = (
                    sub_copy["check_time"]
                    .astype(str)
                    .str.extract(r"Minute\s+(\d+)", expand=False)
                    .astype(float)
                    .fillna(0)
                )
                sub_copy = sub_copy.sort_values(by="_min_num", ascending=True)

                rows = []
                for _, r in sub_copy.iterrows():
                    st_val = str(r.get("status", "PENDING"))
                    act_rng = str(r.get("actual_range", "")).replace("nan", "").strip()
                    act_sc = str(r.get("actual_score", "")).replace("nan", "").strip()
                    fav_rng = str(r.get("highest_prob_range", "")).strip()
                    val_rng = str(r.get("value_range", "")).strip()
                    clu_rng = str(r.get("best_cluster_name", "")).strip()
                    clu_prb = r.get("best_cluster_prob", 0.0)

                    if st_val == "COMPLETED" and act_rng:
                        hits = []
                        if act_rng == fav_rng:
                            hits.append("⭐ FAV TUTTU")
                        if act_rng == val_rng:
                            hits.append("🔥 VALUE TUTTU")
                        if check_cluster_hit(clu_rng, act_rng, league=ACTIVE_LEAGUE):
                            hits.append("🛡️ KÜME TUTTU")
                        outcome_badge = " | ".join(hits) if hits else f"❌ ({act_rng})"
                    else:
                        outcome_badge = "⏳ Bekliyor"

                    def _fmt_cell(odd_col, prob_col, b_lbl=""):
                        o_v = float(r.get(odd_col, 99.0) or 99.0)
                        p_v = float(r.get(prob_col, 0.0) or 0.0)
                        if o_v >= 90.0:
                            if b_lbl and b_lbl == fav_rng:
                                return f"🔒 Kilit (%{p_v:.0f})"
                            return f"- (%{p_v:.0f})"
                        return f"{o_v:.2f} (%{p_v:.0f})"

                    fav_col_name = BAREM_TO_ODD_COL.get(fav_rng, "")
                    fav_raw_odd = float(r.get(fav_col_name, 0.0) or 0.0) if fav_col_name else 0.0
                    fav_cell_disp = f"{fav_rng} (🔒 Kilit)" if fav_raw_odd >= 90.0 else fav_rng

                    row_dict = {
                        "Dakika": r.get("check_time", ""),
                        "Anlık Skor": r.get("current_score", ""),
                    }
                    for b_name, _, _ in BAREM_RANGES:
                        short_b = LEAGUE_CFG["short_labels"].get(b_name, b_name)
                        row_dict[short_b] = _fmt_cell(BAREM_TO_ODD_COL[b_name], BAREM_TO_PROB_COL[b_name], b_name)
                    row_dict.update(
                        {
                            "⭐ 1xBet Fav": fav_cell_disp,
                            "🔥 Value Barem": val_rng,
                            "🛡️ 20-Sayı Kümesi": f"{clu_rng} (%{float(clu_prb or 0):.0f})",
                            "Maç Sonu": f"{act_sc} ({act_rng})" if act_sc else "-",
                            "Sonuç Analizi": outcome_badge,
                        }
                    )
                    rows.append(row_dict)
                return pd.DataFrame(rows)

            def _render_match_flow_card(gid: str):
                g_sub = df_view[df_view["game_id"] == gid]
                if g_sub.empty:
                    return
                df_m = _build_single_match_df(g_sub)
                g_name = str(g_sub.iloc[-1].get("game_name", gid))
                g_status = str(g_sub.iloc[-1].get("status", "PENDING"))
                act_sc = str(g_sub.iloc[-1].get("actual_score", "")).replace("nan", "").strip()
                act_rng = str(g_sub.iloc[-1].get("actual_range", "")).replace("nan", "").strip()
                seq_no = gid_to_seq.get(gid, 0)

                first_fav = str(df_m.iloc[0]["⭐ 1xBet Fav"])
                last_fav = str(df_m.iloc[-1]["⭐ 1xBet Fav"])
                last_clu = str(df_m.iloc[-1]["🛡️ 20-Sayı Kümesi"])
                last_min = str(df_m.iloc[-1]["Dakika"])

                if g_status == "COMPLETED" and act_rng:
                    status_pill_html = (
                        '<span style="background:var(--pill-bg); border:1px solid var(--pill-border); '
                        'color:var(--pill-text); padding:0.24rem 0.7rem; border-radius:8px; font-size:0.75rem; font-weight:700;">'
                        f"✅ SONUÇLANDI: {act_sc} &bull; Kazanan Barem: {act_rng}</span>"
                    )
                else:
                    status_pill_html = (
                        '<span style="background:var(--val-bg); border:1px solid var(--val-border); '
                        'color:var(--val-text); padding:0.24rem 0.7rem; border-radius:8px; font-size:0.75rem; font-weight:700;">'
                        f"⏳ CANLI / SONUÇ BEKLENİYOR (Son Kayıt: {last_min})</span>"
                    )

                st.markdown(
                    f"""
                    <div style="background:var(--bg-card); border:1px solid var(--border-subtle); border-left:4px solid var(--score-text); border-radius:10px; padding:0.75rem 1.1rem; margin-top:0.65rem; margin-bottom:0.35rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem; box-shadow:var(--shadow-card);">
                        <div>
                            <span style="font-size:1.0rem; font-weight:800; color:var(--text-primary);">🏀 #{seq_no} &bull; {g_name}</span>
                            <span style="font-size:0.76rem; color:var(--text-muted); margin-left:0.6rem; font-family:'JetBrains Mono',monospace;">[{gid}] &bull; {len(df_m)} Dakika Kaydı</span>
                            <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:0.2rem;">
                                📈 <b>Favori Barem Akışı:</b> {first_fav} ➔ <b>{last_fav}</b> &nbsp;&bull;&nbsp; 🛡️ <b>Son Küme:</b> {last_clu}
                            </div>
                        </div>
                        <div>{status_pill_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    df_m,
                    use_container_width=True,
                    hide_index=True,
                    height=min(340, 52 + 35 * len(df_m)),
                )

            if not sorted_gids:
                st.info("Seçilen sayfa aralığında maç bulunmuyor.")
            elif "Sekmede" in layout_mode and len(sorted_gids) > 1:
                tab_labels = []
                for gid in sorted_gids:
                    g_sub = df_view[df_view["game_id"] == gid]
                    g_name = str(g_sub.iloc[-1].get("game_name", gid)).replace(" (cyber)", "")
                    g_st = "⏳" if (g_sub["status"] == "PENDING").any() else "✅"
                    seq_no = gid_to_seq.get(gid, 0)
                    tab_labels.append(f"#{seq_no} {g_st} {g_name} ({len(g_sub)})")
                m_tabs = st.tabs(tab_labels)
                for idx_t, gid in enumerate(sorted_gids):
                    with m_tabs[idx_t]:
                        _render_match_flow_card(gid)
            else:
                for gid in sorted_gids:
                    _render_match_flow_card(gid)


render_live_dashboard()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2.5: v8.1 LIVE TRIAL BANKROLL TRACKER (3.000 TL FRESH START FROM NOV 26 13:00)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _build_v81_combined_trial_records() -> tuple[list[dict], list[dict]]:
    """Evaluates all NBA 2K26 + Cyber Euro League matches with v8.1 rules and returns (completed_eval, live_decided)."""

    def _rule_nba_v81(r: dict):
        gap = r["gap"]
        o1, o2 = r["o1"], r["o2"]
        f1, f2 = r["f1"], r["f2"]
        req1 = r["req1"]
        up = r["up"]
        f2_up = r["f2_is_upper"]
        ps = r["ps"]
        odiff = r["odiff"]
        if gap < 6:
            return None, f"Başa baş maç ({gap} sayı fark < 6 sınırı)"
        if o1 < 1.50:
            return None, f"1.Fav oranı çok düşük ({o1:.2f} < 1.50 taban sınırı)"
        if 1.50 <= o1 < 1.85:
            if gap <= 24 and ps != 1 and not f2_up and req1 <= 59.5:
                return (1, f1, o1), "1.50-1.85 bandı güvenli 1.Fav"
            reasons = []
            if gap > 24:
                reasons.append(f"Fark {gap} > 24")
            if ps == 1:
                reasons.append("Önceki maç üst bareme kaydı (ps=+1)")
            if f2_up:
                reasons.append("2.Fav üst baremde")
            if req1 > 59.5:
                reasons.append(f"İstenen sayı yüksek ({req1:.1f} > 59.5)")
            return None, "1.50-1.85 filtre dışı: " + ", ".join(reasons)
        if 1.85 <= o1 < 2.10:
            if ps != 1 and gap <= 32:
                if gap >= 20 and f2_up and req1 < 47.5:
                    return None, f"20+ farkta ({gap} fark) 1.Fav alt barem ve sadece {req1:.1f} sayı istiyor (< 47.5 tuzağı)"
                return (1, f1, o1), "1.85-2.10 ana merkez 1.Fav"
            if ps == 1:
                return None, "Önceki maç üst bareme kaydığı için (ps=+1) 1.85-2.10 pas"
            return None, f"Aşırı fark ({gap} > 32)"
        if 2.10 <= o1 < 2.30:
            if gap >= 25 and f2_up and req1 <= 53.5:
                return (1, f1, o1), "25+ farkta düşük kalan sayı 1.Fav"
            if req1 <= 48.5 and ps != 1 and gap <= 24:
                return (1, f1, o1), f"Düşük kalan sayı ({req1:.1f} <= 48.5) 1.Fav"
            if 48.5 < req1 <= 56.5 and 10 <= gap <= 24 and o2 <= 3.05:
                if gap >= 16 and f2_up and req1 <= 50.5:
                    return (1, f1, o1), "16-24 farkta düşük kalan sayı 1.Fav"
                if up or (10 <= gap <= 15) or not f2_up or ps != 0:
                    return (2, f2, o2), f"2.10-2.30 bandı 2.Fav ({req1:.1f} kalan sayı)"
            return None, f"2.10-2.30 seçici filtre dışı (Fark:{gap}, İstenen:{req1:.1f}, 2.Fav:{o2:.2f}, ps:{ps:+d})"
        if 2.30 <= o1 < 2.50:
            if 16 <= gap <= 28 and not f2_up and req1 >= 51.5 and o2 <= 3.10:
                return (2, f2, o2), f"16-28 fark yavaşlaması ({gap} fark, 1.Fav {req1:.1f} istiyor ➔ Alt 2.Fav)"
            if req1 <= 50.5 and ps != -1 and gap <= 24:
                return (1, f1, o1), f"2.30-2.50 düşük kalan sayı ({req1:.1f}) 1.Fav"
            if gap <= 15 and ps >= 0 and req1 <= 56.5 and (odiff <= 0.25 or up or ps == 1 or req1 <= 54.5):
                return (1, f1, o1), "2.30-2.50 dengeli 1.Fav"
            if req1 > 56.5 and 10 <= gap <= 24 and not f2_up and o2 <= 3.05:
                return (2, f2, o2), f"Yüksek kalan sayı ({req1:.1f} > 56.5) ➔ Alt 2.Fav"
            return None, f"2.30-2.50 seçici filtre dışı (Fark:{gap}, İstenen:{req1:.1f}, 2.Fav:{o2:.2f})"
        if o1 >= 2.50:
            if (ps == 1 or req1 <= 50.5) and 10 <= gap <= 18 and o2 <= 3.10:
                return (2, f2, o2), "2.50+ makas dar ikiz barem 2.Fav"
            return None, f"1.Fav oranı yüksek ({o1:.2f} >= 2.50) ve şartlar sağlanmadı"
        return None, "Seçici pas"

    def _rule_euro_v81(r: dict):
        gap = r["gap"]
        o1, o2 = r["o1"], r["o2"]
        f1, f2 = r["f1"], r["f2"]
        req1 = r["req1"]
        f2_up = r["f2_is_upper"]
        ps = r["ps"]
        odiff = r["odiff"]
        if gap < 6:
            return None, f"Başa baş maç ({gap} sayı fark < 6 sınırı)"
        if gap >= 29:
            return None, f"Aşırı fark ({gap} >= 29 kopmuş maç)"
        if o1 < 1.20:
            return None, f"1.Fav oranı çok düşük ({o1:.2f} < 1.20 taban sınırı)"
        if o1 >= 2.30:
            return None, f"EuroLeague kararsız piyasa (1.Fav oranı {o1:.2f} >= 2.30 tavan sınırı)"
        if 1.20 <= o1 < 1.50:
            if gap <= 15 and f2_up and req1 <= 38.5:
                return (1, f1, o1), "1.20-1.50 güvenli 1.Fav"
            return None, f"1.20-1.50 filtre dışı (Fark:{gap}, İstenen:{req1:.1f})"
        if 1.50 <= o1 < 1.85:
            if f2_up and req1 <= 37.5:
                return (1, f1, o1), f"1.50-1.85 düşük kalan sayı ({req1:.1f}) 1.Fav"
            if gap >= 21 and req1 <= 41.5:
                return (1, f1, o1), "21+ farkta 1.50-1.85 1.Fav"
            return None, f"1.50-1.85 filtre dışı (Fark:{gap}, İstenen:{req1:.1f})"
        if 1.85 <= o1 < 2.10:
            if 16 <= gap <= 28 and not f2_up and req1 >= 37.5 and o2 <= 3.40:
                return (2, f2, o2), f"16-28 fark yavaşlaması ({gap} fark, 1.Fav {req1:.1f} sayı istiyor ➔ Alt 2.Fav)"
            if gap <= 15 and ((f2_up and req1 <= 39.5) or (odiff <= 0.18 and req1 <= 37.5)):
                return (1, f1, o1), f"1.85-2.10 normal fark ({gap}) ve düşük kalan sayı ({req1:.1f}) 1.Fav"
            return None, f"1.85-2.10 filtre dışı (Fark:{gap}, İstenen:{req1:.1f}, 2.Fav:{f2}@{o2:.2f})"
        if 2.10 <= o1 < 2.30:
            if req1 <= 36.5 and (f2_up or ps == 1 or gap >= 16):
                return (1, f1, o1), f"2.10-2.30 düşük kalan sayı ({req1:.1f} <= 36.5) 1.Fav"
            if 36.5 < req1 <= 41.5 and o2 <= 3.40 and gap <= 24:
                return (2, f2, o2), f"2.10-2.30 yüksek kalan sayı ({req1:.1f} > 36.5) ➔ 2.Fav"
            return None, f"2.10-2.30 filtre dışı (Fark:{gap}, İstenen:{req1:.1f}, 2.Fav:{o2:.2f})"
        return None, "Seçici pas"

    all_records = []
    live_decided = []

    for lg in ("nba", "euro"):
        cfg = get_league_config(lg)
        paths = _get_scraper_paths(lg)
        files = cfg["files"]
        df_m = load_completed_matches(lg)
        m_map = {}
        if not df_m.empty:
            for _, mr in df_m.iterrows():
                gid = str(mr.get("game_id", "")).strip()
                try:
                    tot = int(float(mr.get("total_score", 0) or 0))
                    ht = int(float(mr.get("home_total", 0) or 0))
                    at = int(float(mr.get("away_total", 0) or 0))
                except Exception:
                    continue
                if tot >= int(cfg["min_total_valid"]):
                    m_map[gid] = (tot, f"{ht}-{at}", get_range_from_score(tot, league=lg), str(mr.get("date", "")).strip())

        frames = [
            _read_csv_safe(paths["csv_path"]),
            _read_csv_safe(os.path.join(BASE_DIR, files["log_seed"])),
            _read_csv_safe(os.path.join(BASE_DIR, files["log_cloud"])),
            _fetch_github_csv(files["log_cloud"]),
        ]
        frames = [f for f in frames if not f.empty and "game_id" in f.columns and "check_time" in f.columns]
        if not frames:
            continue
        df_s = pd.concat(frames, ignore_index=True)
        df_s["game_id"] = df_s["game_id"].astype(str).str.strip()
        df_s = df_s.drop_duplicates(subset=["game_id", "check_time"], keep="last").copy()

        for idx, sr in df_s.iterrows():
            gid = str(sr["game_id"]).strip()
            if gid in m_map:
                tot, sc, act_b, dt = m_map[gid]
                if pd.isna(sr.get("actual_range")) or str(sr.get("actual_range", "")).strip() in ("", "nan", "None"):
                    df_s.at[idx, "actual_score"] = sc
                    df_s.at[idx, "actual_range"] = act_b
                    df_s.at[idx, "status"] = "COMPLETED"

        df_s["_min"] = df_s["check_time"].astype(str).str.extract(r"Minute\s+(\d+)")[0].fillna(-1).astype(int)
        df_s["_gid_num"] = df_s["game_id"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)
        df_s = df_s[df_s["_gid_num"] >= int(cfg["min_clean_gid"])].sort_values(["_gid_num", "_min"]).reset_index(drop=True)

        dec_min = int(cfg["decision_min"])
        close_start = int(cfg["closing_start_min"])
        barem_names = [b[0] for b in cfg["barem_ranges"]]
        b_cols = cfg["barem_odd_cols"]
        p_cols = cfg["barem_prob_cols"]
        mid_map = {}
        for bn, blo, bhi in cfg["barem_ranges"]:
            elo = max(float(cfg["bar_lower_floor"]), float(blo))
            ehi = min(float(bhi), float(cfg["bar_upper_cap"]))
            mid_map[bn] = (elo + ehi) / 2.0

        ps = 0
        ps2 = 0
        for gid, grp in df_s.groupby("game_id", sort=False):
            cands = grp[(grp["_min"] >= close_start) & (grp["_min"] <= dec_min)]
            if cands.empty:
                continue
            last_r = None
            for _, cr in cands.iloc[::-1].iterrows():
                if any(1.001 < float(cr.get(b_cols[bn], 99.0) or 99.0) < 85.0 for bn in barem_names):
                    last_r = cr
                    break
            if last_r is None:
                continue
            o_raw = {bn: float(last_r.get(b_cols[bn], 99.0) or 99.0) for bn in barem_names}
            p_raw = {bn: float(last_r.get(p_cols[bn], 0.0) or 0.0) for bn in barem_names}
            om, _ = resolve_effective_barem_odds(o_raw, p_raw, league=lg)
            if len(om) < 2:
                continue
            arr = sorted(om.items(), key=lambda x: x[1])
            f1, o1 = arr[0]
            f2, o2 = arr[1]
            i1 = barem_names.index(f1)
            i2 = barem_names.index(f2)
            om1 = om.get(barem_names[i1 - 1], 99.0) if i1 > 0 else 99.0
            op1 = om.get(barem_names[i1 + 1], 99.0) if i1 < len(barem_names) - 1 else 99.0
            sc_str = str(last_r.get("current_score", "0-0"))
            if "-" not in sc_str:
                continue
            try:
                s1, s2 = map(int, sc_str.split("-"))
            except Exception:
                continue
            cur_tot = s1 + s2
            gap = abs(s1 - s2)
            full_dt = m_map.get(gid, (0, "", "", str(last_r.get("date", "")).strip()))[3]
            if len(full_dt) < 10:
                full_dt = str(last_r.get("date", "")).strip()

            rec = {
                "gid": gid,
                "gid_num": int(gid.replace("CB-", "")),
                "lg": "NBA" if lg == "nba" else "EURO",
                "teams": str(last_r.get("game_name", "")).replace(" (cyber)", ""),
                "full_dt": full_dt,
                "day": full_dt[:10],
                "min": int(last_r["_min"]),
                "cur_sc": sc_str,
                "cur_tot": cur_tot,
                "gap": gap,
                "f1": f1,
                "o1": o1,
                "f2": f2,
                "o2": o2,
                "i1": i1,
                "i2": i2,
                "f2_is_upper": i2 > i1,
                "up": op1 < om1,
                "odiff": o2 - o1,
                "req1": mid_map[f1] - cur_tot,
                "req2": mid_map[f2] - cur_tot,
                "ps": ps,
                "ps2": ps2,
            }
            dec, reason = _rule_nba_v81(rec) if lg == "nba" else _rule_euro_v81(rec)
            rec["dec"] = dec
            rec["reason"] = reason

            act_b = str(last_r.get("actual_range", "")).strip()
            if act_b not in barem_names:
                if int(last_r["_min"]) == dec_min:
                    live_decided.append(rec)
                continue

            act_sc = str(last_r.get("actual_score", "0-0"))
            a1, a2 = map(int, act_sc.split("-")) if "-" in act_sc else (0, 0)
            fin_tot = a1 + a2
            i_act = barem_names.index(act_b)
            shift = i_act - i1
            rec["fin_sc"] = act_sc
            rec["fin_tot"] = fin_tot
            rec["act_b"] = act_b
            all_records.append(rec)

            ps2 = ps
            ps = 1 if shift > 0 else (-1 if shift < 0 else 0)

    all_records.sort(key=lambda x: x["gid_num"])
    return all_records, live_decided


@st.fragment(run_every="20s" if auto_ui_refresh else None)
def render_v81_live_trial_bankroll():
    st.markdown(
        '<div class="section-hdr">💰 v8.1 Canlı Deneme Kasası — 3.000 TL Başlangıç (Oynanan & Pas Geçilen Maçlar Anlık Kazanç Tablosu)</div>',
        unsafe_allow_html=True,
    )
    all_eval, live_decided = _build_v81_combined_trial_records()
    if not all_eval:
        st.info("Canlı deneme kasası için maç verisi yükleniyor...")
        return

    c_p1, c_p2, c_p3 = st.columns([2.3, 1.8, 1.1])
    with c_p1:
        period_mode = st.radio(
            "📅 Takip Başlangıç Noktası:",
            [
                "🆕 Yeni Sistem (v8.1) Kurulduktan Sonrası (26 Kasım 13:00+)",
                "📆 Bugünün Tüm Maçları (26 Kasım 00:00+)",
                "🏆 Tüm 15 Günlük Canlı Veri (12 Kasım – Bugün)",
            ],
            index=0,
            horizontal=False,
            key="v81_trial_period_mode",
        )
    with c_p2:
        filter_mode = st.radio(
            "🔎 Maç Filtresi:",
            [
                "📋 Tüm Maçlar (Oynanan + Pas Geçilen)",
                "🎯 Sadece Oynanan Kuponlar (OYNA)",
                "⏭️ Sadece Pas Geçilenler (PAS)",
            ],
            index=0,
            horizontal=False,
            key="v81_trial_filter_mode",
        )
    with c_p3:
        sort_newest = st.toggle("🔽 En Yeni Maç Üstte", value=False, key="v81_trial_sort_newest")
        include_1235 = st.toggle(
            "12:35-12:40 Maçlarını Dahil Et",
            value=False,
            help="Sistem güncellenirken sahada oynanmakta olan 12:35 (Milano-Baskonia) ve 12:40 (Knicks-Celtics) maçlarını da yeni kasaya dahil eder.",
            key="v81_trial_inc_1235",
        )

    if "Yeni Sistem" in period_mode:
        cutoff_dt = "2026-11-26 12:30" if include_1235 else "2026-11-26 13:00"
        sub_eval = [m.copy() for m in all_eval if m["full_dt"] >= cutoff_dt]
    elif "Bugünün" in period_mode:
        sub_eval = [m.copy() for m in all_eval if m["day"] >= "2026-11-26"]
    else:
        sub_eval = [m.copy() for m in all_eval]

    def _get_tier_3000(b_val: float):
        if b_val < 4200:
            return (1, 150, 120)
        if b_val < 5700:
            return (2, 210, 170)
        if b_val < 7600:
            return (3, 280, 220)
        if b_val < 10400:
            return (4, 380, 300)
        if b_val < 14200:
            return (5, 520, 420)
        if b_val < 19400:
            return (6, 710, 570)
        return (7, 850, 680)

    k_flat = 3000.0
    k_tier = 3000.0
    p_idx = 0
    won_cnt = 0
    lost_cnt = 0
    pass_cnt = 0
    saved_cnt = 0
    nba_p = nba_w = eu_p = eu_w = 0
    daily_stats = {}

    for seq_i, m in enumerate(sub_eval, 1):
        m["seq_i"] = seq_i
        d = m["day"]
        if d not in daily_stats:
            daily_stats[d] = {
                "tot": 0, "p": 0, "pas": 0, "saved": 0,
                "w": 0, "l": 0, "nba_p": 0, "nba_w": 0, "eu_p": 0, "eu_w": 0,
                "net_f": 0.0, "k_flat": k_flat, "net_t": 0.0, "k_tier": k_tier,
            }
        ds = daily_stats[d]
        ds["tot"] += 1

        if m["dec"] is not None:
            p_idx += 1
            f_num, pick_b, pick_o = m["dec"]
            won = (pick_b == m["act_b"])
            st_f = 150.0 if f_num == 1 else 120.0
            net_f = st_f * (pick_o - 1.0) if won else -st_f
            k_flat += net_f

            tk, s1_t, s2_t = _get_tier_3000(k_tier)
            st_t = float(s1_t if f_num == 1 else s2_t)
            net_t = st_t * (pick_o - 1.0) if won else -st_t
            k_tier += net_t

            if won:
                won_cnt += 1
                ds["w"] += 1
            else:
                lost_cnt += 1
                ds["l"] += 1

            if m["lg"] == "NBA":
                nba_p += 1
                ds["nba_p"] += 1
                if won:
                    nba_w += 1
                    ds["nba_w"] += 1
            else:
                eu_p += 1
                ds["eu_p"] += 1
                if won:
                    eu_w += 1
                    ds["eu_w"] += 1

            ds["p"] += 1
            ds["net_f"] += net_f
            ds["k_flat"] = k_flat
            ds["net_t"] += net_t
            ds["k_tier"] = k_tier

            m["kupon_lbl"] = f"Kupon #{p_idx}"
            m["pick_lbl"] = f"OYNA: {f_num}.Fav {pick_b} (@{pick_o:.2f}) — {m['reason']}"
            m["stake_lbl"] = f"{int(st_f)} TL"
            m["res_lbl"] = "✅ KAZANDI" if won else "❌ KAYBETTİ"
            m["net_f"] = net_f
            m["k_flat"] = k_flat
            m["k_tier"] = k_tier
        else:
            pass_cnt += 1
            ds["pas"] += 1
            fav1_lost = (m["f1"] != m["act_b"])
            if fav1_lost:
                saved_cnt += 1
                ds["saved"] += 1
            ds["k_flat"] = k_flat
            ds["k_tier"] = k_tier

            m["kupon_lbl"] = "⏭️ PAS"
            m["pick_lbl"] = f"PAS GEÇ — {m['reason']}"
            m["stake_lbl"] = "0 TL"
            m["res_lbl"] = f"🛡️ Zarardan Korudu ({m['act_b']} bitti)" if fav1_lost else f"⏭️ Pas ({m['act_b']} bitti)"
            m["net_f"] = 0.0
            m["k_flat"] = k_flat
            m["k_tier"] = k_tier

    net_flat_tot = k_flat - 3000.0
    net_tier_tot = k_tier - 3000.0
    hit_pct = (won_cnt / p_idx * 100.0) if p_idx > 0 else 0.0
    roi_pct = (net_flat_tot / 3000.0) * 100.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "💰 3.000 TL Deneme Kasası (Sabit 150/120)",
        f"{k_flat:,.1f} TL",
        f"{net_flat_tot:+,.1f} TL (%{roi_pct:+.1f} Kasa Artışı)",
    )
    k2.metric(
        "🎯 Oynanan Kupon & İsabet",
        f"{p_idx} Kupon ({won_cnt}K / {lost_cnt}Y)",
        f"%{hit_pct:.1f} İsabet Oranı",
    )
    k3.metric(
        "🛡️ Pas Geçilen & Korunan Zarar",
        f"{pass_cnt} Maç Pas Geçildi",
        f"{saved_cnt} Maçta 1.Fav Patladı (+{saved_cnt * 150:,.0f} TL Korudu)",
    )
    k4.metric(
        "🏀🇪🇺 Lig Bazlı İsabet",
        f"NBA: {nba_w}/{nba_p} • Euro: {eu_w}/{eu_p}",
        f"Toplam İncelenen: {len(sub_eval)} Maç",
    )
    k5.metric(
        "🚀 Kademeli Kasa (Kademe 1–7)",
        f"{k_tier:,.1f} TL",
        f"{net_tier_tot:+,.1f} TL Net Kâr",
    )

    if live_decided:
        live_html_items = []
        for lm in live_decided:
            lg_b = "🏀 NBA" if lm["lg"] == "NBA" else "🇪🇺 EURO"
            if lm["dec"] is not None:
                fn_l, pb_l, po_l = lm["dec"]
                st_l = 150 if fn_l == 1 else 120
                dec_badge = f"<b style='color:#10b981;'>🎯 CANLI OYNA: {fn_l}.Fav {pb_l} (@{po_l:.2f}) — {st_l} TL</b> ({lm['reason']})"
            else:
                dec_badge = f"<b style='color:#f59e0b;'>⏭️ CANLI PAS GEÇ:</b> {lm['reason']}"
            live_html_items.append(
                f"<div style='margin-top:0.25rem;'>• <b>{lg_b} | {lm['teams']}</b> "
                f"(Karar Skoru: <code>{lm['cur_sc']}</code> | Fark: <b>{lm['gap']}</b> | 1.Fav: <code>{lm['f1']} @{lm['o1']:.2f}</code>) ➔ {dec_badge} <i>(4. Çeyrek oynanıyor, maç sonu bekleniyor)</i></div>"
            )
        st.markdown(
            f"""
            <div style="background:var(--bg-card); border:1px solid var(--val-border); border-left:4px solid #10b981; border-radius:10px; padding:0.75rem 1rem; margin:0.5rem 0 0.8rem 0;">
                <div style="font-size:0.88rem; font-weight:800; color:var(--text-primary);">🔴 Şu An Karar Dakikasını Geçmiş & 4. Çeyreği Oynanan Canlı Maçlar ({len(live_decided)} Maç)</div>
                <div style="font-size:0.82rem; color:var(--text-secondary);">{''.join(live_html_items)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(daily_stats) >= 1:
        st.markdown("##### 📅 Gün Gün Kazanç & İsabet Özeti (`3.000 TL` Başlangıç Kasası)")
        day_rows = []
        for d in sorted(daily_stats.keys(), reverse=sort_newest):
            s = daily_stats[d]
            d_hit = (s["w"] / s["p"] * 100.0) if s["p"] > 0 else 0.0
            day_rows.append(
                {
                    "Tarih": d,
                    "İncelenen Maç": s["tot"],
                    "Oynanan (OYNA)": s["p"],
                    "Pas Geçilen (PAS)": f"{s['pas']} ({s['saved']} Zarardan Korudu)",
                    "NBA (K/O)": f"{s['nba_w']}/{s['nba_p']}",
                    "Euro (K/O)": f"{s['eu_w']}/{s['eu_p']}",
                    "Toplam K/Y": f"{s['w']}K / {s['l']}Y",
                    "İsabet %": f"%{d_hit:.1f}",
                    "Günlük Net (Sabit 150/120)": f"{s['net_f']:+,.1f} TL",
                    "Güncel Sabit Kasa": f"{s['k_flat']:,.1f} TL",
                    "Güncel Kademeli Kasa": f"{s['k_tier']:,.1f} TL",
                }
            )
        st.dataframe(pd.DataFrame(day_rows), use_container_width=True, hide_index=True)

    view_list = sub_eval
    if "Sadece Oynanan" in filter_mode:
        view_list = [m for m in sub_eval if m["dec"] is not None]
    elif "Sadece Pas" in filter_mode:
        view_list = [m for m in sub_eval if m["dec"] is None]

    if sort_newest:
        view_list = list(reversed(view_list))

    st.markdown(f"##### 🏀🇪🇺 Maç Maç Canlı Takip Tablosu (`{len(view_list)}` Maç Gösteriliyor)")
    table_rows = []
    for m in view_list:
        table_rows.append(
            {
                "Sıra": m["seq_i"],
                "Kupon / Durum": m["kupon_lbl"],
                "Başlangıç Saati": m["full_dt"],
                "Lig": "🏀 NBA" if m["lg"] == "NBA" else "🇪🇺 EURO",
                "Maç": m["teams"],
                "Karar Skoru (Fark)": f"{m['cur_sc']} ({m['gap']} Fark)",
                "1. Favori (İstenen Sayı)": f"{m['f1']} @{m['o1']:.2f} ({m['req1']:.1f} sayı)",
                "2. Favori": f"{m['f2']} @{m['o2']:.2f}",
                "Sistem Kararı & Kural Gerekçesi": m["pick_lbl"],
                "Bahis Tutarı": m["stake_lbl"],
                "Maç Sonu (Toplam ➔ Barem)": f"{m['fin_sc']} ({m['fin_tot']} ➔ {m['act_b']})",
                "Sonuç": m["res_lbl"],
                "Net Kâr / Zarar": f"{m['net_f']:+,.1f} TL" if m["dec"] is not None else "0,0 TL",
                "3.000 TL Deneme Kasası": f"{m['k_flat']:,.1f} TL",
            }
        )
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        height=min(620, 58 + 36 * max(1, len(table_rows))),
    )


render_v81_live_trial_bankroll()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ADVANCED ANALYTICS & HISTORICAL ARCHIVE (ALL 8 MODULES)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="section-hdr">🧠 {LEAGUE_CFG["short_name"]} — Gelişmiş İstatistik, Geçmiş Maçlar & Kasa/Doğrulama Merkezi</div>',
    unsafe_allow_html=True,
)

show_archive_modules = st.toggle(
    "📂 Alt 8 Detaylı Arşiv & İstatistik Sekmesini Aç (10.700+ Geçmiş Maç Analizi & Simülatörler)",
    value=False,
    help="Sitenin canlı takip sırasında çok hızlı çalışması için bu 8 ağır arşiv sekmesi varsayılan olarak kapalıdır. İncelemek istediğinizde tek tıkla açabilirsiniz.",
    key="toggle_heavy_archive_8_tabs",
)
if not show_archive_modules:
    st.caption(
        "⚡ **Hızlı Canlı Takip Modu Aktif:** Sayfanın yağ gibi akması için 10.700+ maçlık 8 alt arşiv sekmesi beklemede. "
        "Detaylı arşiv, H2H veya simülatörleri incelemek istediğinizde yukarıdaki anahtarı açabilirsiniz."
    )
    st.stop()

df_completed_archive = load_completed_matches(ACTIVE_LEAGUE)
df_sym_all = load_symmetry_log(ACTIVE_LEAGUE)
live_state_now = load_live_state(ACTIVE_LEAGUE)
live_matches_now = live_state_now.get("live_matches", [])

tab_dna, tab_sym, tab_q3q4, tab_bankroll, tab_sim, tab_smart35, tab_pair, tab_guard = st.tabs(
    [
        f"🏆 1. Geçmiş Maçlar & Takım Barem DNA ({len(df_completed_archive)} Maç)",
        f"⚖️ 2. Kapanış Oranı ({MID_Q3_MIN}-{CLOSE_MIN_INT}. Dk) Simetri & Tuzak Analizi",
        "⚡ 3. Q3 ➔ Q4 Patlama / Çöküş Dedektörü",
        "💰 4. Canlı Kupon, Kasa (ROI) & Çifte Barem (Dutching) Hesaplayıcı",
        "🧮 5. Manuel Canlı Tempo Simülatörü",
        f"🤖 6. {DEC_MIN}. Dk Akıllı Kural & Sanal Kasa Simülatörü",
        "🤝 7. Takım Çifti Geçmişi (H2H)",
        "🛡️ 8. Veri Doğrulama — Hata Bekçisi",
    ]
)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 6: DECISION-MINUTE SMART RULES, TRANSITIONS & VIRTUAL BANKROLL SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_smart35:
    st.markdown(
        f"##### 🤖 {DEC_MIN}. Dakika Ne Oynamalı? — Gerçek Maçlardan Çıkarılan Sade Kurallar & Kasa Tablosu",
        help=f"Bu sekmedeki tüm tablolar ve kasa sonuçları, {LEAGUE_CFG['short_name']} maçlarının {DEC_MIN}. dakika 1xBet kapanış oranlarından otomatik hesaplanır.",
    )
    sim35 = simulate_35m_strategies(df_sym_all, league=ACTIVE_LEAGUE)
    if sim35["total_matches"] == 0:
        st.info("Sonuçlanmış canlı kapanış kaydı bekleniyor. Maçlar bittikçe bu sekme otomatik dolar.")
    else:
        st.caption(
            f"📊 İncelenen gerçek maç sayısı: **{sim35['total_matches']}** • Her maça sabit **1.000 TL** bahis üzerinden gerçek performans • "
            "Tablo başlıklarındaki **`?`** işaretlerinin veya aşağıdaki kutuların üstüne fareyle gelerek her simetri ve oran kuralının ne işe yaradığını okuyabilirsiniz."
        )
        _n_games = df_sym_all["game_id"].nunique() if not df_sym_all.empty else 0
        _n_done = (
            df_sym_all[df_sym_all["status"] == "COMPLETED"]["game_id"].nunique()
            if (not df_sym_all.empty and "status" in df_sym_all.columns)
            else 0
        )
        st.caption(f"🔎 Veri kaynağı: {len(df_sym_all)} dakika kaydı • {_n_games} maç • {_n_done} sonuçlanmış • sürüm {ENGINE_VERSION}")

        with st.expander("🌙☁️ 7/24 Ücretsiz VDS (Google Cloud / AWS + Tailscale) & Telefon Gece Oto-Pilot Kurulumu", expanded=False):
            st.markdown(
                """
                **Evdeki bilgisayarınız ve telefonunuz tamamen kapalıyken** 7/24 buluttaki ücretsiz sanal bilgisayardan (VDS) veya telefonunuzdan otomatik bahis almak için:
                - **Canlı Bulut Sinyal Adresi (7/24 Otomatik Güncellenir):**  
                  `https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/live_bet_signal.json`
                ---
                #### ☁️ Çözüm B: 7/24 Ücretsiz VDS (Google Cloud / AWS) + Tailscale Tek Komut Kurulumu
                1. **Google Cloud (`$300` Ücretsiz Kredi)** veya **Amazon AWS (`12 Ay Ücretsiz Free Tier`)** üzerinden ücretsiz bir **Windows Server** sanal bilgisayarı (VDS) açın ve Windows'un **"Uzak Masaüstü Bağlantısı"** ile bağlanın.
                2. VDS içinde **PowerShell**'i açıp şu **tek satırlık komutu** yapıştırıp Enter'a basın *(Chrome, Tailscale ve NBA 2K26 v7.1 Oto-Pilot eklentisini otomatik kurar)*:
                   ```powershell
                   irm https://raw.githubusercontent.com/miracgucc/nba2k26-live/main/install_vds_windows.ps1 | iex
                   ```
                3. **Tailscale Ev/Telefon IP Köprüsü:** VDS'te ve kendi telefonunuzda ücretsiz **Tailscale** uygulamasına aynı Gmail ile giriş yapın; telefonunuzda `Use as Exit Node` seçeneğini açıp VDS'te telefonunuzu seçin (böylece Melbet/1xBet sizi %100 kendi ev/mobil internetinizden bağlanıyor görür).
                4. VDS masaüstüne gelen **`NBA2K26_7_24_OtoPilot_Baslat`** kısayoluna çift tıklayın, Melbet'e girip **"Tek Tıkla Bahis (1-Click Bet)"** özelliğini açın ve Uzak Masaüstü penceresini kapatıp uyuyun!

                ---
                #### 📲 Çözüm A: Telefon Tarayıcısı (Kiwi Browser) veya 1xBet Mobil Uygulaması (`MacroDroid`)
                - **Melbet Mobil Tarayıcı:** Android'de **Kiwi Browser + Tampermonkey** kurup aşağıdaki butondan scripti indirin; gece yatarken sağ alttaki **`🌑 Karart`** butonuna basarak ekranı simsiyah yapın.
                """
            )
            _autopilot_js_path = os.path.join(BASE_DIR, "mobile_night_autopilot.user.js")
            if os.path.exists(_autopilot_js_path):
                with open(_autopilot_js_path, "r", encoding="utf-8") as _f_js:
                    _js_content = _f_js.read()
                st.download_button(
                    label="📥 Telefon Gece Oto-Pilot Scriptini İndir (mobile_night_autopilot.user.js)",
                    data=_js_content,
                    file_name="mobile_night_autopilot.user.js",
                    mime="application/javascript",
                )

        _gap_thr = int(LEAGUE_CFG["blowout_brake_gap"]) if IS_EURO else 18
        _slow_q4_avg = 33 if IS_EURO else 48
        _exp_rem_d = int(round(EXP_REM_DEC))
        _m10_center_str = "-".join(str(x) for x in LEAGUE_CFG["mod10_center_digits"])
        _m10_risk_str = "-".join(str(x) for x in LEAGUE_CFG["mod10_risk_digits"])
        _ex_fav1 = LEAGUE_CFG["default_barem"]
        _ex_fav2 = LEAGUE_CFG["default_backup_barem"]
        _rule1_title = (
            "🔥 1. Kural: Alt Kanat & V-Dip 1. Favori [?]"
            if IS_NBA2K27
            else ("🔥 1. Kural: 1.50–2.10 Altın Merkez Favori [?]" if IS_EURO else "🔥 1. Kural: 1.85–2.10 & Seçici 2.30–2.50 1. Favori [?]")
        )
        _rule1_body = (
            "NBA 2K27'de <b>Alt komşu oranı üst komşudan 1.00+ düşükse</b> (1.000 TL başı <b>+157 TL</b>), <b>V-Dip</b> (<b>+119 TL</b>) veya <b>2x Alt sonrası</b> (<b>+396 TL</b>) doğrudan <b>1. Favori</b> oynanır."
            if IS_NBA2K27
            else (
                "Euro'da 1.Fav oranı <b>1.50–2.10</b> ve fark <b>5–20 sayı</b> iken (özellikle <b>Tam İkiz <=0.25: +313 TL</b> veya <b>Barem Düşüşü: +163 TL</b>) doğrudan <b>1. Favori</b> en yüksek net kazancı verir."
                if IS_EURO
                else "NBA 2K26'da <b>1.85–2.10</b> (önceki maç +1 değilken: <b>%60.7 / +222 TL</b>) ve <b>2.30–2.50</b> bandında <b>Tam İkiz (<=0.25) / 10–15 Fark / +1 Üst</b> (<b>%56–%60 / +267 ile +415 TL</b>) doğrudan <b>1. Favori</b> oynanır."
            )
        )
        _rule3_title = (
            "🔄 3. Kural: NBA 2K27 2. Favori Bölgeleri [?]"
            if IS_NBA2K27
            else ("🔄 3. Kural: Euro 21+ Fark & 2.10+ 2. Favori [?]" if IS_EURO else "🔄 3. Kural: NBA 2K26 Seçici 2. Favori Fırsatları [?]")
        )
        _rule3_body = (
            "NBA 2K27'de maç içinde <b>barem aşağı düştüyse</b> (<b>+385 TL</b>), <b>1. ve 2. Fav oran farkı <=0.45</b> ise (<b>+173 TL</b>) veya oran <b>2.50+</b> ise (<b>+945 TL</b>) doğrudan <b>2. Favori Barem</b> oynanır."
            if IS_NBA2K27
            else (
                "Euro'da 29. dk fark <b>21+ sayı</b> olduğunda maç rölantiye girer ve <b>2. Favori</b> 1.000 TL başı <b>+270 TL</b> kazandırır. Ayrıca <b>2.10–2.50</b> bandında üst kanat açıkken <b>2. Favori +433 TL</b> üretir."
                if IS_EURO
                else "NBA 2K26'da <b>2.10–2.30</b> bandında sadece <b>10+ fark ve simetri sinyali varken</b> (6–9 fark pas geçilir!), <b>2.30–2.50 alt kanatta (+310 TL)</b> ve <b>2.50+ kaymalı maçlarda (%61.5 / +750 TL)</b> doğrudan <b>2. Favori</b> oynanır."
            )
        )
        _rule4_body = (
            "Euro'da <b>Önceki Maç +1 Üst</b> (<b>+160 TL</b>), <b>Sarkaç (-1➔+1 / +1➔-1: +180 TL)</b> ve <b>2x Favori Yatması (+168 TL)</b> sonrası <b>1. Favori</b> merkeze döner. NBA 2K26'da ise <b>+1 Üst</b> ve <b>-2 Alt</b> sonrasında <b>2. Favori (+250 ile +265 TL)</b> kazanır!"
        )

        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.55rem; margin:0.5rem 0 0.85rem 0;">
                <div class="info-strip" style="cursor:help; border-left:4px solid #059669;"
                     title="762 gerçek kapanış maçında (358 NBA 2K26 + 59 NBA 2K27 + 345 Euro) hem ilk yarı hem ikinci yarı maçlarında doğrulanmış ana 1. Favori kuralı.">
                    <div class="info-strip-label">{_rule1_title}</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        {_rule1_body}
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #dc2626;"
                     title="{DEC_MIN}. dakikada skor farkı 0-5 sayı (Euro 0-4) ise taktik faul ve uzatma yüzünden barem şaşar (NBA 1.000 TL başı -330 TL zarar). Bu maçlar pas geçilir.">
                    <div class="info-strip-label">⛔ 2. Kural: Başa Baş (0–5 Fark) Pas Geç [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        Fark <b>{"0–4 sayı ve oran 1.20 altı / 2.50+" if IS_EURO else "0–5 sayı"}</b> ise <b>Bahis Alma (Pas Geç)</b>.
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #2563eb;"
                     title="Gerçek maç verilerinde 1. Favorinin zayıflayıp 2. Favorinin öne çıktığı en kârlı fırsat bölgeleri.">
                    <div class="info-strip-label">{_rule3_title}</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        {_rule3_body}
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #e11d48;"
                     title="Ardışık biten maçların barem kayma sırası (Sarkaç -1➔+1 / +1➔-1, +1 Üst ve -2 Alt tekrarsızlığı) sıradaki maçın yönünü belirler.">
                    <div class="info-strip-label">🌊 4. Kural: Ardışık Dalga & Sarkaç Simetrisi [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        {_rule4_body}
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #d97706;"
                     title="İki barem kapatırken 500/500 eşit bölmek yerine 650 TL / 350 TL veya 550 TL / 450 TL ağırlık vermek üç ligde de %83–%88 isabet ve 1.000 TL başı net kâr sağlar.">
                    <div class="info-strip-label">🛡️ 5. Kural: Ağırlıklı İkili Kapatma (650/350 & 550/450 TL) [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        Seçilmiş maçlarda <b>650/350 TL</b> veya <b>550/450 TL</b> ikili kapatma NBA 2K26'da <b>%83.1 (+49 TL)</b>, NBA 2K27'de <b>%88.2 (+97 TL)</b>, Euro'da <b>%84.0 (+193 TL)</b> üretir.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "###### 💰 Sanal Kasa: Hangi Yöntem 1.000 TL Başına Ne Kadar Kazandırdı?",
            help="Gerçek sonuçlanmış maçların tamamında bu yöntemleri her maça 1.000 TL yatırarak uygulasaydık kasamızın ne olacağını gösterir.",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Oynama Yöntemi (Strateji)": s["name"],
                        "Tutma Oranı": f"{s['wins']} / {s['total']} (%{s['win_rate']})",
                        "1.000 TL Başı Net": f"{s['net_profit']/max(1, s['total']):+.0f} TL",
                        "Toplam Yatırılan": f"{s['cost']:.0f} TL",
                        "Toplam Geri Dönüş": f"{s['return']:.0f} TL",
                        "Net Kâr / Zarar": f"{s['net_profit']:+.0f} TL",
                        "Kâr Oranı (%)": f"%{s['roi_pct']:+.1f}",
                    }
                    for s in sim35["strategies"]
                ]
            ),
            column_config={
                "Oynama Yöntemi (Strateji)": st.column_config.TextColumn(
                    "Oynama Yöntemi (Strateji)",
                    help="Test edilen bahis yöntemi. En üstteki yöntem v6.1 1. Tercih (Canlı Kart Karar Motoru) kurallarını uygular.",
                ),
                "Tutma Oranı": st.column_config.TextColumn(
                    "Tutma Oranı",
                    help="Oynanan toplam maç içinden kaç tanesinin kazandığı ve tutturma yüzdesi.",
                ),
                "1.000 TL Başı Net": st.column_config.TextColumn(
                    "1.000 TL Başı Net",
                    help="Oynanan her 1.000 TL'lik kupon başına kasaya kalan ortalama net kâr veya zarar.",
                ),
                "Toplam Yatırılan": st.column_config.TextColumn(
                    "Toplam Yatırılan",
                    help="Oynanan tüm maçlar için cepten çıkan toplam bahis tutarı (Pas geçilen maçlarda para harcanmaz).",
                ),
                "Toplam Geri Dönüş": st.column_config.TextColumn(
                    "Toplam Geri Dönüş",
                    help="Tutan kuponlardan kasaya geri dönen toplam para.",
                ),
                "Net Kâr / Zarar": st.column_config.TextColumn(
                    "Net Kâr / Zarar",
                    help="Toplam Geri Dönüş eksi Toplam Yatırılan = Cebimize kalan net kâr veya zarar.",
                ),
                "Kâr Oranı (%)": st.column_config.TextColumn(
                    "Kâr Oranı (%)",
                    help="Yatırılan her 1.000 TL başına elde edilen net kazanç yüzdesi.",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        if sim35.get("sec_dir_brackets"):
            st.markdown(
                "###### 🧭 2. Favori Üst Baremde mi Yoksa Alt Baremde mi Duruyor?",
                help="1xBet'te 1. Favoriden sonra en düşük oranlı 2. baremin bir üst baremde mi yoksa bir alt baremde mi durduğuna göre maçların nasıl bittiğini gösterir.",
            )
            st.dataframe(pd.DataFrame(sim35["sec_dir_brackets"]), use_container_width=True, hide_index=True)

        if sim35.get("margin_brackets"):
            st.markdown(
                f"###### ⚖️ {DEC_MIN}. Dakikadaki Skor Farkına Göre Maçlar Nasıl Bitiyor?",
                help=f"{DEC_MIN}. dakikada iki takım arasındaki skor farkının kalan sürede atılan sayıya ve barem tutmasına etkisini gösterir.",
            )
            st.dataframe(pd.DataFrame(sim35["margin_brackets"]), use_container_width=True, hide_index=True)

        if sim35["odds_brackets"]:
            st.markdown(
                f"###### 🎚️ {DEC_MIN}. Dakikadaki 1. Favori Oranına Göre Ne Oynanmalı?",
                help=f"{DEC_MIN}. dakikadaki 1. Favori oranının büyüklüğüne göre maçların hangi baremde bittiğini gösterir.",
            )
            st.dataframe(pd.DataFrame(sim35["odds_brackets"]), use_container_width=True, hide_index=True)

        if sim35.get("mod10_brackets"):
            st.markdown(
                f"###### 🔢 SİMETRİ 1: {DEC_MIN}. Dakika Toplam Skorunun Son Rakamına Göre Barem İçi Konum",
                help=f"{DEC_MIN}. dakikadan sonra ortalama {_exp_rem_d} sayı atıldığı için; son rakam {_m10_center_str} ise tam barem ortasına, {_m10_risk_str} ise baremin alt sınırına yaklaşır.",
            )
            st.dataframe(pd.DataFrame(sim35["mod10_brackets"]), use_container_width=True, hide_index=True)

        if sim35.get("twin_conc_brackets"):
            st.markdown(
                "###### 👯 SİMETRİ 2: 1. ve 2. Favori Oranları Birbirine Yakın Olduğunda Ne Olur?",
                help="1. Favori ile 2. Favori oranlarının birbirine yakınlığı, ilk 2 favorinin toplam gücü ve alt-üst komşu baremlerin dengesi.",
            )
            st.dataframe(pd.DataFrame(sim35["twin_conc_brackets"]), use_container_width=True, hide_index=True)

        if sim35.get("pendulum_brackets"):
            st.markdown(
                f"###### 🔄 SİMETRİ 3: Son Çeyrek Öncesi ({MID_Q3_MIN}.Dk ➔ {DEC_MIN}.Dk) Barem Değiştiyse Ne Olur?",
                help=f"{MID_Q3_MIN}. dakikadaki favori barem ile {DEC_MIN}. dakikadaki favori barem arasındaki değişimi inceler.",
            )
            st.dataframe(pd.DataFrame(sim35["pendulum_brackets"]), use_container_width=True, hide_index=True)

        arc_sym = analyze_archive_4500_symmetry(df_completed_archive, league=ACTIVE_LEAGUE)
        if arc_sym.get("mod10_36m"):
            st.markdown(
                f"###### 🏛️ SİMETRİ 4: {arc_sym['total_archive']} Maçlık Arşivde 3. Çeyrek Sonu ({CLOSE_MIN_INT}.Dk) Skorun Son Rakamı (`..0` – `..9`) Analizi",
                help=f"{arc_sym['total_archive']} tamamlanmış maçta 3. çeyrek sonu ({CLOSE_MIN_INT}. dk) toplam skorunun son rakamına göre beklenen barem, 1 alt veya 1 üst barem oranları.",
            )
            st.dataframe(pd.DataFrame(arc_sym["mod10_36m"]), use_container_width=True, hide_index=True)

        if arc_sym.get("half_symmetry"):
            st.markdown(
                f"###### 🪞 SİMETRİ 5: {arc_sym['total_archive']} Maçlık Arşivde İlk Yarı ve İkinci Yarı Sayı Dengesi",
                help="İlk yarıda çok az veya çok fazla sayı çıktığında 2. yarıda oyunun nasıl tepki vererek toplam skoru dengelediğini gösterir.",
            )
            st.dataframe(
                pd.DataFrame(arc_sym["half_symmetry"]),
                use_container_width=True,
                hide_index=True,
            )

        if sim35.get("edge_brackets"):
            st.markdown(
                f"###### 📐 {DEC_MIN}. Dakika Skoru + {_exp_rem_d} Sayı Eklendiğinde Baremin Neresine Düşüyor?",
                help=f"{DEC_MIN}. dakikadaki toplam skora ortalama kalan sayı ({_exp_rem_d}) eklendiğinde favori baremin neresine düştüğünü analiz eder.",
            )
            st.dataframe(pd.DataFrame(sim35["edge_brackets"]), use_container_width=True, hide_index=True)

        if sim35.get("minute_windows"):
            st.markdown(
                "###### ⏱️ Hangi Dakikada Bahis Alınmalı? (Dakika Karşılaştırması)",
                help="Maçın erken, orta ve kapanış dakikalarında bahis alındığında hangisinin kârlı, hangisinin tuzak olduğunu gösterir.",
            )
            st.dataframe(pd.DataFrame(sim35["minute_windows"]), use_container_width=True, hide_index=True)

        if sim35["drift_brackets"]:
            st.markdown(
                f"###### 📉 Maç Başından {DEC_MIN}. Dakikaya Barem Değiştiğinde Ne Olur?",
                help=f"Maçın başındaki favori barem ile {DEC_MIN}. dakikadaki favori barem arasındaki yükseliş veya düşüşe göre maç sonu sonucu.",
            )
            st.dataframe(pd.DataFrame(sim35["drift_brackets"]), use_container_width=True, hide_index=True)

        if sim35["transitions"]:
            st.markdown(
                "###### 🌊 Bir Önceki Maçın Sonucuna Göre Sıradaki Maç Nasıl Bitiyor?",
                help="Bir önceki biten maçta favori mi geldi, alt barem mi geldi, üst barem mi geldi bilgisinin sıradaki maça etkisi.",
            )
            st.dataframe(pd.DataFrame(sim35["transitions"]), use_container_width=True, hide_index=True)

        st.markdown(
            "###### 📒 Maç Maç Sanal Kupon Defteri (Tüm Gerçek Maçlar)",
            help="En son biten maç en üstte olacak şekilde her maçta sistemin hangi baremi önerdiğini ve kuponların ne yaptığını tek tek gösterir.",
        )
        st.dataframe(
            pd.DataFrame(sim35["ledger"]),
            use_container_width=True,
            hide_index=True,
            height=440,
        )
        st.warning(
            f"⚠️ Bu analiz {sim35['total_matches']} gerçek kapanışlı maç üzerinden hesaplanmıştır. "
            "İstatistiksel üstünlük sağlar ancak kesin kazanç garantisi değildir; kasa yönetimine sadık kalın."
        )

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1: HISTORICAL MATCHES CENTER, TEAM BAREM DNA & H2H ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
with tab_dna:
    if df_completed_archive.empty:
        st.info(
            f"Şu an {LEAGUE_CFG['short_name']} geçmiş maç arşivi boş. Yukarıdaki **'{LEAGUE_CFG['archive_btn_label']}'** "
            "butonuna basarak tamamlanmış maçları saniyeler içinde yükleyebilirsiniz."
        )
    else:
        df_arc = df_completed_archive.copy()
        df_arc["Kazanan Barem"] = df_arc["total_score"].apply(lambda v: get_range_from_score(v, league=ACTIVE_LEAGUE))
        df_arc["20'li Küme"] = df_arc["total_score"].apply(lambda v: get_cluster_from_score(v, league=ACTIVE_LEAGUE))
        df_arc["Q1 Toplam"] = df_arc["home_q1"].astype(int) + df_arc["away_q1"].astype(int)
        df_arc["Q2 Toplam"] = df_arc["home_q2"].astype(int) + df_arc["away_q2"].astype(int)
        df_arc["Q3 Toplam"] = df_arc["home_q3"].astype(int) + df_arc["away_q3"].astype(int)
        df_arc["Q4 Toplam"] = df_arc["home_q4"].astype(int) + df_arc["away_q4"].astype(int)

        all_teams = sorted(
            set(df_arc["home_team"].dropna().unique()).union(set(df_arc["away_team"].dropna().unique()))
        )

        flt1, flt2, flt3 = st.columns([1.3, 1.3, 1.1])
        with flt1:
            sel_t1 = st.selectbox("1. Takım Filtresi (H2H veya Tek Takım)", ["Tüm Takımlar"] + all_teams, index=0, key=f"dna_t1_{ACTIVE_LEAGUE}")
        with flt2:
            sel_t2 = st.selectbox("2. Takım Filtresi (Rakip / H2H)", ["Tüm Takımlar"] + all_teams, index=0, key=f"dna_t2_{ACTIVE_LEAGUE}")
        with flt3:
            sel_barem_flt = st.selectbox(
                "Barem Filtresi",
                ["Tüm Baremler"] + [b[0] for b in BAREM_RANGES],
                index=0,
                key=f"dna_b_{ACTIVE_LEAGUE}",
            )

        df_filtered = df_arc.copy()
        if sel_t1 != "Tüm Takımlar" and sel_t2 != "Tüm Takımlar":
            df_filtered = df_filtered[
                ((df_filtered["home_team"] == sel_t1) & (df_filtered["away_team"] == sel_t2))
                | ((df_filtered["home_team"] == sel_t2) & (df_filtered["away_team"] == sel_t1))
            ]
        elif sel_t1 != "Tüm Takımlar":
            df_filtered = df_filtered[
                (df_filtered["home_team"] == sel_t1) | (df_filtered["away_team"] == sel_t1)
            ]
        elif sel_t2 != "Tüm Takımlar":
            df_filtered = df_filtered[
                (df_filtered["home_team"] == sel_t2) | (df_filtered["away_team"] == sel_t2)
            ]

        if sel_barem_flt != "Tüm Baremler":
            df_filtered = df_filtered[df_filtered["Kazanan Barem"] == sel_barem_flt]

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        n_flt = len(df_filtered)
        avg_tot = df_filtered["total_score"].mean() if n_flt > 0 else 0.0
        modal_b = (
            df_filtered["Kazanan Barem"].mode().iloc[0]
            if n_flt > 0 and not df_filtered["Kazanan Barem"].mode().empty
            else "-"
        )
        modal_c = (
            df_filtered["20'li Küme"].mode().iloc[0]
            if n_flt > 0 and not df_filtered["20'li Küme"].mode().empty
            else "-"
        )
        m_c1.metric("📚 Filtrelenen Maç Sayısı", f"{n_flt} Maç")
        m_c2.metric("🏀 Ortalama Toplam Sayı", f"{avg_tot:.1f} Sayı")
        m_c3.metric("🎯 En Çok Çıkan Barem", modal_b)
        m_c4.metric("🛡️ En Çok Çıkan 20'li Küme", modal_c)

        ch_col, dna_col = st.columns([1.05, 1.35])
        with ch_col:
            _ch_lbl = "6'lı" if LEAGUE_BAREM_N == 6 else "9'lu"
            st.markdown(f"##### 📊 Geçmiş Maçların {_ch_lbl} Barem Dağılım Grafiği")
            barem_order = [b[0] for b in BAREM_RANGES]
            b_counts = [int((df_filtered["Kazanan Barem"] == b).sum()) for b in barem_order]
            b_pcts = [(c / n_flt * 100.0) if n_flt > 0 else 0.0 for c in b_counts]
            fig_hist_b = go.Figure(
                go.Bar(
                    x=barem_order,
                    y=b_counts,
                    text=[f"{c} (%{p:.1f})" for c, p in zip(b_counts, b_pcts)],
                    textposition="outside",
                    marker_color="#059669",
                )
            )
            fig_hist_b.update_layout(
                height=310,
                margin=dict(l=15, r=15, t=25, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#475569" if not IS_DARK else "#cbd5e1", size=11),
                yaxis=dict(title="Maç Adedi", range=[0, max(b_counts + [5]) * 1.25]),
            )
            st.plotly_chart(fig_hist_b, use_container_width=True)

        with dna_col:
            st.markdown(f"##### 🧬 Takım Bazlı Barem DNA Tablosu ({LEAGUE_CFG['short_name']})")
            df_dna_table = compute_team_dna_table(df_arc, league=ACTIVE_LEAGUE)
            st.dataframe(df_dna_table, use_container_width=True, hide_index=True, height=310)

        st.markdown("##### 📜 Tamamlanmış Tüm Maçların Detaylı Barem & Periyot Arşivi")
        show_cols = [
            "date",
            "home_team",
            "away_team",
            "home_total",
            "away_total",
            "total_score",
            "Kazanan Barem",
            "20'li Küme",
            "Q1 Toplam",
            "Q2 Toplam",
            "Q3 Toplam",
            "Q4 Toplam",
        ]
        avail_cols = [c for c in show_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[avail_cols].iloc[::-1], use_container_width=True, hide_index=True, height=340)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2: CLOSING ODDS SYMMETRY & SEQUENTIAL WAVE BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with tab_sym:
    st.markdown(
        f"##### 🌊 Ardışık Maçlarda {CLOSE_MIN_INT}. Dakika Favorisi Tuttu mu, Komşu Bareme mi Kaydı? (`2 Alt / 1 Alt / Favori / 1 Üst / 2 Üst`)"
    )
    st.caption(
        f"Son 4 maçın nasıl bittiğini seçerek ({len(df_completed_archive)} gerçek {LEAGUE_CFG['short_name']} maçı üzerinde) "
        "sıradaki maçın hangi baremde biteceğini test edebilirsiniz."
    )

    w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns([1, 1, 1, 1, 1.2])
    seq_map = {
        "❄️ 2 Alt Barem Geldi": -2,
        "📉 1 Alt Barem Geldi": -1,
        "🎯 Tam Favori Barem Geldi": 0,
        "📈 1 Üst Barem Geldi": 1,
        "🔥 2 Üst Barem Geldi": 2,
    }
    rev_map = {v: k for k, v in seq_map.items()}

    default_wave = analyze_sequential_barem_wave(df_completed_archive, df_sym_all, league=ACTIVE_LEAGUE)
    act_pat = default_wave.get("active_pattern", [0, -1, -1, 0])
    while len(act_pat) < 4:
        act_pat = [0] + act_pat

    opts_list = list(seq_map.keys())
    with w_col1:
        s_m1 = st.selectbox("1. Önceki Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[0], opts_list[2])), key=f"sym_m1_{ACTIVE_LEAGUE}")
    with w_col2:
        s_m2 = st.selectbox("2. Önceki Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[1], opts_list[1])), key=f"sym_m2_{ACTIVE_LEAGUE}")
    with w_col3:
        s_m3 = st.selectbox("3. Önceki Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[2], opts_list[1])), key=f"sym_m3_{ACTIVE_LEAGUE}")
    with w_col4:
        s_m4 = st.selectbox("4. (Son Bitmiş) Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[3], opts_list[2])), key=f"sym_m4_{ACTIVE_LEAGUE}")
    with w_col5:
        _b_opts_sym = [b[0] for b in BAREM_RANGES]
        _def_b_idx = _b_opts_sym.index(LEAGUE_CFG["default_barem"]) if LEAGUE_CFG["default_barem"] in _b_opts_sym else 0
        test_fav_b = st.selectbox(
            f"Canlı Maçın {CLOSE_MIN_INT}.Dk Favori Baremi",
            _b_opts_sym,
            index=_def_b_idx,
            key=f"sym_favb_{ACTIVE_LEAGUE}",
        )

    custom_pattern = [seq_map[s_m1], seq_map[s_m2], seq_map[s_m3], seq_map[s_m4]]
    sim_wave = analyze_sequential_barem_wave(
        df_completed_archive, df_sym_all, current_fav_barem=test_fav_b, custom_seq=custom_pattern, league=ACTIVE_LEAGUE
    )

    wk_m2, wk_m1, wk_0, wk_p1, wk_p2, wk_pair = st.columns(6)
    wk_m2.metric(
        f"❄️ 2 Alt Barem ({sim_wave['target_minus_2_barem']})",
        f"%{sim_wave['prob_minus_2']}",
        "2 Kademe Düşük Biter",
    )
    wk_m1.metric(
        f"📉 1 Alt Barem ({sim_wave['target_minus_1_barem']})",
        f"%{sim_wave['prob_minus_1']}",
        f"Toplam Alt: %{sim_wave['prob_down_total']}",
    )
    wk_0.metric(
        f"🎯 Favori Barem ({sim_wave['target_0_barem']})",
        f"%{sim_wave['prob_0']}",
        f"{sim_wave['pattern_matches_count']} Benzer Maç",
    )
    wk_p1.metric(
        f"📈 1 Üst Barem ({sim_wave['target_plus_1_barem']})",
        f"%{sim_wave['prob_plus_1']}",
        f"Toplam Üst: %{sim_wave['prob_up_total']}",
    )
    wk_p2.metric(
        f"🔥 2 Üst Barem ({sim_wave['target_plus_2_barem']})",
        f"%{sim_wave['prob_plus_2']}",
        "2 Kademe Yüksek Biter",
    )
    wk_pair.metric(
        "🛡️ En Güçlü İkili Barem",
        f"%{sim_wave['recommended_pair_prob']}",
        sim_wave["recommended_pair"],
    )

    barem_names_list = [b[0] for b in BAREM_RANGES]

    def _get_shifted_b(base_b_str: str, delta_idx: int) -> str:
        if base_b_str not in barem_names_list:
            return "-"
        n_idx = max(0, min(len(barem_names_list) - 1, barem_names_list.index(base_b_str) + delta_idx))
        return barem_names_list[n_idx]

    if default_wave.get("recent_chain"):
        st.markdown(f"###### 📜 Canlı Takip Edilip Tamamlanan Maçların `{CLOSE_MIN_INT}. Dakika Favori Barem vs Biten Barem` Tablosu")
        rc_rows = []
        for item in default_wave["recent_chain"][::-1]:
            eb = item["exp_barem_36"]
            rc_rows.append(
                {
                    "Tarih / Saat": item["date"],
                    "Maç": f"{item['home_team']} vs {item['away_team']}",
                    f"{CLOSE_MIN_INT}. Dk Skor": item["s36"],
                    "Maç Sonu Toplam": f"{item['home_total']}:{item['away_total']} ({item['total_score']})",
                    "❄️ 2 Alt Barem": _get_shifted_b(eb, -2),
                    "📉 1 Alt Barem": _get_shifted_b(eb, -1),
                    f"🎯 {CLOSE_MIN_INT}.Dk Favori Barem": eb,
                    "📈 1 Üst Barem": _get_shifted_b(eb, 1),
                    "🔥 2 Üst Barem": _get_shifted_b(eb, 2),
                    "Biten Gerçek Barem": item["act_barem"],
                    "Nasıl Sonuçlandı?": item["shift_label"],
                    "Referans": item["source"],
                }
            )
        st.dataframe(pd.DataFrame(rc_rows), use_container_width=True, hide_index=True, height=min(320, 60 + 38 * len(rc_rows)))

    st.markdown("---")
    st.markdown(
        f"##### ⚖️ 1xBet Kapanış Oranı ({CLOSING_START}-{CLOSE_MIN_INT}. Dk) Simetri Tuzağı & Komşu Bareme (`2 Alt / 1 Alt / Favori / 1 Üst / 2 Üst`) Kayma Analizi"
    )
    df_comp_sym = (
        df_sym_all[df_sym_all["status"] == "COMPLETED"].copy()
        if not df_sym_all.empty and "status" in df_sym_all.columns
        else pd.DataFrame()
    )

    if df_comp_sym.empty:
        st.info(
            f"Şu an {LEAGUE_CFG['short_name']} logunda sonuçlanmış (COMPLETED) canlı barem kaydı bekleniyor. "
            f"Canlı maçlar bittikçe burada **{CLOSING_START}-{CLOSE_MIN_INT}. Dk Kapanış İsabet Oranları** ve **Dakika Penceresi** istatistikleri otomatik hesaplanır."
        )
    else:
        _win_close = f"{CLOSING_START}-{CLOSE_MIN_INT}. Dk (Kapanış Bölgesi)"
        _win_mid = f"{HALFTIME_MIN}-{CLOSING_START - 1}. Dk (Q3 Ortası / Sonu)"
        _win_early = f"0-{HALFTIME_MIN - 1}. Dk (İlk Yarı)"
        sym_records = []
        for _, r in df_comp_sym.iterrows():
            odds_row = {b: float(r.get(col, 99.0) or 99.0) for b, col in BAREM_TO_ODD_COL.items()}
            valid_sorted = sorted([(k, v) for k, v in odds_row.items() if v < 90.0], key=lambda x: x[1])
            if not valid_sorted:
                continue
            fav1_b, fav1_o = valid_sorted[0]
            fav2_b, fav2_o = valid_sorted[1] if len(valid_sorted) > 1 else ("-", 99.0)
            gap = round(fav2_o - fav1_o, 2) if fav2_o < 90.0 else 99.0
            act_b = str(r.get("actual_range", "")).strip()
            val_b = str(r.get("value_range", "")).strip()
            clu_b = str(r.get("best_cluster_name", "")).strip()

            if fav1_b in barem_names_list and act_b in barem_names_list:
                b_shift = barem_names_list.index(act_b) - barem_names_list.index(fav1_b)
            else:
                b_shift = 99

            m_match = re.search(r"Minute\s+(\d+)", str(r.get("check_time", "")))
            el_min = int(m_match.group(1)) if m_match else EARLY_Q2_MIN
            if el_min >= CLOSING_START:
                win_bucket = _win_close
            elif el_min >= HALFTIME_MIN:
                win_bucket = _win_mid
            else:
                win_bucket = _win_early

            sym_records.append(
                {
                    "game_id": str(r.get("game_id", r.get("game_name", ""))),
                    "game_name": r.get("game_name", ""),
                    "check_time": r.get("check_time", ""),
                    "el_min": el_min,
                    "window": win_bucket,
                    "fav1_b": fav1_b,
                    "fav1_o": fav1_o,
                    "fav2_b": fav2_b,
                    "fav2_o": fav2_o,
                    "gap": gap,
                    "is_trap": gap <= 0.65,
                    "actual_b": act_b,
                    "b_shift": b_shift,
                    "hit_m2": b_shift <= -2,
                    "hit_m1": b_shift == -1,
                    "hit_0": b_shift == 0,
                    "hit_p1": b_shift == 1,
                    "hit_p2": b_shift >= 2,
                    "fav1_hit": act_b == fav1_b,
                    "fav2_hit": act_b == fav2_b,
                    "either_fav_hit": act_b in (fav1_b, fav2_b),
                    "val_hit": act_b == val_b,
                    "clu_hit": check_cluster_hit(clu_b, act_b, league=ACTIVE_LEAGUE),
                }
            )

        df_sr = pd.DataFrame(sym_records)
        if not df_sr.empty:
            df_sr_32 = df_sr[df_sr["el_min"] >= CLOSING_START].copy()
            base_closing = df_sr_32 if not df_sr_32.empty else df_sr
            df_closing = base_closing.sort_values("el_min").groupby("game_id", as_index=False).last()
            tot_n = len(df_closing)
            m2_pct = df_closing["hit_m2"].mean() * 100.0
            m1_pct = df_closing["hit_m1"].mean() * 100.0
            f0_pct = df_closing["hit_0"].mean() * 100.0
            p1_pct = df_closing["hit_p1"].mean() * 100.0
            p2_pct = df_closing["hit_p2"].mean() * 100.0
            both_pct = df_closing["either_fav_hit"].mean() * 100.0

            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("❄️ 2 Alt Bareme Düştü", f"%{m2_pct:.1f}", f"{int(df_closing['hit_m2'].sum())} / {tot_n} Maç ({CLOSING_START}+ Dk)")
            s2.metric("📉 1 Alt Bareme Düştü", f"%{m1_pct:.1f}", f"{int(df_closing['hit_m1'].sum())} / {tot_n} Maç ({CLOSING_START}+ Dk)")
            s3.metric("🎯 Tam Favori Geldi", f"%{f0_pct:.1f}", f"{int(df_closing['hit_0'].sum())} / {tot_n} Maç ({CLOSING_START}+ Dk)")
            s4.metric("📈 1 Üst Bareme Çıktı", f"%{p1_pct:.1f}", f"{int(df_closing['hit_p1'].sum())} / {tot_n} Maç ({CLOSING_START}+ Dk)")
            s5.metric("🔥 2 Üst Bareme Çıktı", f"%{p2_pct:.1f}", f"{int(df_closing['hit_p2'].sum())} / {tot_n} Maç ({CLOSING_START}+ Dk)")
            s6.metric(
                "🛡️ 1. veya 2. Favori",
                f"%{both_pct:.1f}",
                f"{int(df_closing['either_fav_hit'].sum())} / {tot_n} Maç ({CLOSING_START}+ Dk)",
            )

            win_rows = []
            for w_name in [_win_early, _win_mid, _win_close]:
                w_sub = df_sr[df_sr["window"] == w_name]
                if w_sub.empty:
                    continue
                win_rows.append(
                    {
                        "Dakika Penceresi": w_name,
                        "Kayıt Sayısı": len(w_sub),
                        "❄️ 2 Alt Barem (%)": f"%{w_sub['hit_m2'].mean()*100:.1f}",
                        "📉 1 Alt Barem (%)": f"%{w_sub['hit_m1'].mean()*100:.1f}",
                        "🎯 Tam Favori (%)": f"%{w_sub['hit_0'].mean()*100:.1f}",
                        "📈 1 Üst Barem (%)": f"%{w_sub['hit_p1'].mean()*100:.1f}",
                        "🔥 2 Üst Barem (%)": f"%{w_sub['hit_p2'].mean()*100:.1f}",
                        "🥈 2. Favori (%)": f"%{w_sub['fav2_hit'].mean()*100:.1f}",
                        "🛡️ 20-Sayı Küme (%)": f"%{w_sub['clu_hit'].mean()*100:.1f}",
                    }
                )
            st.dataframe(pd.DataFrame(win_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3: Q3 ➔ Q4 BURST / COLLAPSE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_q3q4:
    st.markdown(
        "##### ⚡ 3. Çeyrek (Q3) Temposuna Göre 4. Çeyrek (Q4) Sayı Patlaması / Çöküş Dedektörü"
    )
    if df_completed_archive.empty:
        st.info("Q3 ➔ Q4 analizi için geçmiş maç verisi bekleniyor.")
    else:
        q_teams = ["Tüm Takımlar"] + sorted(
            set(df_completed_archive["home_team"].dropna().unique()).union(
                set(df_completed_archive["away_team"].dropna().unique())
            )
        )
        _q3_c_disp = int(LEAGUE_CFG["q3_collapse_disp"])
        _q3_n_lo = int(LEAGUE_CFG["q3_norm_lo"])
        _q3_n_hi = int(LEAGUE_CFG["q3_norm_hi"])
        _q3_b_disp = int(LEAGUE_CFG["q3_burst_disp"])
        _q3_max_c = float(LEAGUE_CFG["q3_collapse_max"])
        _q3_min_b = float(LEAGUE_CFG["q3_burst_min"])

        qc1, qc2, qc3 = st.columns([1.3, 1.3, 1.2])
        with qc1:
            q_t1 = st.selectbox("Ev Sahibi / 1. Takım", q_teams, index=0, key=f"q3_t1_{ACTIVE_LEAGUE}")
        with qc2:
            q_t2 = st.selectbox("Deplasman / 2. Takım", q_teams, index=0, key=f"q3_t2_{ACTIVE_LEAGUE}")
        with qc3:
            _sl_min = 20 if IS_EURO else 35
            _sl_max = 60 if IS_EURO else 72
            _sl_def = int(round(LEAGUE_CFG["prior_q"])) if IS_EURO else 56
            q3_test_score = st.slider("3. Çeyrek (Q3) Toplam Sayısı", _sl_min, _sl_max, _sl_def, 1, key=f"q3_sl_{ACTIVE_LEAGUE}")

        t1_arg = "" if q_t1 == "Tüm Takımlar" else q_t1
        t2_arg = "" if q_t2 == "Tüm Takımlar" else q_t2
        sig_res = build_q3_q4_streak_signal(df_completed_archive, t1_arg, t2_arg, q3_score=q3_test_score, league=ACTIVE_LEAGUE)

        k_q1, k_q2, k_q3, k_q4 = st.columns(4)
        k_q1.metric("🔔 Tespit Edilen Q4 Sinyali", sig_res["signal"])
        k_q2.metric(f"🔥 Q4 Sayı Patlaması ({_q3_b_disp}+ Sayı)", f"%{sig_res['burst_prob']:.1f}")
        k_q3.metric(f"⚖️ Q4 Normal Tempo ({_q3_n_lo}-{_q3_n_hi} Sayı)", f"%{sig_res['normal_prob']:.1f}")
        k_q4.metric(f"❄️ Q4 Sayı Çöküşü (≤ {_q3_c_disp} Sayı)", f"%{sig_res['collapse_prob']:.1f}")

        st.info(f"📌 **Analiz Özeti ({sig_res['sample_size']} Benzer Maç):** {sig_res['detail']}")

        df_q_arc = df_completed_archive.copy()
        df_q_arc["Q3_Tot"] = df_q_arc["home_q3"].astype(float) + df_q_arc["away_q3"].astype(float)
        df_q_arc["Q4_Tot"] = df_q_arc["home_q4"].astype(float) + df_q_arc["away_q4"].astype(float)
        if t1_arg:
            df_q_arc = df_q_arc[(df_q_arc["home_team"] == t1_arg) | (df_q_arc["away_team"] == t1_arg)]
        if t2_arg:
            df_q_arc = df_q_arc[(df_q_arc["home_team"] == t2_arg) | (df_q_arc["away_team"] == t2_arg)]

        q_buckets = [
            (f"Düşük Q3 (≤ {_q3_c_disp} Sayı)", df_q_arc[df_q_arc["Q3_Tot"] <= _q3_max_c]),
            (f"Orta Q3 ({_q3_n_lo} - {_q3_n_hi} Sayı)", df_q_arc[(df_q_arc["Q3_Tot"] > _q3_max_c) & (df_q_arc["Q3_Tot"] < _q3_min_b)]),
            (f"Yüksek Q3 ({_q3_b_disp}+ Sayı)", df_q_arc[df_q_arc["Q3_Tot"] >= _q3_min_b]),
        ]
        q_table_rows = []
        for b_label, sub_q in q_buckets:
            if sub_q.empty:
                continue
            q_table_rows.append(
                {
                    "3. Çeyrek (Q3) Tempo Grubu": b_label,
                    "Maç Sayısı": len(sub_q),
                    "Q3 Ortalaması": round(sub_q["Q3_Tot"].mean(), 1),
                    "Takip Eden Q4 Ortalaması": round(sub_q["Q4_Tot"].mean(), 1),
                    f"🔥 Q4 Patlama (≥{_q3_b_disp} Sayı) %": f"%{(sub_q['Q4_Tot'] >= _q3_b_disp).mean()*100:.1f}",
                    f"❄️ Q4 Çöküş (≤{_q3_c_disp} Sayı) %": f"%{(sub_q['Q4_Tot'] <= _q3_c_disp).mean()*100:.1f}",
                    "Maç Sonu Toplam Ort.": round(sub_q["total_score"].mean(), 1),
                }
            )
        st.dataframe(pd.DataFrame(q_table_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4: LIVE BANKROLL / ROI & DUTCHING (DOUBLE BAREM) STAKE CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_bankroll:
    d_col1, d_col2 = st.columns([1.15, 1.25])

    with d_col1:
        st.markdown("##### 🛡️ Çifte Barem (Dutching) Kasa & Tutar Hesaplayıcı")
        st.caption(
            "Komşu iki bareme (20-sayı kümesi) toplam kasayı öyle böler ki, **hangisi gelirse gelsin aynı net kârı** elde edersiniz."
        )

        open_live_list = [m for m in live_matches_now if m.get("is_barem_open")]
        preset_options = ["Manuel Oran Girişi"] + [
            f"{m['team_home']} vs {m['team_away']} ({m.get('bookie_fav_1')} & {m.get('bookie_fav_2')})"
            for m in open_live_list
        ]
        sel_preset = st.selectbox("Canlı Maçtan Otomatik Doldur", preset_options, index=0, key=f"bk_preset_{ACTIVE_LEAGUE}")

        def_b1_name = LEAGUE_CFG["default_barem"]
        def_b2_name = LEAGUE_CFG["default_backup_barem"]
        def_o1, def_o2 = 4.50, 4.50
        if sel_preset != "Manuel Oran Girişi":
            idx_m = preset_options.index(sel_preset) - 1
            if 0 <= idx_m < len(open_live_list):
                m_sel = open_live_list[idx_m]
                def_b1_name = m_sel.get("bookie_fav_1") or def_b1_name
                def_b2_name = m_sel.get("bookie_fav_2") or def_b2_name
                o_1 = float(m_sel.get("bookie_fav_1_odd", 4.5))
                o_2 = float(m_sel.get("bookie_fav_2_odd", 4.5))
                if o_1 < 90.0:
                    def_o1 = o_1
                if o_2 < 90.0:
                    def_o2 = o_2

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            total_stake = st.number_input("Toplam Kupon Tutarı (TL)", min_value=10.0, max_value=100000.0, value=1000.0, step=100.0, key=f"bk_stk_{ACTIVE_LEAGUE}")
        with dc2:
            odd_1 = st.number_input(f"1. Barem Oranı ({def_b1_name})", min_value=1.50, max_value=50.0, value=float(def_o1), step=0.10, key=f"bk_o1_{ACTIVE_LEAGUE}")
        with dc3:
            odd_2 = st.number_input(f"2. Barem Oranı ({def_b2_name})", min_value=1.50, max_value=50.0, value=float(def_o2), step=0.10, key=f"bk_o2_{ACTIVE_LEAGUE}")

        split_mode = st.selectbox(
            "⚖️ İkili Kapatma Bölüşüm Yöntemi (1.000 TL Bazlı)",
            [
                "🛡️ %55 2. Favori / %45 1. Favori (550 TL / 450 TL — NBA 10-20 Fark Önerilen)",
                "💎 %65 1. Barem / %35 2. Barem (650 TL / 350 TL — Ana Tercih + Sigorta)",
                "⚖️ Eşit Garanti Dönüş (Klasik Oran Orantılı Bölüşüm)",
            ],
            index=0 if not IS_EURO else 1,
            key=f"bk_split_mode_{ACTIVE_LEAGUE}",
        )

        inv_sum = (1.0 / odd_1) + (1.0 / odd_2)
        eff_dutch_odd = 1.0 / inv_sum if inv_sum > 0 else 0.0
        if split_mode.startswith("🛡️"):
            stake_1 = total_stake * 0.45
            stake_2 = total_stake * 0.55
        elif split_mode.startswith("💎"):
            stake_1 = total_stake * 0.65
            stake_2 = total_stake * 0.35
        else:
            stake_1 = total_stake * ((1.0 / odd_1) / inv_sum)
            stake_2 = total_stake * ((1.0 / odd_2) / inv_sum)

        ret_1 = stake_1 * odd_1
        ret_2 = stake_2 * odd_2
        net_1 = ret_1 - total_stake
        net_2 = ret_2 - total_stake

        st.success(
            f"💰 **1. Bareme ({def_b1_name}) Yatırılacak:** `{stake_1:.0f} TL` (Oran: {odd_1:.2f}) ➔ Tutarsa Dönüş: `{ret_1:.0f} TL` (**Net: `{net_1:+.0f} TL`**)\n\n"
            f"💰 **2. Bareme ({def_b2_name}) Yatırılacak:** `{stake_2:.0f} TL` (Oran: {odd_2:.2f}) ➔ Tutarsa Dönüş: `{ret_2:.0f} TL` (**Net: `{net_2:+.0f} TL`**)\n\n"
            f"✅ **Klasik Eşit Çifte Şans Oranı:** `{eff_dutch_odd:.2f}` | **Toplam Kupon:** `{total_stake:.0f} TL`"
        )

    with d_col2:
        st.markdown(
            f"##### 📈 Strateji Bazlı Gerçek Kasa (Bankroll) & ROI Performansı ({CLOSING_START}+ Dk Kapanış Odaklı)",
            help=f"Erken dakikaların yanıltıcı gürültüsünü temizlemek için varsayılan olarak sadece {CLOSING_START}. dakika ve sonrasındaki kapanış verilerini baz alır.",
        )
        _roi_opt_1 = f"🎯 {CLOSING_START}-{CLOSE_MIN_INT}. Dk Kapanış: Her Maça 1 Kupon (Son Kapanış Kaydı) [Önerilen]"
        _roi_opt_2 = f"🔥 {FINAL_WIN_START}-{CLOSE_MIN_INT}. Dk Tam Kapanış: Sadece {FINAL_WIN_START}-{DEC_MIN}. Dk Son Kayıtlar (Her Maça 1 Kupon)"
        _roi_opt_3 = f"⏱️ {CLOSING_START}-{CLOSE_MIN_INT}. Dk Arası Tüm Kapanış Kayıtları"
        _roi_opt_4 = f"📊 Tüm Dakikalar (0-{CLOSE_MIN_INT}. Dk Ham Kayıtlar — Erken Dakikalar Dahil / Önerilmez)"
        roi_scope = st.selectbox(
            "⏱️ Kasa & ROI Hesaplama Bazı (Dakika Filtresi)",
            [_roi_opt_1, _roi_opt_2, _roi_opt_3, _roi_opt_4],
            index=0,
            key=f"bk_scope_{ACTIVE_LEAGUE}",
            help=f"Gerçek bahis stratejisinde kuponlar {CLOSING_START}. dakikadan sonraki kapanış bölgesinde ve maç başına 1 kez oynandığı için ilk seçenek en sağlıklı sonucu verir.",
        )
        df_comp_roi = (
            df_sym_all[df_sym_all["status"] == "COMPLETED"].copy()
            if not df_sym_all.empty and "status" in df_sym_all.columns
            else pd.DataFrame()
        )
        if df_comp_roi.empty:
            st.info(
                "Henüz sonuçlanmış (`COMPLETED`) canlı barem kaydı yok. Takip edilen canlı maçlar sonuçlandıkça "
                f"burada **{CLOSING_START}+ Dk Kapanış** stratejilerinin net TL kârı ve ROI yüzdesi listelenir."
            )
        else:
            df_comp_roi["_min_num"] = (
                df_comp_roi["check_time"]
                .astype(str)
                .str.extract(r"Minute\s+(\d+)", expand=False)
                .astype(float)
                .fillna(0)
            )
            if roi_scope == _roi_opt_1:
                df_eval_roi = df_comp_roi[df_comp_roi["_min_num"] >= CLOSING_START].copy()
                if not df_eval_roi.empty and "game_id" in df_eval_roi.columns:
                    df_eval_roi = df_eval_roi.sort_values("_min_num").groupby("game_id", as_index=False).last()
            elif roi_scope == _roi_opt_2:
                df_eval_roi = df_comp_roi[df_comp_roi["_min_num"] >= FINAL_WIN_START].copy()
                if not df_eval_roi.empty and "game_id" in df_eval_roi.columns:
                    df_eval_roi = df_eval_roi.sort_values("_min_num").groupby("game_id", as_index=False).last()
            elif roi_scope == _roi_opt_3:
                df_eval_roi = df_comp_roi[df_comp_roi["_min_num"] >= CLOSING_START].copy()
            else:
                df_eval_roi = df_comp_roi.copy()

            unit_bet = 1000.0
            fav_pnl, fav_hits = 0.0, 0
            val_pnl, val_hits = 0.0, 0
            clu_pnl, clu_hits = 0.0, 0
            v3_all_pnl, v3_all_hits = 0.0, 0
            v3_sel_pnl, v3_sel_hits, v3_sel_bets = 0.0, 0, 0
            dutch_sel_pnl, dutch_sel_hits = 0.0, 0
            dutch_v2_pnl, dutch_v2_hits = 0.0, 0
            n_bets = len(df_eval_roi)
            b_names_roi = [b[0] for b in BAREM_RANGES]

            for _, r in df_eval_roi.iterrows():
                act_b = str(r.get("actual_range", "")).strip()
                fav_b = str(r.get("highest_prob_range", "")).strip()
                val_b = str(r.get("value_range", "")).strip()
                clu_b = str(r.get("best_cluster_name", "")).strip()

                row_odds = {}
                for b_n, col_n in BAREM_TO_ODD_COL.items():
                    ov = float(r.get(col_n, 99.0) or 99.0)
                    if 1.0 < ov < 90.0:
                        row_odds[b_n] = ov
                sorted_ro = sorted(row_odds.items(), key=lambda x: x[1])
                if sorted_ro:
                    fav_b, fav_odd = sorted_ro[0]
                    sec_b, sec_odd = sorted_ro[1] if len(sorted_ro) > 1 else sorted_ro[0]
                else:
                    fav_odd = float(r.get(BAREM_TO_ODD_COL.get(fav_b, ""), 2.25) or 2.25)
                    if fav_odd >= 90.0:
                        fav_odd = 2.25
                    sec_b, sec_odd = fav_b, 2.50

                val_odd = float(r.get(BAREM_TO_ODD_COL.get(val_b, ""), 3.20) or 3.20)
                if val_odd >= 90.0:
                    val_odd = 3.20

                sc_str = str(r.get("current_score", ""))
                try:
                    sp = sc_str.strip().split("-")
                    sc_gap = abs(int(sp[0]) - int(sp[1]))
                except Exception:
                    sc_gap = 8

                idx_fav_r = b_names_roi.index(fav_b) if fav_b in b_names_roi else min(4, len(b_names_roi) // 2)
                idx_sec_r = b_names_roi.index(sec_b) if sec_b in b_names_roi else max(0, idx_fav_r - 1)
                sec_dir_r = idx_sec_r - idx_fav_r
                m1_b_r = b_names_roi[max(0, idx_fav_r - 1)]
                odd_m1_r = row_odds.get(m1_b_r, 3.20)

                if fav_odd < 1.85:
                    v3_pick_b, v3_pick_o = sec_b, sec_odd
                elif fav_odd < 2.10 and sec_dir_r <= -1:
                    v3_pick_b, v3_pick_o = m1_b_r, odd_m1_r
                else:
                    v3_pick_b, v3_pick_o = fav_b, fav_odd

                if act_b == v3_pick_b:
                    v3_all_pnl += unit_bet * (v3_pick_o - 1.0)
                    v3_all_hits += 1
                else:
                    v3_all_pnl -= unit_bet

                if sc_gap >= 6:
                    v3_sel_bets += 1
                    if act_b == v3_pick_b:
                        v3_sel_pnl += unit_bet * (v3_pick_o - 1.0)
                        v3_sel_hits += 1
                    else:
                        v3_sel_pnl -= unit_bet

                if sc_gap >= int(LEAGUE_CFG["blowout_brake_gap"]):
                    db1, do1 = fav_b, fav_odd
                    db2, do2 = m1_b_r, odd_m1_r
                else:
                    db1, do1 = fav_b, fav_odd
                    db2, do2 = sec_b, sec_odd
                eff_dv2 = 1.0 / ((1.0 / do1) + (1.0 / do2)) if (do1 > 0 and do2 > 0 and db1 != db2) else do1
                if act_b in (db1, db2):
                    dutch_v2_pnl += unit_bet * (eff_dv2 - 1.0)
                    dutch_v2_hits += 1
                else:
                    dutch_v2_pnl -= unit_bet

                if sc_gap >= 6:
                    if act_b in (db1, db2):
                        dutch_sel_pnl += unit_bet * (eff_dv2 - 1.0)
                        dutch_sel_hits += 1
                    else:
                        dutch_sel_pnl -= unit_bet

                if act_b == fav_b:
                    fav_pnl += unit_bet * (fav_odd - 1.0)
                    fav_hits += 1
                else:
                    fav_pnl -= unit_bet

                if act_b == val_b:
                    val_pnl += unit_bet * (val_odd - 1.0)
                    val_hits += 1
                else:
                    val_pnl -= unit_bet

                c_members = []
                for c_name, _, _, mems in CLUSTER_RANGES:
                    if c_name == clu_b:
                        c_members = list(mems)
                        break
                if len(c_members) == 2:
                    o1 = float(r.get(BAREM_TO_ODD_COL.get(c_members[0], ""), 2.6) or 2.6)
                    o2 = float(r.get(BAREM_TO_ODD_COL.get(c_members[1], ""), 2.6) or 2.6)
                    if o1 >= 90.0:
                        o1 = 2.6
                    if o2 >= 90.0:
                        o2 = 2.6
                    eff_o = 1.0 / ((1.0 / o1) + (1.0 / o2))
                else:
                    eff_o = 1.35

                if check_cluster_hit(clu_b, act_b, league=ACTIVE_LEAGUE):
                    clu_pnl += unit_bet * (eff_o - 1.0)
                    clu_hits += 1
                else:
                    clu_pnl -= unit_bet

            tot_wagered = max(1.0, n_bets * unit_bet)
            sel_wagered = max(1.0, v3_sel_bets * unit_bet)
            raw_comp_cnt = len(df_comp_roi)
            st.caption(
                f"✅ **Hesaplanan Kapanış Maç Sayısı: {n_bets} Maç** "
                f"*(Toplam {raw_comp_cnt} ham dakika kaydından erken dakikalardaki {max(0, raw_comp_cnt - n_bets)} kayıt temizlendi; sadece {CLOSING_START}+ dk kapanışlar baz alındı)*"
            )
            roi_rows = [
                {
                    "Oynama Yöntemi (Strateji)": f"🔥 Seçici Akıllı Strateji (0-5 Sayı Farklı Yakın Maçları Pas Geç + {CLOSING_START}-{CLOSE_MIN_INT}.Dk Kural)",
                    "Oynanan Maç": v3_sel_bets,
                    "Tutma Oranı": f"{v3_sel_hits} / {v3_sel_bets} (%{(v3_sel_hits/max(1,v3_sel_bets))*100:.1f})",
                    "Toplam Yatırılan": f"{sel_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{v3_sel_pnl:+,.0f} TL",
                    "Kâr Oranı (%)": f"%{(v3_sel_pnl / sel_wagered)*100:+.1f}",
                },
                {
                    "Oynama Yöntemi (Strateji)": "🔥 Akıllı Value Barem (Simetri + Piyasa + Model Hibrit Değerli Tek Barem)",
                    "Oynanan Maç": n_bets,
                    "Tutma Oranı": f"{val_hits} / {n_bets} (%{(val_hits/max(1,n_bets))*100:.1f})",
                    "Toplam Yatırılan": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{val_pnl:+,.0f} TL",
                    "Kâr Oranı (%)": f"%{(val_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Oynama Yöntemi (Strateji)": "🤖 Tüm Maçlara Akıllı Tek Barem (Oran <1.85➔2.Fav | Oran <2.10&Alt➔1 Alt | Diğer➔1.Fav)",
                    "Oynanan Maç": n_bets,
                    "Tutma Oranı": f"{v3_all_hits} / {n_bets} (%{(v3_all_hits/max(1,n_bets))*100:.1f})",
                    "Toplam Yatırılan": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{v3_all_pnl:+,.0f} TL",
                    "Kâr Oranı (%)": f"%{(v3_all_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Oynama Yöntemi (Strateji)": "🛡️ Seçici İki Barem Kapatma (0-5 Sayı Farklı Yakın Maçları Pas Geç + 1. & 2. Favori İkilisi)",
                    "Oynanan Maç": v3_sel_bets,
                    "Tutma Oranı": f"{dutch_sel_hits} / {v3_sel_bets} (%{(dutch_sel_hits/max(1,v3_sel_bets))*100:.1f})",
                    "Toplam Yatırılan": f"{sel_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{dutch_sel_pnl:+,.0f} TL",
                    "Kâr Oranı (%)": f"%{(dutch_sel_pnl / sel_wagered)*100:+.1f}",
                },
                {
                    "Oynama Yöntemi (Strateji)": f"⭐ Her Maç 1xBet 1. Favori Baremi Oynamak ({CLOSING_START}+ Dk Kapanış)",
                    "Oynanan Maç": n_bets,
                    "Tutma Oranı": f"{fav_hits} / {n_bets} (%{(fav_hits/max(1,n_bets))*100:.1f})",
                    "Toplam Yatırılan": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{fav_pnl:+,.0f} TL",
                    "Kâr Oranı (%)": f"%{(fav_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Oynama Yöntemi (Strateji)": "🛡️ Tüm Maçlara İki Barem Kapatma (1. & 2. Favori İkilisi / Farkta Favori & 1 Alt)",
                    "Oynanan Maç": n_bets,
                    "Tutma Oranı": f"{dutch_v2_hits} / {n_bets} (%{(dutch_v2_hits/max(1,n_bets))*100:.1f})",
                    "Toplam Yatırılan": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{dutch_v2_pnl:+,.0f} TL",
                    "Kâr Oranı (%)": f"%{(dutch_v2_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Oynama Yöntemi (Strateji)": "🛡️ 20-Sayı Kümesi (Yan Yana En Güçlü İki Baremi Kapatma)",
                    "Oynanan Maç": n_bets,
                    "Tutma Oranı": f"{clu_hits} / {n_bets} (%{(clu_hits/max(1,n_bets))*100:.1f})",
                    "Toplam Yatırılan": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{clu_pnl:+,.0f} TL",
                    "Kâr Oranı (%)": f"%{(clu_pnl / tot_wagered)*100:+.1f}",
                },
            ]
            st.dataframe(
                pd.DataFrame(roi_rows),
                column_config={
                    "Oynama Yöntemi (Strateji)": st.column_config.TextColumn(
                        help=f"Test edilen bahis yöntemi (Sadece {CLOSING_START}. dakika ve sonrası gerçek kapanış oranlarıyla hesaplanır)."
                    ),
                    "Oynanan Maç": st.column_config.NumberColumn(
                        help=f"{CLOSING_START}. dakika ve sonrasında kapanış kaydı bulunan toplam maç sayısı (her maça 1 kupon)."
                    ),
                    "Tutma Oranı": st.column_config.TextColumn(help="Tutan kupon sayısı ve yüzdesi."),
                    "Toplam Yatırılan": st.column_config.TextColumn(help="Her maça sabit 1.000 TL yatırıldığında kasadan çıkan toplam tutar."),
                    "Net Kâr / Zarar": st.column_config.TextColumn(help="Tüm kuponlar sonuçlandıktan sonra cebimizde kalan net kâr veya zarar."),
                    "Kâr Oranı (%)": st.column_config.TextColumn(help="Yatırılan toplam paraya göre yüzdesel net kâr oranı."),
                },
                use_container_width=True,
                hide_index=True,
            )

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5: MANUAL LIVE TEMPO & MONTE CARLO SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_sim:
    _def_q = 18 if IS_EURO else 26
    _def_q3 = 13 if IS_EURO else 18
    _def_rem_s = round(Q_LEN * 0.4, 1) if IS_EURO else 4.5
    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        sim_q = st.selectbox("Aktif Çeyrek", [1, 2, 3, 4], index=2, key=f"sim_q_{ACTIVE_LEAGUE}")
        sim_rem = st.slider("Çeyrekte Kalan Dakika", 0.0, Q_LEN, _def_rem_s, 0.5, key=f"sim_rem_{ACTIVE_LEAGUE}")
    with s_c2:
        sim_hq1 = st.number_input("Ev Q1", 0, 60, _def_q, key=f"sim_hq1_{ACTIVE_LEAGUE}")
        sim_hq2 = st.number_input("Ev Q2", 0, 60, _def_q, key=f"sim_hq2_{ACTIVE_LEAGUE}")
        sim_hq3 = st.number_input("Ev Q3 (veya Canlı)", 0, 60, _def_q3, key=f"sim_hq3_{ACTIVE_LEAGUE}")
    with s_c3:
        sim_aq1 = st.number_input("Dep Q1", 0, 60, _def_q, key=f"sim_aq1_{ACTIVE_LEAGUE}")
        sim_aq2 = st.number_input("Dep Q2", 0, 60, _def_q, key=f"sim_aq2_{ACTIVE_LEAGUE}")
        sim_aq3 = st.number_input("Dep Q3 (veya Canlı)", 0, 60, _def_q3, key=f"sim_aq3_{ACTIVE_LEAGUE}")

    comp_q_list = []
    if sim_q == 1:
        h_live, a_live = sim_hq1, sim_aq1
    elif sim_q == 2:
        comp_q_list = [(sim_hq1, sim_aq1)]
        h_live, a_live = sim_hq2, sim_aq2
    elif sim_q == 3:
        comp_q_list = [(sim_hq1, sim_aq1), (sim_hq2, sim_aq2)]
        h_live, a_live = sim_hq3, sim_aq3
    else:
        comp_q_list = [(sim_hq1, sim_aq1), (sim_hq2, sim_aq2), (sim_hq3, sim_aq3)]
        h_live, a_live = (10, 10) if IS_EURO else (15, 15)

    sim_res = predict_live_inplay(
        df=df_completed_archive,
        team_home="",
        team_away="",
        current_quarter=sim_q,
        home_live_score=h_live,
        away_live_score=a_live,
        remaining_mins=sim_rem,
        completed_quarters=comp_q_list,
        quarter_length=Q_LEN,
        league=ACTIVE_LEAGUE,
    )
    st.success(
        f"🎯 **Tahmini Maç Sonu:** {sim_res['predicted_home_final']:.0f} - {sim_res['predicted_away_final']:.0f} "
        f"(**Toplam: {sim_res['predicted_total']:.1f} Sayı**) | "
        f"**En Güçlü 20-Sayı Kümesi:** `{sim_res['best_cluster_name']}` (%{sim_res['best_cluster_prob']:.1f})"
    )
    b_probs = sim_res.get("barem_probs", {})
    fig_sim = go.Figure(
        go.Bar(
            x=list(b_probs.keys()),
            y=list(b_probs.values()),
            text=[f"%{v:.1f}" for v in b_probs.values()],
            textposition="outside",
            marker_color="#2563eb",
        )
    )
    fig_sim.update_layout(
        height=260,
        margin=dict(l=15, r=15, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569" if not IS_DARK else "#cbd5e1", size=11),
        yaxis=dict(title="Olasılık (%)", range=[0, max(list(b_probs.values()) + [35]) * 1.2]),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 7: TEAM PAIR H2H HISTORY ("TAKIM ÇİFTİ GEÇMİŞİ")
# ═════════════════════════════════════════════════════════════════════════════
with tab_pair:
    st.markdown(
        f"##### 🤝 Takım Çifti Bazlı Birebir Geçmiş (H2H) — {LEAGUE_CFG['short_name']}",
        help="İki takımın kendi aralarındaki gerçek karşılaşmalarda çıkan barem, tempo ve uzatma eğilimini inceler.",
    )
    if df_completed_archive.empty:
        st.info("Takım çifti analizi için tamamlanmış geçmiş maç arşivi bekleniyor.")
    else:
        pair_teams_all = sorted(
            set(df_completed_archive["home_team"].dropna().unique()).union(
                set(df_completed_archive["away_team"].dropna().unique())
            )
        )
        live_pair_presets = ["Canlı Maç Seç (Opsiyonel)"] + [
            f"{m.get('team_home', '')} vs {m.get('team_away', '')}" for m in live_matches_now
        ]
        sel_live_pair = st.selectbox("Canlı Maçtan Takım Çifti Getir", live_pair_presets, index=0, key=f"pair_live_preset_{ACTIVE_LEAGUE}")
        def_p1_idx = 0
        def_p2_idx = min(1, max(0, len(pair_teams_all) - 1))
        if sel_live_pair != "Canlı Maç Seç (Opsiyonel)" and " vs " in sel_live_pair:
            pt_h, pt_a = sel_live_pair.split(" vs ", 1)
            if pt_h in pair_teams_all:
                def_p1_idx = pair_teams_all.index(pt_h)
            if pt_a in pair_teams_all:
                def_p2_idx = pair_teams_all.index(pt_a)

        pc1, pc2 = st.columns(2)
        with pc1:
            pair_t1 = st.selectbox("1. Takım", pair_teams_all, index=def_p1_idx, key=f"pair_t1_{ACTIVE_LEAGUE}")
        with pc2:
            pair_t2 = st.selectbox("2. Takım", pair_teams_all, index=def_p2_idx, key=f"pair_t2_{ACTIVE_LEAGUE}")

        pair_res = analyze_team_pair_history(df_completed_archive, pair_t1, pair_t2, last_n=10, league=ACTIVE_LEAGUE)
        if pair_res.get("n", 0) == 0:
            st.warning("Seçilen iki takım arasında geçmiş arşivde tamamlanmış karşılaşma bulunamadı.")
        else:
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("🤝 Karşılaşma Sayısı", f"{pair_res['n']} Maç", pair_res.get("confidence", "").upper())
            pm2.metric("🏀 Ort. Toplam Sayı", f"{pair_res['avg_total']} Sayı", f"Lig Ort. Fark: {pair_res['diff_vs_league']:+.1f}")
            pm3.metric("🎯 En Sık Barem", f"{pair_res['top_barem']} (%{pair_res['top_barem_pct']:.0f})", f"2. En Sık: {pair_res['second_barem']}")
            pm4.metric("⚡ Q4 Ort. / Uzatma", f"{pair_res['avg_q4']} Sayı", f"OT: %{pair_res['ot_rate']} • Ort. Fark: {pair_res['avg_margin']}")
            st.info(f"📌 **Takım Çifti Özeti:** {pair_res['hint']}")
            if pair_res.get("last_games"):
                st.dataframe(pd.DataFrame(pair_res["last_games"]), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(f"###### 📚 Tüm {LEAGUE_CFG['short_name']} Takım Çiftleri Barem & Tempo Tablosu")
        df_pairs_all = build_team_pair_table(df_completed_archive, min_games=2, league=ACTIVE_LEAGUE)
        if not df_pairs_all.empty:
            st.dataframe(df_pairs_all, use_container_width=True, hide_index=True, height=340)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 8: DATA VALIDATION GUARD ("VERİ DOĞRULAMA — HATA BEKÇİSİ")
# ═════════════════════════════════════════════════════════════════════════════
with tab_guard:
    st.markdown(
        f"##### 🛡️ Veri Doğrulama — Hata Bekçisi ({LEAGUE_CFG['short_name']})",
        help="Canlı barem kayıtlarındaki favori uyuşmazlıkları, kilitli oran tuzakları, skor geri gitmeleri ve eksik sonuçları otomatik denetler.",
    )
    guard_res = validate_symmetry_log(df_sym_all, league=ACTIVE_LEAGUE)
    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.metric("📋 Taranan Satır", f"{guard_res['n_rows']} Kayıt", f"{guard_res['n_games']} Maç")
    gc2.metric("🔴 Kritik Hata", f"{guard_res['counts'].get('🔴 Hata', 0)}")
    gc3.metric("🟠 Uyarı", f"{guard_res['counts'].get('🟠 Uyarı', 0)}")
    gc4.metric("🔵 Bilgi Notu", f"{guard_res['counts'].get('🔵 Bilgi', 0)}")

    issues_list = guard_res.get("issues", [])
    if guard_res["n_rows"] == 0:
        st.info(f"{LEAGUE_CFG['short_name']} için taranacak barem kaydı henüz yok; canlı veriler geldikçe otomatik denetlenecektir.")
    elif not issues_list:
        st.success(f"✅ Tüm {guard_res['n_rows']} kayıt ({guard_res['n_games']} maç) temiz! Hiçbir veri hatası veya oran anomalisi tespit edilmedi.")
    else:
        st.dataframe(pd.DataFrame(issues_list), use_container_width=True, hide_index=True, height=380)
