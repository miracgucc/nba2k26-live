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

import period_predictor as _pp_mod
import update_all_checkpoints as _uac_mod
import pull_today_results as _ptr_mod
import run_live_auto_scraper as _rlas_mod

importlib.reload(_pp_mod)
importlib.reload(_uac_mod)
importlib.reload(_ptr_mod)
importlib.reload(_rlas_mod)

from period_predictor import (
    BAREM_RANGES,
    CLUSTER_RANGES,
    analyze_barem_symmetry,
    analyze_sequential_barem_wave,
    analyze_smart_35m_decision,
    build_q3_q4_streak_signal,
    get_cluster_from_score,
    get_range_from_score,
    predict_live_inplay,
    predict_period_scores,
    simulate_35m_strategies,
)
from pull_today_results import discover_active_domain, pull_today_results
from run_live_auto_scraper import (
    CACHE_FILE,
    CSV_HEADERS,
    CSV_PATH,
    LIVE_STATE_FILE,
    MATCHES_CSV_PATH,
    TS_FILE,
    init_csv,
    merge_cloud_backups,
    run_auto_scraper,
)
from update_all_checkpoints import auto_settle_checkpoints

TR_TZ = timezone(timedelta(hours=3))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NBA 2K26 CyberLeague Pro v2.0",
    page_icon="ğŸ€",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "soft_light"


def toggle_ui_theme():
    st.session_state.ui_theme = "matte_dark" if st.session_state.ui_theme == "soft_light" else "soft_light"


IS_DARK = st.session_state.ui_theme == "matte_dark"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SINGLETON BACKGROUND DAEMON THREAD (24/7 ZERO-CRASH AUTO SCRAPER)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCRAPE_LOCK = threading.Lock()


ENGINE_VERSION = "v2.9-live-gh"


class BackgroundEngineState:
    def __init__(self):
        self.version = ENGINE_VERSION
        self.is_running = False
        self.last_run_ts = 0.0
        self.last_run_str = "HazÄ±rlanÄ±yor"
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DATA LOADERS & ANALYTICS HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def reset_all_data_clean_slate():
    with SCRAPE_LOCK:
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write(",".join(CSV_HEADERS) + "\n")
        matches_headers = (
            "game_id,date,home_team,away_team,home_q1,away_q1,home_q2,away_q2,"
            "home_q3,away_q3,home_q4,away_q4,ot_home,ot_away,home_total,away_total,"
            "winner,total_score,margin,is_ot,top_scorer,top_scorer_points\n"
        )
        with open(MATCHES_CSV_PATH, "w", encoding="utf-8") as f:
            f.write(matches_headers)
        for jf in [CACHE_FILE, TS_FILE, LIVE_STATE_FILE]:
            with open(jf, "w", encoding="utf-8") as f:
                json.dump({}, f)


def load_live_state() -> dict:
    if os.path.exists(LIVE_STATE_FILE) and os.path.getsize(LIVE_STATE_FILE) > 0:
        try:
            with open(LIVE_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
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


def load_symmetry_log() -> pd.DataFrame:
    """
    Stateless loader: combines local log, local backup, AND live GitHub Raw cloud-bot backup.
    Never writes files, so no thread race / stale-file / redeploy-overwrite can ever hide matches.
    COMPLETED rows always win over PENDING rows for the same (game_id, check_time).
    """
    init_csv()
    frames = [
        _read_csv_safe(CSV_PATH),
        _read_csv_safe(os.path.join(BASE_DIR, "cloud_symmetry_backup.csv")),
        _fetch_github_csv("cloud_symmetry_backup.csv"),
    ]
    frames = [f for f in frames if not f.empty and "game_id" in f.columns and "check_time" in f.columns]
    if not frames:
        return pd.DataFrame(columns=CSV_HEADERS)
    df = pd.concat(frames, ignore_index=True)
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["_gnum"] = df["game_id"].str.extract(r"(\d+)")[0].fillna(0).astype(int)
    df = df[df["_gnum"] >= 770265000].copy()
    if df.empty:
        return pd.DataFrame(columns=CSV_HEADERS)
    df["_done"] = (df["status"].astype(str) == "COMPLETED").astype(int)
    df = df.sort_values("_done", kind="stable").drop_duplicates(subset=["game_id", "check_time"], keep="last")
    df["_min"] = df["check_time"].astype(str).str.extract(r"Minute\s+(\d+)")[0].fillna(0).astype(int)
    df = df.sort_values(["_gnum", "_min"], kind="stable").drop(columns=["_gnum", "_done", "_min"])
    return df.reset_index(drop=True)


def load_completed_matches() -> pd.DataFrame:
    frames = [
        _read_csv_safe(MATCHES_CSV_PATH),
        _read_csv_safe(os.path.join(BASE_DIR, "cloud_matches_backup.csv")),
        _fetch_github_csv("cloud_matches_backup.csv"),
    ]
    frames = [f for f in frames if not f.empty and "game_id" in f.columns]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_id"] = df["game_id"].astype(str).str.strip()
    return df.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)


def check_cluster_hit(cluster_name: str, actual_range: str) -> bool:
    if not cluster_name or not actual_range or pd.isna(cluster_name) or pd.isna(actual_range):
        return False
    c_str = str(cluster_name).strip()
    a_str = str(actual_range).strip()
    for name, _, _, member_tuple in CLUSTER_RANGES:
        if name == c_str:
            return a_str in member_tuple
    return False


BAREM_TO_ODD_COL = {
    "160 Or Less": "odds_160_or_less",
    "161-170": "odds_161_170",
    "171-180": "odds_171_180",
    "181-190": "odds_181_190",
    "191-200": "odds_191_200",
    "201-210": "odds_201_210",
    "211-220": "odds_211_220",
    "221-230": "odds_221_230",
    "231+": "odds_231_plus",
}


def compute_team_dna_table(df_matches: pd.DataFrame) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()
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

        barems = tot_scores.apply(get_range_from_score)
        clusters = tot_scores.apply(get_cluster_from_score)
        top_barem = barems.mode().iloc[0] if not barems.mode().empty else "-"
        top_barem_pct = (barems == top_barem).mean() * 100.0
        top_cluster = clusters.mode().iloc[0] if not clusters.mode().empty else "-"
        top_cluster_pct = (clusters == top_cluster).mean() * 100.0
        over_210_pct = (tot_scores >= 211).mean() * 100.0

        rows.append(
            {
                "TakÄ±m": t,
                "MaÃ§ SayÄ±sÄ±": n_games,
                "Ort. Toplam SayÄ±": round(tot_scores.mean(), 1),
                "Std. Sapma": round(tot_scores.std() if n_games > 1 else 0.0, 1),
                "Q1 Ort.": round(q1_tot, 1),
                "Q2 Ort.": round(q2_tot, 1),
                "Q3 Ort.": round(q3_tot, 1),
                "Q4 Ort.": round(q4_tot, 1),
                "En Ã‡ok Ã‡Ä±kan Barem": f"{top_barem} (%{top_barem_pct:.0f})",
                "En GÃ¼Ã§lÃ¼ 20'li KÃ¼me": f"{top_cluster} (%{top_cluster_pct:.0f})",
                "211+ Ãœst OranÄ±": f"%{over_210_pct:.1f}",
            }
        )
    df_dna = pd.DataFrame(rows)
    if not df_dna.empty:
        df_dna = df_dna.sort_values(by=["MaÃ§ SayÄ±sÄ±", "Ort. Toplam SayÄ±"], ascending=[False, False])
    return df_dna


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# EYE-FRIENDLY NORDIC SOFT SLATE / WARM MATTE THEME PALETTE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

.block-container {{
    padding-top: 1.2rem !important;
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

/* 9-Column Barem Grid */
.barem-grid {{
    display: grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 0.45rem;
    margin-top: 0.8rem;
}}

@media (max-width: 1100px) {{
    .barem-grid {{
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TOP COMMAND BAR & CONTROLS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
live_state = load_live_state()
active_domain = live_state.get("domain") or "https://1xbet.global"
last_update_str = live_state.get("updated_at") or bg_engine.last_run_str
scan_dur = live_state.get("duration_sec", 0.0)
next_in = (
    max(0, int((bg_engine.last_run_ts + bg_engine.next_poll_sec) - time.time()))
    if bg_engine.last_run_ts > 0
    else 5
)

st.markdown(
    f"""
<div class="cmd-header">
    <div>
        <div class="cmd-title">ğŸ€ NBA 2K26 CyberLeague Pro</div>
        <div class="cmd-sub">CanlÄ± Barem (Exact Points), KapanÄ±ÅŸ Simetrisi, TakÄ±m Barem DNA'sÄ± & Kasa/Dutching Terminali</div>
    </div>
    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;">
        <div class="status-pill">
            <span>â—</span>
            <span>CanlÄ± AkÄ±ÅŸ Aktif ({active_domain.replace('https://', '')})</span>
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

ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5, ctrl_c6 = st.columns([1.25, 1.25, 1.3, 1.0, 0.85, 0.9])

with ctrl_c1:
    if st.button("âš¡ AnlÄ±k CanlÄ± Barem Ã‡ek", use_container_width=True, type="primary"):
        with st.spinner("1xBet / Mirror Ã¼zerinden canlÄ± maÃ§lar ve baremler Ã§ekiliyor (~2 sn)..."):
            with SCRAPE_LOCK:
                run_auto_scraper(force_manual=True)
                bg_engine.last_run_ts = time.time()
                bg_engine.last_run_str = datetime.now().strftime("%H:%M:%S")
        st.toast("âœ… CanlÄ± maÃ§lar ve barem oranlarÄ± gÃ¼ncellendi!", icon="ğŸ€")
        st.rerun()

with ctrl_c2:
    if st.button("ğŸ”„ BitmiÅŸ MaÃ§larÄ± SonuÃ§landÄ±r", use_container_width=True):
        with st.spinner("BitmiÅŸ maÃ§ sonuÃ§larÄ± kontrol ediliyor..."):
            pull_today_results(only_tracked=True)
            settled = auto_settle_checkpoints()
        st.toast(f"âœ… {settled} barem kaydÄ± sonuÃ§landÄ±rÄ±ldÄ±!", icon="ğŸ¯")
        st.rerun()

with ctrl_c3:
    if st.button("ğŸ“¥ Son 4 Ay ArÅŸivini GÃ¼ncelle (120 GÃ¼n)", use_container_width=True):
        with st.spinner("Son 4 aydaki (120 gÃ¼n) tÃ¼m tamamlanmÄ±ÅŸ CyberLeague maÃ§larÄ± Ã§ekiliyor..."):
            added = pull_today_results(only_tracked=False, days_back=125)
            auto_settle_checkpoints()
        st.toast(f"ğŸ“¥ {added} yeni tamamlanmÄ±ÅŸ maÃ§ 4 aylÄ±k geÃ§miÅŸ arÅŸivine eklendi!", icon="ğŸ“Š")
        st.rerun()

with ctrl_c4:
    if st.button("ğŸ—‘ï¸ Verileri SÄ±fÄ±rla", use_container_width=True):
        reset_all_data_clean_slate()
        st.toast("ğŸ§¹ TÃ¼m kayÄ±tlar sÄ±fÄ±rlandÄ± (0 kayÄ±t). Temiz sayfa hazÄ±r!", icon="âœ¨")
        st.rerun()

with ctrl_c5:
    theme_btn_label = "ğŸŒ™ Mat Gece" if not IS_DARK else "ğŸŒ¤ï¸ Ferah GÃ¼n"
    st.button(theme_btn_label, on_click=toggle_ui_theme, use_container_width=True)

with ctrl_c6:
    auto_ui_refresh = st.toggle("â±ï¸ Oto-Yenile", value=True)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN DASHBOARD FRAGMENT (AUTO-REFRESHING WITHOUT FULL PAGE FLICKER)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.fragment(run_every="15s" if auto_ui_refresh else None)
def render_live_dashboard():
    state = load_live_state()
    df_log = load_symmetry_log()
    df_hist = load_completed_matches()
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
        fav_wins = sum(
            1 for _, r in df_comp.iterrows() if str(r.get("actual_range")) == str(r.get("highest_prob_range"))
        )
        val_wins = sum(
            1 for _, r in df_comp.iterrows() if str(r.get("actual_range")) == str(r.get("value_range"))
        )
        clu_wins = sum(
            1 for _, r in df_comp.iterrows() if check_cluster_hit(r.get("best_cluster_name"), r.get("actual_range"))
        )
        fav_rate = (fav_wins / comp_count) * 100.0
        val_rate = (val_wins / comp_count) * 100.0
        clu_rate = (clu_wins / comp_count) * 100.0
    else:
        fav_wins = val_wins = clu_wins = 0
        fav_rate = val_rate = clu_rate = 0.0

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">ğŸ”´ CanlÄ± CyberLeague MaÃ§Ä±</div>
            <div class="kpi-val">{active_count} MaÃ§</div>
            <div class="kpi-sub">{open_barem_count} MaÃ§ta Barem AÃ§Ä±k &bull; {hist_count} GeÃ§miÅŸ ArÅŸiv</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">ğŸ“‹ Barem KayÄ±t AkÄ±ÅŸÄ±</div>
            <div class="kpi-val">{total_logs}</div>
            <div class="kpi-sub">{pend_count} Bekleyen &bull; {comp_count} SonuÃ§lanan</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">â­ 1xBet Favori Barem</div>
            <div class="kpi-val" style="color:#059669;">%{fav_rate:.1f}</div>
            <div class="kpi-sub">{fav_wins} / {comp_count} Ä°sabet (En DÃ¼ÅŸÃ¼k Oran)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">ğŸ”¥ Model Value Barem</div>
            <div class="kpi-val" style="color:#d97706;">%{val_rate:.1f}</div>
            <div class="kpi-sub">{val_wins} / {comp_count} Ä°sabet (Pozitif EV)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">ğŸ›¡ï¸ 20-SayÄ± KÃ¼me (Ã‡ifte Barem)</div>
            <div class="kpi-val" style="color:#2563eb;">%{clu_rate:.1f}</div>
            <div class="kpi-sub">{clu_wins} / {comp_count} Ä°sabet (Ä°kili Blok)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SEQUENTIAL BAREM WAVE STRIP (LAST 8 MATCHES: 0, -1, +1 WAVE DETECTOR)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    wave_global = analyze_sequential_barem_wave(df_hist, df_log)
    recent_chain = wave_global.get("recent_chain", [])
    if recent_chain:
        pills_html = ""
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
                p_bg = "var(--fav1-bg)"
                p_bd = "var(--fav1-border)"
                p_tx = "var(--fav1-text)"
                tag_txt = "ğŸ¯ 0 (Beklenen)"
            elif sh < 0:
                p_bg = "var(--fav2-bg)"
                p_bd = "var(--fav2-border)"
                p_tx = "var(--fav2-text)"
                tag_txt = f"ğŸ“‰ {sh} ({abs(sh)} Alt Barem)"
            else:
                p_bg = "var(--val-bg)"
                p_bd = "var(--val-border)"
                p_tx = "var(--val-text)"
                tag_txt = f"ğŸ“ˆ +{sh} ({sh} Ãœst Barem)"

            odd_36_val = float(item.get("exp_odd_36", 0.0) or 0.0)
            if odd_36_val > 0:
                ref_line = f'ğŸ“Š 36.Dk Oran: <b>{item["exp_barem_36"]} ({odd_36_val:.2f})</b> â” Biten: <b>{item["act_barem"]}</b>'
            else:
                ref_line = f'ğŸ§® 36.Dk SkorTahmin: <b>{item["exp_barem_36"]}</b> â” Biten: <b>{item["act_barem"]}</b>'
            drift_line = item.get("drift_label", "")

            pills_html += (
                f'<div style="background:{p_bg}; border:1.5px solid {p_bd}; color:{p_tx}; border-radius:10px; padding:0.48rem 0.65rem; min-width:155px; flex:1 1 180px;">'
                f'<div style="font-size:0.67rem; font-weight:700; opacity:0.85;">#{idx_c} &bull; ğŸ•’ {time_only} | {t_short}</div>'
                f'<div style="font-size:0.71rem; font-weight:700; margin-top:0.08rem;">Skor: {sc_pair} ({item["total_score"]} SayÄ±)</div>'
                f'<div style="font-size:0.82rem; font-weight:800; margin:0.12rem 0;">{tag_txt}</div>'
                f'<div style="font-size:0.66rem; opacity:0.95;">{ref_line}</div>'
                f'<div style="font-size:0.63rem; opacity:0.85; margin-top:0.08rem;">{drift_line}</div>'
                f"</div>"
            )

        st.markdown(
            f"""
            <div style="background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:13px; padding:0.9rem 1.2rem; margin-top:0.9rem; box-shadow:var(--shadow-card);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.6rem;">
                    <div style="font-size:0.92rem; font-weight:800; color:var(--text-primary);">
                        ğŸŒŠ ArdÄ±ÅŸÄ±k MaÃ§larda 36. Dakika Simetri Dalga Zinciri (Toplam {len(recent_chain)} MaÃ§ | Son Zincir: <span style="font-family:'JetBrains Mono',monospace; color:var(--score-text);">{wave_global['last_5_labels']}</span>)
                    </div>
                    <div style="font-size:0.8rem; font-weight:700; color:var(--fav1-text); background:var(--pill-bg); border:1px solid var(--pill-border); padding:0.25rem 0.75rem; border-radius:999px;">
                        ğŸ”® SÄ±radaki MaÃ§ Dalga Beklentisi ({wave_global['pattern_matches_count']} Benzer Zincir): â„ï¸ -2 Alt: %{wave_global['prob_minus_2']} &bull; ğŸ“‰ -1 Alt: %{wave_global['prob_minus_1']} &bull; ğŸ¯ Merkez (0): %{wave_global['prob_0']} &bull; ğŸ“ˆ +1 Ãœst: %{wave_global['prob_plus_1']} &bull; ğŸ”¥ +2 Ãœst: %{wave_global['prob_plus_2']}
                    </div>
                </div>
                <div style="display:flex; gap:0.45rem; flex-wrap:wrap; max-height:340px; overflow-y:auto; padding-right:4px;">
                    {pills_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SECTION 1: LIVE MATCH CARDS & 9-BAREM MATRIX
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown(
        '<div class="section-hdr">ğŸ”´ CanlÄ± MaÃ§ KartlarÄ± & AnlÄ±k 9\'lu Barem Matrisi (0 - 36. Dakika)</div>',
        unsafe_allow_html=True,
    )

    if not live_matches:
        st.info(
            "Åu an aktif canlÄ± maÃ§ bekleniyor veya tarama yapÄ±lÄ±yor. "
            "YukarÄ±daki **'âš¡ AnlÄ±k CanlÄ± Barem Ã‡ek'** butonuna basarak anÄ±nda kontrol edebilirsiniz."
        )
    else:
        for m in live_matches:
            t_home = m.get("team_home", "")
            t_away = m.get("team_away", "")
            h_sc = m.get("home_score", 0)
            a_sc = m.get("away_score", 0)
            tot_sc = h_sc + a_sc
            q_num = m.get("quarter", 1)
            el_m = m.get("elapsed_mins", 0.0)
            rem_q = m.get("remaining_mins", 12.0)
            is_open = m.get("is_barem_open", False)
            status_code = m.get("status", "")
            h2h_n = m.get("h2h_sample_size", 0)

            comp_qs = m.get("completed_quarters", [])
            q_str_parts = [f"Q{idx+1}: {hq}-{aq} ({hq+aq})" for idx, (hq, aq) in enumerate(comp_qs)]
            q_str_parts.append(f"Q{q_num} (CanlÄ±): {m.get('live_q_score', '0-0')}")
            quarters_summary = " &nbsp;&bull;&nbsp; ".join(q_str_parts)

            if status_code == "PRE_GAME":
                badge_html = (
                    '<span style="background:var(--bg-subcard); border:1px solid var(--border-subtle); '
                    'color:var(--text-secondary); padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    "â³ MAÃ‡ BAÅLAMADI</span>"
                )
            elif is_open:
                rem_to_36 = max(0.0, 36.0 - el_m)
                badge_html = (
                    '<span style="background:var(--pill-bg); border:1px solid var(--pill-border); '
                    'color:var(--pill-text); padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    f"ğŸŸ¢ BAREM AÃ‡IK (KapanÄ±ÅŸa {rem_to_36:.1f} dk)</span>"
                )
            else:
                badge_html = (
                    '<span style="background:rgba(225,29,72,0.1); border:1px solid rgba(225,29,72,0.28); '
                    'color:#e11d48; padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    "ğŸ”’ 36+ DK BAREM KAPANDI â€” MaÃ§ Sonucu Bekleniyor</span>"
                )

            pred_tot = m.get("predicted_total", 212.0)
            pred_h = m.get("predicted_home_final", 106.0)
            pred_a = m.get("predicted_away_final", 106.0)
            fav1 = m.get("bookie_fav_1") or "-"
            fav1_odd = m.get("bookie_fav_1_odd", 99.0)
            fav2 = m.get("bookie_fav_2") or "-"
            val_rng = m.get("value_range") or "-"
            clu_name = m.get("best_cluster_name") or "-"
            clu_prob = m.get("best_cluster_prob", 0.0)
            dutch_odd = m.get("dutch_odd", 0.0)
            sym_gap = m.get("symmetry_gap", 99.0)

            q3_sig = m.get("q3_q4_signal") or {}
            q3_sig_label = q3_sig.get("signal", "NÃ¶tr Q4 Beklentisi")
            q4_burst_p = q3_sig.get("burst_prob", 35.0)
            q4_coll_p = q3_sig.get("collapse_prob", 25.0)

            match_wave = analyze_sequential_barem_wave(df_hist, df_log, current_fav_barem=fav1)
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
                score_gap=abs(int(h_sc) - int(a_sc)),
            )
            smart_color = {"TRAP_LOW": "#e11d48", "GOLDEN": "#059669"}.get(smart["zone_type"], "var(--text-secondary)")
            smart_html = (
                f'<div class="info-strip" style="grid-column: 1 / -1; border-left:4px solid {smart_color};">'
                f'<div class="info-strip-label">ğŸ¤– 35. Dk AkÄ±llÄ± Karar & Tuzak Kutusu</div>'
                f'<div class="info-strip-val" style="font-size:0.84rem;">'
                f'<b style="color:{smart_color};">{smart["zone_badge"]}</b> &nbsp;&bull;&nbsp; {smart["drift_badge"]}<br>'
                f'Ana Tercih: <b>{smart["primary_pick"]}</b> &nbsp;&bull;&nbsp; Koruma: <b>{smart["backup_pick"]}</b> '
                f'&nbsp;(GÃ¼ven: %{smart["confidence_pct"]})'
                f'<div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.15rem;">{smart["reasoning"]}</div>'
                f"</div></div>"
            )

            odds_map = m.get("odds", {})
            probs_map = m.get("barem_probs", {})
            boxes_html = ""
            for b_name, _, _ in BAREM_RANGES:
                odd_v = float(odds_map.get(b_name, 99.0))
                prob_v = float(probs_map.get(b_name, 0.0))
                odd_disp = f"{odd_v:.2f}" if odd_v < 90.0 else "ğŸ”’"

                cls_list = ["barem-box"]
                tag_html = ""
                if b_name == fav1 and odd_v < 90.0:
                    cls_list.append("fav1")
                    tag_html = '<div class="b-tag" style="background:var(--fav1-border); color:#ffffff;">â­ 1. FAVORÄ°</div>'
                elif b_name == fav2 and odd_v < 90.0:
                    cls_list.append("fav2")
                    tag_html = '<div class="b-tag" style="background:var(--fav2-border); color:#ffffff;">ğŸ¥ˆ 2. FAVORÄ°</div>'

                if b_name == val_rng and odd_v < 90.0 and b_name != fav1:
                    cls_list.append("val-pick")
                    tag_html = '<div class="b-tag" style="background:var(--val-border); color:#ffffff;">ğŸ”¥ VALUE</div>'
                elif b_name == val_rng and b_name == fav1 and odd_v < 90.0:
                    tag_html = '<div class="b-tag" style="background:var(--fav1-border); color:#ffffff;">â­ FAV + VALUE</div>'

                boxes_html += (
                    f'<div class="{" ".join(cls_list)}">'
                    f'<div class="b-name">{b_name}</div>'
                    f'<div class="b-odd">{odd_disp}</div>'
                    f'<div class="b-prob">Model: %{prob_v:.1f}</div>'
                    f"{tag_html}"
                    f"</div>"
                )

            sym_text = (
                f"{fav1} & {fav2} (Fark: {sym_gap:.2f} | Ã‡ifte Åans: {dutch_odd:.2f})"
                if sym_gap < 2.0
                else f"Favori Barem: {fav1} ({fav1_odd if fav1_odd < 90 else '-'})"
            )

            st.markdown(
                f"""
            <div class="match-card">
                <div class="match-top">
                    <div>
                        <div class="teams-title">ğŸ€ {t_home} &nbsp;<span style="color:var(--text-muted); font-weight:500;">vs</span>&nbsp; {t_away}</div>
                        <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.22rem; font-weight:500;">
                            â±ï¸ <b>Q{q_num} â€” {el_m:.1f}. Dakika</b> (Ã‡eyrekte kalan: {rem_q:.1f} dk) &nbsp;&bull;&nbsp; {quarters_summary} &nbsp;&bull;&nbsp; ğŸ“š <b>GeÃ§miÅŸ Veri: {h2h_n} MaÃ§</b>
                        </div>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;">
                        {badge_html}
                        <div class="score-badge">{h_sc} - {a_sc} <span style="font-size:0.85rem; opacity:0.8;">({tot_sc})</span></div>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.65rem; margin-bottom:0.4rem;">
                    <div class="info-strip">
                        <div class="info-strip-label">ğŸ¯ Monte Carlo MaÃ§ Sonu Tahmini</div>
                        <div class="info-strip-val">
                            {pred_h:.0f} - {pred_a:.0f} &nbsp;<span style="color:var(--score-text);">(Toplam: {pred_tot:.1f} SayÄ±)</span>
                        </div>
                    </div>
                    <div class="info-strip">
                        <div class="info-strip-label">âš–ï¸ Barem Simetri & Ã‡ifte Åans</div>
                        <div class="info-strip-val" style="color:var(--fav1-text);">
                            {sym_text}
                        </div>
                    </div>
                    <div class="info-strip">
                        <div class="info-strip-label">ğŸ›¡ï¸ En GÃ¼Ã§lÃ¼ 20-SayÄ± KÃ¼mesi & Value</div>
                        <div class="info-strip-val" style="color:var(--val-text);">
                            KÃ¼me: {clu_name} (%{clu_prob:.1f}) &nbsp;&bull;&nbsp; Value: {val_rng}
                        </div>
                    </div>
                    <div class="info-strip">
                        <div class="info-strip-label">âš¡ Q3 â” Q4 Patlama / Ã‡Ã¶kÃ¼ÅŸ DedektÃ¶rÃ¼</div>
                        <div class="info-strip-val">
                            {q3_sig_label} <span style="font-size:0.76rem; color:var(--text-muted);">(Patlama %{q4_burst_p:.0f} / Ã‡Ã¶kÃ¼ÅŸ %{q4_coll_p:.0f})</span>
                        </div>
                    </div>
                    <div class="info-strip">
                        <div class="info-strip-label">ğŸŒŠ ArdÄ±ÅŸÄ±k MaÃ§ Dalga Hedefi (-2 / -1 / 0 / +1 / +2)</div>
                        <div class="info-strip-val" style="color:var(--score-text); font-size:0.84rem;">
                            -2: {w_bm2} &bull; -1: {w_bm1} &bull; <b>0: {w_b0}</b> &bull; +1: {w_bp1} &bull; +2: {w_bp2}
                            <div style="font-size:0.74rem; color:var(--fav1-text); margin-top:0.1rem;">Ã–nerilen Dalga Ä°kilisi: {w_rec_pair} (%{w_rec_prob})</div>
                        </div>
                    </div>
                    {smart_html}
                </div>
                <div class="barem-grid">
                    {boxes_html}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # SECTION 2: MATCH-ISOLATED BAREM FLOW & CLOSING SYMMETRY TABLES (0 - 36m)
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown(
        '<div class="section-hdr">ğŸ“Š MaÃ§ BazlÄ± AyrÄ±ÅŸtÄ±rÄ±lmÄ±ÅŸ Barem & Simetri AkÄ±ÅŸ Tablosu (0 - 36. Dakika)</div>',
        unsafe_allow_html=True,
    )

    if df_log.empty:
        st.info(
            "HenÃ¼z kaydedilmiÅŸ barem akÄ±ÅŸ verisi bulunmuyor. CanlÄ± bir maÃ§ta barem aÃ§Ä±kken (0-36. dk) her maÃ§ kendi baÄŸÄ±msÄ±z kutusunda buraya eklenecektir."
        )
    else:
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.15, 1.35, 1.25, 1.25])
        with f_col1:
            status_filter = st.selectbox(
                "Durum Filtresi",
                ["TÃ¼mÃ¼", "â³ CanlÄ± / Bekleyen (PENDING)", "âœ… SonuÃ§lanan (COMPLETED)"],
                index=0,
            )
        with f_col2:
            unique_games = ["TÃ¼m MaÃ§lar (Her MaÃ§ AyrÄ± Tabloda)"] + list(df_log["game_name"].dropna().unique())[::-1]
            game_filter = st.selectbox("MaÃ§ Filtresi", unique_games, index=0)
        with f_col3:
            min_filter = st.selectbox(
                "Dakika AralÄ±ÄŸÄ±",
                ["TÃ¼m Dakikalar (0-36. dk)", "ğŸ”¥ KapanÄ±ÅŸ BÃ¶lgesi (30-36. dk)", "âš¡ Erken/Orta Periyot (0-29. dk)"],
                index=0,
            )
        with f_col4:
            layout_mode = st.selectbox(
                "GÃ¶rÃ¼nÃ¼m Modu",
                ["ğŸ—‚ï¸ Her MaÃ§ AyrÄ± Kutuda (Alt Alta)", "ğŸ“‘ Her MaÃ§ AyrÄ± Sekmede (Tab)"],
                index=1,
            )

        df_view = df_log.copy()
        if "PENDING" in status_filter:
            df_view = df_view[df_view["status"] == "PENDING"]
        elif "COMPLETED" in status_filter:
            df_view = df_view[df_view["status"] == "COMPLETED"]

        if not game_filter.startswith("TÃ¼m MaÃ§lar"):
            df_view = df_view[df_view["game_name"] == game_filter]

        if "30-36" in min_filter:
            df_view = df_view[
                df_view["check_time"]
                .astype(str)
                .str.extract(r"Minute\s+(\d+)", expand=False)
                .astype(float)
                .fillna(0)
                >= 30
            ]
        elif "0-29" in min_filter:
            df_view = df_view[
                df_view["check_time"]
                .astype(str)
                .str.extract(r"Minute\s+(\d+)", expand=False)
                .astype(float)
                .fillna(0)
                < 30
            ]

        if df_view.empty:
            st.info("SeÃ§ilen filtreye uygun barem kaydÄ± bulunamadÄ±.")
        else:
            # Group strictly by game_id so simultaneous games NEVER mix in a single table!
            # Order games so PENDING (active live) games appear first, then most recent games.
            game_ids_ordered = []
            for gid in df_view["game_id"].dropna().unique()[::-1]:
                if gid not in game_ids_ordered:
                    game_ids_ordered.append(gid)

            pending_gids = [
                gid for gid in game_ids_ordered if (df_view[df_view["game_id"] == gid]["status"] == "PENDING").any()
            ]
            completed_gids = [gid for gid in game_ids_ordered if gid not in pending_gids]
            sorted_gids = pending_gids + completed_gids

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
                            hits.append("â­ FAV TUTTU")
                        if act_rng == val_rng:
                            hits.append("ğŸ”¥ VALUE TUTTU")
                        if check_cluster_hit(clu_rng, act_rng):
                            hits.append("ğŸ›¡ï¸ KÃœME TUTTU")
                        outcome_badge = " | ".join(hits) if hits else f"âŒ ({act_rng})"
                    else:
                        outcome_badge = "â³ Bekliyor"

                    def _fmt_cell(odd_col, prob_col):
                        o_v = float(r.get(odd_col, 99.0) or 99.0)
                        p_v = float(r.get(prob_col, 0.0) or 0.0)
                        if o_v >= 90.0:
                            return f"- (%{p_v:.0f})"
                        return f"{o_v:.2f} (%{p_v:.0f})"

                    rows.append(
                        {
                            "Dakika": r.get("check_time", ""),
                            "AnlÄ±k Skor": r.get("current_score", ""),
                            "160-": _fmt_cell("odds_160_or_less", "model_prob_160_or_less"),
                            "161-170": _fmt_cell("odds_161_170", "model_prob_161_170"),
                            "171-180": _fmt_cell("odds_171_180", "model_prob_171_180"),
                            "181-190": _fmt_cell("odds_181_190", "model_prob_181_190"),
                            "191-200": _fmt_cell("odds_191_200", "model_prob_191_200"),
                            "201-210": _fmt_cell("odds_201_210", "model_prob_201_210"),
                            "211-220": _fmt_cell("odds_211_220", "model_prob_211_220"),
                            "221-230": _fmt_cell("odds_221_230", "model_prob_221_230"),
                            "231+": _fmt_cell("odds_231_plus", "model_prob_231_plus"),
                            "â­ 1xBet Fav": fav_rng,
                            "ğŸ”¥ Value Barem": val_rng,
                            "ğŸ›¡ï¸ 20-SayÄ± KÃ¼mesi": f"{clu_rng} (%{float(clu_prb or 0):.0f})",
                            "MaÃ§ Sonu": f"{act_sc} ({act_rng})" if act_sc else "-",
                            "SonuÃ§ Analizi": outcome_badge,
                        }
                    )
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

                first_fav = str(df_m.iloc[0]["â­ 1xBet Fav"])
                last_fav = str(df_m.iloc[-1]["â­ 1xBet Fav"])
                last_clu = str(df_m.iloc[-1]["ğŸ›¡ï¸ 20-SayÄ± KÃ¼mesi"])
                last_min = str(df_m.iloc[-1]["Dakika"])

                if g_status == "COMPLETED" and act_rng:
                    status_pill_html = (
                        '<span style="background:var(--pill-bg); border:1px solid var(--pill-border); '
                        'color:var(--pill-text); padding:0.24rem 0.7rem; border-radius:8px; font-size:0.75rem; font-weight:700;">'
                        f"âœ… SONUÃ‡LANDI: {act_sc} &bull; Kazanan Barem: {act_rng}</span>"
                    )
                else:
                    status_pill_html = (
                        '<span style="background:var(--val-bg); border:1px solid var(--val-border); '
                        'color:var(--val-text); padding:0.24rem 0.7rem; border-radius:8px; font-size:0.75rem; font-weight:700;">'
                        f"â³ CANLI / SONUÃ‡ BEKLENÄ°YOR (Son KayÄ±t: {last_min})</span>"
                    )

                st.markdown(
                    f"""
                    <div style="background:var(--bg-card); border:1px solid var(--border-subtle); border-left:4px solid var(--score-text); border-radius:10px; padding:0.75rem 1.1rem; margin-top:0.65rem; margin-bottom:0.35rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem; box-shadow:var(--shadow-card);">
                        <div>
                            <span style="font-size:1.0rem; font-weight:800; color:var(--text-primary);">ğŸ€ {g_name}</span>
                            <span style="font-size:0.76rem; color:var(--text-muted); margin-left:0.6rem; font-family:'JetBrains Mono',monospace;">[{gid}] &bull; {len(df_m)} Dakika KaydÄ±</span>
                            <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:0.2rem;">
                                ğŸ“ˆ <b>Favori Barem AkÄ±ÅŸÄ±:</b> {first_fav} â” <b>{last_fav}</b> &nbsp;&bull;&nbsp; ğŸ›¡ï¸ <b>Son KÃ¼me:</b> {last_clu}
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

            if "Sekmede" in layout_mode and len(sorted_gids) > 1:
                tab_labels = []
                for gid in sorted_gids[:12]:
                    g_sub = df_view[df_view["game_id"] == gid]
                    g_name = str(g_sub.iloc[-1].get("game_name", gid)).replace(" (cyber)", "")
                    g_st = "â³" if (g_sub["status"] == "PENDING").any() else "âœ…"
                    tab_labels.append(f"{g_st} {g_name} ({len(g_sub)})")
                m_tabs = st.tabs(tab_labels)
                for idx_t, gid in enumerate(sorted_gids[:12]):
                    with m_tabs[idx_t]:
                        _render_match_flow_card(gid)
            else:
                for gid in sorted_gids:
                    _render_match_flow_card(gid)


render_live_dashboard()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 3: ADVANCED ANALYTICS & HISTORICAL ARCHIVE (ALL 4 MODULES)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown(
    '<div class="section-hdr">ğŸ§  GeliÅŸmiÅŸ Ä°statistik, GeÃ§miÅŸ TÃ¼m MaÃ§lar ArÅŸivi & Kasa/Dutching Merkezi</div>',
    unsafe_allow_html=True,
)

df_completed_archive = load_completed_matches()
df_sym_all = load_symmetry_log()
live_state_now = load_live_state()
live_matches_now = live_state_now.get("live_matches", [])

tab_dna, tab_sym, tab_q3q4, tab_bankroll, tab_sim, tab_smart35 = st.tabs(
    [
        f"ğŸ† 1. GeÃ§miÅŸ MaÃ§lar & TakÄ±m Barem DNA ({len(df_completed_archive)} MaÃ§)",
        "âš–ï¸ 2. KapanÄ±ÅŸ OranÄ± (30-36. Dk) Simetri & Tuzak Analizi",
        "âš¡ 3. Q3 â” Q4 Patlama / Ã‡Ã¶kÃ¼ÅŸ DedektÃ¶rÃ¼",
        "ğŸ’° 4. CanlÄ± Kupon, Kasa (ROI) & Ã‡ifte Barem (Dutching) HesaplayÄ±cÄ±",
        "ğŸ§® 5. Manuel CanlÄ± Tempo SimÃ¼latÃ¶rÃ¼",
        "ğŸ¤– 6. 35. Dk AkÄ±llÄ± Kural & Sanal Kasa SimÃ¼latÃ¶rÃ¼",
    ]
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODULE 6: 35th-MINUTE SMART RULES, TRANSITIONS & VIRTUAL BANKROLL SIMULATOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_smart35:
    st.markdown("##### ğŸ¤– 35. Dakika AkÄ±llÄ± Kural Motoru â€” GerÃ§ek KapanÄ±ÅŸ OranlÄ± MaÃ§lardan HesaplanÄ±r")
    sim35 = simulate_35m_strategies(df_sym_all)
    if sim35["total_matches"] == 0:
        st.info("SonuÃ§lanmÄ±ÅŸ canlÄ± kapanÄ±ÅŸ kaydÄ± bekleniyor. MaÃ§lar bittikÃ§e bu sekme otomatik dolar.")
    else:
        st.caption(
            f"ğŸ“Š Analiz edilen gerÃ§ek maÃ§ sayÄ±sÄ±: **{sim35['total_matches']}** â€¢ Her maÃ§a sabit **1000 TL** bahis â€¢ "
            "Oranlar 35. dakikadaki gerÃ§ek 1xBet kapanÄ±ÅŸ oranlarÄ±dÄ±r."
        )
        _n_games = df_sym_all["game_id"].nunique() if not df_sym_all.empty else 0
        _n_done = (
            df_sym_all[df_sym_all["status"] == "COMPLETED"]["game_id"].nunique()
            if (not df_sym_all.empty and "status" in df_sym_all.columns)
            else 0
        )
        st.caption(f"ğŸ” Veri kaynaÄŸÄ±: {len(df_sym_all)} dakika kaydÄ± â€¢ {_n_games} maÃ§ â€¢ {_n_done} sonuÃ§lanmÄ±ÅŸ â€¢ sÃ¼rÃ¼m {ENGINE_VERSION}")
        st.markdown("###### ğŸ’° Sanal Kasa: Hangi Strateji KazandÄ±rÄ±rdÄ±?")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Strateji": s["name"],
                        "Ä°sabet": f"{s['wins']} / {s['total']} (%{s['win_rate']})",
                        "Toplam YatÄ±rÄ±m": f"{s['cost']:.0f} TL",
                        "Toplam DÃ¶nÃ¼ÅŸ": f"{s['return']:.0f} TL",
                        "Net KÃ¢r / Zarar": f"{s['net_profit']:+.0f} TL",
                        "ROI": f"%{s['roi_pct']:+.1f}",
                    }
                    for s in sim35["strategies"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if sim35["odds_brackets"]:
            st.markdown("###### ğŸšï¸ 35. Dk Favori Oran BandÄ±na GÃ¶re DavranÄ±ÅŸ")
            st.dataframe(pd.DataFrame(sim35["odds_brackets"]), use_container_width=True, hide_index=True)
        if sim35["drift_brackets"]:
            st.markdown("###### ğŸ“‰ MaÃ§ Ä°Ã§i Barem GÃ¶Ã§Ã¼ (Q1 â” 35. Dk) DavranÄ±ÅŸÄ±")
            st.dataframe(pd.DataFrame(sim35["drift_brackets"]), use_container_width=True, hide_index=True)
        if sim35["transitions"]:
            st.markdown("###### ğŸŒŠ ArdÄ±ÅŸÄ±k MaÃ§ GeÃ§iÅŸ Matrisi (Ã–nceki MaÃ§ â” SÄ±radaki MaÃ§)")
            st.dataframe(pd.DataFrame(sim35["transitions"]), use_container_width=True, hide_index=True)
        st.markdown("###### ğŸ“’ MaÃ§ MaÃ§ Sanal Kupon Defteri")
        st.dataframe(pd.DataFrame(sim35["ledger"]), use_container_width=True, hide_index=True, height=420)
        st.warning(
            "âš ï¸ Ã–rneklem kÃ¼Ã§Ã¼ktÃ¼r (onlarca maÃ§). YÃ¼zdeler maÃ§ sayÄ±sÄ± arttÄ±kÃ§a deÄŸiÅŸir; kesin kazanÃ§ garantisi deÄŸildir. "
            "Sistem istatistiksel eÄŸilim gÃ¶sterir."
        )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODULE 1: HISTORICAL MATCHES CENTER, TEAM BAREM DNA & H2H ANALYZER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_dna:
    if df_completed_archive.empty:
        st.info(
            "Åu an geÃ§miÅŸ maÃ§ arÅŸivi boÅŸ. YukarÄ±daki **'ğŸ“¥ Son 4 Ay ArÅŸivini GÃ¼ncelle (120 GÃ¼n)'** butonuna basarak "
            "son 4 ayÄ±n tamamlanmÄ±ÅŸ tÃ¼m CyberLeague maÃ§larÄ±nÄ± saniyeler iÃ§inde yÃ¼kleyebilirsiniz."
        )
    else:
        df_arc = df_completed_archive.copy()
        df_arc["Kazanan Barem"] = df_arc["total_score"].apply(get_range_from_score)
        df_arc["20'li KÃ¼me"] = df_arc["total_score"].apply(get_cluster_from_score)
        df_arc["Q1 Toplam"] = df_arc["home_q1"].astype(int) + df_arc["away_q1"].astype(int)
        df_arc["Q2 Toplam"] = df_arc["home_q2"].astype(int) + df_arc["away_q2"].astype(int)
        df_arc["Q3 Toplam"] = df_arc["home_q3"].astype(int) + df_arc["away_q3"].astype(int)
        df_arc["Q4 Toplam"] = df_arc["home_q4"].astype(int) + df_arc["away_q4"].astype(int)

        all_teams = sorted(
            set(df_arc["home_team"].dropna().unique()).union(set(df_arc["away_team"].dropna().unique()))
        )

        flt1, flt2, flt3 = st.columns([1.3, 1.3, 1.1])
        with flt1:
            sel_t1 = st.selectbox("1. TakÄ±m Filtresi (H2H veya Tek TakÄ±m)", ["TÃ¼m TakÄ±mlar"] + all_teams, index=0)
        with flt2:
            sel_t2 = st.selectbox("2. TakÄ±m Filtresi (Rakip / H2H)", ["TÃ¼m TakÄ±mlar"] + all_teams, index=0)
        with flt3:
            sel_barem_flt = st.selectbox(
                "Barem Filtresi",
                ["TÃ¼m Baremler"] + [b[0] for b in BAREM_RANGES],
                index=0,
            )

        df_filtered = df_arc.copy()
        if sel_t1 != "TÃ¼m TakÄ±mlar" and sel_t2 != "TÃ¼m TakÄ±mlar":
            df_filtered = df_filtered[
                ((df_filtered["home_team"] == sel_t1) & (df_filtered["away_team"] == sel_t2))
                | ((df_filtered["home_team"] == sel_t2) & (df_filtered["away_team"] == sel_t1))
            ]
        elif sel_t1 != "TÃ¼m TakÄ±mlar":
            df_filtered = df_filtered[
                (df_filtered["home_team"] == sel_t1) | (df_filtered["away_team"] == sel_t1)
            ]
        elif sel_t2 != "TÃ¼m TakÄ±mlar":
            df_filtered = df_filtered[
                (df_filtered["home_team"] == sel_t2) | (df_filtered["away_team"] == sel_t2)
            ]

        if sel_barem_flt != "TÃ¼m Baremler":
            df_filtered = df_filtered[df_filtered["Kazanan Barem"] == sel_barem_flt]

        # Summary metrics for filtered historical matches
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        n_flt = len(df_filtered)
        avg_tot = df_filtered["total_score"].mean() if n_flt > 0 else 0.0
        modal_b = (
            df_filtered["Kazanan Barem"].mode().iloc[0]
            if n_flt > 0 and not df_filtered["Kazanan Barem"].mode().empty
            else "-"
        )
        modal_c = (
            df_filtered["20'li KÃ¼me"].mode().iloc[0]
            if n_flt > 0 and not df_filtered["20'li KÃ¼me"].mode().empty
            else "-"
        )
        m_c1.metric("ğŸ“š Filtrelenen MaÃ§ SayÄ±sÄ±", f"{n_flt} MaÃ§")
        m_c2.metric("ğŸ€ Ortalama Toplam SayÄ±", f"{avg_tot:.1f} SayÄ±")
        m_c3.metric("ğŸ¯ En Ã‡ok Ã‡Ä±kan 10'lu Barem", modal_b)
        m_c4.metric("ğŸ›¡ï¸ En Ã‡ok Ã‡Ä±kan 20'li KÃ¼me", modal_c)

        ch_col, dna_col = st.columns([1.05, 1.35])
        with ch_col:
            st.markdown("##### ğŸ“Š GeÃ§miÅŸ MaÃ§larÄ±n 9'lu Barem DaÄŸÄ±lÄ±m GrafiÄŸi")
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
                yaxis=dict(title="MaÃ§ Adedi", range=[0, max(b_counts + [5]) * 1.25]),
            )
            st.plotly_chart(fig_hist_b, use_container_width=True)

        with dna_col:
            st.markdown("##### ğŸ§¬ TakÄ±m BazlÄ± Barem DNA Tablosu (TÃ¼m CyberLeague TakÄ±mlarÄ±)")
            df_dna_table = compute_team_dna_table(df_arc)
            st.dataframe(df_dna_table, use_container_width=True, hide_index=True, height=310)

        st.markdown("##### ğŸ“œ TamamlanmÄ±ÅŸ TÃ¼m MaÃ§larÄ±n DetaylÄ± Barem & Periyot ArÅŸivi")
        show_cols = [
            "date",
            "home_team",
            "away_team",
            "home_total",
            "away_total",
            "total_score",
            "Kazanan Barem",
            "20'li KÃ¼me",
            "Q1 Toplam",
            "Q2 Toplam",
            "Q3 Toplam",
            "Q4 Toplam",
        ]
        avail_cols = [c for c in show_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[avail_cols].iloc[::-1], use_container_width=True, hide_index=True, height=340)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODULE 2: CLOSING ODDS (30-36 MIN) SYMMETRY & SEQUENTIAL WAVE BACKTEST
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_sym:
    st.markdown(
        "##### ğŸŒŠ ArdÄ±ÅŸÄ±k MaÃ§larda 36. Dakika Simetri Dalga Analizi (`-2 Alt` / `-1 Alt` / `0 Beklenen` / `+1 Ãœst` / `+2 Ãœst`)"
    )
    st.caption(
        "SezdiÄŸiniz ardÄ±ÅŸÄ±k maÃ§ dalga Ã¶rÃ¼ntÃ¼sÃ¼nÃ¼ (**-2 Alt Barem, -1 Alt Barem, 0 Beklenen Merkez, +1 Ãœst Barem, +2 Ãœst Barem**) "
        "4.504 gerÃ§ek maÃ§ Ã¼zerinde test eder ve sÄ±radaki maÃ§Ä±n hangi bareme kayacaÄŸÄ±nÄ± hesaplar."
    )

    w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns([1, 1, 1, 1, 1.2])
    seq_map = {
        "â„ï¸ -2 (2 Alt Barem Geldi)": -2,
        "ğŸ“‰ -1 (1 Alt Barem Geldi)": -1,
        "ğŸ¯ 0 (36.Dk Beklenen Geldi)": 0,
        "ğŸ“ˆ +1 (1 Ãœst Barem Geldi)": 1,
        "ğŸ”¥ +2 (2 Ãœst Barem Geldi)": 2,
    }
    rev_map = {v: k for k, v in seq_map.items()}

    default_wave = analyze_sequential_barem_wave(df_completed_archive, df_sym_all)
    act_pat = default_wave.get("active_pattern", [0, -1, -1, 0])
    while len(act_pat) < 4:
        act_pat = [0] + act_pat

    opts_list = list(seq_map.keys())
    with w_col1:
        s_m1 = st.selectbox("1. Ã–nceki MaÃ§", opts_list, index=opts_list.index(rev_map.get(act_pat[0], opts_list[2])))
    with w_col2:
        s_m2 = st.selectbox("2. Ã–nceki MaÃ§", opts_list, index=opts_list.index(rev_map.get(act_pat[1], opts_list[1])))
    with w_col3:
        s_m3 = st.selectbox("3. Ã–nceki MaÃ§", opts_list, index=opts_list.index(rev_map.get(act_pat[2], opts_list[1])))
    with w_col4:
        s_m4 = st.selectbox("4. (Son BitmiÅŸ) MaÃ§", opts_list, index=opts_list.index(rev_map.get(act_pat[3], opts_list[2])))
    with w_col5:
        test_fav_b = st.selectbox(
            "CanlÄ± MaÃ§Ä±n 36.Dk Beklenen Baremi",
            [b[0] for b in BAREM_RANGES],
            index=5,
        )

    custom_pattern = [seq_map[s_m1], seq_map[s_m2], seq_map[s_m3], seq_map[s_m4]]
    sim_wave = analyze_sequential_barem_wave(
        df_completed_archive, df_sym_all, current_fav_barem=test_fav_b, custom_seq=custom_pattern
    )

    wk_m2, wk_m1, wk_0, wk_p1, wk_p2, wk_pair = st.columns(6)
    wk_m2.metric(
        f"â„ï¸ -2 Alt ({sim_wave['target_minus_2_barem']})",
        f"%{sim_wave['prob_minus_2']}",
        "2 Alt Barem KaymasÄ±",
    )
    wk_m1.metric(
        f"ğŸ“‰ -1 Alt ({sim_wave['target_minus_1_barem']})",
        f"%{sim_wave['prob_minus_1']}",
        f"Toplam Alt: %{sim_wave['prob_down_total']}",
    )
    wk_0.metric(
        f"ğŸ¯ 0 Beklenen ({sim_wave['target_0_barem']})",
        f"%{sim_wave['prob_0']}",
        f"{sim_wave['pattern_matches_count']} Benzer Zincir",
    )
    wk_p1.metric(
        f"ğŸ“ˆ +1 Ãœst ({sim_wave['target_plus_1_barem']})",
        f"%{sim_wave['prob_plus_1']}",
        f"Toplam Ãœst: %{sim_wave['prob_up_total']}",
    )
    wk_p2.metric(
        f"ğŸ”¥ +2 Ãœst ({sim_wave['target_plus_2_barem']})",
        f"%{sim_wave['prob_plus_2']}",
        "2 Ãœst Barem KaymasÄ±",
    )
    wk_pair.metric(
        "ğŸ›¡ï¸ Ã–nerilen Ä°kili KÃ¼me",
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
        st.markdown("###### ğŸ“œ CanlÄ± Takip Edilip Tamamlanan MaÃ§larÄ±n `36. Dakika Beklenen vs Biten Barem (-2 / -1 / 0 / +1 / +2)` Zinciri")
        rc_rows = []
        for item in default_wave["recent_chain"][::-1]:
            eb = item["exp_barem_36"]
            rc_rows.append(
                {
                    "Tarih / Saat": item["date"],
                    "MaÃ§": f"{item['home_team']} vs {item['away_team']}",
                    "36. Dk Skor": item["s36"],
                    "MaÃ§ Sonu Toplam": f"{item['home_total']}:{item['away_total']} ({item['total_score']})",
                    "â„ï¸ -2 Alt Barem": _get_shifted_b(eb, -2),
                    "ğŸ“‰ -1 Alt Barem": _get_shifted_b(eb, -1),
                    "ğŸ¯ 0 (36.Dk Beklenen)": eb,
                    "ğŸ“ˆ +1 Ãœst Barem": _get_shifted_b(eb, 1),
                    "ğŸ”¥ +2 Ãœst Barem": _get_shifted_b(eb, 2),
                    "Biten GerÃ§ek Barem": item["act_barem"],
                    "Simetri Dalga SapmasÄ± (-2 / -1 / 0 / +1 / +2)": item["shift_label"],
                    "Referans": item["source"],
                }
            )
        st.dataframe(pd.DataFrame(rc_rows), use_container_width=True, hide_index=True, height=min(320, 60 + 38 * len(rc_rows)))

    st.markdown("---")
    st.markdown(
        "##### âš–ï¸ 1xBet KapanÄ±ÅŸ OranÄ± (30-36. Dk) Simetri TuzaÄŸÄ± & KomÅŸu Bareme (`-2 / -1 / 0 / +1 / +2`) Kayma Analizi"
    )
    df_comp_sym = (
        df_sym_all[df_sym_all["status"] == "COMPLETED"].copy()
        if not df_sym_all.empty and "status" in df_sym_all.columns
        else pd.DataFrame()
    )

    if df_comp_sym.empty:
        st.info(
            "Åu an `symmetry_analysis_log.csv` iÃ§inde sonuÃ§lanmÄ±ÅŸ (COMPLETED) canlÄ± barem kaydÄ± bekleniyor. "
            "CanlÄ± maÃ§lar bittikÃ§e burada **-2 Alt, -1 Alt, 0 Favori, +1 Ãœst, +2 Ãœst Ä°sabet OranlarÄ±**, **Simetri TuzaÄŸÄ± (`Fark <= 0.65`) Kayma OranÄ±** "
            "ve **Dakika Penceresi (`0-24 dk`, `24-30 dk`, `30-36 dk`)** istatistikleri otomatik hesaplanÄ±r."
        )
    else:
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
            el_min = int(m_match.group(1)) if m_match else 18
            if el_min >= 30:
                win_bucket = "30-36. Dk (KapanÄ±ÅŸ)"
            elif el_min >= 24:
                win_bucket = "24-29. Dk (Q3 BaÅŸÄ±/OrtasÄ±)"
            else:
                win_bucket = "0-23. Dk (Ä°lk YarÄ±)"

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
                    "clu_hit": check_cluster_hit(clu_b, act_b),
                }
            )

        df_sr = pd.DataFrame(sym_records)
        if not df_sr.empty:
            # Her maÃ§Ä±n SON (35-36. Dk KapanÄ±ÅŸ) kaydÄ±nÄ± baz alarak MaÃ§ BazlÄ± KapanÄ±ÅŸ Simetrisini hesapla
            df_closing = df_sr.sort_values("el_min").groupby("game_id", as_index=False).last()
            tot_n = len(df_closing)
            m2_pct = df_closing["hit_m2"].mean() * 100.0
            m1_pct = df_closing["hit_m1"].mean() * 100.0
            f0_pct = df_closing["hit_0"].mean() * 100.0
            p1_pct = df_closing["hit_p1"].mean() * 100.0
            p2_pct = df_closing["hit_p2"].mean() * 100.0
            both_pct = df_closing["either_fav_hit"].mean() * 100.0

            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("â„ï¸ -2 Alt Bareme Kayma", f"%{m2_pct:.1f}", f"{int(df_closing['hit_m2'].sum())} / {tot_n} MaÃ§")
            s2.metric("ğŸ“‰ -1 Alt Bareme Kayma", f"%{m1_pct:.1f}", f"{int(df_closing['hit_m1'].sum())} / {tot_n} MaÃ§")
            s3.metric("ğŸ¯ 0 (1. Favori Merkez)", f"%{f0_pct:.1f}", f"{int(df_closing['hit_0'].sum())} / {tot_n} MaÃ§")
            s4.metric("ğŸ“ˆ +1 Ãœst Bareme Kayma", f"%{p1_pct:.1f}", f"{int(df_closing['hit_p1'].sum())} / {tot_n} MaÃ§")
            s5.metric("ğŸ”¥ +2 Ãœst Bareme Kayma", f"%{p2_pct:.1f}", f"{int(df_closing['hit_p2'].sum())} / {tot_n} MaÃ§")
            s6.metric(
                "ğŸ›¡ï¸ 1. veya 2. Favori",
                f"%{both_pct:.1f}",
                f"{int(df_closing['either_fav_hit'].sum())} / {tot_n} MaÃ§",
            )

            win_rows = []
            for w_name in ["0-23. Dk (Ä°lk YarÄ±)", "24-29. Dk (Q3 BaÅŸÄ±/OrtasÄ±)", "30-36. Dk (KapanÄ±ÅŸ)"]:
                w_sub = df_sr[df_sr["window"] == w_name]
                if w_sub.empty:
                    continue
                win_rows.append(
                    {
                        "Dakika Penceresi": w_name,
                        "KayÄ±t SayÄ±sÄ±": len(w_sub),
                        "â„ï¸ -2 Alt Barem (%)": f"%{w_sub['hit_m2'].mean()*100:.1f}",
                        "ğŸ“‰ -1 Alt Barem (%)": f"%{w_sub['hit_m1'].mean()*100:.1f}",
                        "ğŸ¯ 0 (1. Favori) (%)": f"%{w_sub['hit_0'].mean()*100:.1f}",
                        "ğŸ“ˆ +1 Ãœst Barem (%)": f"%{w_sub['hit_p1'].mean()*100:.1f}",
                        "ğŸ”¥ +2 Ãœst Barem (%)": f"%{w_sub['hit_p2'].mean()*100:.1f}",
                        "ğŸ¥ˆ 2. Favori (%)": f"%{w_sub['fav2_hit'].mean()*100:.1f}",
                        "ğŸ›¡ï¸ 20-SayÄ± KÃ¼me (%)": f"%{w_sub['clu_hit'].mean()*100:.1f}",
                    }
                )
            st.dataframe(pd.DataFrame(win_rows), use_container_width=True, hide_index=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODULE 3: Q3 â” Q4 BURST / COLLAPSE DETECTOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_q3q4:
    st.markdown(
        "##### âš¡ 3. Ã‡eyrek (Q3) Temposuna GÃ¶re 4. Ã‡eyrek (Q4) SayÄ± PatlamasÄ± / Ã‡Ã¶kÃ¼ÅŸ DedektÃ¶rÃ¼"
    )
    if df_completed_archive.empty:
        st.info("Q3 â” Q4 analizi iÃ§in geÃ§miÅŸ maÃ§ verisi bekleniyor.")
    else:
        q_teams = ["TÃ¼m TakÄ±mlar"] + sorted(
            set(df_completed_archive["home_team"].dropna().unique()).union(
                set(df_completed_archive["away_team"].dropna().unique())
            )
        )
        qc1, qc2, qc3 = st.columns([1.3, 1.3, 1.2])
        with qc1:
            q_t1 = st.selectbox("Ev Sahibi / 1. TakÄ±m", q_teams, index=0, key="q3_t1")
        with qc2:
            q_t2 = st.selectbox("Deplasman / 2. TakÄ±m", q_teams, index=0, key="q3_t2")
        with qc3:
            q3_test_score = st.slider("3. Ã‡eyrek (Q3) Toplam SayÄ±sÄ±", 35, 72, 56, 1)

        t1_arg = "" if q_t1 == "TÃ¼m TakÄ±mlar" else q_t1
        t2_arg = "" if q_t2 == "TÃ¼m TakÄ±mlar" else q_t2
        sig_res = build_q3_q4_streak_signal(df_completed_archive, t1_arg, t2_arg, q3_score=q3_test_score)

        k_q1, k_q2, k_q3, k_q4 = st.columns(4)
        k_q1.metric("ğŸ”” Tespit Edilen Q4 Sinyali", sig_res["signal"])
        k_q2.metric("ğŸ”¥ Q4 SayÄ± PatlamasÄ± (56+ SayÄ±)", f"%{sig_res['burst_prob']:.1f}")
        k_q3.metric("âš–ï¸ Q4 Normal Tempo (49-55 SayÄ±)", f"%{sig_res['normal_prob']:.1f}")
        k_q4.metric("â„ï¸ Q4 SayÄ± Ã‡Ã¶kÃ¼ÅŸÃ¼ (â‰¤ 48 SayÄ±)", f"%{sig_res['collapse_prob']:.1f}")

        st.info(f"ğŸ“Œ **Analiz Ã–zeti ({sig_res['sample_size']} Benzer MaÃ§):** {sig_res['detail']}")

        # Show actual Q3 vs Q4 scatter/distribution from archive
        df_q_arc = df_completed_archive.copy()
        df_q_arc["Q3_Tot"] = df_q_arc["home_q3"].astype(float) + df_q_arc["away_q3"].astype(float)
        df_q_arc["Q4_Tot"] = df_q_arc["home_q4"].astype(float) + df_q_arc["away_q4"].astype(float)
        if t1_arg:
            df_q_arc = df_q_arc[(df_q_arc["home_team"] == t1_arg) | (df_q_arc["away_team"] == t1_arg)]
        if t2_arg:
            df_q_arc = df_q_arc[(df_q_arc["home_team"] == t2_arg) | (df_q_arc["away_team"] == t2_arg)]

        q_buckets = [
            ("DÃ¼ÅŸÃ¼k Q3 (â‰¤ 48 SayÄ±)", df_q_arc[df_q_arc["Q3_Tot"] <= 48]),
            ("Orta Q3 (49 - 54 SayÄ±)", df_q_arc[(df_q_arc["Q3_Tot"] >= 49) & (df_q_arc["Q3_Tot"] <= 54)]),
            ("YÃ¼ksek Q3 (55+ SayÄ±)", df_q_arc[df_q_arc["Q3_Tot"] >= 55]),
        ]
        q_table_rows = []
        for b_label, sub_q in q_buckets:
            if sub_q.empty:
                continue
            q_table_rows.append(
                {
                    "3. Ã‡eyrek (Q3) Tempo Grubu": b_label,
                    "MaÃ§ SayÄ±sÄ±": len(sub_q),
                    "Q3 OrtalamasÄ±": round(sub_q["Q3_Tot"].mean(), 1),
                    "Takip Eden Q4 OrtalamasÄ±": round(sub_q["Q4_Tot"].mean(), 1),
                    "ğŸ”¥ Q4 Patlama (â‰¥56 SayÄ±) %": f"%{(sub_q['Q4_Tot'] >= 56).mean()*100:.1f}",
                    "â„ï¸ Q4 Ã‡Ã¶kÃ¼ÅŸ (â‰¤48 SayÄ±) %": f"%{(sub_q['Q4_Tot'] <= 48).mean()*100:.1f}",
                    "MaÃ§ Sonu Toplam Ort.": round(sub_q["total_score"].mean(), 1),
                }
            )
        st.dataframe(pd.DataFrame(q_table_rows), use_container_width=True, hide_index=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODULE 4: LIVE BANKROLL / ROI & DUTCHING (DOUBLE BAREM) STAKE CALCULATOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_bankroll:
    d_col1, d_col2 = st.columns([1.15, 1.25])

    with d_col1:
        st.markdown("##### ğŸ›¡ï¸ Ã‡ifte Barem (Dutching) Kasa & Tutar HesaplayÄ±cÄ±")
        st.caption(
            "KomÅŸu iki bareme (20-sayÄ± kÃ¼mesi) toplam kasayÄ± Ã¶yle bÃ¶ler ki, **hangisi gelirse gelsin aynÄ± net kÃ¢rÄ±** elde edersiniz."
        )

        open_live_list = [m for m in live_matches_now if m.get("is_barem_open")]
        preset_options = ["Manuel Oran GiriÅŸi"] + [
            f"{m['team_home']} vs {m['team_away']} ({m.get('bookie_fav_1')} & {m.get('bookie_fav_2')})"
            for m in open_live_list
        ]
        sel_preset = st.selectbox("CanlÄ± MaÃ§tan Otomatik Doldur", preset_options, index=0)

        def_b1_name, def_b2_name = "201-210", "211-220"
        def_o1, def_o2 = 4.50, 4.50
        if sel_preset != "Manuel Oran GiriÅŸi":
            idx_m = preset_options.index(sel_preset) - 1
            if 0 <= idx_m < len(open_live_list):
                m_sel = open_live_list[idx_m]
                def_b1_name = m_sel.get("bookie_fav_1") or "201-210"
                def_b2_name = m_sel.get("bookie_fav_2") or "211-220"
                o_1 = float(m_sel.get("bookie_fav_1_odd", 4.5))
                o_2 = float(m_sel.get("bookie_fav_2_odd", 4.5))
                if o_1 < 90.0:
                    def_o1 = o_1
                if o_2 < 90.0:
                    def_o2 = o_2

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            total_stake = st.number_input("Toplam Kupon TutarÄ± (TL)", min_value=10.0, max_value=100000.0, value=1000.0, step=100.0)
        with dc2:
            odd_1 = st.number_input(f"1. Barem OranÄ± ({def_b1_name})", min_value=1.50, max_value=50.0, value=float(def_o1), step=0.10)
        with dc3:
            odd_2 = st.number_input(f"2. Barem OranÄ± ({def_b2_name})", min_value=1.50, max_value=50.0, value=float(def_o2), step=0.10)

        inv_sum = (1.0 / odd_1) + (1.0 / odd_2)
        eff_dutch_odd = 1.0 / inv_sum if inv_sum > 0 else 0.0
        stake_1 = total_stake * ((1.0 / odd_1) / inv_sum)
        stake_2 = total_stake * ((1.0 / odd_2) / inv_sum)
        gross_return = stake_1 * odd_1
        net_profit = gross_return - total_stake
        roi_pct = (net_profit / total_stake) * 100.0 if total_stake > 0 else 0.0

        st.success(
            f"ğŸ’° **1. Bareme ({def_b1_name}) YatÄ±rÄ±lacak:** `{stake_1:.2f} TL` (Oran: {odd_1:.2f})\n\n"
            f"ğŸ’° **2. Bareme ({def_b2_name}) YatÄ±rÄ±lacak:** `{stake_2:.2f} TL` (Oran: {odd_2:.2f})\n\n"
            f"âœ… **Efektif Ã‡ifte Åans OranÄ±:** `{eff_dutch_odd:.2f}` | "
            f"**Garanti Toplam DÃ¶nÃ¼ÅŸ:** `{gross_return:.2f} TL` | "
            f"**Net KÃ¢r:** `+{net_profit:.2f} TL` (**+%{roi_pct:.1f} ROI**)"
        )

    with d_col2:
        st.markdown("##### ğŸ“ˆ Strateji BazlÄ± GerÃ§ek Kasa (Bankroll) & ROI PerformansÄ±")
        st.caption(
            "SonuÃ§lanan (`COMPLETED`) her canlÄ± barem kaydÄ±na sabit **1.000 TL** yatÄ±rÄ±ldÄ±ÄŸÄ±nda 3 ana stratejinin gerÃ§ek net kÃ¢r/zarar tablosu:"
        )
        df_comp_roi = (
            df_sym_all[df_sym_all["status"] == "COMPLETED"].copy()
            if not df_sym_all.empty and "status" in df_sym_all.columns
            else pd.DataFrame()
        )
        if df_comp_roi.empty:
            st.info(
                "HenÃ¼z sonuÃ§lanmÄ±ÅŸ (`COMPLETED`) canlÄ± barem kaydÄ± yok. Takip edilen canlÄ± maÃ§lar sonuÃ§landÄ±kÃ§a "
                "burada **1xBet Favori**, **Model Value** ve **20-SayÄ± KÃ¼mesi Dutching** stratejilerinin net TL kÃ¢rÄ± ve ROI yÃ¼zdesi listelenir."
            )
        else:
            unit_bet = 1000.0
            fav_pnl = 0.0
            val_pnl = 0.0
            clu_pnl = 0.0
            n_bets = len(df_comp_roi)

            for _, r in df_comp_roi.iterrows():
                act_b = str(r.get("actual_range", "")).strip()
                fav_b = str(r.get("highest_prob_range", "")).strip()
                val_b = str(r.get("value_range", "")).strip()
                clu_b = str(r.get("best_cluster_name", "")).strip()

                fav_odd = float(r.get(BAREM_TO_ODD_COL.get(fav_b, ""), 4.3) or 4.3)
                if fav_odd >= 90.0:
                    fav_odd = 4.3
                val_odd = float(r.get(BAREM_TO_ODD_COL.get(val_b, ""), 4.8) or 4.8)
                if val_odd >= 90.0:
                    val_odd = 4.8

                # 1. Fav Strategy
                if act_b == fav_b:
                    fav_pnl += unit_bet * (fav_odd - 1.0)
                else:
                    fav_pnl -= unit_bet

                # 2. Value Strategy
                if act_b == val_b:
                    val_pnl += unit_bet * (val_odd - 1.0)
                else:
                    val_pnl -= unit_bet

                # 3. 20-Point Cluster Dutching Strategy
                c_members = []
                for c_name, _, _, mems in CLUSTER_RANGES:
                    if c_name == clu_b:
                        c_members = list(mems)
                        break
                if len(c_members) == 2:
                    o1 = float(r.get(BAREM_TO_ODD_COL.get(c_members[0], ""), 4.5) or 4.5)
                    o2 = float(r.get(BAREM_TO_ODD_COL.get(c_members[1], ""), 4.5) or 4.5)
                    if o1 >= 90.0:
                        o1 = 4.5
                    if o2 >= 90.0:
                        o2 = 4.5
                    eff_o = 1.0 / ((1.0 / o1) + (1.0 / o2))
                else:
                    eff_o = 2.25

                if check_cluster_hit(clu_b, act_b):
                    clu_pnl += unit_bet * (eff_o - 1.0)
                else:
                    clu_pnl -= unit_bet

            tot_wagered = n_bets * unit_bet
            roi_rows = [
                {
                    "Strateji": "â­ 1xBet 1. Favori Tekli Barem",
                    "Kupon SayÄ±sÄ±": n_bets,
                    "Toplam YatÄ±rÄ±m": f"{tot_wagered:,.0f} TL",
                    "Net KÃ¢r / Zarar": f"{fav_pnl:+,.0f} TL",
                    "ROI (%)": f"%{(fav_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Strateji": "ğŸ”¥ Model Value (Pozitif EV) Tekli Barem",
                    "Kupon SayÄ±sÄ±": n_bets,
                    "Toplam YatÄ±rÄ±m": f"{tot_wagered:,.0f} TL",
                    "Net KÃ¢r / Zarar": f"{val_pnl:+,.0f} TL",
                    "ROI (%)": f"%{(val_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Strateji": "ğŸ›¡ï¸ 20-SayÄ± KÃ¼mesi (Ã‡ifte Barem Dutching)",
                    "Kupon SayÄ±sÄ±": n_bets,
                    "Toplam YatÄ±rÄ±m": f"{tot_wagered:,.0f} TL",
                    "Net KÃ¢r / Zarar": f"{clu_pnl:+,.0f} TL",
                    "ROI (%)": f"%{(clu_pnl / tot_wagered)*100:+.1f}",
                },
            ]
            st.dataframe(pd.DataFrame(roi_rows), use_container_width=True, hide_index=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODULE 5: MANUAL LIVE TEMPO & MONTE CARLO SIMULATOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_sim:
    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        sim_q = st.selectbox("Aktif Ã‡eyrek", [1, 2, 3, 4], index=2)
        sim_rem = st.slider("Ã‡eyrekte Kalan Dakika", 0.0, 12.0, 4.5, 0.5)
    with s_c2:
        sim_hq1 = st.number_input("Ev Q1", 0, 60, 26)
        sim_hq2 = st.number_input("Ev Q2", 0, 60, 27)
        sim_hq3 = st.number_input("Ev Q3 (veya CanlÄ±)", 0, 60, 18)
    with s_c3:
        sim_aq1 = st.number_input("Dep Q1", 0, 60, 25)
        sim_aq2 = st.number_input("Dep Q2", 0, 60, 26)
        sim_aq3 = st.number_input("Dep Q3 (veya CanlÄ±)", 0, 60, 19)

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
        h_live, a_live = 15, 15

    sim_res = predict_live_inplay(
        df=df_completed_archive,
        team_home="",
        team_away="",
        current_quarter=sim_q,
        home_live_score=h_live,
        away_live_score=a_live,
        remaining_mins=sim_rem,
        completed_quarters=comp_q_list,
        quarter_length=12.0,
    )
    st.success(
        f"ğŸ¯ **Tahmini MaÃ§ Sonu:** {sim_res['predicted_home_final']:.0f} - {sim_res['predicted_away_final']:.0f} "
        f"(**Toplam: {sim_res['predicted_total']:.1f} SayÄ±**) | "
        f"**En GÃ¼Ã§lÃ¼ 20-SayÄ± KÃ¼mesi:** `{sim_res['best_cluster_name']}` (%{sim_res['best_cluster_prob']:.1f})"
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
        yaxis=dict(title="OlasÄ±lÄ±k (%)", range=[0, max(list(b_probs.values()) + [35]) * 1.2]),
    )
    st.plotly_chart(fig_sim, use_container_width=True)
