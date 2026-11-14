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
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "soft_light"


def toggle_ui_theme():
    st.session_state.ui_theme = "matte_dark" if st.session_state.ui_theme == "soft_light" else "soft_light"


IS_DARK = st.session_state.ui_theme == "matte_dark"

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON BACKGROUND DAEMON THREAD (24/7 ZERO-CRASH AUTO SCRAPER)
# ─────────────────────────────────────────────────────────────────────────────
SCRAPE_LOCK = threading.Lock()


ENGINE_VERSION = "v3.2-smart-62m"


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
                "211+ Üst Oranı": f"%{over_210_pct:.1f}",
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


# ─────────────────────────────────────────────────────────────────────────────
# TOP COMMAND BAR & CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
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
        <div class="cmd-title">🏀 NBA 2K26 CyberLeague Pro</div>
        <div class="cmd-sub">Canlı Barem (Exact Points), Kapanış Simetrisi, Takım Barem DNA'sı & Kasa/Dutching Terminali</div>
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

ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5, ctrl_c6 = st.columns([1.25, 1.25, 1.3, 1.0, 0.85, 0.9])

with ctrl_c1:
    if st.button("⚡ Anlık Canlı Barem Çek", use_container_width=True, type="primary"):
        with st.spinner("1xBet / Mirror üzerinden canlı maçlar ve baremler çekiliyor (~2 sn)..."):
            with SCRAPE_LOCK:
                run_auto_scraper(force_manual=True)
                bg_engine.last_run_ts = time.time()
                bg_engine.last_run_str = datetime.now().strftime("%H:%M:%S")
        st.toast("✅ Canlı maçlar ve barem oranları güncellendi!", icon="🏀")
        st.rerun()

with ctrl_c2:
    if st.button("🔄 Bitmiş Maçları Sonuçlandır", use_container_width=True):
        with st.spinner("Bitmiş maç sonuçları kontrol ediliyor..."):
            pull_today_results(only_tracked=True)
            settled = auto_settle_checkpoints()
        st.toast(f"✅ {settled} barem kaydı sonuçlandırıldı!", icon="🎯")
        st.rerun()

with ctrl_c3:
    if st.button("📥 Son 4 Ay Arşivini Güncelle (120 Gün)", use_container_width=True):
        with st.spinner("Son 4 aydaki (120 gün) tüm tamamlanmış CyberLeague maçları çekiliyor..."):
            added = pull_today_results(only_tracked=False, days_back=125)
            auto_settle_checkpoints()
        st.toast(f"📥 {added} yeni tamamlanmış maç 4 aylık geçmiş arşivine eklendi!", icon="📊")
        st.rerun()

with ctrl_c4:
    if st.button("🗑️ Verileri Sıfırla", use_container_width=True):
        reset_all_data_clean_slate()
        st.toast("🧹 Tüm kayıtlar sıfırlandı (0 kayıt). Temiz sayfa hazır!", icon="✨")
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
            <div class="kpi-label">🔴 Canlı CyberLeague Maçı</div>
            <div class="kpi-val">{active_count} Maç</div>
            <div class="kpi-sub">{open_barem_count} Maçta Barem Açık &bull; {hist_count} Geçmiş Arşiv</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">📋 Barem Kayıt Akışı</div>
            <div class="kpi-val">{total_logs}</div>
            <div class="kpi-sub">{pend_count} Bekleyen &bull; {comp_count} Sonuçlanan</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">⭐ 1xBet Favori Barem</div>
            <div class="kpi-val" style="color:#059669;">%{fav_rate:.1f}</div>
            <div class="kpi-sub">{fav_wins} / {comp_count} İsabet (En Düşük Oran)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">🔥 Model Value Barem</div>
            <div class="kpi-val" style="color:#d97706;">%{val_rate:.1f}</div>
            <div class="kpi-sub">{val_wins} / {comp_count} İsabet (Pozitif EV)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">🛡️ 20-Sayı Küme (Çifte Barem)</div>
            <div class="kpi-val" style="color:#2563eb;">%{clu_rate:.1f}</div>
            <div class="kpi-sub">{clu_wins} / {comp_count} İsabet (İkili Blok)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SEQUENTIAL BAREM WAVE STRIP (LAST 8 MATCHES: 0, -1, +1 WAVE DETECTOR)
    # ─────────────────────────────────────────────────────────────────────────
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
                tag_txt = "🎯 Tam Favori Geldi"
            elif sh < 0:
                p_bg = "var(--fav2-bg)"
                p_bd = "var(--fav2-border)"
                p_tx = "var(--fav2-text)"
                tag_txt = f"📉 {abs(sh)} Alt Bareme Düştü"
            else:
                p_bg = "var(--val-bg)"
                p_bd = "var(--val-border)"
                p_tx = "var(--val-text)"
                tag_txt = f"📈 {sh} Üst Bareme Çıktı"

            odd_36_val = float(item.get("exp_odd_36", 0.0) or 0.0)
            if odd_36_val > 0:
                ref_line = f'📊 36.Dk Favori: <b>{item["exp_barem_36"]} ({odd_36_val:.2f})</b> ➔ Biten: <b>{item["act_barem"]}</b>'
            else:
                ref_line = f'🧮 36.Dk Tahmin: <b>{item["exp_barem_36"]}</b> ➔ Biten: <b>{item["act_barem"]}</b>'
            drift_line = item.get("drift_label", "")

            pills_html += (
                f'<div style="background:{p_bg}; border:1.5px solid {p_bd}; color:{p_tx}; border-radius:10px; padding:0.48rem 0.65rem; min-width:155px; flex:1 1 180px;">'
                f'<div style="font-size:0.67rem; font-weight:700; opacity:0.85;">#{idx_c} &bull; 🕒 {time_only} | {t_short}</div>'
                f'<div style="font-size:0.71rem; font-weight:700; margin-top:0.08rem;">Skor: {sc_pair} ({item["total_score"]} Sayı)</div>'
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
                        🌊 Son Biten Maçlarda 36. Dakika Favorisi Tuttu mu, Kaydı mı? (Toplam {len(recent_chain)} Maç | Son 5 Maç: <span style="font-family:'JetBrains Mono',monospace; color:var(--score-text);">{wave_global['last_5_labels']}</span>)
                    </div>
                    <div style="font-size:0.8rem; font-weight:700; color:var(--fav1-text); background:var(--pill-bg); border:1px solid var(--pill-border); padding:0.25rem 0.75rem; border-radius:999px;">
                        🔮 Sıradaki Maç Beklentisi ({wave_global['pattern_matches_count']} Benzer Maç): ❄️ 2 Alt Barem: %{wave_global['prob_minus_2']} &bull; 📉 1 Alt Barem: %{wave_global['prob_minus_1']} &bull; 🎯 Tam Favori: %{wave_global['prob_0']} &bull; 📈 1 Üst Barem: %{wave_global['prob_plus_1']} &bull; 🔥 2 Üst Barem: %{wave_global['prob_plus_2']}
                    </div>
                </div>
                <div style="display:flex; gap:0.45rem; flex-wrap:wrap; max-height:340px; overflow-y:auto; padding-right:4px;">
                    {pills_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: LIVE MATCH CARDS & 9-BAREM MATRIX
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🔴 Canlı Maç Kartları & Anlık 9\'lu Barem Matrisi (0 - 36. Dakika)</div>',
        unsafe_allow_html=True,
    )

    if not live_matches:
        st.info(
            "Şu an aktif canlı maç bekleniyor veya tarama yapılıyor. "
            "Yukarıdaki **'⚡ Anlık Canlı Barem Çek'** butonuna basarak anında kontrol edebilirsiniz."
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
            q_str_parts.append(f"Q{q_num} (Canlı): {m.get('live_q_score', '0-0')}")
            quarters_summary = " &nbsp;&bull;&nbsp; ".join(q_str_parts)

            if status_code == "PRE_GAME":
                badge_html = (
                    '<span style="background:var(--bg-subcard); border:1px solid var(--border-subtle); '
                    'color:var(--text-secondary); padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    "⏳ MAÇ BAŞLAMADI</span>"
                )
            elif is_open:
                rem_to_36 = max(0.0, 36.0 - el_m)
                badge_html = (
                    '<span style="background:var(--pill-bg); border:1px solid var(--pill-border); '
                    'color:var(--pill-text); padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    f"🟢 BAREM AÇIK (Kapanışa {rem_to_36:.1f} dk)</span>"
                )
            else:
                badge_html = (
                    '<span style="background:rgba(225,29,72,0.1); border:1px solid rgba(225,29,72,0.28); '
                    'color:#e11d48; padding:0.28rem 0.75rem; border-radius:8px; font-size:0.76rem; font-weight:700;">'
                    "🔒 36+ DK BAREM KAPANDI — Maç Sonucu Bekleniyor</span>"
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
            q3_sig_label = q3_sig.get("signal", "Nötr Q4 Beklentisi")
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
                cur_tot=int(tot_sc),
                elapsed_mins=float(el_m),
            )
            smart_color = {
                "SUPER_SIGNAL": "#059669",
                "GOLDEN": "#059669",
                "PASS_CLUTCH": "#dc2626",
                "TRAP_EXTREME": "#e11d48",
                "TRAP_LOW": "#d97706",
            }.get(smart["zone_type"], "var(--text-secondary)")
            smart_html = (
                f'<div class="info-strip" style="grid-column: 1 / -1; border-left:4px solid {smart_color};" '
                f'title="Fareyi aşağıdaki rozetlerin üstüne getirerek her terimin ne işe yaradığını okuyabilirsiniz.">'
                f'<div class="info-strip-label">🤖 35. Dk Akıllı Karar & Tuzak Kutusu v3.1 '
                f'<span style="font-weight:500; opacity:0.8;">(Açıklama için rozetlerin üstüne gelin ℹ️)</span></div>'
                f'<div class="info-strip-val" style="font-size:0.84rem; line-height:1.55;">'
                f'<span title="{smart["reasoning"]}" style="cursor:help; border-bottom:1px dotted currentColor;">'
                f'<b style="color:{smart_color};">{smart["zone_badge"]} [?]</b></span> &nbsp;&bull;&nbsp; '
                f'<span title="{smart["sec_dir_tip"]}" style="cursor:help; border-bottom:1px dotted var(--text-muted);">'
                f'{smart["sec_dir_badge"]} [?]</span> &nbsp;&bull;&nbsp; '
                f'<span title="{smart["margin_tip"]}" style="cursor:help; border-bottom:1px dotted var(--text-muted);">'
                f'{smart["margin_badge"]} [?]</span> &nbsp;&bull;&nbsp; '
                f'<span title="{smart["drift_tip"]}" style="cursor:help; border-bottom:1px dotted var(--text-muted);">'
                f'{smart["drift_badge"]} [?]</span><br>'
                f'<span title="Tek barem oynayacaksanız sistemin 62 maçlık veriye göre önerdiği 1 numaralı tercih." style="cursor:help;">'
                f'🎯 Ana Tercih: <b>{smart["primary_pick"]}</b></span> &nbsp;&bull;&nbsp; '
                f'<span title="İkili Koruma (Dutching): İki bareme birden bahis bölerek ikisinden biri tuttuğunda kâr etme yöntemi." style="cursor:help;">'
                f'🛡️ İkili Koruma: <b>{smart["backup_pick"]}</b></span> &nbsp;&bull;&nbsp; '
                f'<span title="Kademeli Kasa Önerisi: Yakın maçta (0-5 fark) 0 TL Pas, normal maçta 1000 TL, süper sinyalde 2000 TL." style="cursor:help;">'
                f'💰 Kasa: <b>{smart["stake_badge"]}</b></span> &nbsp;&bull;&nbsp; '
                f'<span title="{smart["edge_tip"]}" style="cursor:help; border-bottom:1px dotted var(--text-muted);">'
                f'{smart["edge_badge"]} [?]</span>'
                f'<div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.18rem;">💡 {smart["reasoning"]}</div>'
                f"</div></div>"
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

                if b_name == val_rng and odd_v < 90.0 and b_name != fav1:
                    cls_list.append("val-pick")
                    tag_html = '<div class="b-tag" style="background:var(--val-border); color:#ffffff;">🔥 DEĞERLİ ORAN</div>'
                elif b_name == val_rng and b_name == fav1 and odd_v < 90.0:
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
                f"1.Fav: {fav1} ({fav1_odd:.2f}) &bull; 2.Fav: {fav2} (İkisini Kapatma Oranı: {dutch_odd:.2f})"
                if sym_gap < 2.0 and fav1_odd < 90
                else f"1. Favori Barem: {fav1} ({fav1_odd if fav1_odd < 90 else '-'})"
            )

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
                    <div class="info-strip" style="cursor:help;" title="1xBet'teki en düşük oranlı 1. Favori ve 2. Favori baremleri gösterir. Çifte Şans (Dutching) oranı, iki bareme aynı kazancı verecek şekilde parayı böldüğünüzde oluşan net orandır.">
                        <div class="info-strip-label">⚖️ Barem Simetri & Çifte Şans [?]</div>
                        <div class="info-strip-val" style="color:var(--fav1-text);">
                            {sym_text}
                        </div>
                    </div>
                    <div class="info-strip" style="cursor:help;" title="20-Sayı Kümesi: Yan yana iki baremin (toplam 20 sayılık geniş aralığın) gelme ihtimalidir. Value ise 1xBet oranının model olasılığına göre normalden yüksek (değerli) kaldığı baremdir.">
                        <div class="info-strip-label">🛡️ En Güçlü 20-Sayı Kümesi & Value [?]</div>
                        <div class="info-strip-val" style="color:var(--val-text);">
                            Küme: {clu_name} (%{clu_prob:.1f}) &nbsp;&bull;&nbsp; Value: {val_rng}
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
                <div class="barem-grid">
                    {boxes_html}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: MATCH-ISOLATED BAREM FLOW & CLOSING SYMMETRY TABLES (0 - 36m)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">📊 Maç Bazlı Ayrıştırılmış Barem & Simetri Akış Tablosu (0 - 36. Dakika)</div>',
        unsafe_allow_html=True,
    )

    if df_log.empty:
        st.info(
            "Henüz kaydedilmiş barem akış verisi bulunmuyor. Canlı bir maçta barem açıkken (0-36. dk) her maç kendi bağımsız kutusunda buraya eklenecektir."
        )
    else:
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.15, 1.35, 1.25, 1.25])
        with f_col1:
            status_filter = st.selectbox(
                "Durum Filtresi",
                ["Tümü", "⏳ Canlı / Bekleyen (PENDING)", "✅ Sonuçlanan (COMPLETED)"],
                index=0,
            )
        with f_col2:
            unique_games = ["Tüm Maçlar (Her Maç Ayrı Tabloda)"] + list(df_log["game_name"].dropna().unique())[::-1]
            game_filter = st.selectbox("Maç Filtresi", unique_games, index=0)
        with f_col3:
            min_filter = st.selectbox(
                "Dakika Aralığı",
                ["Tüm Dakikalar (0-36. dk)", "🔥 Kapanış Bölgesi (30-36. dk)", "⚡ Erken/Orta Periyot (0-29. dk)"],
                index=0,
            )
        with f_col4:
            layout_mode = st.selectbox(
                "Görünüm Modu",
                ["🗂️ Her Maç Ayrı Kutuda (Alt Alta)", "📑 Her Maç Ayrı Sekmede (Tab)"],
                index=1,
            )

        df_view = df_log.copy()
        if "PENDING" in status_filter:
            df_view = df_view[df_view["status"] == "PENDING"]
        elif "COMPLETED" in status_filter:
            df_view = df_view[df_view["status"] == "COMPLETED"]

        if not game_filter.startswith("Tüm Maçlar"):
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
            st.info("Seçilen filtreye uygun barem kaydı bulunamadı.")
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
                            hits.append("⭐ FAV TUTTU")
                        if act_rng == val_rng:
                            hits.append("🔥 VALUE TUTTU")
                        if check_cluster_hit(clu_rng, act_rng):
                            hits.append("🛡️ KÜME TUTTU")
                        outcome_badge = " | ".join(hits) if hits else f"❌ ({act_rng})"
                    else:
                        outcome_badge = "⏳ Bekliyor"

                    def _fmt_cell(odd_col, prob_col):
                        o_v = float(r.get(odd_col, 99.0) or 99.0)
                        p_v = float(r.get(prob_col, 0.0) or 0.0)
                        if o_v >= 90.0:
                            return f"- (%{p_v:.0f})"
                        return f"{o_v:.2f} (%{p_v:.0f})"

                    rows.append(
                        {
                            "Dakika": r.get("check_time", ""),
                            "Anlık Skor": r.get("current_score", ""),
                            "160-": _fmt_cell("odds_160_or_less", "model_prob_160_or_less"),
                            "161-170": _fmt_cell("odds_161_170", "model_prob_161_170"),
                            "171-180": _fmt_cell("odds_171_180", "model_prob_171_180"),
                            "181-190": _fmt_cell("odds_181_190", "model_prob_181_190"),
                            "191-200": _fmt_cell("odds_191_200", "model_prob_191_200"),
                            "201-210": _fmt_cell("odds_201_210", "model_prob_201_210"),
                            "211-220": _fmt_cell("odds_211_220", "model_prob_211_220"),
                            "221-230": _fmt_cell("odds_221_230", "model_prob_221_230"),
                            "231+": _fmt_cell("odds_231_plus", "model_prob_231_plus"),
                            "⭐ 1xBet Fav": fav_rng,
                            "🔥 Value Barem": val_rng,
                            "🛡️ 20-Sayı Kümesi": f"{clu_rng} (%{float(clu_prb or 0):.0f})",
                            "Maç Sonu": f"{act_sc} ({act_rng})" if act_sc else "-",
                            "Sonuç Analizi": outcome_badge,
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
                            <span style="font-size:1.0rem; font-weight:800; color:var(--text-primary);">🏀 {g_name}</span>
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

            if "Sekmede" in layout_mode and len(sorted_gids) > 1:
                tab_labels = []
                for gid in sorted_gids[:12]:
                    g_sub = df_view[df_view["game_id"] == gid]
                    g_name = str(g_sub.iloc[-1].get("game_name", gid)).replace(" (cyber)", "")
                    g_st = "⏳" if (g_sub["status"] == "PENDING").any() else "✅"
                    tab_labels.append(f"{g_st} {g_name} ({len(g_sub)})")
                m_tabs = st.tabs(tab_labels)
                for idx_t, gid in enumerate(sorted_gids[:12]):
                    with m_tabs[idx_t]:
                        _render_match_flow_card(gid)
            else:
                for gid in sorted_gids:
                    _render_match_flow_card(gid)


render_live_dashboard()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ADVANCED ANALYTICS & HISTORICAL ARCHIVE (ALL 4 MODULES)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-hdr">🧠 Gelişmiş İstatistik, Geçmiş Tüm Maçlar Arşivi & Kasa/Dutching Merkezi</div>',
    unsafe_allow_html=True,
)

df_completed_archive = load_completed_matches()
df_sym_all = load_symmetry_log()
live_state_now = load_live_state()
live_matches_now = live_state_now.get("live_matches", [])

tab_dna, tab_sym, tab_q3q4, tab_bankroll, tab_sim, tab_smart35 = st.tabs(
    [
        f"🏆 1. Geçmiş Maçlar & Takım Barem DNA ({len(df_completed_archive)} Maç)",
        "⚖️ 2. Kapanış Oranı (30-36. Dk) Simetri & Tuzak Analizi",
        "⚡ 3. Q3 ➔ Q4 Patlama / Çöküş Dedektörü",
        "💰 4. Canlı Kupon, Kasa (ROI) & Çifte Barem (Dutching) Hesaplayıcı",
        "🧮 5. Manuel Canlı Tempo Simülatörü",
        "🤖 6. 35. Dk Akıllı Kural & Sanal Kasa Simülatörü",
    ]
)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 6: 35th-MINUTE SMART RULES, TRANSITIONS & VIRTUAL BANKROLL SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_smart35:
    st.markdown(
        "##### 🤖 35. Dakika Akıllı Kural Motoru v3.1 — Gerçek Kapanış Oranlı Maçlardan Hesaplanır",
        help="Bu sekmedeki tüm tablolar ve sanal kasa sonuçları, canlı takip edilip tamamlanmış gerçek maçların 35. dakika 1xBet kapanış oranlarından otomatik hesaplanır.",
    )
    sim35 = simulate_35m_strategies(df_sym_all)
    if sim35["total_matches"] == 0:
        st.info("Sonuçlanmış canlı kapanış kaydı bekleniyor. Maçlar bittikçe bu sekme otomatik dolar.")
    else:
        st.caption(
            f"📊 Analiz edilen gerçek maç sayısı: **{sim35['total_matches']}** • Her maça sabit **1000 TL** (veya Kademeli Kasa: **0 / 1000 / 2000 TL**) bahis • "
            "Tablo başlıklarındaki **`?`** ikonlarının veya aşağıdaki kutuların üstüne fareyle gelerek her terimin ne işe yaradığını okuyabilirsiniz."
        )
        _n_games = df_sym_all["game_id"].nunique() if not df_sym_all.empty else 0
        _n_done = (
            df_sym_all[df_sym_all["status"] == "COMPLETED"]["game_id"].nunique()
            if (not df_sym_all.empty and "status" in df_sym_all.columns)
            else 0
        )
        st.caption(f"🔎 Veri kaynağı: {len(df_sym_all)} dakika kaydı • {_n_games} maç • {_n_done} sonuçlanmış • sürüm {ENGINE_VERSION}")

        st.markdown(
            """
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.55rem; margin:0.5rem 0 0.85rem 0;">
                <div class="info-strip" style="cursor:help; border-left:4px solid #059669;"
                     title="1xBet'te en düşük oranlı 1. Favori baremin hemen yanında 2. Favori vardır. Eğer 2. Favori BİR ÜST baremdeyse (ör. 1.Fav 201-210 iken 2.Fav 211-220 ise) ve 1.Fav oranı 2.10 üstüyse, maç neredeyse hiç alt bareme düşmez (%0) ve 1. Favori %61.1 isabetle +%39.9 kâr verir!">
                    <div class="info-strip-label">🔥 1. Kural: 2. Favori Yönü [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        <b>2. Favori Üstteyse (+1)</b> maç alta düşmez, <b>0 Merkez</b> oyna (<b>+%39.9 ROI</b>). <b>Alttaysa (-1)</b> ve oran &lt;2.10 ise <b>-1 Alt</b> oyna.
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #dc2626;"
                     title="35. dakikada iki takım arasındaki skor farkı 0 ile 5 sayı arasındaysa maç başa baş demektir. Son dakikalarda taktik fauller ve uzatma (OT) riski yüzünden normalden +6 sayı fazla (ort. 59 sayı) atılır ve barem yukarı patlar. Bu maçlara bahis almamak (PAS) genel kârı +%17.6'dan +%29.1'e çıkarır!">
                    <div class="info-strip-label">⛔ 2. Kural: 0-5 Fark Yakın Maç Tuzağı [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        35. dk'da fark <b>0-5 sayı</b> ise faul/uzatma yüzünden barem patlar. <b>PAS GEÇMEK (0 TL)</b> seni zarardan korur.
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #2563eb;"
                     title="35. dakikada fark 18 sayı veya daha fazlaysa maç kopmuş demektir. Takımlar son çeyrekte rölantiye geçer, ortalama sadece 47.4 sayı atılır (yakın maçtan 11.6 sayı daha az!). Üst bareme kaçma ihtimali %5'tir; 0 Merkez %60 tutar (+%27.5 ROI).">
                    <div class="info-strip-label">🧊 3. Kural: 18+ Fark Rölanti Freni [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        Fark <b>18+ sayı</b> ise son çeyrek yavaşlar (ort. 47.4 sayı). Üste kaçmaz; <b>0 Merkez %60</b> tutar (<b>+%27.5 ROI</b>).
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #e11d48;"
                     title="35. dakikada 1. Favori oranı 1.85'in altındaysa (ör. 1.65, 1.75), herkes oraya yüklendiği için büyük tuzaktır ve -%37 zarar yazar! Bu maçlarda 1. Favori yerine doğrudan 2. Favori baremi oynamak +%41.8 kâr (ROI) sağlamıştır.">
                    <div class="info-strip-label">🚨 4. Kural: &lt;1.85 Aşırı Favori Tuzağı [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        1. Favori oranı <b>1.85 altındaysa</b> oynama (-%37 zarar)! Doğrudan <b>2. Favori</b> oyna (<b>+%41.8 ROI</b>).
                    </div>
                </div>
                <div class="info-strip" style="cursor:help; border-left:4px solid #d97706;"
                     title="35. dakikadan maç sonuna kadar ortalama 53.3 sayı atılır. Mevcut skora 53.3 eklediğimizde çıkan sayı favori baremin alt sınırına çok yakınsa (Alt Kenar), maçların %55.6'sı bir alt bareme (-1) düşer. Tam ortasındaysa 0 Merkez %56 tutar.">
                    <div class="info-strip-label">📐 5. Kural: Barem İçi Kenar Konumu [?]</div>
                    <div style="font-size:0.77rem; color:var(--text-primary); margin-top:0.15rem;">
                        <b>35.Dk Skor + 53.3</b> favori baremin alt sınırındaysa <b>-1 Alt</b> riski %56'dır; tam ortasındaysa <b>0 Merkez</b> güvenlidir.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "###### 💰 Sanal Kasa: Hangi Strateji Ne Kadar Kazandırdı?",
            help="Gerçek sonuçlanmış maçların tamamında bu 7 stratejiyi uygulasaydık kasamızın ne olacağını gösterir. ROI: Yatırılan toplam paraya göre yüzdelik net kâr oranıdır.",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Strateji": s["name"],
                        "İsabet": f"{s['wins']} / {s['total']} (%{s['win_rate']})",
                        "Toplam Yatırım": f"{s['cost']:.0f} TL",
                        "Toplam Dönüş": f"{s['return']:.0f} TL",
                        "Net Kâr / Zarar": f"{s['net_profit']:+.0f} TL",
                        "ROI": f"%{s['roi_pct']:+.1f}",
                    }
                    for s in sim35["strategies"]
                ]
            ),
            column_config={
                "Strateji": st.column_config.TextColumn(
                    "Strateji",
                    help="Test edilen bahis yöntemi. En üstteki 'v3 Seçici' yakın maçları (0-5 fark) pas geçip diğerlerine 1000 TL oynar.",
                ),
                "İsabet": st.column_config.TextColumn(
                    "İsabet",
                    help="Oynanan toplam maç içinden kaç tanesinin kazandığı ve tutturma yüzdesi.",
                ),
                "Toplam Yatırım": st.column_config.TextColumn(
                    "Toplam Yatırım",
                    help="Oynanan tüm maçlar için cepten çıkan toplam bahis tutarı (Pas geçilen maçlarda para harcanmaz).",
                ),
                "Toplam Dönüş": st.column_config.TextColumn(
                    "Toplam Dönüş",
                    help="Tutan kuponlardan kasaya geri dönen brüt toplam para.",
                ),
                "Net Kâr / Zarar": st.column_config.TextColumn(
                    "Net Kâr / Zarar",
                    help="Toplam Dönüş eksi Toplam Yatırım = Cebimize kalan net kâr veya zarar.",
                ),
                "ROI": st.column_config.TextColumn(
                    "ROI (Verim %)",
                    help="Yatırılan her 100 TL başına elde edilen net yüzdelik kâr. Örneğin +%29.1 ROI, her 1000 TL yatırıma ortalama +291 TL net kâr demektir.",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        if sim35.get("sec_dir_brackets"):
            st.markdown(
                "###### 🧭 2. Favorinin Yönüne Göre Davranış (Piyasa Eğilimi: Üstte mi Altta mı?)",
                help="1xBet'te 1. Favoriden sonra en düşük oranlı 2. baremin üstte (+1) mi yoksa altta (-1) mı durduğuna göre maçların nasıl bittiğini gösterir.",
            )
            st.dataframe(
                pd.DataFrame(sim35["sec_dir_brackets"]),
                column_config={
                    "2. Favorinin Konumu (Piyasa Eğilimi)": st.column_config.TextColumn(
                        help="2. Favori Üstte (+1): Piyasa sayıların yukarı gitmesini bekliyor. 2. Favori Altta (-1): Piyasa sayıların aşağı düşmesini bekliyor."
                    ),
                    "Maç Sayısı": st.column_config.NumberColumn(help="Bu koşula uyan tamamlanmış gerçek maç sayısı."),
                    "🎯 0 Merkez (ROI)": st.column_config.TextColumn(
                        help="35. dakikadaki 1. Favori (Merkez) baremin tutma oranı ve tek başına oynandığında bıraktığı yüzdelik kâr (ROI)."
                    ),
                    "📉 -1 Alt (ROI)": st.column_config.TextColumn(
                        help="Favorinin 1 kademe altındaki baremin tutma oranı ve tek başına oynandığında bıraktığı yüzdelik kâr (ROI)."
                    ),
                    "❄️ -2 Alt": st.column_config.TextColumn(help="Favorinin 2 veya daha fazla kademe altında bitme oranı."),
                    "📈 +1/+2 Üst": st.column_config.TextColumn(help="Favorinin 1 veya 2 kademe üstünde bitme oranı."),
                    "💡 Ne İşe Yarar / Ne Oynanır?": st.column_config.TextColumn(
                        help="Bu senaryo canlı maçta karşınıza çıktığında ne oynamanız gerektiğini özetler."
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

        if sim35.get("margin_brackets"):
            st.markdown(
                "###### ⚖️ 35. Dk Skor Farkına Göre Davranış (Yakın Maç Uzatma Tuzağı vs 18+ Fark Rölanti Freni)",
                help="35. dakikada iki takım arasındaki skor farkının son çeyrekte atılan sayıya ve barem isabetine etkisini gösterir.",
            )
            st.dataframe(
                pd.DataFrame(sim35["margin_brackets"]),
                column_config={
                    "35. Dk Skor Farkı Bandı": st.column_config.TextColumn(
                        help="35. dakikada ev sahibi ile deplasman skoru arasındaki mutlak sayı farkı."
                    ),
                    "Maç Sayısı": st.column_config.NumberColumn(help="Bu fark aralığındaki gerçek maç sayısı."),
                    "Son 13 Dk Ort. Atılan Sayı": st.column_config.TextColumn(
                        help="35. dakikadan maç bitimine kadar atılan ortalama toplam sayı. Yakın maçta 59.0 sayı atılırken 18+ farkta 47.4 sayıya düşer!"
                    ),
                    "🎯 0 Merkez (ROI)": st.column_config.TextColumn(
                        help="1. Favori (0 Merkez) baremin tutma yüzdesi ve net kâr oranı (ROI)."
                    ),
                    "📉 -1/-2 Alt Barem": st.column_config.TextColumn(help="Maçın favori baremin altında bitme oranı."),
                    "📈 +1/+2 Üst Barem": st.column_config.TextColumn(help="Maçın favori baremin üstünde bitme oranı."),
                    "💡 Ne İşe Yarar / Ne Oynanır?": st.column_config.TextColumn(
                        help="Canlı maçta bu skor farkını gördüğünüzde uygulanacak kural."
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

        if sim35["odds_brackets"]:
            st.markdown(
                "###### 🎚️ 35. Dk Favori Oran Bandına Göre Davranış (<1.85 Tuzağı & Altın Bant)",
                help="35. dakikadaki 1. Favori oranının büyüklüğüne göre maçların hangi baremde bittiğini gösterir.",
            )
            st.dataframe(
                pd.DataFrame(sim35["odds_brackets"]),
                column_config={
                    "35. Dk 1. Favori Oran Bandı": st.column_config.TextColumn(
                        help="35. dakikada en düşük oranlı 1. Favori baremin 1xBet oranı."
                    ),
                    "🎯 0 (Merkez) İsabet & ROI": st.column_config.TextColumn(
                        help="1. Favorinin doğrudan gelme yüzdesi ve yüzdelik net kârı (ROI)."
                    ),
                    "💡 Ne Yapmalı? (Kural Özeti)": st.column_config.TextColumn(
                        help="Oran bu aralıktayken hangi seçeneğin kârlı olduğunu açıklar."
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

        if sim35.get("edge_brackets"):
            st.markdown(
                "###### 📐 35. Dk Skorun Barem İçindeki Yeri (Mevcut Skor + 53.3 Kalan Sayı Hesabı)",
                help="35. dakikadaki toplam skora ortalama kalan sayı (53.3) eklendiğinde favori baremin neresine (alt sınır, orta, üst sınır) düştüğünü analiz eder.",
            )
            st.dataframe(
                pd.DataFrame(sim35["edge_brackets"]),
                column_config={
                    "35. Dk Skorun Barem İçindeki Yeri": st.column_config.TextColumn(
                        help="Örn: Favori barem 201-210 (ortası 205.5) iken 35.dk skoru + 53.3 = 201 yapıyorsa 'Alt Kenarda' demektir."
                    ),
                    "📉 -1/-2 Alta Kayma": st.column_config.TextColumn(help="Alt baremlere düşme yüzdesi."),
                    "🎯 0 Merkezde Kalma (ROI)": st.column_config.TextColumn(help="Favori baremde kalma yüzdesi ve ROI verimi."),
                    "📈 +1/+2 Üste Çıkma": st.column_config.TextColumn(help="Üst baremlere çıkma yüzdesi."),
                    "💡 Ne İşe Yarar?": st.column_config.TextColumn(help="Kenar konumuna göre nasıl aksiyon alınacağını açıklar."),
                },
                use_container_width=True,
                hide_index=True,
            )

        if sim35.get("minute_windows"):
            st.markdown(
                "###### ⏱️ Hangi Dakikada Bahis Alınmalı? (Dakika Penceresi İsabet & ROI Analizi)",
                help="Maçın 18., 24., 30., 32-33. ve 34-35. dakikalarında bahis alındığında hangisinin kârlı, hangisinin tuzak olduğunu gösterir.",
            )
            st.dataframe(
                pd.DataFrame(sim35["minute_windows"]),
                column_config={
                    "Bahis Giriş Dakikası": st.column_config.TextColumn(help="Bahsin alındığı maç dakikası."),
                    "🎯 1. Favori İsabet & ROI": st.column_config.TextColumn(
                        help="O dakikadaki 1. Favori bareme tekli bahis alındığında tutma yüzdesi ve ROI kârı."
                    ),
                    "🛡️ 1. & 2. Favori İkili Koruma": st.column_config.TextColumn(
                        help="O dakikadaki 1. ve 2. Favori bareme ikili koruma (Dutching) yapıldığında tutma yüzdesi ve ROI kârı."
                    ),
                    "💡 Zamanlama Tavsiyesi": st.column_config.TextColumn(
                        help="32-33. dakikalarda neden bahis alınmaması gerektiğini ve en doğru giriş dakikasını açıklar."
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

        if sim35["drift_brackets"]:
            st.markdown(
                "###### 📉 Maç İçi Barem Göçü (Q1 ➔ 35. Dk) Davranışı",
                help="Maçın 1. çeyreğindeki favori barem ile 35. dakikadaki favori barem arasındaki yükseliş/düşüş farkına göre maç sonu davranışı.",
            )
            st.dataframe(pd.DataFrame(sim35["drift_brackets"]), use_container_width=True, hide_index=True)

        if sim35["transitions"]:
            st.markdown(
                "###### 🌊 Ardışık Maç Geçiş Matrisi (Önceki Maç ➔ Sıradaki Maç)",
                help="Bir önceki biten maçın sonucuna (0 Merkez, -1 Alt, +1 Üst) göre sıradaki maçın nasıl bittiğini gösterir.",
            )
            st.dataframe(pd.DataFrame(sim35["transitions"]), use_container_width=True, hide_index=True)

        st.markdown(
            "###### 📒 Maç Maç Sanal Kupon Defteri (Tüm Gerçek Maçlar)",
            help="En son biten maç en üstte olacak şekilde her maçta v3 Seçici, Kademeli Kasa ve Akıllı İkili Koruma kuponlarının ne yaptığını tek tek gösterir.",
        )
        st.dataframe(
            pd.DataFrame(sim35["ledger"]),
            column_config={
                "Maç": st.column_config.TextColumn(help="Karşılaşan takımlar."),
                "Kapanış Dk & Fark": st.column_config.TextColumn(
                    help="Kapanış oranının alındığı dakika ve o andaki skor farkı (0-5 fark yakın maçtır, pas geçilir)."
                ),
                "35.Dk Favori (0)": st.column_config.TextColumn(help="35. dakikada 1xBet'in en düşük oranlı 1. Favori baremi."),
                "2. Favori Yönü": st.column_config.TextColumn(
                    help="2. Favorinin 1. Favoriye göre üstte mi (📈 Üst) yoksa altta mı (📉 Alt) durduğu ve oranı."
                ),
                "🤖 v3 Akıllı Seçim": st.column_config.TextColumn(
                    help="v3 Akıllı Kural Motoru'nun bu maç için seçtiği barem ve hangi kuralı uyguladığı."
                ),
                "Biten Barem (Skor)": st.column_config.TextColumn(help="Maçın gerçekte bittiği barem ve maç sonu skoru."),
                "v3 Seçici (1000 TL / Pas)": st.column_config.TextColumn(
                    help="0-5 sayı fark olan yakın maçları PAS geçip diğerlerine 1000 TL oynayan stratejinin bu maçtaki net sonucu."
                ),
                "Kademeli Kasa (0/1000/2000)": st.column_config.TextColumn(
                    help="Yakın maça 0 TL, normal maça 1000 TL, Süper Sinyal maçına 2000 TL yatıran kademeli kasanın bu maçtaki sonucu."
                ),
                "Akıllı İkili Koruma": st.column_config.TextColumn(
                    help="10+ farkta (0 & -1), diğer maçlarda (1.Fav & 2.Fav) çifte şans oynayan İkili Koruma kuponunun sonucu."
                ),
            },
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
            "Şu an geçmiş maç arşivi boş. Yukarıdaki **'📥 Son 4 Ay Arşivini Güncelle (120 Gün)'** butonuna basarak "
            "son 4 ayın tamamlanmış tüm CyberLeague maçlarını saniyeler içinde yükleyebilirsiniz."
        )
    else:
        df_arc = df_completed_archive.copy()
        df_arc["Kazanan Barem"] = df_arc["total_score"].apply(get_range_from_score)
        df_arc["20'li Küme"] = df_arc["total_score"].apply(get_cluster_from_score)
        df_arc["Q1 Toplam"] = df_arc["home_q1"].astype(int) + df_arc["away_q1"].astype(int)
        df_arc["Q2 Toplam"] = df_arc["home_q2"].astype(int) + df_arc["away_q2"].astype(int)
        df_arc["Q3 Toplam"] = df_arc["home_q3"].astype(int) + df_arc["away_q3"].astype(int)
        df_arc["Q4 Toplam"] = df_arc["home_q4"].astype(int) + df_arc["away_q4"].astype(int)

        all_teams = sorted(
            set(df_arc["home_team"].dropna().unique()).union(set(df_arc["away_team"].dropna().unique()))
        )

        flt1, flt2, flt3 = st.columns([1.3, 1.3, 1.1])
        with flt1:
            sel_t1 = st.selectbox("1. Takım Filtresi (H2H veya Tek Takım)", ["Tüm Takımlar"] + all_teams, index=0)
        with flt2:
            sel_t2 = st.selectbox("2. Takım Filtresi (Rakip / H2H)", ["Tüm Takımlar"] + all_teams, index=0)
        with flt3:
            sel_barem_flt = st.selectbox(
                "Barem Filtresi",
                ["Tüm Baremler"] + [b[0] for b in BAREM_RANGES],
                index=0,
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
            df_filtered["20'li Küme"].mode().iloc[0]
            if n_flt > 0 and not df_filtered["20'li Küme"].mode().empty
            else "-"
        )
        m_c1.metric("📚 Filtrelenen Maç Sayısı", f"{n_flt} Maç")
        m_c2.metric("🏀 Ortalama Toplam Sayı", f"{avg_tot:.1f} Sayı")
        m_c3.metric("🎯 En Çok Çıkan 10'lu Barem", modal_b)
        m_c4.metric("🛡️ En Çok Çıkan 20'li Küme", modal_c)

        ch_col, dna_col = st.columns([1.05, 1.35])
        with ch_col:
            st.markdown("##### 📊 Geçmiş Maçların 9'lu Barem Dağılım Grafiği")
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
            st.markdown("##### 🧬 Takım Bazlı Barem DNA Tablosu (Tüm CyberLeague Takımları)")
            df_dna_table = compute_team_dna_table(df_arc)
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
# MODULE 2: CLOSING ODDS (30-36 MIN) SYMMETRY & SEQUENTIAL WAVE BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with tab_sym:
    st.markdown(
        "##### 🌊 Ardışık Maçlarda 36. Dakika Simetri Dalga Analizi (`-2 Alt` / `-1 Alt` / `0 Beklenen` / `+1 Üst` / `+2 Üst`)"
    )
    st.caption(
        "Sezdiğiniz ardışık maç dalga örüntüsünü (**-2 Alt Barem, -1 Alt Barem, 0 Beklenen Merkez, +1 Üst Barem, +2 Üst Barem**) "
        "4.504 gerçek maç üzerinde test eder ve sıradaki maçın hangi bareme kayacağını hesaplar."
    )

    w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns([1, 1, 1, 1, 1.2])
    seq_map = {
        "❄️ -2 (2 Alt Barem Geldi)": -2,
        "📉 -1 (1 Alt Barem Geldi)": -1,
        "🎯 0 (36.Dk Beklenen Geldi)": 0,
        "📈 +1 (1 Üst Barem Geldi)": 1,
        "🔥 +2 (2 Üst Barem Geldi)": 2,
    }
    rev_map = {v: k for k, v in seq_map.items()}

    default_wave = analyze_sequential_barem_wave(df_completed_archive, df_sym_all)
    act_pat = default_wave.get("active_pattern", [0, -1, -1, 0])
    while len(act_pat) < 4:
        act_pat = [0] + act_pat

    opts_list = list(seq_map.keys())
    with w_col1:
        s_m1 = st.selectbox("1. Önceki Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[0], opts_list[2])))
    with w_col2:
        s_m2 = st.selectbox("2. Önceki Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[1], opts_list[1])))
    with w_col3:
        s_m3 = st.selectbox("3. Önceki Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[2], opts_list[1])))
    with w_col4:
        s_m4 = st.selectbox("4. (Son Bitmiş) Maç", opts_list, index=opts_list.index(rev_map.get(act_pat[3], opts_list[2])))
    with w_col5:
        test_fav_b = st.selectbox(
            "Canlı Maçın 36.Dk Beklenen Baremi",
            [b[0] for b in BAREM_RANGES],
            index=5,
        )

    custom_pattern = [seq_map[s_m1], seq_map[s_m2], seq_map[s_m3], seq_map[s_m4]]
    sim_wave = analyze_sequential_barem_wave(
        df_completed_archive, df_sym_all, current_fav_barem=test_fav_b, custom_seq=custom_pattern
    )

    wk_m2, wk_m1, wk_0, wk_p1, wk_p2, wk_pair = st.columns(6)
    wk_m2.metric(
        f"❄️ -2 Alt ({sim_wave['target_minus_2_barem']})",
        f"%{sim_wave['prob_minus_2']}",
        "2 Alt Barem Kayması",
    )
    wk_m1.metric(
        f"📉 -1 Alt ({sim_wave['target_minus_1_barem']})",
        f"%{sim_wave['prob_minus_1']}",
        f"Toplam Alt: %{sim_wave['prob_down_total']}",
    )
    wk_0.metric(
        f"🎯 0 Beklenen ({sim_wave['target_0_barem']})",
        f"%{sim_wave['prob_0']}",
        f"{sim_wave['pattern_matches_count']} Benzer Zincir",
    )
    wk_p1.metric(
        f"📈 +1 Üst ({sim_wave['target_plus_1_barem']})",
        f"%{sim_wave['prob_plus_1']}",
        f"Toplam Üst: %{sim_wave['prob_up_total']}",
    )
    wk_p2.metric(
        f"🔥 +2 Üst ({sim_wave['target_plus_2_barem']})",
        f"%{sim_wave['prob_plus_2']}",
        "2 Üst Barem Kayması",
    )
    wk_pair.metric(
        "🛡️ Önerilen İkili Küme",
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
        st.markdown("###### 📜 Canlı Takip Edilip Tamamlanan Maçların `36. Dakika Beklenen vs Biten Barem (-2 / -1 / 0 / +1 / +2)` Zinciri")
        rc_rows = []
        for item in default_wave["recent_chain"][::-1]:
            eb = item["exp_barem_36"]
            rc_rows.append(
                {
                    "Tarih / Saat": item["date"],
                    "Maç": f"{item['home_team']} vs {item['away_team']}",
                    "36. Dk Skor": item["s36"],
                    "Maç Sonu Toplam": f"{item['home_total']}:{item['away_total']} ({item['total_score']})",
                    "❄️ -2 Alt Barem": _get_shifted_b(eb, -2),
                    "📉 -1 Alt Barem": _get_shifted_b(eb, -1),
                    "🎯 0 (36.Dk Beklenen)": eb,
                    "📈 +1 Üst Barem": _get_shifted_b(eb, 1),
                    "🔥 +2 Üst Barem": _get_shifted_b(eb, 2),
                    "Biten Gerçek Barem": item["act_barem"],
                    "Simetri Dalga Sapması (-2 / -1 / 0 / +1 / +2)": item["shift_label"],
                    "Referans": item["source"],
                }
            )
        st.dataframe(pd.DataFrame(rc_rows), use_container_width=True, hide_index=True, height=min(320, 60 + 38 * len(rc_rows)))

    st.markdown("---")
    st.markdown(
        "##### ⚖️ 1xBet Kapanış Oranı (30-36. Dk) Simetri Tuzağı & Komşu Bareme (`-2 / -1 / 0 / +1 / +2`) Kayma Analizi"
    )
    df_comp_sym = (
        df_sym_all[df_sym_all["status"] == "COMPLETED"].copy()
        if not df_sym_all.empty and "status" in df_sym_all.columns
        else pd.DataFrame()
    )

    if df_comp_sym.empty:
        st.info(
            "Şu an `symmetry_analysis_log.csv` içinde sonuçlanmış (COMPLETED) canlı barem kaydı bekleniyor. "
            "Canlı maçlar bittikçe burada **-2 Alt, -1 Alt, 0 Favori, +1 Üst, +2 Üst İsabet Oranları**, **Simetri Tuzağı (`Fark <= 0.65`) Kayma Oranı** "
            "ve **Dakika Penceresi (`0-24 dk`, `24-30 dk`, `30-36 dk`)** istatistikleri otomatik hesaplanır."
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
                win_bucket = "30-36. Dk (Kapanış)"
            elif el_min >= 24:
                win_bucket = "24-29. Dk (Q3 Başı/Ortası)"
            else:
                win_bucket = "0-23. Dk (İlk Yarı)"

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
            # Her maçın SON (35-36. Dk Kapanış) kaydını baz alarak Maç Bazlı Kapanış Simetrisini hesapla
            df_closing = df_sr.sort_values("el_min").groupby("game_id", as_index=False).last()
            tot_n = len(df_closing)
            m2_pct = df_closing["hit_m2"].mean() * 100.0
            m1_pct = df_closing["hit_m1"].mean() * 100.0
            f0_pct = df_closing["hit_0"].mean() * 100.0
            p1_pct = df_closing["hit_p1"].mean() * 100.0
            p2_pct = df_closing["hit_p2"].mean() * 100.0
            both_pct = df_closing["either_fav_hit"].mean() * 100.0

            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("❄️ -2 Alt Bareme Kayma", f"%{m2_pct:.1f}", f"{int(df_closing['hit_m2'].sum())} / {tot_n} Maç")
            s2.metric("📉 -1 Alt Bareme Kayma", f"%{m1_pct:.1f}", f"{int(df_closing['hit_m1'].sum())} / {tot_n} Maç")
            s3.metric("🎯 0 (1. Favori Merkez)", f"%{f0_pct:.1f}", f"{int(df_closing['hit_0'].sum())} / {tot_n} Maç")
            s4.metric("📈 +1 Üst Bareme Kayma", f"%{p1_pct:.1f}", f"{int(df_closing['hit_p1'].sum())} / {tot_n} Maç")
            s5.metric("🔥 +2 Üst Bareme Kayma", f"%{p2_pct:.1f}", f"{int(df_closing['hit_p2'].sum())} / {tot_n} Maç")
            s6.metric(
                "🛡️ 1. veya 2. Favori",
                f"%{both_pct:.1f}",
                f"{int(df_closing['either_fav_hit'].sum())} / {tot_n} Maç",
            )

            win_rows = []
            for w_name in ["0-23. Dk (İlk Yarı)", "24-29. Dk (Q3 Başı/Ortası)", "30-36. Dk (Kapanış)"]:
                w_sub = df_sr[df_sr["window"] == w_name]
                if w_sub.empty:
                    continue
                win_rows.append(
                    {
                        "Dakika Penceresi": w_name,
                        "Kayıt Sayısı": len(w_sub),
                        "❄️ -2 Alt Barem (%)": f"%{w_sub['hit_m2'].mean()*100:.1f}",
                        "📉 -1 Alt Barem (%)": f"%{w_sub['hit_m1'].mean()*100:.1f}",
                        "🎯 0 (1. Favori) (%)": f"%{w_sub['hit_0'].mean()*100:.1f}",
                        "📈 +1 Üst Barem (%)": f"%{w_sub['hit_p1'].mean()*100:.1f}",
                        "🔥 +2 Üst Barem (%)": f"%{w_sub['hit_p2'].mean()*100:.1f}",
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
        qc1, qc2, qc3 = st.columns([1.3, 1.3, 1.2])
        with qc1:
            q_t1 = st.selectbox("Ev Sahibi / 1. Takım", q_teams, index=0, key="q3_t1")
        with qc2:
            q_t2 = st.selectbox("Deplasman / 2. Takım", q_teams, index=0, key="q3_t2")
        with qc3:
            q3_test_score = st.slider("3. Çeyrek (Q3) Toplam Sayısı", 35, 72, 56, 1)

        t1_arg = "" if q_t1 == "Tüm Takımlar" else q_t1
        t2_arg = "" if q_t2 == "Tüm Takımlar" else q_t2
        sig_res = build_q3_q4_streak_signal(df_completed_archive, t1_arg, t2_arg, q3_score=q3_test_score)

        k_q1, k_q2, k_q3, k_q4 = st.columns(4)
        k_q1.metric("🔔 Tespit Edilen Q4 Sinyali", sig_res["signal"])
        k_q2.metric("🔥 Q4 Sayı Patlaması (56+ Sayı)", f"%{sig_res['burst_prob']:.1f}")
        k_q3.metric("⚖️ Q4 Normal Tempo (49-55 Sayı)", f"%{sig_res['normal_prob']:.1f}")
        k_q4.metric("❄️ Q4 Sayı Çöküşü (≤ 48 Sayı)", f"%{sig_res['collapse_prob']:.1f}")

        st.info(f"📌 **Analiz Özeti ({sig_res['sample_size']} Benzer Maç):** {sig_res['detail']}")

        # Show actual Q3 vs Q4 scatter/distribution from archive
        df_q_arc = df_completed_archive.copy()
        df_q_arc["Q3_Tot"] = df_q_arc["home_q3"].astype(float) + df_q_arc["away_q3"].astype(float)
        df_q_arc["Q4_Tot"] = df_q_arc["home_q4"].astype(float) + df_q_arc["away_q4"].astype(float)
        if t1_arg:
            df_q_arc = df_q_arc[(df_q_arc["home_team"] == t1_arg) | (df_q_arc["away_team"] == t1_arg)]
        if t2_arg:
            df_q_arc = df_q_arc[(df_q_arc["home_team"] == t2_arg) | (df_q_arc["away_team"] == t2_arg)]

        q_buckets = [
            ("Düşük Q3 (≤ 48 Sayı)", df_q_arc[df_q_arc["Q3_Tot"] <= 48]),
            ("Orta Q3 (49 - 54 Sayı)", df_q_arc[(df_q_arc["Q3_Tot"] >= 49) & (df_q_arc["Q3_Tot"] <= 54)]),
            ("Yüksek Q3 (55+ Sayı)", df_q_arc[df_q_arc["Q3_Tot"] >= 55]),
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
                    "🔥 Q4 Patlama (≥56 Sayı) %": f"%{(sub_q['Q4_Tot'] >= 56).mean()*100:.1f}",
                    "❄️ Q4 Çöküş (≤48 Sayı) %": f"%{(sub_q['Q4_Tot'] <= 48).mean()*100:.1f}",
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
        sel_preset = st.selectbox("Canlı Maçtan Otomatik Doldur", preset_options, index=0)

        def_b1_name, def_b2_name = "201-210", "211-220"
        def_o1, def_o2 = 4.50, 4.50
        if sel_preset != "Manuel Oran Girişi":
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
            total_stake = st.number_input("Toplam Kupon Tutarı (TL)", min_value=10.0, max_value=100000.0, value=1000.0, step=100.0)
        with dc2:
            odd_1 = st.number_input(f"1. Barem Oranı ({def_b1_name})", min_value=1.50, max_value=50.0, value=float(def_o1), step=0.10)
        with dc3:
            odd_2 = st.number_input(f"2. Barem Oranı ({def_b2_name})", min_value=1.50, max_value=50.0, value=float(def_o2), step=0.10)

        inv_sum = (1.0 / odd_1) + (1.0 / odd_2)
        eff_dutch_odd = 1.0 / inv_sum if inv_sum > 0 else 0.0
        stake_1 = total_stake * ((1.0 / odd_1) / inv_sum)
        stake_2 = total_stake * ((1.0 / odd_2) / inv_sum)
        gross_return = stake_1 * odd_1
        net_profit = gross_return - total_stake
        roi_pct = (net_profit / total_stake) * 100.0 if total_stake > 0 else 0.0

        st.success(
            f"💰 **1. Bareme ({def_b1_name}) Yatırılacak:** `{stake_1:.2f} TL` (Oran: {odd_1:.2f})\n\n"
            f"💰 **2. Bareme ({def_b2_name}) Yatırılacak:** `{stake_2:.2f} TL` (Oran: {odd_2:.2f})\n\n"
            f"✅ **Efektif Çifte Şans Oranı:** `{eff_dutch_odd:.2f}` | "
            f"**Garanti Toplam Dönüş:** `{gross_return:.2f} TL` | "
            f"**Net Kâr:** `+{net_profit:.2f} TL` (**+%{roi_pct:.1f} ROI**)"
        )

    with d_col2:
        st.markdown("##### 📈 Strateji Bazlı Gerçek Kasa (Bankroll) & ROI Performansı")
        st.caption(
            "Sonuçlanan (`COMPLETED`) her canlı barem kaydına sabit **1.000 TL** yatırıldığında 3 ana stratejinin gerçek net kâr/zarar tablosu:"
        )
        df_comp_roi = (
            df_sym_all[df_sym_all["status"] == "COMPLETED"].copy()
            if not df_sym_all.empty and "status" in df_sym_all.columns
            else pd.DataFrame()
        )
        if df_comp_roi.empty:
            st.info(
                "Henüz sonuçlanmış (`COMPLETED`) canlı barem kaydı yok. Takip edilen canlı maçlar sonuçlandıkça "
                "burada **1xBet Favori**, **Model Value** ve **20-Sayı Kümesi Dutching** stratejilerinin net TL kârı ve ROI yüzdesi listelenir."
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
                    "Strateji": "⭐ 1xBet 1. Favori Tekli Barem",
                    "Kupon Sayısı": n_bets,
                    "Toplam Yatırım": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{fav_pnl:+,.0f} TL",
                    "ROI (%)": f"%{(fav_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Strateji": "🔥 Model Value (Pozitif EV) Tekli Barem",
                    "Kupon Sayısı": n_bets,
                    "Toplam Yatırım": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{val_pnl:+,.0f} TL",
                    "ROI (%)": f"%{(val_pnl / tot_wagered)*100:+.1f}",
                },
                {
                    "Strateji": "🛡️ 20-Sayı Kümesi (Çifte Barem Dutching)",
                    "Kupon Sayısı": n_bets,
                    "Toplam Yatırım": f"{tot_wagered:,.0f} TL",
                    "Net Kâr / Zarar": f"{clu_pnl:+,.0f} TL",
                    "ROI (%)": f"%{(clu_pnl / tot_wagered)*100:+.1f}",
                },
            ]
            st.dataframe(pd.DataFrame(roi_rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5: MANUAL LIVE TEMPO & MONTE CARLO SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_sim:
    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        sim_q = st.selectbox("Aktif Çeyrek", [1, 2, 3, 4], index=2)
        sim_rem = st.slider("Çeyrekte Kalan Dakika", 0.0, 12.0, 4.5, 0.5)
    with s_c2:
        sim_hq1 = st.number_input("Ev Q1", 0, 60, 26)
        sim_hq2 = st.number_input("Ev Q2", 0, 60, 27)
        sim_hq3 = st.number_input("Ev Q3 (veya Canlı)", 0, 60, 18)
    with s_c3:
        sim_aq1 = st.number_input("Dep Q1", 0, 60, 25)
        sim_aq2 = st.number_input("Dep Q2", 0, 60, 26)
        sim_aq3 = st.number_input("Dep Q3 (veya Canlı)", 0, 60, 19)

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
