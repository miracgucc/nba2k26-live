import csv
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone

TR_TZ = timezone(timedelta(hours=3))

import numpy as np
import pandas as pd

from period_predictor import (
    BAREM_RANGES,
    analyze_barem_symmetry,
    build_q3_q4_streak_signal,
    get_range_from_score,
    predict_live_inplay,
)
from pull_today_results import (
    _open_keepalive_results_conn,
    discover_active_domain,
    pull_today_results,
)
from update_all_checkpoints import auto_settle_checkpoints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "symmetry_analysis_log.csv")
MATCHES_CSV_PATH = os.path.join(BASE_DIR, "matches_data.csv")
CACHE_FILE = os.path.join(BASE_DIR, "match_detail_cache.json")
TS_FILE = os.path.join(BASE_DIR, "match_log_timestamps.json")
LIVE_STATE_FILE = os.path.join(BASE_DIR, "live_matches_state.json")

CSV_HEADERS = [
    "game_id",
    "game_name",
    "date",
    "check_time",
    "current_score",
    "odds_160_or_less",
    "odds_161_170",
    "odds_171_180",
    "odds_181_190",
    "odds_191_200",
    "odds_201_210",
    "odds_211_220",
    "odds_221_230",
    "odds_231_plus",
    "model_prob_160_or_less",
    "model_prob_161_170",
    "model_prob_171_180",
    "model_prob_181_190",
    "model_prob_191_200",
    "model_prob_201_210",
    "model_prob_211_220",
    "model_prob_221_230",
    "model_prob_231_plus",
    "highest_prob_range",
    "value_range",
    "best_cluster_name",
    "best_cluster_prob",
    "actual_score",
    "actual_range",
    "status",
]


def init_csv():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def _extract_minute_num(check_time_str: str) -> int:
    m = re.search(r"Minute\s+(\d+)", str(check_time_str or ""))
    return int(m.group(1)) if m else 0


def upsert_symmetry_log_row(new_row: dict):
    """
    Upsert a checkpoint row by (game_id, check_time) so simultaneous games never create
    duplicate minute rows and rows stay cleanly grouped by game_id and sorted by minute.
    """
    init_csv()
    existing_rows = []
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        try:
            with open(CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
        except Exception:
            existing_rows = []

    MIN_CLEAN_GID = 770265000

    def _gid_int(g_str: str) -> int:
        m = re.search(r"(\d+)", str(g_str or ""))
        return int(m.group(1)) if m else 0

    dedup_map = {}
    game_order = []
    for r in existing_rows:
        gid = str(r.get("game_id", "")).strip()
        ctime = str(r.get("check_time", "")).strip()
        if not gid or _gid_int(gid) < MIN_CLEAN_GID:
            continue
        if gid not in game_order:
            game_order.append(gid)
        dedup_map[(gid, ctime)] = r

    new_gid = str(new_row.get("game_id", "")).strip()
    if _gid_int(new_gid) < MIN_CLEAN_GID:
        return
    new_ctime = str(new_row.get("check_time", "")).strip()
    if new_gid not in game_order:
        game_order.append(new_gid)

    if (new_gid, new_ctime) in dedup_map:
        prev = dedup_map[(new_gid, new_ctime)]
        if prev.get("status") == "COMPLETED":
            new_row["actual_score"] = prev.get("actual_score", "")
            new_row["actual_range"] = prev.get("actual_range", "")
            new_row["status"] = "COMPLETED"
    dedup_map[(new_gid, new_ctime)] = new_row

    # Group by game_id (in order of game appearance) and sort each game's checkpoints by minute
    final_rows = []
    for gid in game_order:
        g_rows = [r for (g, _), r in dedup_map.items() if g == gid]
        g_rows.sort(key=lambda x: _extract_minute_num(x.get("check_time", "")))
        final_rows.extend(g_rows)

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(final_rows)

    try:
        tmp_runtime_sym = os.path.join(tempfile.gettempdir(), "nba2k26_symmetry_runtime_v2.csv")
        with open(tmp_runtime_sym, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(final_rows)
    except Exception:
        pass


def _map_barem_name(raw_name: str, params_list: list, parameter_val) -> str:
    """Map 1xBet / Fansport event parameters or market name to standard 9 Barem brackets."""
    joined_params = "-".join(str(x).strip() for x in (params_list or []) if str(x).strip())
    if joined_params in {
        "161-170",
        "171-180",
        "181-190",
        "191-200",
        "201-210",
        "211-220",
        "221-230",
    }:
        return joined_params
    if "160" in joined_params:
        return "160 Or Less"
    if "231" in joined_params:
        return "231+"

    if parameter_val is not None:
        try:
            p_float = float(parameter_val)
            if p_float <= 160.5:
                return "160 Or Less"
            elif 160.9 <= p_float <= 162.0:
                return "161-170"
            elif 170.9 <= p_float <= 172.0:
                return "171-180"
            elif 180.9 <= p_float <= 182.0:
                return "181-190"
            elif 190.9 <= p_float <= 192.0:
                return "191-200"
            elif 200.9 <= p_float <= 202.0:
                return "201-210"
            elif 210.9 <= p_float <= 212.0:
                return "211-220"
            elif 220.9 <= p_float <= 222.0:
                return "221-230"
            elif p_float >= 230.9:
                return "231+"
        except Exception:
            pass

    rn = str(raw_name or "").strip().lower()
    if "160" in rn:
        return "160 Or Less"
    elif "161" in rn:
        return "161-170"
    elif "171" in rn:
        return "171-180"
    elif "181" in rn:
        return "181-190"
    elif "191" in rn:
        return "191-200"
    elif "201" in rn:
        return "201-210"
    elif "211" in rn:
        return "211-220"
    elif "221" in rn:
        return "221-230"
    elif "231" in rn:
        return "231+"
    return ""


def extract_ssr_match_data(match_url: str) -> dict:
    """
    Fetch match detail page over HTTP in ~0.15s and evaluate SSR Pinia store (`window.__V3_HOST_APP__`)
    using Node.js in ~0.03s. Zero browser overhead, zero memory leaks, 100% exact Pinia data.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(match_url, headers=headers)
    html = urllib.request.urlopen(req, context=ctx, timeout=8).read().decode("utf-8", errors="ignore")

    scripts = [
        s for s in re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL) if "window.__V3_HOST_APP__=" in s
    ]
    if not scripts:
        return {"error": "no __V3_HOST_APP__ script in HTML"}

    js_payload = scripts[0]
    node_code = (
        """
const window = {};
"""
        + js_payload
        + """
try {
    const smf = window.__V3_HOST_APP__.state["$smf-/sys-betting-app-front"];
    if (!smf || !smf.codeToEval) {
        console.log(JSON.stringify({error: "no codeToEval"}));
        process.exit(0);
    }
    eval(smf.codeToEval);
    const bApp = window.__BETTING_APP__;
    const game = (bApp && bApp.pinia && bApp.pinia.game) ? bApp.pinia.game : {};
    const gamesById = game.gamesById || {};
    let g = gamesById[game.currentGameId];
    if (!g) {
        const keys = Object.keys(gamesById);
        if (keys.length > 0) g = gamesById[keys[0]];
    }
    if (!g) {
        console.log(JSON.stringify({error: "no game in store"}));
        process.exit(0);
    }

    let outcomes = [];
    // 1. Check parsed marketGroups if available
    let exactGroup = (game.marketGroups || []).find(grp => grp && grp.name === 'Exact Number Of Points');
    if (exactGroup && Array.isArray(exactGroup.marketColumns)) {
        for (let col of exactGroup.marketColumns) {
            if (Array.isArray(col)) {
                for (let m of col) {
                    outcomes.push({
                        name: m.name || "",
                        coefView: m.coefByView || m.cfView || m.cf || "",
                        isLocked: !!m.isLocked,
                        params: [],
                        parameter: m.parameter || null
                    });
                }
            }
        }
    }

    // 2. Check SSR unparsedEventGroups (groupId 3319 / shortGroupId 1322 = Exact Number Of Points)
    if (outcomes.length === 0 && Array.isArray(g.unparsedEventGroups)) {
        let rawExact = g.unparsedEventGroups.find(grp => grp && (grp.groupId === 3319 || grp.shortGroupId === 1322));
        if (rawExact && Array.isArray(rawExact.events)) {
            for (let col of rawExact.events) {
                if (Array.isArray(col)) {
                    for (let ev of col) {
                        outcomes.push({
                            name: ev.name || "",
                            coefView: ev.cfView || ev.cf || "",
                            isLocked: !!(ev.isLocked || ev.locked),
                            params: (ev.eventParams && Array.isArray(ev.eventParams.params)) ? ev.eventParams.params : [],
                            parameter: ev.parameter !== undefined ? ev.parameter : null
                        });
                    }
                }
            }
        }
    }

    console.log(JSON.stringify({
        id: g.id,
        team1: g.firstOpponentName,
        team2: g.secondOpponentName,
        score1: g.firstOpponentFullScore,
        score2: g.secondOpponentFullScore,
        period: g.currentPeriodLabel,
        quarter: g.currentPeriod,
        periodScores: g.periodScores || [],
        timeInSeconds: g.timeInSeconds,
        isNotStarted: !!g.isNotStarted,
        isGameOver: !!(g.isGameOver || g.isFinal),
        outcomes: outcomes
    }));
} catch (err) {
    console.log(JSON.stringify({error: err.toString()}));
}
"""
    )

    tmp_js = os.path.join(tempfile.gettempdir(), f"nba2k_ssr_{os.getpid()}.js")
    try:
        with open(tmp_js, "w", encoding="utf-8") as f:
            f.write(node_code)
        res = subprocess.run(["node", tmp_js], capture_output=True, text=True, encoding="utf-8", timeout=5)
        out = (res.stdout or "").strip()
        if out:
            return json.loads(out.splitlines()[-1])
        return {"error": f"empty node output ({res.stderr})"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            if os.path.exists(tmp_js):
                os.remove(tmp_js)
        except Exception:
            pass


def fetch_all_live_games_via_api(domain: str = "") -> list:
    """
    Fetch ALL simultaneous live NBA 2K26 CyberLeague games and their exact Barem (G=3319 / GS=1322)
    odds directly via 1xBet's LiveFeed JSON API (`Get1x2_VZip` + `GetGameZip`) in ~0.3s.
    Never misses a second or third simultaneous live match!
    """
    ctx = ssl._create_unverified_context()
    conn, host_hdr = _open_keepalive_results_conn(ctx)
    req_headers = {
        "Host": host_hdr,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Connection": "keep-alive",
    }

    list_path = "/service-api/LiveFeed/Get1x2_VZip?sports=3&champs=2935701&count=50&lng=en&mode=4"
    raw_list = []
    try:
        conn.request("GET", list_path, headers=req_headers)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="ignore")
        if resp.status == 200 and body:
            raw_list = json.loads(body).get("Value", []) or []
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        conn, host_hdr = _open_keepalive_results_conn(ctx)
        req_headers["Host"] = host_hdr
        try:
            conn.request("GET", list_path, headers=req_headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="ignore")
            if resp.status == 200 and body:
                raw_list = json.loads(body).get("Value", []) or []
        except Exception:
            pass

    parsed_games = []
    for item in raw_list:
        gid = item.get("I")
        if not gid:
            continue
        detail_path = (
            f"/service-api/LiveFeed/GetGameZip?id={gid}&lng=en&isSubGames=true"
            f"&GroupEvents=true&allEventsGroupSubGames=true&countevents=500&grMode=4&marketType=1"
        )
        g_val = None
        try:
            conn.request("GET", detail_path, headers=req_headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="ignore")
            if resp.status == 200 and body:
                g_val = json.loads(body).get("Value")
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            conn, host_hdr = _open_keepalive_results_conn(ctx)
            req_headers["Host"] = host_hdr
            try:
                conn.request("GET", detail_path, headers=req_headers)
                resp = conn.getresponse()
                body = resp.read().decode("utf-8", errors="ignore")
                if resp.status == 200 and body:
                    g_val = json.loads(body).get("Value")
            except Exception:
                pass

        src = g_val if isinstance(g_val, dict) else item
        sc = src.get("SC") or {}
        fs = sc.get("FS") or {}
        ps_raw = sc.get("PS") or []
        ps_sorted = sorted(ps_raw, key=lambda x: int(x.get("Key", 0) or 0))
        period_scores = []
        for p_entry in ps_sorted:
            val_d = p_entry.get("Value") or {}
            period_scores.append(
                {
                    "periodTitle": val_d.get("NF", ""),
                    "firstTeamScoreLabel": str(val_d.get("S1", 0) or 0),
                    "secondTeamScoreLabel": str(val_d.get("S2", 0) or 0),
                }
            )

        outcomes = []
        for ge_grp in src.get("GE") or []:
            if ge_grp.get("G") == 3319 or ge_grp.get("GS") == 1322:
                for col in ge_grp.get("E") or []:
                    if isinstance(col, list):
                        for ev in col:
                            outcomes.append(
                                {
                                    "name": "",
                                    "coefView": str(ev.get("CV") or ev.get("C") or ""),
                                    "isLocked": bool(ev.get("B") or ev.get("L")),
                                    "params": [],
                                    "parameter": ev.get("P"),
                                }
                            )

        cp = sc.get("CP")
        cps = str(sc.get("CPS", "") or "")
        info_i = str(sc.get("I", "") or "").lower()
        sls_str = str(sc.get("SLS", "") or "").lower()
        s1_val = int(fs.get("S1", 0) or 0)
        s2_val = int(fs.get("S2", 0) or 0)
        is_not_started = (
            (cp is None or int(cp or 0) == 0)
            or (s1_val == 0 and s2_val == 0)
            or "pre-match" in info_i
            or "starting" in sls_str
        )
        is_game_over = "final" in cps.lower() or "ended" in cps.lower() or bool(src.get("F"))

        parsed_games.append(
            {
                "id": gid,
                "team1": str(src.get("O1", "")).strip(),
                "team2": str(src.get("O2", "")).strip(),
                "score1": int(fs.get("S1", 0) or 0),
                "score2": int(fs.get("S2", 0) or 0),
                "period": cps,
                "quarter": int(cp) if cp is not None else 1,
                "periodScores": period_scores,
                "timeInSeconds": int(sc.get("TS", 0) or 0),
                "isNotStarted": is_not_started,
                "isGameOver": is_game_over,
                "outcomes": outcomes,
            }
        )

    try:
        conn.close()
    except Exception:
        pass

    return parsed_games


def fetch_active_match_urls(domain: str) -> list:
    """Fetch active NBA 2K26 CyberLeague match URLs from the league page in ~0.15s."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    ctx = ssl._create_unverified_context()
    league_url = f"{domain}/en/live/basketball/2935701-nba-2k26-cyber-league"
    req = urllib.request.Request(league_url, headers=headers)
    html = urllib.request.urlopen(req, context=ctx, timeout=8).read().decode("utf-8", errors="ignore")

    paths = re.findall(
        r"(/en/live/basketball/2935701-nba-2k26-cyber-league/\d+-[0-9a-z-]+)", html
    )
    unique_urls = []
    seen_ids = set()
    for p in paths:
        m = re.search(r"/(\d{6,})-", p)
        if m:
            gid = m.group(1)
            if gid not in seen_ids:
                seen_ids.add(gid)
                unique_urls.append(f"{domain}{p}")
    return unique_urls


def get_required_interval(elapsed_mins: float) -> int:
    """
    Polling schedule for 0 - 36.0 minutes:
    - 0 to 24m: 300s (5 min)
    - 24 to 30m: 180s (3 min)
    - 30 to 32m: 120s (2 min)
    - 32 to 34m: 60s (1 min)
    - 34 to 35m: 30s
    - 35 to 36m: 10s (Barem closing window!)
    - 36m+: 60s (Only waiting for final score, no CSV logging)
    """
    if elapsed_mins >= 36.0:
        return 60
    elif elapsed_mins >= 35.0:
        return 10
    elif elapsed_mins >= 34.0:
        return 30
    elif elapsed_mins >= 32.0:
        return 60
    elif elapsed_mins >= 30.0:
        return 120
    elif elapsed_mins >= 24.0:
        return 180
    else:
        return 300


def merge_cloud_backups():
    """Merge 24/7 GitHub Actions cloud backup CSVs and /tmp runtime cache into local CSVs without git conflicts."""
    for sym_src in [
        os.path.join(tempfile.gettempdir(), "nba2k26_symmetry_runtime_v2.csv"),
        os.path.join(BASE_DIR, "cloud_symmetry_backup.csv"),
    ]:
        if os.path.exists(sym_src) and os.path.getsize(sym_src) > 0:
            try:
                with open(sym_src, "r", encoding="utf-8") as f:
                    c_rows = list(csv.DictReader(f))
                for r in c_rows:
                    upsert_symmetry_log_row(r)
            except Exception:
                pass
    cloud_m = os.path.join(BASE_DIR, "cloud_matches_backup.csv")
    if os.path.exists(cloud_m) and os.path.getsize(cloud_m) > 0:
        try:
            df_c = pd.read_csv(cloud_m)
            if os.path.exists(MATCHES_CSV_PATH) and os.path.getsize(MATCHES_CSV_PATH) > 0:
                df_l = pd.read_csv(MATCHES_CSV_PATH)
                df_merged = (
                    pd.concat([df_c, df_l], ignore_index=True)
                    .sort_values("date")
                    .drop_duplicates(subset=["game_id"], keep="last")
                )
                df_merged.to_csv(MATCHES_CSV_PATH, index=False, encoding="utf-8")
        except Exception:
            pass


def run_auto_scraper(force_manual=False):
    init_csv()
    merge_cloud_backups()
    start_ts = time.time()

    domain = discover_active_domain()
    print(f"[v2.0 Engine] Active 1xBet/Mirror domain: {domain}")

    api_games = fetch_all_live_games_via_api(domain)
    if api_games:
        print(f"Found {len(api_games)} active NBA 2K26 CyberLeague matches via LiveFeed API.")
    else:
        try:
            match_urls = fetch_active_match_urls(domain)
        except Exception as e:
            print(f"Primary domain fetch failed ({e}), refreshing mirror...")
            domain = discover_active_domain(force_refresh=True)
            try:
                match_urls = fetch_active_match_urls(domain)
            except Exception as e2:
                print(f"Mirror fetch failed ({e2})")
                match_urls = []
        print(f"Found {len(match_urls)} active NBA 2K26 CyberLeague matches via SSR fallback.")
        for match_url in match_urls:
            d = extract_ssr_match_data(match_url)
            if "error" not in d:
                api_games.append(d)

    # Load historical matches (may be empty on clean slate!)
    if os.path.exists(MATCHES_CSV_PATH) and os.path.getsize(MATCHES_CSV_PATH) > 0:
        try:
            df_matches = pd.read_csv(MATCHES_CSV_PATH)
        except Exception:
            df_matches = pd.DataFrame()
    else:
        df_matches = pd.DataFrame()

    # Load timestamps
    log_timestamps = {}
    if os.path.exists(TS_FILE):
        try:
            with open(TS_FILE, "r", encoding="utf-8") as f:
                log_timestamps = json.load(f)
        except Exception:
            log_timestamps = {}

    recommended_poll_speed = 180
    live_cards = []

    for data in api_games:

        game_id_str = f"CB-{data.get('id')}"
        team1 = str(data.get("team1", "")).strip()
        team2 = str(data.get("team2", "")).strip()
        total_home = int(data.get("score1") or 0)
        total_away = int(data.get("score2") or 0)

        if data.get("isNotStarted") or (total_home == 0 and total_away == 0 and not data.get("periodScores")):
            print(f"  PRE-GAME: {team1} vs {team2} has not started yet.")
            live_cards.append(
                {
                    "game_id": game_id_str,
                    "team_home": team1,
                    "team_away": team2,
                    "home_score": 0,
                    "away_score": 0,
                    "quarter": 1,
                    "elapsed_mins": 0.0,
                    "remaining_mins": 12.0,
                    "status": "PRE_GAME",
                    "completed_quarters": [],
                    "live_q_score": "0-0",
                    "odds": {},
                    "barem_probs": {},
                    "predicted_total": 212.0,
                    "predicted_home_final": 106.0,
                    "predicted_away_final": 106.0,
                    "best_cluster_name": "201-220",
                    "best_cluster_prob": 0.0,
                    "value_range": "-",
                    "highest_prob_range": "-",
                    "dutch_odd": 0.0,
                    "is_barem_open": False,
                    "updated_at": datetime.now(TR_TZ).strftime("%H:%M:%S"),
                }
            )
            continue

        raw_q = data.get("quarter")
        try:
            current_quarter = int(raw_q) if raw_q is not None else 1
        except Exception:
            current_quarter = 1

        period_scores = data.get("periodScores") or []
        if len(period_scores) > current_quarter:
            current_quarter = len(period_scores)
        current_quarter = max(1, min(5, current_quarter))

        completed_quarters = []
        home_live_score = 0
        away_live_score = 0
        for q_idx, ps in enumerate(period_scores):
            q_num = q_idx + 1
            h_s = int(ps.get("firstTeamScoreLabel", 0) or 0)
            a_s = int(ps.get("secondTeamScoreLabel", 0) or 0)
            if q_num < current_quarter:
                completed_quarters.append((h_s, a_s))
            elif q_num == current_quarter:
                home_live_score = h_s
                away_live_score = a_s

        time_in_seconds = float(data.get("timeInSeconds") or 0.0)
        if time_in_seconds > 720.0:
            total_match_elapsed = time_in_seconds / 60.0
            quarter_elapsed = max(0.0, total_match_elapsed - (current_quarter - 1) * 12.0)
        elif time_in_seconds > 0:
            quarter_elapsed = time_in_seconds / 60.0
            total_match_elapsed = (current_quarter - 1) * 12.0 + quarter_elapsed
        else:
            # Estimate from current quarter score if clock temporarily 0
            est_q_mins = min(12.0, (home_live_score + away_live_score) / 4.4)
            quarter_elapsed = est_q_mins
            total_match_elapsed = (current_quarter - 1) * 12.0 + quarter_elapsed

        remaining_mins = max(0.0, min(12.0, 12.0 - quarter_elapsed))
        check_time = f"Q{current_quarter} - Minute {int(total_match_elapsed)}"

        # Map 9 Barem odds
        odds_dict = {}
        for o in data.get("outcomes", []):
            if o.get("isLocked"):
                continue
            b_name = _map_barem_name(o.get("name", ""), o.get("params", []), o.get("parameter"))
            if not b_name:
                continue
            coef_str = str(o.get("coefView", "")).strip()
            if coef_str and coef_str != "-":
                try:
                    val = float(coef_str.replace(",", "."))
                    if val > 1.0:
                        odds_dict[b_name] = val
                except Exception:
                    pass

        full_odds = {b[0]: odds_dict.get(b[0], 99.0) for b in BAREM_RANGES}
        is_barem_open = any(v < 90.0 for v in full_odds.values()) and total_match_elapsed < 36.0

        # Run Monte Carlo & Symmetry Analysis
        res_pred = predict_live_inplay(
            df=df_matches,
            team_home=team1,
            team_away=team2,
            current_quarter=min(4, current_quarter),
            home_live_score=home_live_score,
            away_live_score=away_live_score,
            remaining_mins=remaining_mins,
            completed_quarters=completed_quarters[:3],
            quarter_length=12.0,
        )
        barem_probs = res_pred.get("barem_probs", {})
        sym_info = analyze_barem_symmetry(full_odds, barem_probs)

        # Q3 -> Q4 Burst / Collapse Signal
        if len(completed_quarters) >= 3:
            q3_total_for_signal = completed_quarters[2][0] + completed_quarters[2][1]
        elif current_quarter == 3:
            el_q3 = max(1.0, 12.0 - remaining_mins)
            q3_total_for_signal = int(round(((home_live_score + away_live_score) / el_q3) * 12.0))
        else:
            q3_total_for_signal = 53
        q3_q4_sig = build_q3_q4_streak_signal(df_matches, team1, team2, q3_score=q3_total_for_signal)

        req_interval = get_required_interval(total_match_elapsed)
        recommended_poll_speed = min(recommended_poll_speed, req_interval)

        # Save to live dashboard state card
        live_cards.append(
            {
                "game_id": game_id_str,
                "team_home": team1,
                "team_away": team2,
                "home_score": total_home,
                "away_score": total_away,
                "quarter": current_quarter,
                "elapsed_mins": round(total_match_elapsed, 1),
                "remaining_mins": round(remaining_mins, 1),
                "check_time": check_time,
                "status": "BAREM_OPEN" if is_barem_open else "WAITING_FINAL",
                "completed_quarters": completed_quarters,
                "live_q_score": f"{home_live_score}-{away_live_score}",
                "odds": full_odds,
                "barem_probs": barem_probs,
                "clusters": res_pred.get("clusters", {}),
                "predicted_total": res_pred.get("predicted_total", 0.0),
                "predicted_home_final": res_pred.get("predicted_home_final", 0.0),
                "predicted_away_final": res_pred.get("predicted_away_final", 0.0),
                "best_cluster_name": res_pred.get("best_cluster_name", ""),
                "best_cluster_prob": res_pred.get("best_cluster_prob", 0.0),
                "h2h_sample_size": res_pred.get("sample_size", 0),
                "q3_q4_signal": q3_q4_sig,
                "value_range": sym_info["value_range"],
                "highest_prob_range": sym_info["highest_prob_range"],
                "bookie_fav_1": sym_info["bookie_fav_1"],
                "bookie_fav_1_odd": sym_info["bookie_fav_1_odd"],
                "bookie_fav_2": sym_info["bookie_fav_2"],
                "bookie_fav_2_odd": sym_info["bookie_fav_2_odd"],
                "symmetry_gap": sym_info["symmetry_gap"],
                "is_symmetric_trap": sym_info["is_symmetric_trap"],
                "dutch_odd": sym_info["dutch_odd"],
                "best_ev_pct": sym_info["best_ev_pct"],
                "is_barem_open": is_barem_open,
                "updated_at": datetime.now(TR_TZ).strftime("%H:%M:%S"),
            }
        )

        # STRICT RULE: Never log snapshots to symmetry_analysis_log.csv after Minute 36.0 or when barem is closed!
        if total_match_elapsed >= 36.0 or not is_barem_open:
            print(
                f"  [SKIP CSV] {team1} vs {team2} ({check_time}, Score {total_home}-{total_away}): "
                f"Minute >= 36 or Barem closed. Waiting for final result only."
            )
            continue

        now_ts = time.time()
        last_game_ts = log_timestamps.get(game_id_str, 0)
        elapsed_since_last = now_ts - last_game_ts

        if force_manual or (elapsed_since_last >= req_interval - 5):
            row = {
                "game_id": game_id_str,
                "game_name": f"{team1} vs {team2}",
                "date": datetime.now(TR_TZ).strftime("%Y-%m-%d"),
                "check_time": check_time,
                "current_score": f"{total_home}-{total_away}",
                "odds_160_or_less": full_odds["160 Or Less"],
                "odds_161_170": full_odds["161-170"],
                "odds_171_180": full_odds["171-180"],
                "odds_181_190": full_odds["181-190"],
                "odds_191_200": full_odds["191-200"],
                "odds_201_210": full_odds["201-210"],
                "odds_211_220": full_odds["211-220"],
                "odds_221_230": full_odds["221-230"],
                "odds_231_plus": full_odds["231+"],
                "model_prob_160_or_less": barem_probs.get("160 Or Less", 0.0),
                "model_prob_161_170": barem_probs.get("161-170", 0.0),
                "model_prob_171_180": barem_probs.get("171-180", 0.0),
                "model_prob_181_190": barem_probs.get("181-190", 0.0),
                "model_prob_191_200": barem_probs.get("191-200", 0.0),
                "model_prob_201_210": barem_probs.get("201-210", 0.0),
                "model_prob_211_220": barem_probs.get("211-220", 0.0),
                "model_prob_221_230": barem_probs.get("221-230", 0.0),
                "model_prob_231_plus": barem_probs.get("231+", 0.0),
                "highest_prob_range": sym_info["highest_prob_range"],
                "value_range": sym_info["value_range"],
                "best_cluster_name": res_pred.get("best_cluster_name", ""),
                "best_cluster_prob": res_pred.get("best_cluster_prob", 0.0),
                "actual_score": "",
                "actual_range": "",
                "status": "PENDING",
            }
            upsert_symmetry_log_row(row)
            log_timestamps[game_id_str] = now_ts
            with open(TS_FILE, "w", encoding="utf-8") as f:
                json.dump(log_timestamps, f)

            print(
                f"  [LOGGED] {team1} vs {team2} | {check_time} | Score: {total_home}-{total_away} | "
                f"Fav: {sym_info['highest_prob_range']} ({sym_info['bookie_fav_1_odd']}) | "
                f"Value: {sym_info['value_range']} | Cluster: {res_pred.get('best_cluster_name')} ({res_pred.get('best_cluster_prob')}%)"
            )
        else:
            rem_wait = int(req_interval - elapsed_since_last)
            print(f"  [THROTTLED] {team1} vs {team2} ({check_time}): next log in {rem_wait}s.")

    # Keep live dashboard refreshed every 20s (or 10s in 35-36m closing window) so new games appear immediately
    live_poll_speed = min(20, recommended_poll_speed)

    # Save live cards state for instant dashboard rendering
    state_payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        "domain": domain,
        "recommended_poll_speed": live_poll_speed,
        "duration_sec": round(time.time() - start_ts, 2),
        "live_matches": live_cards,
    }
    with open(LIVE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state_payload, f, ensure_ascii=False, indent=2)

    # Check if any match has finished and settle checkpoints immediately
    try:
        pull_today_results(only_tracked=False, days_back=1)
        auto_settle_checkpoints()
    except Exception as e:
        print(f"Auto-settle warning: {e}")

    print(f"[POLL_SPEED] {live_poll_speed} (Completed in {round(time.time() - start_ts, 2)}s)")
    return live_poll_speed


run_single_cycle = run_auto_scraper


if __name__ == "__main__":
    force_flag = "--manual" in sys.argv
    run_auto_scraper(force_manual=force_flag)
