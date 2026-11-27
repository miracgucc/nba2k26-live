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
import threading
from datetime import datetime, timedelta, timezone

_FILE_LOCK = threading.RLock()
_LAST_SETTLE_TS = {}

TR_TZ = timezone(timedelta(hours=3))

import numpy as np
import pandas as pd

from period_predictor import (
    BAREM_RANGES,
    analyze_barem_symmetry,
    analyze_smart_35m_decision,
    build_q3_q4_streak_signal,
    get_csv_headers,
    get_league_config,
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
_IS_GH_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


def _get_scraper_paths(league: str = "nba") -> dict:
    """Resolve runtime/GH Actions paths for a given league without cross-contamination."""
    cfg = get_league_config(league)
    files = cfg["files"]
    key = cfg["key"]
    csv_path = os.path.join(BASE_DIR, files["log_seed"] if _IS_GH_ACTIONS else files["log_runtime"])
    matches_path = os.path.join(BASE_DIR, files["matches_seed"] if _IS_GH_ACTIONS else files["matches_runtime"])
    if _IS_GH_ACTIONS:
        state_name = files["state_file"]
    elif key == "nba":
        state_name = "runtime_live_state.json"
    elif key == "nba2k27":
        state_name = "runtime_nba2k27_live_state.json"
    else:
        state_name = "runtime_euro_live_state.json"
    if key == "nba":
        ts_name = "match_log_timestamps.json"
    elif key == "nba2k27":
        ts_name = "nba2k27_match_log_timestamps.json"
    else:
        ts_name = "euro_match_log_timestamps.json"
    return {
        "csv_path": csv_path,
        "seed_csv": os.path.join(BASE_DIR, files["log_seed"]),
        "cloud_csv": os.path.join(BASE_DIR, files["log_cloud"]),
        "matches_path": matches_path,
        "seed_matches": os.path.join(BASE_DIR, files["matches_seed"]),
        "cloud_matches": os.path.join(BASE_DIR, files["matches_cloud"]),
        "state_file": os.path.join(BASE_DIR, state_name),
        "ts_file": os.path.join(BASE_DIR, ts_name),
    }


_NBA_PATHS = _get_scraper_paths("nba")
CSV_PATH = _NBA_PATHS["csv_path"]
MATCHES_CSV_PATH = _NBA_PATHS["matches_path"]
CACHE_FILE = os.path.join(BASE_DIR, "match_detail_cache.json")
TS_FILE = _NBA_PATHS["ts_file"]
LIVE_STATE_FILE = _NBA_PATHS["state_file"]
CSV_HEADERS = get_csv_headers("nba")


def init_csv(league: str = "nba"):
    paths = _get_scraper_paths(league)
    csv_path = paths["csv_path"]
    if (not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0) and not _IS_GH_ACTIONS:
        if os.path.exists(paths["seed_csv"]) and os.path.getsize(paths["seed_csv"]) > 0:
            try:
                import shutil
                shutil.copyfile(paths["seed_csv"], csv_path)
            except Exception:
                pass
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        headers = get_csv_headers(league)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def _extract_minute_num(check_time_str: str) -> int:
    m = re.search(r"Minute\s+(\d+)", str(check_time_str or ""))
    return int(m.group(1)) if m else 0


def upsert_symmetry_log_row(new_row: dict, league: str = "nba"):
    """Thread-safe wrapper: serialises all read-modify-write cycles on a league's symmetry log CSV."""
    with _FILE_LOCK:
        _upsert_symmetry_log_rows_unlocked([new_row], league=league)


def _upsert_symmetry_log_row_unlocked(new_row: dict, league: str = "nba"):
    _upsert_symmetry_log_rows_unlocked([new_row], league=league)


def _is_valid_cyber_match(game_name: str) -> bool:
    """Reject non-cyber leagues (e.g. IPBL, TBL, Chinese Women) leaked by mirror fallbacks."""
    gn = str(game_name or "").lower()
    return "(cyber)" in gn


def _score_sum(sc_str: str) -> int:
    try:
        a, b = str(sc_str or "").split("-")
        return int(a.strip()) + int(b.strip())
    except Exception:
        return -1


def _upsert_symmetry_log_rows_unlocked(new_rows: list, league: str = "nba"):
    """
    Batch upsert checkpoint rows by (game_id, check_time) so simultaneous games never create
    duplicate minute rows, foreign non-cyber matches are blocked, and halftime score rollbacks
    are rejected automatically.
    """
    if not new_rows:
        return
    init_csv(league)
    cfg = get_league_config(league)
    csv_path = _get_scraper_paths(league)["csv_path"]
    headers = get_csv_headers(league)
    min_clean_gid = int(cfg.get("min_clean_gid", 0)) if cfg["key"] == "nba" else 770000000

    existing_rows = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
        except Exception:
            existing_rows = []

    def _gid_int(g_str: str) -> int:
        m = re.search(r"(\d+)", str(g_str or ""))
        return int(m.group(1)) if m else 0

    dedup_map = {}
    game_order = []
    changed = False
    for r in existing_rows:
        gid = str(r.get("game_id", "")).strip()
        ctime = str(r.get("check_time", "")).strip()
        if not gid or _gid_int(gid) < min_clean_gid or not _is_valid_cyber_match(r.get("game_name", "")):
            changed = True
            continue
        if gid not in game_order:
            game_order.append(gid)
        dedup_map[(gid, ctime)] = {h: r.get(h, "") for h in headers}

    for new_row in new_rows:
        new_gid = str(new_row.get("game_id", "")).strip()
        if _gid_int(new_gid) < min_clean_gid or not _is_valid_cyber_match(new_row.get("game_name", "")):
            continue
        new_ctime = str(new_row.get("check_time", "")).strip()
        new_m = _extract_minute_num(new_ctime)
        new_tot = _score_sum(new_row.get("current_score", ""))

        # Reject transient halftime / quarter-transition score reset glitches (cur_tot < prev_tot)
        if new_tot >= 0:
            max_prior_tot = max(
                (
                    _score_sum(er.get("current_score", ""))
                    for (eg, ec), er in dedup_map.items()
                    if eg == new_gid and _extract_minute_num(ec) <= new_m
                ),
                default=-1,
            )
            if max_prior_tot > 0 and new_tot < max_prior_tot:
                continue

        if new_gid not in game_order:
            game_order.append(new_gid)

        clean_new = {h: new_row.get(h, "") for h in headers}
        if (new_gid, new_ctime) in dedup_map:
            prev = dedup_map[(new_gid, new_ctime)]
            if prev.get("status") == "COMPLETED":
                clean_new["actual_score"] = prev.get("actual_score", "")
                clean_new["actual_range"] = prev.get("actual_range", "")
                clean_new["status"] = "COMPLETED"
        dedup_map[(new_gid, new_ctime)] = clean_new
        changed = True

    final_rows = []
    for gid in game_order:
        g_rows = [r for (g, _), r in dedup_map.items() if g == gid]
        g_rows.sort(key=lambda x: _extract_minute_num(x.get("check_time", "")))
        # Self-heal transient halftime score resets (e.g. Q2 - Minute 24 dropping 103 -> 55)
        max_sc_seen = -1
        for gr in g_rows:
            cur_sc_sum = _score_sum(gr.get("current_score", ""))
            if cur_sc_sum >= 0 and max_sc_seen > 0 and cur_sc_sum < max_sc_seen - 2:
                changed = True
                continue
            if cur_sc_sum > max_sc_seen:
                max_sc_seen = cur_sc_sum
            final_rows.append(gr)

    if not changed:
        return

    tmp_main = csv_path + ".tmp"
    with open(tmp_main, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_rows)
    os.replace(tmp_main, csv_path)


def _map_barem_name(raw_name: str, params_list: list, parameter_val, league: str = "nba") -> str:
    """Map 1xBet / Fansport event parameters or market name to a league's Barem brackets."""
    joined_params = "-".join(str(x).strip() for x in (params_list or []) if str(x).strip())
    rn = str(raw_name or "").strip().lower()

    if str(league or "nba").lower() == "euro":
        if joined_params in {"131-140", "141-150", "151-160", "161-170"}:
            return joined_params
        if "130" in joined_params or ("131" in joined_params and "140" not in joined_params):
            return "131'den Az"
        if "171" in joined_params or ("170" in joined_params and "161" not in joined_params):
            return "170'den Fazla"

        if parameter_val is not None:
            try:
                p_float = float(parameter_val)
                if p_float <= 130.5:
                    return "131'den Az"
                elif 130.9 <= p_float <= 132.5:
                    return "131-140"
                elif 140.9 <= p_float <= 142.5:
                    return "141-150"
                elif 150.9 <= p_float <= 152.5:
                    return "151-160"
                elif 160.9 <= p_float <= 162.5:
                    return "161-170"
                elif p_float >= 170.5:
                    return "170'den Fazla"
            except Exception:
                pass

        for bracket in ("131-140", "141-150", "151-160", "161-170"):
            if bracket in rn:
                return bracket
        if "130" in rn or "az" in rn or "or less" in rn:
            return "131'den Az"
        if "171" in rn or "170" in rn or "fazla" in rn or "more" in rn:
            return "170'den Fazla"
        if "131" in rn:
            return "131-140"
        if "141" in rn:
            return "141-150"
        if "151" in rn:
            return "151-160"
        if "161" in rn:
            return "161-170"
        return ""

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


def fetch_all_live_games_via_api(domain: str = "", league: str = "nba") -> list:
    """
    Fetch ALL simultaneous live CyberLeague / Cyber Euro League games and their exact Barem (G=3319 / GS=1322)
    odds directly via 1xBet's LiveFeed JSON API (`Get1x2_VZip` + `GetGameZip`) in ~0.3s.
    """
    cfg = get_league_config(league)
    champ_id = cfg["champ_id"]
    ctx = ssl._create_unverified_context()
    conn, host_hdr = _open_keepalive_results_conn(ctx)
    req_headers = {
        "Host": host_hdr,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Connection": "keep-alive",
    }

    list_path = f"/service-api/LiveFeed/Get1x2_VZip?sports=3&champs={champ_id}&count=50&lng=en&mode=4"
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
    target_champ = str(champ_id).strip()
    for item in raw_list:
        gid = item.get("I")
        if not gid:
            continue
        item_li = str(item.get("LI") or "").strip()
        if item_li and item_li != target_champ:
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
        src_li = str(src.get("LI") or item_li or "").strip()
        if src_li and src_li != target_champ:
            continue
        t1_name = str(src.get("O1", "")).strip()
        t2_name = str(src.get("O2", "")).strip()
        if not (_is_valid_cyber_match(t1_name) and _is_valid_cyber_match(t2_name)):
            continue
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


def fetch_active_match_urls(domain: str, league: str = "nba") -> list:
    """Fetch active league match URLs from the league page in ~0.15s."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    ctx = ssl._create_unverified_context()
    _lk = str(league or "nba").lower()
    if _lk == "euro":
        slug = "2983166-nba-2k26-cyber-euro-league"
    elif _lk == "nba2k27":
        slug = "3125701-nba-2k27-cyber-league"
    else:
        slug = "2935701-nba-2k26-cyber-league"
    league_url = f"{domain}/en/live/basketball/{slug}"
    req = urllib.request.Request(league_url, headers=headers)
    html = urllib.request.urlopen(req, context=ctx, timeout=8).read().decode("utf-8", errors="ignore")

    paths = re.findall(
        rf"(/en/live/basketball/{re.escape(slug)}/\d+-[0-9a-z-]+)", html
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


def get_required_interval(elapsed_mins: float, close_min: float = 36.0) -> int:
    """
    Adaptive polling schedule relative to league Barem close minute (36.0 NBA / 30.0 Euro):
    - close_min - 1 to close_min: 10s (Barem closing decision window!)
    - close_min - 2 to close_min - 1: 30s
    - close_min - 4 to close_min - 2: 60s
    - close_min - 6 to close_min - 4: 120s
    - close_min - 12 to close_min - 6: 180s
    - earlier: 300s
    - close_min+: 60s (Waiting for final score, no CSV logging)
    """
    if elapsed_mins >= close_min:
        return 60
    elif elapsed_mins >= close_min - 1.0:
        return 10
    elif elapsed_mins >= close_min - 2.0:
        return 30
    elif elapsed_mins >= close_min - 4.0:
        return 60
    elif elapsed_mins >= close_min - 6.0:
        return 120
    elif elapsed_mins >= close_min - 12.0:
        return 180
    else:
        return 300


_LAST_MERGE_SIG = {}


def merge_cloud_backups(league: str = "nba"):
    """
    Merge the 24/7 GitHub Actions cloud backup CSVs into the local league CSVs.
    Runs only when a backup file actually changed (size/mtime signature), under a global lock,
    with atomic writes so concurrent UI/engine threads can never truncate or corrupt the log.
    """
    paths = _get_scraper_paths(league)
    cloud_sym = paths["cloud_csv"]
    cloud_m = paths["cloud_matches"]
    matches_path = paths["matches_path"]

    def _sig(p):
        try:
            st_ = os.stat(p)
            return (st_.st_size, int(st_.st_mtime))
        except Exception:
            return None

    cur_sig = (_sig(cloud_sym), _sig(cloud_m))
    with _FILE_LOCK:
        if _LAST_MERGE_SIG.get(league) == cur_sig:
            return
        if os.path.exists(cloud_sym) and os.path.getsize(cloud_sym) > 0:
            try:
                with open(cloud_sym, "r", encoding="utf-8") as f:
                    c_rows = list(csv.DictReader(f))
                if c_rows:
                    _upsert_symmetry_log_rows_unlocked(c_rows, league=league)
            except Exception:
                pass
        if os.path.exists(cloud_m) and os.path.getsize(cloud_m) > 0:
            try:
                df_c = pd.read_csv(cloud_m)
                if os.path.exists(matches_path) and os.path.getsize(matches_path) > 0:
                    df_l = pd.read_csv(matches_path)
                    df_merged = pd.concat([df_c, df_l], ignore_index=True)
                else:
                    df_merged = df_c.copy()
                if "game_id" in df_merged.columns and "total_score" in df_merged.columns:
                    df_merged["game_id"] = df_merged["game_id"].astype(str).str.strip()
                    df_merged = df_merged[df_merged["game_id"].str.startswith("CB-")].copy()
                    for _mc in ("home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"):
                        if _mc in df_merged.columns:
                            df_merged[_mc] = pd.to_numeric(df_merged[_mc], errors="coerce")
                    df_merged = df_merged.dropna(subset=["home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"]).copy()
                    df_merged = df_merged[df_merged["total_score"] > 0].copy()
                    df_merged = (
                        df_merged.sort_values("date")
                        .drop_duplicates(subset=["game_id"], keep="last")
                    )
                tmp_m = matches_path + ".tmp"
                df_merged.to_csv(tmp_m, index=False, encoding="utf-8")
                os.replace(tmp_m, matches_path)
            except Exception:
                pass
        _LAST_MERGE_SIG[league] = cur_sig


def _scrape_single_league(domain: str, league: str = "nba", force_manual: bool = False) -> int:
    cfg = get_league_config(league)
    paths = _get_scraper_paths(league)
    tag = cfg["key"].upper()
    q_len = float(cfg["quarter_length"])
    q_secs = q_len * 60.0
    close_min = float(cfg["close_min"])
    pace_ppm = float(cfg.get("pace_pts_per_min", 4.4))
    barem_ranges = cfg["barem_ranges"]
    default_cluster = cfg["cluster_ranges"][len(cfg["cluster_ranges"]) // 2][0]

    init_csv(league)
    merge_cloud_backups(league)
    start_ts = time.time()

    api_games = fetch_all_live_games_via_api(domain, league=league)
    if api_games:
        print(f"[{tag}] Found {len(api_games)} active {cfg['short_name']} matches via LiveFeed API.")
    else:
        try:
            match_urls = fetch_active_match_urls(domain, league=league)
        except Exception as e:
            print(f"[{tag}] Primary domain fetch failed ({e}), refreshing mirror...")
            domain = discover_active_domain(force_refresh=True)
            try:
                match_urls = fetch_active_match_urls(domain, league=league)
            except Exception as e2:
                print(f"[{tag}] Mirror fetch failed ({e2})")
                match_urls = []
        print(f"[{tag}] Found {len(match_urls)} active {cfg['short_name']} matches via SSR fallback.")
        for match_url in match_urls:
            d = extract_ssr_match_data(match_url)
            if "error" not in d:
                api_games.append(d)

    matches_path = paths["matches_path"]
    if not os.path.exists(matches_path) and os.path.exists(paths["seed_matches"]) and not _IS_GH_ACTIONS:
        try:
            import shutil
            shutil.copyfile(paths["seed_matches"], matches_path)
        except Exception:
            pass
    if os.path.exists(matches_path) and os.path.getsize(matches_path) > 0:
        try:
            df_matches = pd.read_csv(matches_path)
        except Exception:
            df_matches = pd.DataFrame()
    else:
        df_matches = pd.DataFrame()

    ts_file = paths["ts_file"]
    log_timestamps = {}
    if os.path.exists(ts_file):
        try:
            with open(ts_file, "r", encoding="utf-8") as f:
                log_timestamps = json.load(f)
        except Exception:
            log_timestamps = {}

    recommended_poll_speed = 180
    live_cards = []

    for data in api_games:
        game_id_str = f"CB-{data.get('id')}"
        team1 = str(data.get("team1", "")).strip()
        team2 = str(data.get("team2", "")).strip()
        if not (_is_valid_cyber_match(team1) and _is_valid_cyber_match(team2)):
            continue
        total_home = int(data.get("score1") or 0)
        total_away = int(data.get("score2") or 0)

        if data.get("isNotStarted") or (total_home == 0 and total_away == 0 and not data.get("periodScores")):
            print(f"  [{tag}] PRE-GAME: {team1} vs {team2} has not started yet.")
            live_cards.append(
                {
                    "game_id": game_id_str,
                    "league": cfg["key"],
                    "team_home": team1,
                    "team_away": team2,
                    "home_score": 0,
                    "away_score": 0,
                    "quarter": 1,
                    "elapsed_mins": 0.0,
                    "remaining_mins": q_len,
                    "status": "PRE_GAME",
                    "completed_quarters": [],
                    "live_q_score": "0-0",
                    "odds": {},
                    "barem_probs": {},
                    "predicted_total": round(cfg["prior_q"] * 4.0, 1),
                    "predicted_home_final": round(cfg["prior_q"] * 2.0, 1),
                    "predicted_away_final": round(cfg["prior_q"] * 2.0, 1),
                    "best_cluster_name": default_cluster,
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

        # Recover true cumulative score if BetB2B briefly emits a quarter-only total at halftime
        ps_cum_h = sum(q[0] for q in completed_quarters) + home_live_score
        ps_cum_a = sum(q[1] for q in completed_quarters) + away_live_score
        if (ps_cum_h + ps_cum_a) > (total_home + total_away) and ps_cum_h >= total_home and ps_cum_a >= total_away:
            total_home = ps_cum_h
            total_away = ps_cum_a

        time_in_seconds = float(data.get("timeInSeconds") or 0.0)
        if time_in_seconds > q_secs:
            total_match_elapsed = time_in_seconds / 60.0
            quarter_elapsed = max(0.0, total_match_elapsed - (current_quarter - 1) * q_len)
        elif time_in_seconds > 0:
            quarter_elapsed = time_in_seconds / 60.0
            total_match_elapsed = (current_quarter - 1) * q_len + quarter_elapsed
        else:
            est_q_mins = min(q_len, (home_live_score + away_live_score) / pace_ppm)
            quarter_elapsed = est_q_mins
            total_match_elapsed = (current_quarter - 1) * q_len + quarter_elapsed

        remaining_mins = max(0.0, min(q_len, q_len - quarter_elapsed))
        check_time = f"Q{current_quarter} - Minute {int(total_match_elapsed)}"

        odds_dict = {}
        for o in data.get("outcomes", []):
            if o.get("isLocked"):
                continue
            b_name = _map_barem_name(o.get("name", ""), o.get("params", []), o.get("parameter"), league=league)
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

        full_odds = {b[0]: odds_dict.get(b[0], 99.0) for b in barem_ranges}

        res_pred = predict_live_inplay(
            df=df_matches,
            team_home=team1,
            team_away=team2,
            current_quarter=min(4, current_quarter),
            home_live_score=home_live_score,
            away_live_score=away_live_score,
            remaining_mins=remaining_mins,
            completed_quarters=completed_quarters[:3],
            quarter_length=q_len,
            league=league,
        )
        barem_probs = res_pred.get("barem_probs", {})
        _cur_h_sc = sum(int(q[0]) for q in completed_quarters[:3]) + int(home_live_score)
        _cur_a_sc = sum(int(q[1]) for q in completed_quarters[:3]) + int(away_live_score)
        sym_info = analyze_barem_symmetry(
            full_odds,
            barem_probs,
            league=league,
            proj_tot=float(res_pred.get("predicted_total", 0.0) or 0.0),
            score_gap=abs(_cur_h_sc - _cur_a_sc),
        )
        is_barem_open = (
            any(v < 90.0 for v in full_odds.values()) or bool(sym_info.get("is_locked_fav", False))
        ) and total_match_elapsed < close_min

        if len(completed_quarters) >= 3:
            q3_total_for_signal = completed_quarters[2][0] + completed_quarters[2][1]
        elif current_quarter == 3:
            el_q3 = max(1.0, q_len - remaining_mins)
            q3_total_for_signal = int(round(((home_live_score + away_live_score) / el_q3) * q_len))
        else:
            q3_total_for_signal = int(round(cfg["prior_q"]))
        q3_q4_sig = build_q3_q4_streak_signal(df_matches, team1, team2, q3_score=q3_total_for_signal, league=league)

        req_interval = get_required_interval(total_match_elapsed, close_min=close_min)
        recommended_poll_speed = min(recommended_poll_speed, req_interval)

        live_cards.append(
            {
                "game_id": game_id_str,
                "league": cfg["key"],
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
                "best_cluster_name": sym_info.get("best_cluster_name") or res_pred.get("best_cluster_name", ""),
                "best_cluster_prob": sym_info.get("best_cluster_prob") or res_pred.get("best_cluster_prob", 0.0),
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

        # STRICT RULE: Never log snapshots to symmetry log after close_min or when barem is closed!
        if total_match_elapsed >= close_min or not is_barem_open:
            print(
                f"  [{tag} SKIP CSV] {team1} vs {team2} ({check_time}, Score {total_home}-{total_away}): "
                f"Minute >= {int(close_min)} or Barem closed. Waiting for final result only."
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
            }
            for b_name, col_name in cfg["barem_odd_cols"].items():
                row[col_name] = full_odds[b_name]
            for b_name, col_name in cfg["barem_prob_cols"].items():
                row[col_name] = barem_probs.get(b_name, 0.0)
            row.update(
                {
                    "highest_prob_range": sym_info["highest_prob_range"],
                    "value_range": sym_info["value_range"],
                    "best_cluster_name": res_pred.get("best_cluster_name", ""),
                    "best_cluster_prob": res_pred.get("best_cluster_prob", 0.0),
                    "actual_score": "",
                    "actual_range": "",
                    "status": "PENDING",
                }
            )
            upsert_symmetry_log_row(row, league=league)
            log_timestamps[game_id_str] = now_ts
            with open(ts_file, "w", encoding="utf-8") as f:
                json.dump(log_timestamps, f)

            print(
                f"  [{tag} LOGGED] {team1} vs {team2} | {check_time} | Score: {total_home}-{total_away} | "
                f"Fav: {sym_info['highest_prob_range']} ({sym_info['bookie_fav_1_odd']}) | "
                f"Value: {sym_info['value_range']} | Cluster: {res_pred.get('best_cluster_name')} ({res_pred.get('best_cluster_prob')}%)"
            )
        else:
            rem_wait = int(req_interval - elapsed_since_last)
            print(f"  [{tag} THROTTLED] {team1} vs {team2} ({check_time}): next log in {rem_wait}s.")

    live_poll_speed = min(20, recommended_poll_speed)

    state_payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        "league": cfg["key"],
        "domain": domain,
        "recommended_poll_speed": live_poll_speed,
        "duration_sec": round(time.time() - start_ts, 2),
        "live_matches": live_cards,
    }
    tmp_state = paths["state_file"] + ".tmp"
    with open(tmp_state, "w", encoding="utf-8") as f:
        json.dump(state_payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_state, paths["state_file"])

    try:
        now_settle = time.time()
        last_settle = _LAST_SETTLE_TS.get(league, 0.0)
        if force_manual or (now_settle - last_settle >= 90.0):
            _LAST_SETTLE_TS[league] = now_settle
            pull_today_results(only_tracked=False, days_back=1, league=league)
            auto_settle_checkpoints(league=league)
    except Exception as e:
        print(f"[{tag}] Auto-settle warning: {e}")

    if cfg["key"] == "nba" and os.environ.get("GITHUB_ACTIONS") == "true":
        try:
            _update_live_bet_signal_file(live_cards, paths, domain)
        except Exception as e:
            print(f"[{tag}] Live bet signal update warning: {e}")

    print(f"[{tag} POLL_SPEED] {live_poll_speed} (Completed in {round(time.time() - start_ts, 2)}s)")
    return live_poll_speed


_BAREM_META_2K26 = {
    "160_or_less": {"tr": "160 veya daha az", "en": "160 or less", "t_id": 1838},
    "161-170": {"tr": "161-170", "en": "161-170", "t_id": 1839},
    "171-180": {"tr": "171-180", "en": "171-180", "t_id": 1840},
    "181-190": {"tr": "181-190", "en": "181-190", "t_id": 1841},
    "191-200": {"tr": "191-200", "en": "191-200", "t_id": 1842},
    "201-210": {"tr": "201-210", "en": "201-210", "t_id": 1843},
    "211-220": {"tr": "211-220", "en": "211-220", "t_id": 1844},
    "221-230": {"tr": "221-230", "en": "221-230", "t_id": 1845},
    "231+": {"tr": "231 veya daha fazla", "en": "231 or more", "t_id": 1846},
}


def discover_active_melbet_domain() -> str:
    """Query Melbet's permanent master edge IP (178.253.29.189 / 91.186.206.142) to get the latest unblocked TR domain."""
    import socket
    ctx = ssl._create_unverified_context()
    for ip, host in [("178.253.29.189", "melbet.com"), ("91.186.206.142", "melbet.org")]:
        try:
            with socket.create_connection((ip, 443), timeout=4) as s:
                with ctx.wrap_socket(s, server_hostname=host) as ss:
                    ss.sendall(
                        f"GET /tr/live/basketball HTTP/1.1\r\nHost: {host}\r\nUser-Agent: Mozilla/5.0\r\nConnection: close\r\n\r\n".encode()
                    )
                    head = ss.recv(2048).decode("utf-8", errors="ignore")
                    m = re.search(r"Location:\s*(https?://[^/\r\n]+)", head, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
        except Exception:
            pass
    return "https://melbet-56831.pro"


def _send_telegram_alert_if_configured(sig: dict):
    """Send a Telegram push notification when a new 35m OYNA signal triggers (if configured)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    cfg_file = os.path.join(BASE_DIR, "telegram_config.json")
    if (not token or not chat_id) and os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                tcfg = json.load(f)
                token = token or str(tcfg.get("bot_token", "")).strip()
                chat_id = chat_id or str(tcfg.get("chat_id", "")).strip()
        except Exception:
            pass
    if not token or not chat_id:
        return
    sent_file = os.path.join(BASE_DIR, "sent_telegram_signals.json")
    sent_ids = []
    if os.path.exists(sent_file):
        try:
            with open(sent_file, "r", encoding="utf-8") as f:
                sent_ids = json.load(f)
        except Exception:
            sent_ids = []
    sid = sig.get("signal_id", "")
    if not sid or sid in sent_ids:
        return
    msg = (
        f"🚨 *NBA 2K26 GECE OTO-PİLOT — OYNA SİNYALİ!*\n\n"
        f"🏀 *Maç:* {sig.get('game_name')}\n"
        f"⏱ *Dakika:* {sig.get('elapsed_mins')} dk | *Skor:* `{sig.get('current_score')}` (Fark: {sig.get('score_gap')})\n"
        f"🎯 *Seçilen Barem:* `{sig.get('target_barem_tr')}` ({sig.get('pick_type')})\n"
        f"📈 *Canlı Oran:* *{sig.get('target_odd'):.2f}*\n"
        f"💰 *Önerilen Tutar:*\n"
        f"  • 8.000 TL Kasa: `{sig.get('stakes', {}).get('kasa_8000', 400)} TL`\n"
        f"  • 10.000 TL Kasa: `{sig.get('stakes', {}).get('kasa_10000', 500)} TL`\n"
        f"  • 20.000 TL Kasa: `{sig.get('stakes', {}).get('kasa_20000', 1000)} TL`\n"
        f"📌 *Kural:* {sig.get('zone_badge')}"
    )
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=6)
        sent_ids.append(sid)
        with open(sent_file, "w", encoding="utf-8") as f:
            json.dump(sent_ids[-100:], f)
    except Exception as e:
        print(f"[TELEGRAM] Notification warning: {e}")


def _update_live_bet_signal_file(live_cards: list, paths: dict, domain: str):
    """Build live_bet_signal.json for mobile 1xBet/Melbet night auto-pilot (works 24/7 while PC is off)."""
    cfg = get_league_config("nba")
    barems = [b[0] for b in cfg["barem_ranges"]]
    odd_cols = cfg["barem_odd_cols"]
    csv_path = paths["csv_path"]
    df_log = pd.read_csv(csv_path) if (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0) else pd.DataFrame()

    chain_hist = []
    recent_history = []
    if not df_log.empty and "status" in df_log.columns:
        df_c = df_log.copy()
        df_c["_min"] = df_c["check_time"].astype(str).str.extract(r"Minute\s+(\d+)", expand=False).astype(float).fillna(0.0)
        df_c["_gid_num"] = df_c["game_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(float).fillna(0.0)
        df_cl = df_c[df_c["_min"] >= 32].sort_values(["_gid_num", "_min"]).groupby("game_id", as_index=False).last().sort_values(["_gid_num", "_min"]).tail(25).reset_index(drop=True)
        for _, r in df_cl.iterrows():
            gid = str(r.get("game_id", "")).strip()
            act_b = str(r.get("actual_range", "")).strip()
            om = {b: float(r.get(odd_cols[b], 99) or 99) for b in barems}
            valid_om = sorted([(b, v) for b, v in om.items() if 1.001 < v < 85.0], key=lambda x: x[1])
            f1 = valid_om[0][0] if valid_om else str(r.get("highest_prob_range", "")).strip()
            o1 = valid_om[0][1] if valid_om else 99.0
            f2 = valid_om[1][0] if len(valid_om) >= 2 else "-"
            o2 = valid_om[1][1] if len(valid_om) >= 2 else 99.0
            try:
                p1, p2 = str(r.get("current_score", "0-0")).split("-")
                cur_tot = int(p1.strip()) + int(p2.strip())
                gap = abs(int(p1.strip()) - int(p2.strip()))
            except Exception:
                cur_tot, gap = 0, 0
            g_this = df_c[(df_c["game_id"] == gid) & (df_c["_min"] <= r["_min"])].sort_values("_min")
            sm = analyze_smart_35m_decision(
                fav1_barem=f1,
                fav1_odd=o1,
                fav2_barem=f2,
                fav2_odd=o2,
                game_id=gid,
                df_log=g_this,
                recent_chain=chain_hist[-10:],
                score_gap=gap,
                cur_tot=cur_tot,
                elapsed_mins=float(r["_min"]) + 0.5,
                odds_map=om,
                league="nba",
                proj_tot=208.2,
            )
            is_pass = str(sm.get("zone_type", "")).startswith("PASS_")
            p_bs = sm.get("primary_barems") or [f1]
            tb = p_bs[0] if p_bs else f1
            is_f2 = (tb == f2)
            to = o2 if is_f2 else o1
            meta = _BAREM_META_2K26.get(tb, {"tr": tb, "en": tb, "t_id": 0})
            st_done = str(r.get("status", "")) == "COMPLETED" and act_b in barems
            recent_history.append({
                "signal_id": f"{gid}_35m",
                "game_id": gid,
                "raw_game_id": gid.replace("CB-", ""),
                "date": str(r.get("date", ""))[:10],
                "game_name": str(r.get("game_name", "")).replace(" (cyber)", ""),
                "current_score": str(r.get("current_score", "")),
                "score_gap": gap,
                "decision": "PAS" if is_pass else "OYNA",
                "pick_type": "PAS" if is_pass else ("2. Favori" if is_f2 else "1. Favori"),
                "target_barem": "" if is_pass else tb,
                "target_barem_tr": "" if is_pass else meta["tr"],
                "target_t_id": 0 if is_pass else meta["t_id"],
                "target_odd": 0.0 if is_pass else round(float(to), 2),
                "zone_badge": sm.get("zone_badge", ""),
                "actual_score": str(r.get("actual_score", "")) if st_done else "",
                "actual_range": act_b if st_done else "",
                "result": ("PAS" if is_pass else ("TUTTU" if act_b == tb else "YATTI")) if st_done else "BEKLIYOR",
            })
            if st_done and f1 in barems:
                chain_hist.append({"raw_shift": barems.index(act_b) - barems.index(f1)})

    prev_sh = chain_hist[-1]["raw_shift"] if chain_hist else 0
    prev2_sh = chain_hist[-2]["raw_shift"] if len(chain_hist) >= 2 else 0

    active_signals = []
    live_matches_35m = []
    for lc in live_cards:
        gid = str(lc.get("game_id", "")).strip()
        el = float(lc.get("elapsed_mins", 0.0) or 0.0)
        om = lc.get("odds", {}) or {}
        f1 = str(lc.get("bookie_fav_1") or lc.get("highest_prob_range") or "-")
        o1 = float(lc.get("bookie_fav_1_odd", 99.0) or 99.0)
        f2 = str(lc.get("bookie_fav_2") or "-")
        o2 = float(lc.get("bookie_fav_2_odd", 99.0) or 99.0)
        h_sc = int(lc.get("home_score", 0) or 0)
        a_sc = int(lc.get("away_score", 0) or 0)
        gap = abs(h_sc - a_sc)
        g_this = df_log[df_log["game_id"] == gid] if (not df_log.empty and "game_id" in df_log.columns) else pd.DataFrame()
        sm = analyze_smart_35m_decision(
            fav1_barem=f1,
            fav1_odd=o1,
            fav2_barem=f2,
            fav2_odd=o2,
            game_id=gid,
            df_log=g_this,
            recent_chain=chain_hist[-10:],
            score_gap=gap,
            cur_tot=h_sc + a_sc,
            elapsed_mins=el,
            odds_map=om,
            league="nba",
            proj_tot=float(lc.get("predicted_total", 208.2) or 208.2),
        )
        is_pass = str(sm.get("zone_type", "")).startswith("PASS_")
        p_bs = sm.get("primary_barems") or [f1]
        tb = p_bs[0] if p_bs else f1
        is_f2 = (tb == f2)
        to = o2 if is_f2 else o1
        meta = _BAREM_META_2K26.get(tb, {"tr": tb, "en": tb, "t_id": 0})
        in_window = (33.0 <= el <= 35.9) and bool(lc.get("is_barem_open", False))
        sig_item = {
            "signal_id": f"{gid}_35m",
            "game_id": gid,
            "raw_game_id": gid.replace("CB-", ""),
            "game_name": f"{lc.get('team_home', '')} vs {lc.get('team_away', '')}".replace(" (cyber)", ""),
            "home_team": str(lc.get("team_home", "")).replace(" (cyber)", "").strip(),
            "away_team": str(lc.get("team_away", "")).replace(" (cyber)", "").strip(),
            "quarter": int(lc.get("quarter", 1) or 1),
            "elapsed_mins": round(el, 1),
            "in_decision_window_35m": in_window,
            "is_barem_open": bool(lc.get("is_barem_open", False)),
            "current_score": f"{h_sc}-{a_sc}",
            "score_gap": gap,
            "fav1_barem": f1,
            "fav1_odd": round(o1, 2) if o1 < 85 else 0.0,
            "fav2_barem": f2,
            "fav2_odd": round(o2, 2) if o2 < 85 else 0.0,
            "decision": "PAS" if is_pass else "OYNA",
            "pick_type": "PAS" if is_pass else ("2. Favori" if is_f2 else "1. Favori"),
            "target_barem": "" if is_pass else tb,
            "target_barem_tr": "" if is_pass else meta["tr"],
            "target_barem_en": "" if is_pass else meta["en"],
            "target_t_id": 0 if is_pass else meta["t_id"],
            "target_odd": 0.0 if is_pass else round(float(to), 2),
            "min_acceptable_odd": 0.0 if is_pass else round(max(1.50, float(to) - 0.15), 2),
            "zone_type": sm.get("zone_type", ""),
            "zone_badge": sm.get("zone_badge", ""),
            "stakes": {
                "kasa_8000": 0 if is_pass else (320 if is_f2 else 400),
                "kasa_10000": 0 if is_pass else (400 if is_f2 else 500),
                "kasa_15000": 0 if is_pass else (600 if is_f2 else 750),
                "kasa_20000": 0 if is_pass else (800 if is_f2 else 1000),
            },
            "updated_at_tr": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        }
        live_matches_35m.append(sig_item)
        if in_window and not is_pass and 1.50 <= to < 85.0:
            active_signals.append(sig_item)
            _send_telegram_alert_if_configured(sig_item)

    melbet_dom = discover_active_melbet_domain()
    payload = {
        "engine_version": "v7.1-selective-calibrated",
        "updated_at_tr": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": round(time.time(), 2),
        "active_domain": domain,
        "active_melbet_domain": melbet_dom,
        "active_melbet_nba2k26_url": f"{melbet_dom}/tr/live/basketball/2935701-nba-2k26-cyber-league",
        "chain_state": {
            "prev_shift": prev_sh,
            "prev2_shift": prev2_sh,
        },
        "has_active_play_signal": len(active_signals) > 0,
        "current_play_signal": active_signals[0] if active_signals else None,
        "active_signals": active_signals,
        "live_matches": live_matches_35m,
        "recent_history": recent_history[-18:][::-1],
    }
    sig_path = os.path.join(BASE_DIR, "live_bet_signal.json")
    tmp_sig = sig_path + ".tmp"
    with open(tmp_sig, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_sig, sig_path)


def run_auto_scraper(force_manual=False, league: str = "all"):
    domain = discover_active_domain()
    print(f"[v2.1 Multi-League Engine] Active 1xBet/Mirror domain: {domain}")
    leagues = ("nba", "nba2k27", "euro") if str(league or "all").lower() == "all" else (str(league).lower(),)
    poll_speeds = []
    for lg in leagues:
        try:
            poll_speeds.append(_scrape_single_league(domain, league=lg, force_manual=force_manual))
        except Exception as e:
            print(f"[{lg.upper()}] Scrape cycle error: {e}")
    return min(poll_speeds) if poll_speeds else 20


run_single_cycle = run_auto_scraper


if __name__ == "__main__":
    force_flag = "--manual" in sys.argv
    arg_league = "all"
    for arg in sys.argv[1:]:
        if arg.startswith("--league="):
            arg_league = arg.split("=", 1)[1].strip().lower()
    run_auto_scraper(force_manual=force_flag, league=arg_league)
