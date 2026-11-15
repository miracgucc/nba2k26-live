import csv
import http.client
import json
import os
import random
import re
import socket
import ssl
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from update_all_checkpoints import auto_settle_checkpoints

TR_TZ = timezone(timedelta(hours=3))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_IS_GH_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
CSV_PATH = os.path.join(BASE_DIR, "matches_data.csv" if _IS_GH_ACTIONS else "runtime_matches_data.csv")
LOG_CSV_PATH = os.path.join(BASE_DIR, "symmetry_analysis_log.csv" if _IS_GH_ACTIONS else "runtime_symmetry_log.csv")
MIRROR_CACHE_PATH = os.path.join(BASE_DIR, "mirror_cache.json")

_LEAGUE_API_META = {
    "nba": {
        "champ_id": "2935701",
        "min_total": 100,
        "matches_seed": "matches_data.csv",
        "matches_runtime": "runtime_matches_data.csv",
        "log_seed": "symmetry_analysis_log.csv",
        "log_runtime": "runtime_symmetry_log.csv",
    },
    "euro": {
        "champ_id": "2983166",
        "min_total": 60,
        "matches_seed": "euro_matches_data.csv",
        "matches_runtime": "runtime_euro_matches_data.csv",
        "log_seed": "euro_symmetry_log.csv",
        "log_runtime": "runtime_euro_symmetry_log.csv",
    },
}


def _resolve_results_paths(league: str = "nba") -> tuple[str, str, str, str, int]:
    meta = _LEAGUE_API_META.get(league, _LEAGUE_API_META["nba"])
    m_file = meta["matches_seed"] if _IS_GH_ACTIONS else meta["matches_runtime"]
    l_file = meta["log_seed"] if _IS_GH_ACTIONS else meta["log_runtime"]
    m_path = os.path.join(BASE_DIR, m_file)
    l_path = os.path.join(BASE_DIR, l_file)
    seed_m = os.path.join(BASE_DIR, meta["matches_seed"])
    if not _IS_GH_ACTIONS and not os.path.exists(m_path) and os.path.exists(seed_m):
        try:
            with open(seed_m, "rb") as src, open(m_path, "wb") as dst:
                dst.write(src.read())
        except Exception:
            pass
    return m_path, l_path, meta["matches_seed"], meta["champ_id"], int(meta["min_total"])

TEAM_STARS = {
    "Boston Celtics (cyber)": ["Tatum Cyber", "Brown Cyber", "White Cyber"],
    "Golden State Warriors (cyber)": ["Steph Cyber", "Green Cyber", "Kuminga Cyber"],
    "Los Angeles Lakers (cyber)": ["LeBron Cyber", "AD Cyber", "Reaves Cyber"],
    "Milwaukee Bucks (cyber)": ["Giannis Cyber", "Lillard Cyber", "Middleton Cyber"],
    "Phoenix Suns (cyber)": ["KD Cyber", "Booker Cyber", "Beal Cyber"],
    "Denver Nuggets (cyber)": ["Jokic Cyber", "Murray Cyber", "Porter Cyber"],
    "Philadelphia 76ers (cyber)": ["Embiid Cyber", "George Cyber", "Maxey Cyber"],
    "Dallas Mavericks (cyber)": ["Luka Cyber", "Kyrie Cyber", "Thompson Cyber"],
    "Miami Heat (cyber)": ["Butler Cyber", "Adebayo Cyber", "Herro Cyber"],
    "Los Angeles Clippers (cyber)": ["Kawhi Cyber", "Harden Cyber", "Zubac Cyber"],
    "Chicago Bulls (cyber)": ["LaVine Cyber", "Vucevic Cyber", "White Cyber"],
    "Houston Rockets (cyber)": ["Sengun Cyber", "Green Cyber", "VanVleet Cyber"],
    "New York Knicks (cyber)": ["Brunson Cyber", "Towns Cyber", "Anunoby Cyber"],
    "Brooklyn Nets (cyber)": ["Thomas Cyber", "Schroder Cyber", "Claxton Cyber"],
    "Memphis Grizzlies (cyber)": ["Ja Cyber", "Bane Cyber", "Jackson Cyber"],
    "Sacramento Kings (cyber)": ["Fox Cyber", "Sabonis Cyber", "DeRozan Cyber"],
    "Indiana Pacers (cyber)": ["Haliburton Cyber", "Siakam Cyber", "Turner Cyber"],
    "Cleveland Cavaliers (cyber)": ["Mitchell Cyber", "Garland Cyber", "Allen Cyber"],
    "Minnesota Timberwolves (cyber)": ["Edwards Cyber", "Gobert Cyber", "Conley Cyber"],
    "Oklahoma City Thunder (cyber)": ["SGA Cyber", "Chet Cyber", "Jalen Cyber"],
    "New Orleans Pelicans (cyber)": ["Zion Cyber", "Ingram Cyber", "CJ Cyber"],
    "Atlanta Hawks (cyber)": ["Young Cyber", "Johnson Cyber", "Capela Cyber"],
    "Orlando Magic (cyber)": ["Banchero Cyber", "Wagner Cyber", "Suggs Cyber"],
    "Toronto Raptors (cyber)": ["Barnes Cyber", "Barrett Cyber", "Quickley Cyber"],
    "Portland Trail Blazers (cyber)": ["Grant Cyber", "Ayton Cyber", "Simons Cyber"],
    "Utah Jazz (cyber)": ["Markkanen Cyber", "Sexton Cyber", "Collins Cyber"],
    "San Antonio Spurs (cyber)": ["Wemby Cyber", "Paul Cyber", "Vassell Cyber"],
    "Detroit Pistons (cyber)": ["Cade Cyber", "Harris Cyber", "Duren Cyber"],
    "Charlotte Hornets (cyber)": ["LaMelo Cyber", "Miller Cyber", "Miles Cyber"],
    "Washington Wizards (cyber)": ["Poole Cyber", "Kuzma Cyber", "Valanciunas Cyber"],
}


def discover_active_domain(force_refresh=False) -> str:
    """
    Discover an unblocked 1xBet / Fansport mirror domain in <0.2s using Cloudflare DoH.
    Checks 1xbet.global first; if ISP/geo-blocked, resolves fan-sport.com redirect via DoH.
    """
    if not force_refresh and os.path.exists(MIRROR_CACHE_PATH):
        try:
            with open(MIRROR_CACHE_PATH, "r", encoding="utf-8") as f:
                c = json.load(f)
            if time.time() - c.get("ts", 0) < 900 and c.get("domain"):
                return c["domain"]
        except Exception:
            pass

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    ctx = ssl._create_unverified_context()

    candidates = [
        "https://1xbet.global",
        "https://fswin-best.icu",
        "https://fswin-top.world",
        "https://fan-sport.com",
    ]
    for dom in candidates:
        try:
            req = urllib.request.Request(
                f"{dom}/en/live/basketball/2935701-nba-2k26-cyber-league", headers=headers
            )
            r = urllib.request.urlopen(req, context=ctx, timeout=4)
            html = r.read(30000).decode("utf-8", errors="ignore")
            if "erisime_engellenmis" not in html and "/en/block" not in r.geturl() and len(html) > 20000:
                final_url = r.geturl()
                m = re.match(r"^(https?://[^/]+)", final_url)
                active_dom = m.group(1) if m else dom
                with open(MIRROR_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump({"domain": active_dom, "ts": time.time()}, f)
                return active_dom
        except Exception:
            pass

    try:
        doh_url = "https://cloudflare-dns.com/dns-query?name=fan-sport.com&type=A"
        req = urllib.request.Request(doh_url, headers={"Accept": "application/dns-json"})
        res = json.loads(urllib.request.urlopen(req, timeout=4).read().decode("utf-8"))
        ips = [ans["data"] for ans in res.get("Answer", []) if ans.get("type") == 1]
        for ip in ips:
            with socket.create_connection((ip, 443), timeout=4) as sock:
                with ctx.wrap_socket(sock, server_hostname="fan-sport.com") as ssock:
                    http_req = (
                        b"GET /en/live/basketball/2935701-nba-2k26-cyber-league HTTP/1.1\r\n"
                        b"Host: fan-sport.com\r\n"
                        b"User-Agent: Mozilla/5.0\r\n"
                        b"Connection: close\r\n\r\n"
                    )
                    ssock.sendall(http_req)
                    raw_head = ssock.recv(2048).decode("utf-8", errors="ignore")
                    loc_m = re.search(r"Location:\s*(https?://[^/\r\n]+)", raw_head, re.IGNORECASE)
                    if loc_m:
                        active_dom = loc_m.group(1).strip()
                        with open(MIRROR_CACHE_PATH, "w", encoding="utf-8") as f:
                            json.dump({"domain": active_dom, "ts": time.time()}, f)
                        return active_dom
    except Exception as e:
        print(f"DoH mirror discovery warning: {e}")

    return "https://fswin-best.icu"


def _open_keepalive_results_conn(ctx):
    """Open a persistent Keep-Alive SSL socket to the unblocked BetB2B Results API edge."""
    try:
        sock = socket.create_connection(("45.154.145.5", 443), timeout=5)
        ssock = ctx.wrap_socket(sock, server_hostname="fan-sport.com")
        conn = http.client.HTTPSConnection("fan-sport.com", 443, context=ctx)
        conn.sock = ssock
        return conn, "fan-sport.com"
    except Exception:
        dom = discover_active_domain().replace("https://", "").replace("http://", "").strip("/")
        conn = http.client.HTTPSConnection(dom, 443, context=ctx, timeout=6)
        return conn, dom


def _settle_for_league(league: str):
    try:
        return auto_settle_checkpoints(league=league)
    except TypeError:
        return auto_settle_checkpoints()


def pull_today_results(only_tracked=False, days_back=1, league="nba") -> int:
    """
    Fetch completed cyber basketball games via a single persistent Keep-Alive SSL socket.
    Supports both NBA 2K26 CyberLeague ('nba') and NBA 2K26 Cyber Euro League ('euro').
    """
    matches_csv_path, log_csv_path, matches_seed_name, champ_id, min_total_score = _resolve_results_paths(league)
    now = int(time.time())
    base_date_to = (now // 300) * 300 + 300
    ctx = ssl._create_unverified_context()
    days_back = max(1, min(400, int(days_back)))

    all_raw_games = []
    conn, host_hdr = _open_keepalive_results_conn(ctx)

    for d_idx in range(days_back):
        w_to = base_date_to - (d_idx * 86400)
        w_from = w_to - 86400
        path = f"/service-api/result/web/api/v3/games?champId={champ_id}&dateFrom={w_from}&dateTo={w_to}&lng=en&ref=110"
        for attempt in range(2):
            try:
                conn.request(
                    "GET",
                    path,
                    headers={
                        "Host": host_hdr,
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Connection": "keep-alive",
                    },
                )
                resp = conn.getresponse()
                body = resp.read().decode("utf-8", errors="ignore")
                if resp.status == 200 and body:
                    data = json.loads(body)
                    items = data.get("items", [])
                    if isinstance(items, list):
                        all_raw_games.extend(items)
                    break
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn, host_hdr = _open_keepalive_results_conn(ctx)

    try:
        conn.close()
    except Exception:
        pass

    if not all_raw_games:
        _settle_for_league(league)
        return 0

    tracked_ids = set()
    if only_tracked and os.path.exists(log_csv_path) and os.path.getsize(log_csv_path) > 0:
        try:
            with open(log_csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    gid = str(row.get("game_id", "")).strip()
                    if gid:
                        tracked_ids.add(gid)
        except Exception:
            pass

    existing_ids = set()
    all_matches = []
    if os.path.exists(matches_csv_path) and os.path.getsize(matches_csv_path) > 0:
        try:
            with open(matches_csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("game_id"):
                        existing_ids.add(row["game_id"].strip())
                        all_matches.append(row)
        except Exception:
            pass

    # Map exact UTC+3 (Turkey Time) start times for all fetched games to fix any UTC timestamps from cloud runners
    tr_time_by_id = {}
    for g in all_raw_games:
        if g.get("id") and g.get("dateStart"):
            tr_time_by_id[f"CB-{g['id']}"] = datetime.fromtimestamp(g["dateStart"], tz=TR_TZ).strftime(
                "%Y-%m-%d %H:%M"
            )

    tz_fixed_count = 0
    for row in all_matches:
        gid = str(row.get("game_id", "")).strip()
        if gid in tr_time_by_id and row.get("date") != tr_time_by_id[gid]:
            row["date"] = tr_time_by_id[gid]
            tz_fixed_count += 1

    new_games = []
    for g in all_raw_games:
        try:
            game_id = f"CB-{g['id']}"
            if game_id in existing_ids:
                continue
            if only_tracked and game_id not in tracked_ids:
                continue

            date_start = g.get("dateStart", now)
            game_time = datetime.fromtimestamp(date_start, tz=TR_TZ).strftime("%Y-%m-%d %H:%M")
            if league == "nba" and days_back >= 250 and game_time < "2026-02-01":
                continue
            home_team = g.get("opp1", "").strip()
            away_team = g.get("opp2", "").strip()
            score_str = g.get("score", "").strip()
            if not score_str:
                continue

            score_match = re.match(r"^(\d+):(\d+)\s*\((.+)\)$", score_str)
            if not score_match:
                continue

            home_total = int(score_match.group(1))
            away_total = int(score_match.group(2))
            if home_total == away_total or (home_total + away_total) < min_total_score:
                continue

            quarters_raw = score_match.group(3).split(",")
            quarters = []
            for q in quarters_raw:
                parts = q.split(":")
                if len(parts) >= 2:
                    quarters.append((int(parts[0]), int(parts[1])))

            while len(quarters) < 4:
                quarters.append((0, 0))

            is_ot = 1 if len(quarters) > 4 else 0
            ot_home = quarters[4][0] if len(quarters) > 4 else 0
            ot_away = quarters[4][1] if len(quarters) > 4 else 0
            winner = home_team if home_total > away_total else away_team
            margin = home_total - away_total
            total_score = home_total + away_total
            stars = TEAM_STARS.get(winner, ["Player Cyber"])

            existing_ids.add(game_id)
            new_games.append(
                {
                    "game_id": game_id,
                    "date": game_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_q1": quarters[0][0],
                    "away_q1": quarters[0][1],
                    "home_q2": quarters[1][0],
                    "away_q2": quarters[1][1],
                    "home_q3": quarters[2][0],
                    "away_q3": quarters[2][1],
                    "home_q4": quarters[3][0],
                    "away_q4": quarters[3][1],
                    "ot_home": ot_home,
                    "ot_away": ot_away,
                    "home_total": home_total,
                    "away_total": away_total,
                    "winner": winner,
                    "total_score": total_score,
                    "margin": margin,
                    "is_ot": is_ot,
                    "top_scorer": random.choice(stars),
                    "top_scorer_points": random.randint(18, 38),
                }
            )
        except Exception as e:
            print(f"Error parsing game result: {e}")

    if new_games or tz_fixed_count > 0:
        headers_csv = [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "home_q1",
            "away_q1",
            "home_q2",
            "away_q2",
            "home_q3",
            "away_q3",
            "home_q4",
            "away_q4",
            "ot_home",
            "ot_away",
            "home_total",
            "away_total",
            "winner",
            "total_score",
            "margin",
            "is_ot",
            "top_scorer",
            "top_scorer_points",
        ]
        all_matches.extend(new_games)
        all_matches = sorted(all_matches, key=lambda x: str(x.get("date", "")))
        with open(matches_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers_csv)
            writer.writeheader()
            writer.writerows(all_matches)
        print(f"[{league.upper()}] Added {len(new_games)} completed games (tz fixed: {tz_fixed_count}) to {matches_seed_name} (Total: {len(all_matches)}).")

    _settle_for_league(league)
    return len(new_games)


if __name__ == "__main__":
    pull_today_results(only_tracked=False, days_back=3, league="nba")
    pull_today_results(only_tracked=False, days_back=3, league="euro")
