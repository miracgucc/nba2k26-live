import os
import shutil
import pandas as pd
from period_predictor import get_league_config, get_range_from_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_IS_GH_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


def _resolve_league_paths(league: str = "nba") -> tuple[str, str]:
    """Resolve checkpoint log CSV and completed matches CSV paths for a league."""
    cfg = get_league_config(league)
    files = cfg["files"]
    sym_file = files["log_seed"] if _IS_GH_ACTIONS else files["log_runtime"]
    matches_file = files["matches_seed"] if _IS_GH_ACTIONS else files["matches_runtime"]
    csv_path = os.path.join(BASE_DIR, sym_file)
    matches_csv = os.path.join(BASE_DIR, matches_file)

    if not _IS_GH_ACTIONS:
        seed_sym = os.path.join(BASE_DIR, files["log_seed"])
        seed_m = os.path.join(BASE_DIR, files["matches_seed"])
        if not os.path.exists(csv_path) and os.path.exists(seed_sym):
            try:
                shutil.copyfile(seed_sym, csv_path)
            except Exception:
                pass
        if not os.path.exists(matches_csv) and os.path.exists(seed_m):
            try:
                shutil.copyfile(seed_m, matches_csv)
            except Exception:
                pass
    return csv_path, matches_csv


CSV_PATH, MATCHES_CSV = _resolve_league_paths("nba")


def auto_settle_checkpoints(league: str = "nba") -> int:
    """Settle PENDING checkpoints in a league's symmetry log CSV using its completed matches CSV."""
    csv_path, matches_csv = _resolve_league_paths(league)
    cfg = get_league_config(league)
    min_total_valid = int(cfg.get("min_total_valid", 50))

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0

    if not os.path.exists(matches_csv) or os.path.getsize(matches_csv) == 0:
        return 0

    try:
        df_logs = pd.read_csv(csv_path)
        df_results = pd.read_csv(matches_csv)
    except Exception as e:
        print(f"[{cfg['key'].upper()}] Error reading CSVs for settlement: {e}")
        return 0

    if df_logs.empty or df_results.empty:
        return 0

    for col in ["actual_score", "actual_range", "status"]:
        if col not in df_logs.columns:
            df_logs[col] = ""

    df_logs["actual_score"] = df_logs["actual_score"].astype(str).replace("nan", "")
    df_logs["actual_range"] = df_logs["actual_range"].astype(str).replace("nan", "")
    df_logs["status"] = df_logs["status"].astype(str).replace("nan", "PENDING")

    pending_mask = df_logs["status"] == "PENDING"
    if pending_mask.sum() == 0:
        return 0

    settled_count = 0
    for idx, row in df_logs[pending_mask].iterrows():
        game_id = row.get("game_id")
        if game_id is None or pd.isna(game_id) or str(game_id).strip() == "":
            continue

        clean_gid = str(game_id).strip()
        match_results = df_results[
            (df_results["game_id"].astype(str).str.strip() == clean_gid)
            | (df_results["game_id"].astype(str).str.strip() == clean_gid.replace("CB-", ""))
        ]

        if match_results.empty:
            continue

        latest_match = match_results.iloc[-1]
        try:
            total_home = int(float(latest_match.get("home_total", 0)))
            total_away = int(float(latest_match.get("away_total", 0)))
            total_score = total_home + total_away
            if total_score <= min_total_valid and pd.notna(latest_match.get("total_score")):
                total_score = int(float(latest_match.get("total_score", 0)))
        except Exception:
            total_score = 0

        if total_score > min_total_valid:
            actual_score_str = f"{total_home}-{total_away}"
            actual_range = get_range_from_score(total_score, league=league)
            df_logs.at[idx, "actual_score"] = actual_score_str
            df_logs.at[idx, "actual_range"] = actual_range
            df_logs.at[idx, "status"] = "COMPLETED"
            settled_count += 1
            print(f"[{cfg['key'].upper()}] Settled: {row.get('game_name')} ({row.get('check_time')}) -> Final: {actual_score_str} ({actual_range})")

    if settled_count > 0:
        df_logs.to_csv(csv_path, index=False)

    return settled_count


if __name__ == "__main__":
    for _lg in ("nba", "euro"):
        auto_settle_checkpoints(_lg)
