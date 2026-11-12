import os
import pandas as pd
from period_predictor import get_range_from_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "symmetry_analysis_log.csv")
MATCHES_CSV = os.path.join(BASE_DIR, "matches_data.csv")


def auto_settle_checkpoints():
    """Settle PENDING checkpoints in symmetry_analysis_log.csv using completed games in matches_data.csv."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return 0

    if not os.path.exists(MATCHES_CSV) or os.path.getsize(MATCHES_CSV) == 0:
        return 0

    try:
        df_logs = pd.read_csv(CSV_PATH)
        df_results = pd.read_csv(MATCHES_CSV)
    except Exception as e:
        print(f"Error reading CSVs for settlement: {e}")
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
            if total_score <= 50 and pd.notna(latest_match.get("total_score")):
                total_score = int(float(latest_match.get("total_score", 0)))
        except Exception:
            total_score = 0

        if total_score > 50:
            actual_score_str = f"{total_home}-{total_away}"
            actual_range = get_range_from_score(total_score)
            df_logs.at[idx, "actual_score"] = actual_score_str
            df_logs.at[idx, "actual_range"] = actual_range
            df_logs.at[idx, "status"] = "COMPLETED"
            settled_count += 1
            print(f"Settled: {row.get('game_name')} ({row.get('check_time')}) -> Final: {actual_score_str} ({actual_range})")

    if settled_count > 0:
        df_logs.to_csv(CSV_PATH, index=False)

    return settled_count


if __name__ == "__main__":
    auto_settle_checkpoints()
