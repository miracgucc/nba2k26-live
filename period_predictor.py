from datetime import datetime, timedelta
import numpy as np
import pandas as pd

__all__ = [
    "BAREM_RANGES",
    "CLUSTER_RANGES",
    "LEAGUE_CONFIGS",
    "get_league_config",
    "get_csv_headers",
    "get_range_from_score",
    "get_cluster_from_score",
    "build_q3_q4_streak_signal",
    "predict_period_scores",
    "predict_live_inplay",
    "resolve_effective_barem_odds",
    "analyze_barem_symmetry",
    "analyze_sequential_barem_wave",
    "analyze_smart_35m_decision",
    "analyze_matchup_live_summary",
    "analyze_archive_4500_symmetry",
    "simulate_35m_strategies",
    "analyze_team_pair_history",
    "build_team_pair_table",
    "validate_symmetry_log",
]

NBA_BAREM_RANGES = [
    ("160 Or Less", 0.0, 160.5),
    ("161-170", 160.5, 170.5),
    ("171-180", 170.5, 180.5),
    ("181-190", 180.5, 190.5),
    ("191-200", 190.5, 200.5),
    ("201-210", 200.5, 210.5),
    ("211-220", 210.5, 220.5),
    ("221-230", 220.5, 230.5),
    ("231+", 230.5, 999.0),
]

NBA_CLUSTER_RANGES = [
    ("161-180", 160.5, 180.5, ("161-170", "171-180")),
    ("171-190", 170.5, 190.5, ("171-180", "181-190")),
    ("181-200", 180.5, 200.5, ("181-190", "191-200")),
    ("191-210", 190.5, 210.5, ("191-200", "201-210")),
    ("201-220", 200.5, 220.5, ("201-210", "211-220")),
    ("211-230", 210.5, 230.5, ("211-220", "221-230")),
    ("221-240", 220.5, 240.5, ("221-230", "231+")),
]

EURO_BAREM_RANGES = [
    ("131'den Az", 0.0, 130.5),
    ("131-140", 130.5, 140.5),
    ("141-150", 140.5, 150.5),
    ("151-160", 150.5, 160.5),
    ("161-170", 160.5, 170.5),
    ("170'den Fazla", 170.5, 999.0),
]

EURO_CLUSTER_RANGES = [
    ("121-140", 0.0, 140.5, ("131'den Az", "131-140")),
    ("131-150", 130.5, 150.5, ("131-140", "141-150")),
    ("141-160", 140.5, 160.5, ("141-150", "151-160")),
    ("151-170", 150.5, 170.5, ("151-160", "161-170")),
    ("161+", 160.5, 999.0, ("161-170", "170'den Fazla")),
]

BAREM_RANGES = NBA_BAREM_RANGES
CLUSTER_RANGES = NBA_CLUSTER_RANGES
_NBA_FEB_DAYS = max(290, (datetime.now().date() - datetime(2026, 2, 1).date()).days + 2)

LEAGUE_CONFIGS = {
    "nba": {
        "key": "nba",
        "champ_id": "2935701",
        "title": "NBA 2K26 CyberLeague Pro",
        "short_name": "NBA 2K26 CyberLeague",
        "btn_label": "🏀 NBA 2K26 CyberLeague",
        "quarter_length": 12.0,
        "total_minutes": 48.0,
        "close_min": 36.0,
        "close_min_int": 36,
        "decision_min": 35,
        "closing_start_min": 32,
        "final_win_start": 34,
        "trap_min": 33,
        "mid_q3_min": 30,
        "halftime_min": 24,
        "early_q2_min": 18,
        "q3_pre_low": 27,
        "q3_pre_high": 31,
        "rem_span_decision": 13.0,
        "prior_q": 53.0,
        "exp_rem_close": 52,
        "exp_rem_decision": 54.4,
        "exp_rem_clutch": 58.3,
        "exp_rem_blowout": 49.8,
        "exp_rem_ctrl": 54.9,
        "avg_q4_sum": 52.5,
        "q4_slow_max": 51.6,
        "q4_fast_min": 53.6,
        "q4_half_default": 26.2,
        "q3_collapse_max": 48.5,
        "q3_burst_min": 55.5,
        "q3_collapse_disp": 48,
        "q3_burst_disp": 56,
        "q3_norm_lo": 49,
        "q3_norm_hi": 55,
        "blowout_brake_gap": 28,
        "team_std_dev": 8.5,
        "pace_pts_per_min": 4.4,
        "min_clean_gid": 770265000,
        "min_total_valid": 100,
        "archive_min_total": 140,
        "archive_min_s3": 100,
        "archive_days": _NBA_FEB_DAYS,
        "archive_btn_label": f"📥 Şubat 2026+ Arşivini Güncelle ({_NBA_FEB_DAYS} Gün)",
        "default_barem": "201-210",
        "default_backup_barem": "211-220",
        "high_over_threshold": 211,
        "bar_lower_floor": 150,
        "bar_upper_cap": 240,
        "archive_h1_slow_max": 98,
        "archive_h1_fast_min": 108,
        "mod10_center_digits": (1, 2, 3, 4),
        "mod10_mid_digits": (5, 6, 7),
        "mod10_risk_digits": (8, 9, 0),
        "default_wave_probs": (40.6, 26.3, 20.0, 8.6, 4.6),
        "barem_ranges": NBA_BAREM_RANGES,
        "cluster_ranges": NBA_CLUSTER_RANGES,
        "barem_odd_cols": {
            "160 Or Less": "odds_160_or_less",
            "161-170": "odds_161_170",
            "171-180": "odds_171_180",
            "181-190": "odds_181_190",
            "191-200": "odds_191_200",
            "201-210": "odds_201_210",
            "211-220": "odds_211_220",
            "221-230": "odds_221_230",
            "231+": "odds_231_plus",
        },
        "barem_prob_cols": {
            "160 Or Less": "model_prob_160_or_less",
            "161-170": "model_prob_161_170",
            "171-180": "model_prob_171_180",
            "181-190": "model_prob_181_190",
            "191-200": "model_prob_191_200",
            "201-210": "model_prob_201_210",
            "211-220": "model_prob_211_220",
            "221-230": "model_prob_221_230",
            "231+": "model_prob_231_plus",
        },
        "short_labels": {
            "160 Or Less": "160-",
            "161-170": "161-170",
            "171-180": "171-180",
            "181-190": "181-190",
            "191-200": "191-200",
            "201-210": "201-210",
            "211-220": "211-220",
            "221-230": "221-230",
            "231+": "231+",
        },
        "files": {
            "log_seed": "symmetry_analysis_log.csv",
            "log_runtime": "runtime_symmetry_log.csv",
            "log_cloud": "cloud_symmetry_backup.csv",
            "matches_seed": "matches_data.csv",
            "matches_runtime": "runtime_matches_data.csv",
            "matches_cloud": "cloud_matches_backup.csv",
            "state_file": "live_matches_state.json",
            "ts_file": "last_log_timestamps.json",
        },
    },
    "nba2k27": {
        "key": "nba2k27",
        "champ_id": "3125701",
        "slug": "3125701-nba-2k27-cyber-league",
        "title": "NBA 2K27 CyberLeague Pro",
        "short_name": "NBA 2K27 CyberLeague",
        "btn_label": "🔥 NBA 2K27 CyberLeague",
        "quarter_length": 12.0,
        "total_minutes": 48.0,
        "close_min": 36.0,
        "close_min_int": 36,
        "decision_min": 35,
        "closing_start_min": 32,
        "final_win_start": 34,
        "trap_min": 33,
        "mid_q3_min": 30,
        "halftime_min": 24,
        "early_q2_min": 18,
        "q3_pre_low": 27,
        "q3_pre_high": 31,
        "rem_span_decision": 13.0,
        "prior_q": 52.5,
        "exp_rem_close": 52,
        "exp_rem_decision": 53.7,
        "exp_rem_clutch": 56.7,
        "exp_rem_blowout": 48.5,
        "exp_rem_ctrl": 53.5,
        "avg_q4_sum": 51.9,
        "q4_slow_max": 51.0,
        "q4_fast_min": 53.0,
        "q4_half_default": 25.9,
        "q3_collapse_max": 48.5,
        "q3_burst_min": 55.5,
        "q3_collapse_disp": 48,
        "q3_burst_disp": 56,
        "q3_norm_lo": 49,
        "q3_norm_hi": 55,
        "blowout_brake_gap": 28,
        "team_std_dev": 8.5,
        "pace_pts_per_min": 4.4,
        "min_clean_gid": 770000000,
        "min_total_valid": 100,
        "archive_min_total": 140,
        "archive_min_s3": 100,
        "archive_days": 30,
        "archive_btn_label": "📥 NBA 2K27 Arşivini Güncelle (30 Gün)",
        "default_barem": "201-210",
        "default_backup_barem": "211-220",
        "high_over_threshold": 211,
        "bar_lower_floor": 150,
        "bar_upper_cap": 240,
        "archive_h1_slow_max": 98,
        "archive_h1_fast_min": 108,
        "mod10_center_digits": (1, 2, 3, 4),
        "mod10_mid_digits": (5, 6, 7),
        "mod10_risk_digits": (8, 9, 0),
        "default_wave_probs": (40.6, 26.3, 20.0, 8.6, 4.6),
        "barem_ranges": NBA_BAREM_RANGES,
        "cluster_ranges": NBA_CLUSTER_RANGES,
        "barem_odd_cols": {
            "160 Or Less": "odds_160_or_less",
            "161-170": "odds_161_170",
            "171-180": "odds_171_180",
            "181-190": "odds_181_190",
            "191-200": "odds_191_200",
            "201-210": "odds_201_210",
            "211-220": "odds_211_220",
            "221-230": "odds_221_230",
            "231+": "odds_231_plus",
        },
        "barem_prob_cols": {
            "160 Or Less": "model_prob_160_or_less",
            "161-170": "model_prob_161_170",
            "171-180": "model_prob_171_180",
            "181-190": "model_prob_181_190",
            "191-200": "model_prob_191_200",
            "201-210": "model_prob_201_210",
            "211-220": "model_prob_211_220",
            "221-230": "model_prob_221_230",
            "231+": "model_prob_231_plus",
        },
        "short_labels": {
            "160 Or Less": "160-",
            "161-170": "161-170",
            "171-180": "171-180",
            "181-190": "181-190",
            "191-200": "191-200",
            "201-210": "201-210",
            "211-220": "211-220",
            "221-230": "221-230",
            "231+": "231+",
        },
        "files": {
            "log_seed": "nba2k27_symmetry_log.csv",
            "log_runtime": "runtime_nba2k27_symmetry_log.csv",
            "log_cloud": "cloud_nba2k27_symmetry_backup.csv",
            "matches_seed": "nba2k27_matches_data.csv",
            "matches_runtime": "runtime_nba2k27_matches_data.csv",
            "matches_cloud": "cloud_nba2k27_matches_backup.csv",
            "state_file": "nba2k27_live_matches_state.json",
            "ts_file": "nba2k27_last_log_timestamps.json",
        },
    },
    "euro": {
        "key": "euro",
        "champ_id": "2983166",
        "title": "NBA 2K26 Cyber Euro League Pro",
        "short_name": "NBA 2K26 Cyber Euro League",
        "btn_label": "🇪🇺 NBA 2K26 Cyber Euro League",
        "quarter_length": 10.0,
        "total_minutes": 40.0,
        "close_min": 30.0,
        "close_min_int": 30,
        "decision_min": 29,
        "closing_start_min": 26,
        "final_win_start": 28,
        "trap_min": 27,
        "mid_q3_min": 25,
        "halftime_min": 20,
        "early_q2_min": 15,
        "q3_pre_low": 22,
        "q3_pre_high": 25,
        "rem_span_decision": 11.0,
        "prior_q": 36.3,
        "exp_rem_close": 37,
        "exp_rem_decision": 38.1,
        "exp_rem_clutch": 42.3,
        "exp_rem_blowout": 34.6,
        "exp_rem_ctrl": 38.6,
        "avg_q4_sum": 36.7,
        "q4_slow_max": 35.9,
        "q4_fast_min": 37.5,
        "q4_half_default": 18.4,
        "q3_collapse_max": 32.5,
        "q3_burst_min": 39.5,
        "q3_collapse_disp": 32,
        "q3_burst_disp": 40,
        "q3_norm_lo": 33,
        "q3_norm_hi": 39,
        "blowout_brake_gap": 20,
        "team_std_dev": 9.5,
        "pace_pts_per_min": 3.63,
        "min_clean_gid": 771090000,
        "min_total_valid": 60,
        "archive_min_total": 80,
        "archive_min_s3": 55,
        "archive_days": 185,
        "archive_btn_label": "📥 Son 6 Ay Arşivini Güncelle (185 Gün)",
        "default_barem": "141-150",
        "default_backup_barem": "151-160",
        "high_over_threshold": 151,
        "bar_lower_floor": 120,
        "bar_upper_cap": 180,
        "archive_h1_slow_max": 66,
        "archive_h1_fast_min": 76,
        "mod10_center_digits": (7, 8, 9, 0),
        "mod10_mid_digits": (1, 2, 3),
        "mod10_risk_digits": (4, 5, 6),
        "default_wave_probs": (56.7, 20.2, 15.4, 2.9, 4.8),
        "barem_ranges": EURO_BAREM_RANGES,
        "cluster_ranges": EURO_CLUSTER_RANGES,
        "barem_odd_cols": {
            "131'den Az": "odds_130_or_less",
            "131-140": "odds_131_140",
            "141-150": "odds_141_150",
            "151-160": "odds_151_160",
            "161-170": "odds_161_170",
            "170'den Fazla": "odds_171_plus",
        },
        "barem_prob_cols": {
            "131'den Az": "model_prob_130_or_less",
            "131-140": "model_prob_131_140",
            "141-150": "model_prob_141_150",
            "151-160": "model_prob_151_160",
            "161-170": "model_prob_161_170",
            "170'den Fazla": "model_prob_171_plus",
        },
        "short_labels": {
            "131'den Az": "131-",
            "131-140": "131-140",
            "141-150": "141-150",
            "151-160": "151-160",
            "161-170": "161-170",
            "170'den Fazla": "170+",
        },
        "files": {
            "log_seed": "euro_symmetry_log.csv",
            "log_runtime": "runtime_euro_symmetry_log.csv",
            "log_cloud": "cloud_euro_symmetry_backup.csv",
            "matches_seed": "euro_matches_data.csv",
            "matches_runtime": "runtime_euro_matches_data.csv",
            "matches_cloud": "cloud_euro_matches_backup.csv",
            "state_file": "euro_live_matches_state.json",
            "ts_file": "euro_last_log_timestamps.json",
        },
    },
}


def get_league_config(league: str = "nba") -> dict:
    return LEAGUE_CONFIGS.get(str(league or "nba").lower(), LEAGUE_CONFIGS["nba"])


def get_csv_headers(league: str = "nba") -> list:
    cfg = get_league_config(league)
    b_names = [b[0] for b in cfg["barem_ranges"]]
    return (
        ["game_id", "game_name", "date", "check_time", "current_score"]
        + [cfg["barem_odd_cols"][b] for b in b_names]
        + [cfg["barem_prob_cols"][b] for b in b_names]
        + ["highest_prob_range", "value_range", "best_cluster_name", "best_cluster_prob", "actual_score", "actual_range", "status"]
    )


def get_range_from_score(total: float, league: str = "nba") -> str:
    cfg = get_league_config(league)
    try:
        val = float(total)
        if np.isnan(val) or np.isinf(val):
            return cfg["default_barem"]
        total_int = int(round(val))
    except (ValueError, TypeError):
        return cfg["default_barem"]
    for b_name, _, b_hi in cfg["barem_ranges"]:
        if total_int <= b_hi:
            return b_name
    return cfg["barem_ranges"][-1][0]


def get_cluster_from_score(total: float, league: str = "nba") -> str:
    cfg = get_league_config(league)
    try:
        val = float(total)
        if np.isnan(val) or np.isinf(val):
            return cfg["cluster_ranges"][len(cfg["cluster_ranges"]) // 2][0]
        total_int = int(round(val))
    except (ValueError, TypeError):
        return cfg["cluster_ranges"][len(cfg["cluster_ranges"]) // 2][0]
    for c_name, _, c_hi, _ in cfg["cluster_ranges"]:
        if total_int <= c_hi:
            return c_name
    return cfg["cluster_ranges"][-1][0]


def build_q3_q4_streak_signal(
    df,
    projected_q3_total=None,
    q3_threshold=None,
    streak_window=3,
    team_home="",
    team_away="",
    q3_score=None,
    league="nba",
):
    """Calculate how often a Q3 total is followed by a Q4 burst or collapse."""
    cfg = get_league_config(league)
    prior_q = float(cfg["prior_q"])
    def_thr = int(round(prior_q)) - 3
    if projected_q3_total is None:
        projected_q3_total = prior_q
    if q3_threshold is None:
        q3_threshold = def_thr

    if isinstance(projected_q3_total, str):
        team_home = projected_q3_total
        if isinstance(q3_threshold, str):
            team_away = q3_threshold
            q3_threshold = def_thr
        projected_q3_total = q3_score if q3_score is not None else prior_q
    elif q3_score is not None:
        projected_q3_total = q3_score

    default_resp = {
        "normal": {"probability": 0.50, "count": 0},
        "burst_positive": {"probability": 0.25, "count": 0},
        "collapse_negative": {"probability": 0.25, "count": 0},
        "signal": "⚖️ Dengeli Q4 Beklentisi",
        "detail": f"Standart {cfg['short_name']} periyot dağılımı ({prior_q:.1f} sayı/çeyrek).",
        "threshold": q3_threshold,
        "sample_size": 0,
        "streak_window": streak_window,
        "burst_probability": 0.25,
        "collapse_probability": 0.25,
        "normal_probability": 0.50,
        "burst_prob": 25.0,
        "collapse_prob": 25.0,
        "normal_prob": 50.0,
    }
    if df is None or df.empty:
        return default_resp

    required = ["home_q3", "away_q3", "home_q4", "away_q4"]
    if not all(col in df.columns for col in required):
        return default_resp

    hist = df.copy()
    for c in required:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.dropna(subset=required).copy()
    if hist.empty:
        return default_resp
    if team_home or team_away:
        mask = pd.Series(False, index=hist.index)
        if team_home:
            mask |= (hist["home_team"] == team_home) | (hist["away_team"] == team_home)
        if team_away:
            mask |= (hist["home_team"] == team_away) | (hist["away_team"] == team_away)
        sub_t = hist[mask]
        if len(sub_t) >= 5:
            hist = sub_t

    if hist.empty:
        return default_resp

    hist["q3_total"] = hist["home_q3"].astype(float) + hist["away_q3"].astype(float)
    hist["q4_total"] = hist["home_q4"].astype(float) + hist["away_q4"].astype(float)
    hist["delta_q4_vs_q3"] = hist["q4_total"] - hist["q3_total"]

    # Filter around similar Q3 pace or >= threshold
    q3_val = float(projected_q3_total)
    similar_q3 = hist[(hist["q3_total"] >= q3_val - 5) & (hist["q3_total"] <= q3_val + 5)].copy()
    if len(similar_q3) < 8:
        similar_q3 = hist[hist["q3_total"] >= min(q3_val, float(q3_threshold))].copy()
    if similar_q3.empty:
        similar_q3 = hist.copy()

    similar_q3["q4_delta_bucket"] = pd.cut(
        similar_q3["q4_total"],
        bins=[-np.inf, float(cfg["q3_collapse_max"]), float(cfg["q3_burst_min"]), np.inf],
        labels=["collapse_negative", "normal", "burst_positive"],
    )

    counts = similar_q3["q4_delta_bucket"].value_counts().reindex(
        ["normal", "burst_positive", "collapse_negative"], fill_value=0
    )
    total = int(counts.sum())
    if total == 0:
        return default_resp

    probs = {}
    for key in ["normal", "burst_positive", "collapse_negative"]:
        prob = float(counts.get(key, 0)) / total
        probs[key] = {"probability": round(prob, 3), "count": int(counts.get(key, 0))}

    b_pct = round(probs["burst_positive"]["probability"] * 100.0, 1)
    c_pct = round(probs["collapse_negative"]["probability"] * 100.0, 1)
    n_pct = round(probs["normal"]["probability"] * 100.0, 1)
    avg_q4 = round(float(similar_q3["q4_total"].mean()), 1)

    if b_pct >= c_pct + 8:
        signal = "🔥 Q4 Sayı Patlaması Eğilimi"
    elif c_pct >= b_pct + 8:
        signal = "❄️ Q4 Tempo Düşüşü / Çöküş Riski"
    else:
        signal = "⚖️ Dengeli Q3 ➔ Q4 Geçişi"

    detail = (
        f"Benzer Q3 temposundaki ({int(q3_val)} sayı) {total} geçmiş maçta Q4 ortalaması {avg_q4} sayı oldu "
        f"(Patlama ≥{cfg['q3_burst_disp']}: %{b_pct}, Normal {cfg['q3_norm_lo']}-{cfg['q3_norm_hi']}: %{n_pct}, Çöküş ≤{cfg['q3_collapse_disp']}: %{c_pct})."
    )

    return {
        **probs,
        "signal": signal,
        "detail": detail,
        "threshold": q3_threshold,
        "sample_size": total,
        "streak_window": streak_window,
        "burst_probability": probs["burst_positive"]["probability"],
        "collapse_probability": probs["collapse_negative"]["probability"],
        "normal_probability": probs["normal"]["probability"],
        "burst_prob": b_pct,
        "collapse_prob": c_pct,
        "normal_prob": n_pct,
    }


def predict_period_scores(df, home_q1, away_q1, home_q2, away_q2, team_home=None, team_away=None, top_k=8, league="nba"):
    """Predict Q3/Q4 scoring using H2H/historical data or pure pace priors when database is empty."""
    cfg = get_league_config(league)
    current_q12_home = float(home_q1 + home_q2)
    current_q12_away = float(away_q1 + away_q2)
    current_q12_total = current_q12_home + current_q12_away
    current_home_share = current_q12_home / current_q12_total if current_q12_total > 0 else 0.5
    current_pace = current_q12_total / 2.0

    required = ["home_q1", "home_q2", "home_q3", "home_q4", "away_q1", "away_q2", "away_q3", "away_q4"]
    has_hist = (
        df is not None
        and not df.empty
        and all(col in df.columns for col in required)
        and len(df.dropna(subset=required)) >= 1
    )

    if not has_hist:
        prior_q = float(cfg["prior_q"])
        blended_q = 0.65 * current_pace + 0.35 * prior_q
        home_split = float(np.clip(0.6 * current_home_share + 0.4 * 0.5, 0.35, 0.65))
        home_q3 = blended_q * home_split
        away_q3 = blended_q * (1.0 - home_split)
        home_q4 = blended_q * home_split
        away_q4 = blended_q * (1.0 - home_split)
        home_final = current_q12_home + home_q3 + home_q4
        away_final = current_q12_away + away_q3 + away_q4
        tempo_mult = current_pace / prior_q if prior_q > 0 else 1.0
        return {
            "predicted_home_q3": round(home_q3, 1),
            "predicted_away_q3": round(away_q3, 1),
            "predicted_home_q4": round(home_q4, 1),
            "predicted_away_q4": round(away_q4, 1),
            "predicted_final_total": round(home_q3 + away_q3 + home_q4 + away_q4, 1),
            "predicted_home_final": round(home_final, 1),
            "predicted_away_final": round(away_final, 1),
            "winner": "Ev Sahibi" if home_final >= away_final else "Deplasman",
            "tempo_multiplier": round(tempo_mult, 2),
            "neighbor_count": 0,
            "tempo_note": "Canlı Tempo Bazlı",
            "recent_shift_signal": 0.0,
            "recent_shift_volatility": 0.0,
            "shift_note": "Sıfır kurulum — Canlı tempo modeli aktif",
        }

    hist_raw = df.copy()
    for c in required:
        hist_raw[c] = pd.to_numeric(hist_raw[c], errors="coerce")
    hist_raw = hist_raw.dropna(subset=required).copy()
    if hist_raw.empty:
        return {
            "predicted_q3_home": round(q3_home, 1),
            "predicted_q3_away": round(q3_away, 1),
            "predicted_q4_home": round(q4_home, 1),
            "predicted_q4_away": round(q4_away, 1),
            "predicted_total": round(tot_final, 1),
            "predicted_home_final": round(home_final, 1),
            "predicted_away_final": round(away_final, 1),
            "winner": "Ev Sahibi" if home_final >= away_final else "Deplasman",
            "tempo_multiplier": round(tempo_mult, 2),
            "neighbor_count": 0,
            "tempo_note": "Canlı Tempo Bazlı",
            "recent_shift_signal": 0.0,
            "recent_shift_volatility": 0.0,
            "shift_note": "Sıfır kurulum — Canlı tempo modeli aktif",
        }
    if team_home and team_away:
        h2h_mask = (
            ((hist_raw["home_team"] == team_home) & (hist_raw["away_team"] == team_away))
            | ((hist_raw["home_team"] == team_away) & (hist_raw["away_team"] == team_home))
        )
        df_h2h = hist_raw[h2h_mask].copy()
        if len(df_h2h) >= 3:
            hist = df_h2h
        else:
            either_mask = hist_raw["home_team"].isin([team_home, team_away]) | hist_raw["away_team"].isin([team_home, team_away])
            df_either = hist_raw[either_mask].copy()
            hist = df_either if len(df_either) >= 3 else hist_raw
    else:
        hist = hist_raw

    for c in required:
        hist[c] = hist[c].astype(float)

    hist["q12_home"] = hist["home_q1"] + hist["home_q2"]
    hist["q12_away"] = hist["away_q1"] + hist["away_q2"]
    hist["q12_total"] = hist["q12_home"] + hist["q12_away"]
    hist["q12_gap"] = hist["q12_home"] - hist["q12_away"]
    hist["q12_pace"] = hist["q12_total"] / 2.0
    hist["q34_total"] = hist["home_q3"] + hist["home_q4"] + hist["away_q3"] + hist["away_q4"]
    hist["q3_total"] = hist["home_q3"] + hist["away_q3"]
    hist["q4_total"] = hist["home_q4"] + hist["away_q4"]
    hist["q34_home_share"] = (hist["home_q3"] + hist["home_q4"]) / hist["q34_total"].replace(0, np.nan)
    hist["q3_share"] = hist["q3_total"] / hist["q34_total"].replace(0, np.nan)

    current_q12_gap = current_q12_home - current_q12_away
    hist["distance"] = (
        0.45 * np.abs(hist["q12_total"] - current_q12_total) / max(current_q12_total, 1.0)
        + 0.30 * np.abs(hist["q12_pace"] - current_pace) / max(current_pace, 1.0)
        + 0.25 * np.abs(hist["q12_gap"] - current_q12_gap) / max(abs(current_q12_gap), 1.0)
    ).fillna(999.0)

    candidates = hist.sort_values("distance").head(min(top_k, len(hist)))
    weights = 1.0 / (1.0 + candidates["distance"].astype(float))
    weights = weights / weights.sum()

    base_q34_total = float(np.average(candidates["q34_total"], weights=weights))
    neighbor_pace = float(np.average(candidates["q12_pace"], weights=weights))
    neighbor_home_share = float(np.average(candidates["q34_home_share"].fillna(0.5), weights=weights))
    neighbor_q3_share = float(np.average(candidates["q3_share"].fillna(0.5), weights=weights))

    tempo_multiplier = float(np.clip(current_pace / neighbor_pace if neighbor_pace > 0 else 1.0, 0.75, 1.25))
    adjusted_q34_total = base_q34_total * tempo_multiplier
    adjusted_q3_total = adjusted_q34_total * float(np.clip(neighbor_q3_share, 0.35, 0.65))
    adjusted_q4_total = max(15.0, adjusted_q34_total - adjusted_q3_total)

    home_split = float(np.clip(0.6 * current_home_share + 0.4 * neighbor_home_share, 0.3, 0.7))
    home_q3 = adjusted_q3_total * home_split
    away_q3 = adjusted_q3_total * (1.0 - home_split)
    home_q4 = adjusted_q4_total * home_split
    away_q4 = adjusted_q4_total * (1.0 - home_split)

    home_final = current_q12_home + home_q3 + home_q4
    away_final = current_q12_away + away_q3 + away_q4

    return {
        "predicted_home_q3": round(home_q3, 1),
        "predicted_away_q3": round(away_q3, 1),
        "predicted_home_q4": round(home_q4, 1),
        "predicted_away_q4": round(away_q4, 1),
        "predicted_final_total": round(adjusted_q3_total + adjusted_q4_total, 1),
        "predicted_home_final": round(home_final, 1),
        "predicted_away_final": round(away_final, 1),
        "winner": "Ev Sahibi" if home_final >= away_final else "Deplasman",
        "tempo_multiplier": round(tempo_multiplier, 2),
        "neighbor_count": int(len(candidates)),
        "tempo_note": "Tempo hızlı" if tempo_multiplier > 1.04 else ("Tempo yavaş" if tempo_multiplier < 0.96 else "Tempo dengeli"),
        "recent_shift_signal": 0.0,
        "recent_shift_volatility": 0.0,
        "shift_note": "H2H + Canlı Tempo Hibrit Modeli",
    }


def predict_live_inplay(
    df,
    team_home,
    team_away,
    current_quarter,
    home_live_score,
    away_live_score,
    remaining_mins,
    completed_quarters,
    quarter_length=None,
    top_k=15,
    league="nba",
):
    """
    Predict live match final score & barem distribution.
    Works 100% seamlessly both when `df` has historical matches AND when `df` is completely empty (clean slate).
    """
    cfg = get_league_config(league)
    if quarter_length is None:
        quarter_length = float(cfg["quarter_length"])
    else:
        quarter_length = float(quarter_length)
    prior_q = float(cfg["prior_q"])
    total_match_len = 4.0 * quarter_length
    current_quarter = max(1, min(4, int(current_quarter or 1)))
    remaining_mins = float(np.clip(remaining_mins, 0.0, quarter_length))
    elapsed_in_q = max(0.05, quarter_length - remaining_mins)

    current_q_total = float(home_live_score + away_live_score)
    current_q_pace = current_q_total / elapsed_in_q

    required = ["home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4"]
    has_hist = (
        df is not None
        and not df.empty
        and all(col in df.columns for col in required)
        and len(df.dropna(subset=required)) >= 1
    )

    hist = pd.DataFrame()
    if has_hist:
        hist_raw = df.copy()
        for c in required:
            hist_raw[c] = pd.to_numeric(hist_raw[c], errors="coerce")
        hist_raw = hist_raw.dropna(subset=required).copy()
        if not hist_raw.empty and team_home and team_away:
            h2h_mask = (
                ((hist_raw["home_team"] == team_home) & (hist_raw["away_team"] == team_away))
                | ((hist_raw["home_team"] == team_away) & (hist_raw["away_team"] == team_home))
            )
            df_h2h = hist_raw[h2h_mask].copy()
            if len(df_h2h) >= 3:
                hist = df_h2h
            else:
                either_mask = hist_raw["home_team"].isin([team_home, team_away]) | hist_raw["away_team"].isin([team_home, team_away])
                df_either = hist_raw[either_mask].copy()
                hist = df_either if len(df_either) >= 3 else hist_raw
        else:
            hist = hist_raw

    # Calculate baseline quarter score (League prior ~53.0 for 12m NBA, ~36.3 for 10m Euro)
    if not hist.empty:
        avg_match_total = sum(hist[c].mean() for c in required)
        league_avg_q_score = float(avg_match_total) / 4.0
    else:
        if completed_quarters:
            comp_q_avg = sum(h + a for h, a in completed_quarters) / len(completed_quarters)
            league_avg_q_score = 0.65 * comp_q_avg + 0.35 * prior_q
        else:
            league_avg_q_score = prior_q

    default_q_pace = league_avg_q_score / quarter_length
    blend_weight = float(np.clip(elapsed_in_q / quarter_length, 0.0, 1.0))
    blended_q_pace = blend_weight * current_q_pace + (1.0 - blend_weight) * default_q_pace
    predicted_q_rem = blended_q_pace * remaining_mins

    home_share = (home_live_score / current_q_total) if current_q_total > 0 else 0.5
    home_share = float(np.clip(home_share, 0.35, 0.65))

    predicted_home_q_final = float(home_live_score) + (predicted_q_rem * home_share)
    predicted_away_q_final = float(away_live_score) + (predicted_q_rem * (1.0 - home_share))

    pred_subsequent = {}
    if not hist.empty:
        hist["q1_total"] = hist["home_q1"] + hist["away_q1"]
        hist["q1_gap"] = hist["home_q1"] - hist["away_q1"]
        hist["q12_total"] = hist["q1_total"] + hist["home_q2"] + hist["away_q2"]
        hist["q12_gap"] = hist["q1_gap"] + hist["home_q2"] - hist["away_q2"]
        hist["q123_total"] = hist["q12_total"] + hist["home_q3"] + hist["away_q3"]
        hist["q123_gap"] = hist["q12_gap"] + hist["home_q3"] - hist["away_q3"]
        hist["q34_total"] = hist["home_q3"] + hist["away_q3"] + hist["home_q4"] + hist["away_q4"]

        num_completed = len(completed_quarters)
        if num_completed == 1:
            h_q1, a_q1 = completed_quarters[0]
            q1_t, q1_g = h_q1 + a_q1, h_q1 - a_q1
            hist["distance"] = 0.6 * np.abs(hist["q1_total"] - q1_t) / max(q1_t, 1) + 0.4 * np.abs(hist["q1_gap"] - q1_g) / max(abs(q1_g), 1)
        elif num_completed == 2:
            h_q1, a_q1 = completed_quarters[0]
            h_q2, a_q2 = completed_quarters[1]
            q12_t, q12_g = h_q1 + a_q1 + h_q2 + a_q2, (h_q1 - a_q1) + (h_q2 - a_q2)
            hist["distance"] = 0.6 * np.abs(hist["q12_total"] - q12_t) / max(q12_t, 1) + 0.4 * np.abs(hist["q12_gap"] - q12_g) / max(abs(q12_g), 1)
        elif num_completed >= 3:
            h_q1, a_q1 = completed_quarters[0]
            h_q2, a_q2 = completed_quarters[1]
            h_q3, a_q3 = completed_quarters[2]
            q123_t = h_q1 + a_q1 + h_q2 + a_q2 + h_q3 + a_q3
            q123_g = (h_q1 - a_q1) + (h_q2 - a_q2) + (h_q3 - a_q3)
            hist["distance"] = 0.6 * np.abs(hist["q123_total"] - q123_t) / max(q123_t, 1) + 0.4 * np.abs(hist["q123_gap"] - q123_g) / max(abs(q123_g), 1)
        else:
            hist["distance"] = 1.0

        hist["distance"] = hist["distance"].fillna(999.0)
        candidates = hist.sort_values("distance").head(min(top_k, len(hist)))
        weights = 1.0 / (1.0 + candidates["distance"].astype(float))
        weights = weights / weights.sum()

        for q_idx in range(current_quarter + 1, 5):
            pred_subsequent[f"home_q{q_idx}"] = float(np.average(candidates[f"home_q{q_idx}"], weights=weights))
            pred_subsequent[f"away_q{q_idx}"] = float(np.average(candidates[f"away_q{q_idx}"], weights=weights))
        hist_avg_total = float(np.average(candidates["q12_total"] + candidates["q34_total"], weights=weights))
    else:
        # Pure live pace + CyberLeague baseline
        for q_idx in range(current_quarter + 1, 5):
            pred_subsequent[f"home_q{q_idx}"] = league_avg_q_score * home_share
            pred_subsequent[f"away_q{q_idx}"] = league_avg_q_score * (1.0 - home_share)
        hist_avg_total = league_avg_q_score * 4.0

    cur_tot_h = sum(h_q for h_q, _ in completed_quarters) + float(home_live_score)
    cur_tot_a = sum(a_q for _, a_q in completed_quarters) + float(away_live_score)
    cur_tot = cur_tot_h + cur_tot_a
    live_gap = abs(cur_tot_h - cur_tot_a)

    total_elapsed_mins = (current_quarter - 1) * quarter_length + elapsed_in_q
    total_rem_mins = max(0.25, total_match_len - total_elapsed_mins)

    # Calibrated Zero-Bias Empirical Model (v7.0 on 358 NBA 2K26 + 59 NBA 2K27 + 345 Euro = 762 live matches)
    _lg_k = str(league or "nba").lower()
    is_nba_fmt = _lg_k != "euro"
    if _lg_k == "nba2k27":
        gap_slope_q4 = -0.225
        base_full_sd = 16.2
        default_q4_gap0 = 54.80
        _anchor_tot_cfg = 208.3
    elif is_nba_fmt:
        gap_slope_q4 = -0.358
        base_full_sd = 17.5
        default_q4_gap0 = 56.80
        _anchor_tot_cfg = 209.6
    else:
        gap_slope_q4 = -0.345
        base_full_sd = 14.8
        default_q4_gap0 = 40.80
        _anchor_tot_cfg = 145.7
    lg_tot_est = float(hist_avg_total)
    lg_q4_gap0 = default_q4_gap0
    team_tot_delta = 0.0
    team_q4_delta = 0.0

    try:
        if df is not None and not df.empty:
            _df_c = df.dropna(subset=["home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"]).copy()
            for _col in ("home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"):
                _df_c[_col] = pd.to_numeric(_df_c[_col], errors="coerce")
            _df_c = _df_c.dropna(subset=["total_score", "home_q4", "away_q4"])
            if len(_df_c) >= 40:
                if "date" in _df_c.columns:
                    _df_c = _df_c.sort_values("date")
                _df_rec = _df_c.tail(240)
                _tot_all = float(_df_c["total_score"].mean())
                _tot_rec = float(_df_rec["total_score"].mean())
                _anchor_tot = _anchor_tot_cfg
                lg_tot_est = 0.55 * _tot_rec + 0.15 * _tot_all + 0.30 * _anchor_tot

                def _calc_q4_gap0(_sub):
                    _q4 = _sub["home_q4"] + _sub["away_q4"]
                    _g3 = ((_sub["home_q1"] + _sub["home_q2"] + _sub["home_q3"]) - (_sub["away_q1"] + _sub["away_q2"] + _sub["away_q3"])).abs().clip(upper=30.0)
                    _g_sl = 0.22 if _lg_k == "nba2k27" else (0.32 if is_nba_fmt else 0.30)
                    return float((_q4 + _g_sl * _g3).mean())

                _dyn_q4_gap0 = 0.70 * _calc_q4_gap0(_df_rec) + 0.30 * _calc_q4_gap0(_df_c)
                lg_q4_gap0 = 0.50 * _dyn_q4_gap0 + 0.50 * default_q4_gap0
                _lg_q4_raw = float((_df_c["home_q4"] + _df_c["away_q4"]).mean())

                _p1 = _df_c[(_df_c["home_team"] == team_home) | (_df_c["away_team"] == team_home)]
                _p2 = _df_c[(_df_c["home_team"] == team_away) | (_df_c["away_team"] == team_away)]
                if len(_p1) >= 8 and len(_p2) >= 8:
                    team_tot_delta = 0.5 * (float(_p1["total_score"].mean()) + float(_p2["total_score"].mean())) - _tot_all
                    team_q4_delta = 0.5 * (float((_p1["home_q4"] + _p1["away_q4"]).mean()) + float((_p2["home_q4"] + _p2["away_q4"]).mean())) - _lg_q4_raw
    except Exception:
        pass

    lg_early_ppm = max(2.5, (lg_tot_est - (lg_q4_gap0 - 4.4)) / (3.0 * quarter_length))
    live_ppm = (cur_tot / total_elapsed_mins) if total_elapsed_mins >= 2.0 else lg_early_ppm
    pace_dev = (live_ppm - lg_early_ppm) * min(1.0, total_elapsed_mins / (2.0 * quarter_length))

    q4_rem = min(quarter_length, total_rem_mins)
    q13_rem = max(0.0, total_rem_mins - quarter_length)

    q13_rate = (
        lg_early_ppm
        + 0.14 * pace_dev
        + 0.30 * (team_tot_delta / total_match_len)
        - (0.008 * min(25.0, live_gap) * (total_elapsed_mins / (3.0 * quarter_length)))
    )
    q13_pts = q13_rem * max(2.0, q13_rate)

    gap_w = float(np.clip((total_elapsed_mins - 0.5 * quarter_length) / (2.3 * quarter_length), 0.15, 1.0))
    q4_full_exp = lg_q4_gap0 + (0.14 * pace_dev * quarter_length) + gap_w * gap_slope_q4 * min(30.0, live_gap) + 0.35 * team_q4_delta
    q4_pts = max(1.0, q4_full_exp) * (q4_rem / quarter_length)
    pred_rem_total = max(0.5, q13_pts + q4_pts)

    # Distribute projected remaining points across Home / Away proportionally to live quarter projection
    raw_rem_h = max(0.5, predicted_home_q_final - float(home_live_score)) + sum(
        pred_subsequent.get(f"home_q{q_idx}", 0.0) for q_idx in range(current_quarter + 1, 5)
    )
    raw_rem_a = max(0.5, predicted_away_q_final - float(away_live_score)) + sum(
        pred_subsequent.get(f"away_q{q_idx}", 0.0) for q_idx in range(current_quarter + 1, 5)
    )
    raw_rem_sum = raw_rem_h + raw_rem_a
    h_share = float(np.clip(raw_rem_h / raw_rem_sum, 0.38, 0.62)) if raw_rem_sum > 0 else 0.50

    pred_home_final = cur_tot_h + pred_rem_total * h_share
    pred_away_final = cur_tot_a + pred_rem_total * (1.0 - h_share)

    # Calibrated power-law residual std across Q1 -> Q4 (matches empirical 18.0->10.8 NBA, 15.2->8.9 Euro)
    rem_fraction = float(np.clip(total_rem_mins / total_match_len, 0.03, 1.0))
    std_total_target = max(3.2, base_full_sd * (rem_fraction ** 0.38))
    team_std_dev = max(2.2, std_total_target / float(np.sqrt(2.0)))

    rng = np.random.default_rng(int(abs(pred_home_final * 100 + pred_away_final * 10)) % 100000)
    sim_size = 10000
    sim_home_totals = rng.normal(pred_home_final, team_std_dev, sim_size)
    sim_away_totals = rng.normal(pred_away_final, team_std_dev, sim_size)
    sim_totals = sim_home_totals + sim_away_totals

    b_ranges = cfg["barem_ranges"]
    n_b_ranges = len(b_ranges)
    barem_probs = {}
    for b_idx, (b_name, b_low, b_high) in enumerate(b_ranges):
        if b_idx == 0:
            prob = float((sim_totals <= b_high).mean()) * 100.0
        elif b_idx == n_b_ranges - 1:
            prob = float((sim_totals > b_low).mean()) * 100.0
        else:
            prob = float(((sim_totals > b_low) & (sim_totals <= b_high)).mean()) * 100.0
        barem_probs[b_name] = round(prob, 1)

    clusters = {}
    for c_name, c_low, c_high, _ in cfg["cluster_ranges"]:
        c_prob = float(((sim_totals > c_low) & (sim_totals <= c_high)).mean()) * 100.0
        clusters[c_name] = round(c_prob, 1)

    best_cluster_name = max(clusters.keys(), key=lambda k: clusters[k])
    best_cluster_prob = clusters[best_cluster_name]

    prob_home_win = float((sim_home_totals > sim_away_totals).mean())
    prob_away_win = 1.0 - prob_home_win

    return {
        "predicted_home_q_final": round(predicted_home_q_final, 1),
        "predicted_away_q_final": round(predicted_away_q_final, 1),
        "predicted_subsequent": {k: round(v, 1) for k, v in pred_subsequent.items()},
        "predicted_home_final": round(pred_home_final, 1),
        "predicted_away_final": round(pred_away_final, 1),
        "predicted_total": round(pred_home_final + pred_away_final, 1),
        "prob_home_win": round(prob_home_win * 100.0, 1),
        "prob_away_win": round(prob_away_win * 100.0, 1),
        "sim_totals": sim_totals,
        "barem_probs": barem_probs,
        "clusters": clusters,
        "best_cluster_name": best_cluster_name,
        "best_cluster_prob": round(best_cluster_prob, 1),
        "h2h_sample_size": int(len(hist)),
        "sample_size": int(len(hist)),
        "elapsed_match_mins": round(total_elapsed_mins, 1),
        "remaining_match_mins": round(total_rem_mins, 1),
    }


def resolve_effective_barem_odds(odds_dict: dict, barem_probs: dict = None, league: str = "nba") -> tuple[dict, str]:
    """
    Detects when 1xBet has locked (99.0 / closed) an extreme barem (e.g. '231+' / '160 Or Less' or '170'den Fazla' / '131'den Az')
    because it has become virtually certain (< 1.04 odd, 'Banko Kapandı'), preventing the system
    from falsely mistaking the remaining high-odd neighbor (e.g. 9.67 or 20.00) for the 1st Favorite.
    Returns (effective_valid_odds, locked_fav_barem).
    """
    cfg = get_league_config(league)
    barem_names = [b[0] for b in cfg["barem_ranges"]]
    b_low_ext = barem_names[0]
    b_low_p1 = barem_names[1]
    b_low_p2 = barem_names[2]
    b_high_ext = barem_names[-1]
    b_high_m1 = barem_names[-2]
    b_high_m2 = barem_names[-3]

    raw = {}
    for k, v in (odds_dict or {}).items():
        try:
            raw[k] = float(v) if v is not None else 99.0
        except Exception:
            raw[k] = 99.0

    valid = {k: v for k, v in raw.items() if 1.001 <= v < 85.0}
    probs = {}
    for k, v in (barem_probs or {}).items():
        try:
            probs[k] = float(v or 0.0)
        except Exception:
            probs[k] = 0.0

    locked_fav = None

    o_high = raw.get(b_high_ext, 99.0)
    o_hm1 = raw.get(b_high_m1, 99.0)
    o_hm2 = raw.get(b_high_m2, 99.0)
    lower_open = [raw.get(b, 99.0) for b in barem_names[:-3] if 1.001 <= raw.get(b, 99.0) < 85.0]
    if o_high >= 85.0 and not lower_open:
        if (1.001 <= o_hm1 < 85.0 and (o_hm2 >= 85.0 or o_hm2 > o_hm1) and o_hm1 >= 3.80) or probs.get(b_high_ext, 0.0) >= 60.0:
            imp_odd = round(max(1.01, min(1.06, 1.0 / max(0.94, 1.06 - (1.0 / max(o_hm1, 4.0))))), 3) if o_hm1 < 85.0 else 1.01
            valid[b_high_ext] = imp_odd
            locked_fav = b_high_ext

    o_low = raw.get(b_low_ext, 99.0)
    o_lp1 = raw.get(b_low_p1, 99.0)
    o_lp2 = raw.get(b_low_p2, 99.0)
    upper_open = [raw.get(b, 99.0) for b in barem_names[3:] if 1.001 <= raw.get(b, 99.0) < 85.0]
    if o_low >= 85.0 and not upper_open:
        if (1.001 <= o_lp1 < 85.0 and (o_lp2 >= 85.0 or o_lp2 > o_lp1) and o_lp1 >= 3.80) or probs.get(b_low_ext, 0.0) >= 60.0:
            imp_odd = round(max(1.01, min(1.06, 1.0 / max(0.94, 1.06 - (1.0 / max(o_lp1, 4.0))))), 3) if o_lp1 < 85.0 else 1.01
            valid[b_low_ext] = imp_odd
            locked_fav = b_low_ext

    return valid, locked_fav


def analyze_barem_symmetry(
    odds_dict: dict,
    barem_probs: dict,
    league: str = "nba",
    proj_tot: float = None,
    score_gap: int = 0,
    drift: int = 0,
) -> dict:
    """
    Analyze 1xBet Barem Odds vs Calibrated Model Probabilities (v5.5 Wing Pull & Selective Value Engine).
    """
    cfg = get_league_config(league)
    valid_odds, locked_fav = resolve_effective_barem_odds(odds_dict, barem_probs, league=league)
    sorted_by_odds = sorted(valid_odds.items(), key=lambda x: x[1])

    bookie_fav_1 = sorted_by_odds[0][0] if len(sorted_by_odds) >= 1 else None
    bookie_fav_1_odd = sorted_by_odds[0][1] if len(sorted_by_odds) >= 1 else 99.0
    bookie_fav_2 = sorted_by_odds[1][0] if len(sorted_by_odds) >= 2 else None
    bookie_fav_2_odd = sorted_by_odds[1][1] if len(sorted_by_odds) >= 2 else 99.0

    symmetry_gap = round(bookie_fav_2_odd - bookie_fav_1_odd, 3) if (bookie_fav_1 and bookie_fav_2) else 99.0
    is_symmetric_trap = symmetry_gap <= 0.65

    dutch_odd = round(1.0 / ((1.0 / bookie_fav_1_odd) + (1.0 / bookie_fav_2_odd)), 2) if (bookie_fav_1 and bookie_fav_2) else 0.0

    barem_names = [b[0] for b in cfg["barem_ranges"]]
    proj_barem = get_range_from_score(proj_tot, league=league) if (proj_tot is not None and proj_tot > 0) else None
    is_euro_lg = str(league or "nba").lower() == "euro"

    best_value_range = None
    best_ev = -999.0
    best_score = -999.0
    value_reason = "Yeterli canlı oran verisi yok."
    has_value_edge = False
    selective_value_range = ""
    wing_lean = 0
    wing_badge = "⚖️ Alt/Üst Kanat Dengeli"
    sym_guidance = ""

    if valid_odds and bookie_fav_1:
        fav1_idx = barem_names.index(bookie_fav_1) if bookie_fav_1 in barem_names else -1
        fav2_idx = barem_names.index(bookie_fav_2) if (bookie_fav_2 and bookie_fav_2 in barem_names) else fav1_idx
        proj_idx = barem_names.index(proj_barem) if (proj_barem and proj_barem in barem_names) else fav1_idx

        b_m1_nb = barem_names[fav1_idx - 1] if fav1_idx > 0 else None
        b_p1_nb = barem_names[fav1_idx + 1] if (fav1_idx >= 0 and fav1_idx + 1 < len(barem_names)) else None
        odd_m1 = float(valid_odds.get(b_m1_nb, 99.0)) if b_m1_nb else 99.0
        odd_p1 = float(valid_odds.get(b_p1_nb, 99.0)) if b_p1_nb else 99.0
        if b_m1_nb and b_p1_nb and odd_m1 < 85.0 and odd_p1 < 85.0:
            if odd_m1 < odd_p1 - 0.18:
                wing_lean = -1
                wing_badge = f"📉 Alt Kanat Güçlü ({b_m1_nb}: {odd_m1:.2f} vs {b_p1_nb}: {odd_p1:.2f})"
            elif odd_p1 < odd_m1 - 0.18:
                wing_lean = 1
                wing_badge = f"📈 Üst Kanat Güçlü ({b_p1_nb}: {odd_p1:.2f} vs {b_m1_nb}: {odd_m1:.2f})"

        if locked_fav or bookie_fav_1_odd < 1.18:
            sym_guidance = f"🔒 {bookie_fav_1} tek başına kilit favori (1.20 altı tek başına oynanmaz)."
        elif 1.85 <= bookie_fav_1_odd < 2.10:
            sym_guidance = f"💎 {bookie_fav_1} ({bookie_fav_1_odd:.2f}) altın tek favori aralığında ({'Euro %57.3 / 1.000 TL başı +137 TL' if is_euro_lg else 'NBA %60.0 / 1.000 TL başı +206 TL'})."
        elif not is_euro_lg and 2.10 <= bookie_fav_1_odd < 2.30:
            sym_guidance = f"⚠️ NBA'de 2.10–2.30 arası 1. Favori ({bookie_fav_1}) tuzak bölgesi (%33.8); 2. Favori ({bookie_fav_2}) daha değerli."
        elif symmetry_gap <= 0.65:
            sym_guidance = f"⚖️ İki barem neredeyse eşit ({bookie_fav_1} / {bookie_fav_2}) — tek favoriye yüklenmek riskli."
        elif not is_euro_lg and bookie_fav_1_odd < 1.85:
            sym_guidance = f"⚠️ {bookie_fav_1} ({bookie_fav_1_odd:.2f}) kısa oran tuzağı; 2. Favori ({bookie_fav_2}) öne çıkıyor."
        elif not is_euro_lg:
            sym_guidance = f"💡 NBA'de 1.85–2.10 dışı tek favori risklidir; kanat yönü ve skor tahmini belirleyici."
        else:
            sym_guidance = f"📊 Euro {bookie_fav_1} ({bookie_fav_1_odd:.2f}) merkezli dağılım."

        inv_sum = sum(1.0 / o for o in valid_odds.values())
        mkt_probs = {b: (1.0 / o) / inv_sum * 100.0 for b, o in valid_odds.items()} if inv_sum > 0 else {}
        _gap_i = int(score_gap or 0)
        _lg_key = str(league or "nba").lower()
        _is_2k27 = _lg_key == "nba2k27"

        # 0) Locked / Runaway Favorite (all 3 leagues)
        if locked_fav or bookie_fav_1_odd < 1.18 or bookie_fav_2_odd >= 4.50:
            best_value_range = bookie_fav_1
            best_ev = 0.85 * bookie_fav_1_odd - 1.0
            value_reason = (
                f"{bookie_fav_1} ({'🔒 Banko Kilit' if locked_fav else f'{bookie_fav_1_odd:.2f}'}): "
                f"Site {bookie_fav_1} baremini kesin favori olarak kilitledi (oran çok düşük)"
            )
            if not is_euro_lg and not _is_2k27 and _gap_i >= 6 and bookie_fav_2 and 1.05 <= bookie_fav_2_odd < 6.0:
                has_value_edge = True
                selective_value_range = bookie_fav_2
                best_value_range = bookie_fav_2
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): NBA 1.20 altı kilit oranda 2. Favori sürpriz değeri (1.000 TL başı +238 TL)"
        elif is_euro_lg:
            # Euro v7.0 (345 matches: H1+H2 verified):
            # 1) Blowout Gap >= 21 -> 2.Fav (+270 TL; H1 +103, H2 +471)
            # 2) Twin Odds (symmetry_gap <= 0.25 & gap >= 5) -> 1.Fav (+313 TL; H1 +209, H2 +418)
            # 3) 2.10 <= o1 < 2.50 & gap >= 6 & wing_diff >= 1.0 -> 2.Fav (+433 TL; H1 +252, H2 +698)
            # 4) 1.50 <= o1 < 2.10 & 5 <= gap <= 20 & (drift < 0 or symmetry_gap <= 0.45) -> 1.Fav (+163 TL; H1 +74, H2 +258)
            if _gap_i >= 21 and bookie_fav_2 and 1.05 <= bookie_fav_2_odd < 8.0:
                best_value_range = bookie_fav_2
                has_value_edge = True
                selective_value_range = bookie_fav_2
                best_ev = 0.42 * bookie_fav_2_odd - 1.0
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): Euro 21+ sayı fark rölantisinde 2. Favori değeri (1.000 TL başı +270 TL)"
            elif symmetry_gap <= 0.25 and _gap_i >= 5:
                best_value_range = bookie_fav_1
                has_value_edge = True
                selective_value_range = bookie_fav_1
                best_ev = 0.61 * bookie_fav_1_odd - 1.0
                value_reason = f"{bookie_fav_1} ({bookie_fav_1_odd:.2f}): Euro tam ikiz oranlarda 1. Favori merkez üstünlüğü (%60.9 / 1.000 TL başı +313 TL)"
            elif 2.10 <= bookie_fav_1_odd < 2.50 and _gap_i >= 6 and (odd_m1 - odd_p1) >= 1.0 and bookie_fav_2 and bookie_fav_2_odd < 8.0:
                best_value_range = bookie_fav_2
                has_value_edge = True
                selective_value_range = bookie_fav_2
                best_ev = 0.50 * bookie_fav_2_odd - 1.0
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): Euro 2.10+ ve üst kanat açıkken 2. Favori değeri (%53.1 / 1.000 TL başı +433 TL)"
            elif 1.50 <= bookie_fav_1_odd < 2.10:
                best_value_range = bookie_fav_1
                kp1 = float(mkt_probs.get(bookie_fav_1, 0.0)) / 100.0
                best_ev = max(kp1, 0.59) * bookie_fav_1_odd - 1.0
                if 5 <= _gap_i <= 20 and (drift < 0 or symmetry_gap <= 0.45 or bookie_fav_1_odd >= 1.85):
                    has_value_edge = True
                    selective_value_range = bookie_fav_1
                value_reason = f"{bookie_fav_1} ({bookie_fav_1_odd:.2f}): Euro 1.50–2.10 altın merkez aralığı (345 maçta 1.000 TL başı +132 TL)"
            elif 2.10 <= bookie_fav_1_odd < 2.50 and _gap_i >= 6 and bookie_fav_2 and bookie_fav_2_odd < 8.0:
                best_value_range = bookie_fav_2
                has_value_edge = True
                selective_value_range = bookie_fav_2
                best_ev = 0.44 * bookie_fav_2_odd - 1.0
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): Euro 2.10–2.50 aralığında 2. Favori değeri (%44.4 / 1.000 TL başı +163 TL)"
            else:
                best_value_range = bookie_fav_1
                mp1 = float((barem_probs or {}).get(bookie_fav_1, 0.0)) / 100.0
                best_ev = mp1 * bookie_fav_1_odd - 1.0
                value_reason = f"{bookie_fav_1} ({bookie_fav_1_odd:.2f}): Euro merkez favori"
        elif _is_2k27:
            # NBA 2K27 v7.0 (59 matches: H1+H2 verified):
            if ((drift < 0 and _gap_i >= 5) or (symmetry_gap <= 0.45 and _gap_i >= 6) or bookie_fav_1_odd >= 2.50) and bookie_fav_2 and bookie_fav_2_odd < 8.0:
                best_value_range = bookie_fav_2
                has_value_edge = True
                selective_value_range = bookie_fav_2
                best_ev = 0.46 * bookie_fav_2_odd - 1.0
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): NBA 2K27 yakın oran / barem düşüşü 2. Favori değeri (1.000 TL başı +173 ile +385 TL)"
            elif (odd_p1 - odd_m1) >= 1.0 and _gap_i >= 5:
                best_value_range = bookie_fav_1
                has_value_edge = True
                selective_value_range = bookie_fav_1
                best_ev = 0.52 * bookie_fav_1_odd - 1.0
                value_reason = f"{bookie_fav_1} ({bookie_fav_1_odd:.2f}): NBA 2K27 alt kanat düşükken 1. Favori değeri (%52.6 / 1.000 TL başı +157 TL)"
            else:
                best_value_range = bookie_fav_1
                best_ev = 0.45 * bookie_fav_1_odd - 1.0
                value_reason = f"{bookie_fav_1} ({bookie_fav_1_odd:.2f}): NBA 2K27 standart favori"
        else:
            # NBA 2K26 v7.0 (358 matches: H1+H2 verified):
            if bookie_fav_2 and 1.18 <= bookie_fav_1_odd < 1.50 and bookie_fav_2_odd <= 4.80:
                best_value_range = bookie_fav_2
                best_ev = 0.45 * bookie_fav_2_odd - 1.0
                has_value_edge = True
                selective_value_range = bookie_fav_2
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): NBA 1.20–1.50 kısa oran tuzağında 2. Favori değeri (1.000 TL başı +529 TL)"
            elif 1.85 <= bookie_fav_1_odd < 2.10 and _gap_i >= 6 and drift >= 0:
                best_value_range = bookie_fav_1
                best_ev = 0.58 * bookie_fav_1_odd - 1.0
                has_value_edge = True
                selective_value_range = bookie_fav_1
                value_reason = f"{bookie_fav_1} ({bookie_fav_1_odd:.2f}): NBA 1.85–2.10 altın tek favori değeri (%57.4 / 1.000 TL başı +159 TL)"
            elif bookie_fav_2 and 2.10 <= bookie_fav_1_odd < 2.30 and _gap_i >= 6 and bookie_fav_2_odd < 6.0:
                best_value_range = bookie_fav_2
                best_ev = 0.41 * bookie_fav_2_odd - 1.0
                has_value_edge = True
                selective_value_range = bookie_fav_2
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): NBA 2.10–2.30 aralığında 2. Favori değeri (%40.2 / 1.000 TL başı +162 TL)"
            elif bookie_fav_2 and bookie_fav_1_odd >= 2.50 and _gap_i >= 6 and bookie_fav_2_odd < 6.0:
                best_value_range = bookie_fav_2
                best_ev = 0.48 * bookie_fav_2_odd - 1.0
                has_value_edge = True
                selective_value_range = bookie_fav_2
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): NBA 2.50+ kararsız favoride 2. Favori değeri (%50.0 / 1.000 TL başı +407 TL)"
            elif bookie_fav_2 and symmetry_gap <= 0.25 and _gap_i >= 6 and bookie_fav_2_odd < 6.0:
                best_value_range = bookie_fav_2
                best_ev = 0.44 * bookie_fav_2_odd - 1.0
                has_value_edge = True
                selective_value_range = bookie_fav_2
                value_reason = f"{bookie_fav_2} ({bookie_fav_2_odd:.2f}): NBA tam ikiz oranlarda 2. Favori değeri (%44.4 / 1.000 TL başı +128 TL)"
            else:
                cands = []
                for bn in (bookie_fav_1, bookie_fav_2, b_m1_nb, b_p1_nb):
                    if bn and bn not in cands and bn in valid_odds:
                        cands.append(bn)
                for b_name in cands:
                    odd = valid_odds.get(b_name, 99.0)
                    if odd > 4.50 and b_name != bookie_fav_1:
                        continue
                    mp = float((barem_probs or {}).get(b_name, 0.0)) / 100.0
                    kp = float(mkt_probs.get(b_name, 0.0)) / 100.0
                    tp = 0.60 * kp + 0.40 * mp
                    if b_name == b_m1_nb and fav2_idx < fav1_idx:
                        tp += 0.05
                    ev = tp * odd - 1.0
                    sc = tp + 0.50 * ev
                    if sc > best_score:
                        best_score = sc
                        best_ev = ev
                        best_value_range = b_name
                        value_reason = f"{b_name} ({odd:.2f}): Model (%{mp*100:.0f}) vs Piyasa (%{kp*100:.0f}) değeri"

    if not best_value_range:
        if bookie_fav_1:
            best_value_range = bookie_fav_1
            value_reason = f"{bookie_fav_1}: Aktif 1. Favori barem"
        elif barem_probs:
            best_value_range = max(barem_probs.keys(), key=lambda k: barem_probs[k])
            value_reason = f"{best_value_range}: Modelin en yüksek olasılıklı baremi"
        else:
            best_value_range = cfg["default_barem"]

    highest_prob_range = bookie_fav_1 if bookie_fav_1 else (
        max(barem_probs.keys(), key=lambda k: barem_probs[k]) if barem_probs else cfg["default_barem"]
    )
    value_status_label = (
        f"🔥 Değer Fırsatı: {selective_value_range}"
        if has_value_edge and selective_value_range
        else f"⚖️ Büro Dengeli ({best_value_range})"
    )

    return {
        "highest_prob_range": highest_prob_range,
        "bookie_fav_1": bookie_fav_1,
        "bookie_fav_1_odd": bookie_fav_1_odd,
        "bookie_fav_2": bookie_fav_2,
        "bookie_fav_2_odd": bookie_fav_2_odd,
        "is_locked_fav": bool(locked_fav),
        "symmetry_gap": symmetry_gap,
        "is_symmetric_trap": is_symmetric_trap,
        "dutch_odd": dutch_odd,
        "value_range": best_value_range,
        "has_value_edge": has_value_edge,
        "selective_value_range": selective_value_range,
        "value_status_label": value_status_label,
        "value_reason": value_reason,
        "wing_lean": wing_lean,
        "wing_badge": wing_badge,
        "sym_guidance": sym_guidance,
        "best_ev_pct": round(best_ev * 100.0, 1) if best_ev > -900 else 0.0,
    }


def analyze_sequential_barem_wave(
    df_matches: pd.DataFrame,
    df_log: pd.DataFrame = None,
    current_fav_barem: str = None,
    custom_seq: list = None,
    league: str = "nba",
    odds_map: dict = None,
    proj_tot: float = None,
) -> dict:
    """
    Analyze consecutive match-to-match closing Expected Barem vs Actual Final Barem shifts:
      0  -> Expected closing favorite barem hit (Merkez Barem)
     -1  -> Shifted 1 barem bracket DOWN (-1 Alt Barem)
     +1  -> Shifted 1 barem bracket UP (+1 Üst Barem)
    Uses Markov & N-gram sequence matching across completed historical matches.
    """
    cfg = get_league_config(league)
    barem_names = [b[0] for b in cfg["barem_ranges"]]
    def_b = cfg["default_barem"]
    prior_q = float(cfg["prior_q"])
    close_min_int = int(cfg["close_min_int"])
    dec_min = int(cfg["decision_min"])
    w0, wm1, wp1, wm2, wp2 = cfg["default_wave_probs"]

    def _shift_b(base_b: str, delta: int) -> str:
        if base_b not in barem_names:
            return "-"
        idx = max(0, min(len(barem_names) - 1, barem_names.index(base_b) + delta))
        return barem_names[idx]

    if df_matches is None or df_matches.empty:
        b_ref = current_fav_barem or def_b
        return {
            "recent_chain": [],
            "last_5_codes": [],
            "last_5_labels": "-",
            "pattern_matches_count": 0,
            "prob_0": w0,
            "prob_minus_1": wm1,
            "prob_plus_1": wp1,
            "prob_minus_2": wm2,
            "prob_plus_2": wp2,
            "wave_direction": "0 / -1 (Beklenen veya 1 Alt Barem)",
            "target_0_barem": b_ref,
            "target_minus_1_barem": _shift_b(b_ref, -1),
            "target_plus_1_barem": _shift_b(b_ref, 1),
            "target_minus_2_barem": _shift_b(b_ref, -2),
            "target_plus_2_barem": _shift_b(b_ref, 2),
            "recommended_pair": f"{_shift_b(b_ref, -1)} & {b_ref}",
            "recommended_pair_barems": [_shift_b(b_ref, -1), b_ref],
            "recommended_pair_eff_odd": 0.0,
            "recommended_pair_prob": round(w0 + wm1, 1),
        }

    barem_to_col = cfg["barem_odd_cols"]
    barem_to_prob_col = cfg["barem_prob_cols"]
    logged_closing_fav = {}
    logged_closing_odd = {}
    logged_closing_min = {}
    logged_closing_locked = {}
    logged_opening_fav = {}
    if df_log is not None and not df_log.empty and "game_id" in df_log.columns:
        for gid, grp in df_log.groupby("game_id"):
            first_r = grp.iloc[0]
            last_r = grp.iloc[-1]

            o_first = {b: float(first_r.get(barem_to_col[b], 99.0) or 99.0) for b in barem_names}
            p_first = {b: float(first_r.get(barem_to_prob_col[b], 0.0) or 0.0) for b in barem_names}
            eff_first, _ = resolve_effective_barem_odds(o_first, p_first, league=league)
            first_fav = min(eff_first, key=eff_first.get) if eff_first else str(first_r.get("highest_prob_range", "")).strip()

            o_last = {b: float(last_r.get(barem_to_col[b], 99.0) or 99.0) for b in barem_names}
            p_last = {b: float(last_r.get(barem_to_prob_col[b], 0.0) or 0.0) for b in barem_names}
            eff_last, locked_b = resolve_effective_barem_odds(o_last, p_last, league=league)
            fav_val = min(eff_last, key=eff_last.get) if eff_last else str(last_r.get("highest_prob_range", "")).strip()

            if fav_val in barem_names:
                gid_clean = str(gid).strip()
                logged_closing_fav[gid_clean] = fav_val
                logged_opening_fav[gid_clean] = first_fav if first_fav in barem_names else fav_val
                odd_col = barem_to_col.get(fav_val, "")
                odd_val = float(last_r.get(odd_col, 0.0) or 0.0) if odd_col else 0.0
                if locked_b == fav_val or odd_val >= 85.0:
                    logged_closing_locked[gid_clean] = True
                    # Look back in earlier rows of the match for the last open odd before lock
                    last_open_o = eff_last.get(fav_val, 1.04)
                    for _, prev_r in grp.iloc[::-1].iterrows():
                        po = float(prev_r.get(odd_col, 99.0) or 99.0)
                        if 1.001 <= po < 85.0:
                            last_open_o = po
                            break
                    logged_closing_odd[gid_clean] = round(last_open_o, 2)
                else:
                    logged_closing_locked[gid_clean] = False
                    logged_closing_odd[gid_clean] = round(odd_val, 2) if 1.0 < odd_val < 90.0 else 0.0
                logged_closing_min[gid_clean] = str(last_r.get("check_time", "")).strip()

    req_cols = ["home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"]
    hist = df_matches.copy()
    for c in req_cols:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.dropna(subset=req_cols).copy()
    hist = hist[hist["total_score"] > 0].copy()
    if hist.empty:
        b_ref = current_fav_barem or def_b
        return {
            "recent_chain": [],
            "last_5_codes": [],
            "last_5_labels": "-",
            "pattern_matches_count": 0,
            "prob_0": w0,
            "prob_minus_1": wm1,
            "prob_plus_1": wp1,
            "prob_minus_2": wm2,
            "prob_plus_2": wp2,
            "wave_direction": "0 / -1 (Beklenen veya 1 Alt Barem)",
            "target_0_barem": b_ref,
            "target_minus_1_barem": _shift_b(b_ref, -1),
            "target_plus_1_barem": _shift_b(b_ref, 1),
            "target_minus_2_barem": _shift_b(b_ref, -2),
            "target_plus_2_barem": _shift_b(b_ref, 2),
            "recommended_pair": f"{_shift_b(b_ref, -1)} & {b_ref}",
            "recommended_pair_barems": [_shift_b(b_ref, -1), b_ref],
            "recommended_pair_eff_odd": 0.0,
            "recommended_pair_prob": round(w0 + wm1, 1),
        }
    if "date" in hist.columns:
        hist = hist.sort_values(by="date", ascending=True).reset_index(drop=True)

    hist["s36"] = (
        hist["home_q1"].astype(float)
        + hist["away_q1"].astype(float)
        + hist["home_q2"].astype(float)
        + hist["away_q2"].astype(float)
        + hist["home_q3"].astype(float)
        + hist["away_q3"].astype(float)
    )
    hist["q4_tot"] = hist["home_q4"].astype(float) + hist["away_q4"].astype(float)
    # 1xBet Q3-closing expectation: Q3 end score + blended Q4 pace (prior_q baseline + current pace)
    hist["exp_tot_36"] = hist["s36"] + (0.65 * prior_q + 0.35 * (hist["s36"] / 3.0))

    exp_barems = []
    exp_odds = []
    sources = []
    for _, r in hist.iterrows():
        gid_s = str(r.get("game_id", "")).strip()
        if gid_s in logged_closing_fav:
            exp_barems.append(logged_closing_fav[gid_s])
            o_v = logged_closing_odd.get(gid_s, 0.0)
            c_m = logged_closing_min.get(gid_s, f"{dec_min}.Dk")
            is_lck = logged_closing_locked.get(gid_s, False)
            exp_odds.append(o_v)
            if is_lck:
                sources.append(f"🔒 Banko Kilit — Oran Kapandı ({c_m} | Son Açık Oran: {o_v:.2f})")
            elif o_v > 0:
                sources.append(f"✅ Gerçek 1xBet Oranı ({c_m}: {o_v:.2f})")
            else:
                sources.append(f"✅ Gerçek 1xBet Oranı ({c_m})")
        else:
            exp_barems.append(get_range_from_score(r["exp_tot_36"], league=league))
            exp_odds.append(0.0)
            sources.append(f"🧮 {close_min_int}.Dk Skor Tahmini (Oran Kaydı Yok)")

    hist["exp_barem_36"] = exp_barems
    hist["exp_odd_36"] = exp_odds
    hist["source"] = sources
    hist["act_barem"] = hist["total_score"].apply(lambda s: get_range_from_score(s, league=league))

    raw_shifts = []
    wave_codes = []
    for e_b, a_b in zip(hist["exp_barem_36"], hist["act_barem"]):
        sh = barem_names.index(a_b) - barem_names.index(e_b)
        raw_shifts.append(sh)
        if sh == 0:
            wave_codes.append(0)
        elif sh < 0:
            wave_codes.append(-1)
        else:
            wave_codes.append(1)

    hist["raw_shift"] = raw_shifts
    hist["wave_code"] = wave_codes

    # Build recent_chain STRICTLY AND ONLY from matches recorded in log CSV!
    recent_chain = []
    tracked_hist = hist[hist["game_id"].astype(str).str.strip().isin(logged_closing_fav.keys())].copy()
    if not tracked_hist.empty:
        tracked_hist["_gid_num"] = (
            tracked_hist["game_id"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)
        )
        tracked_hist = (
            tracked_hist[tracked_hist["_gid_num"] >= int(cfg["min_clean_gid"])]
            .sort_values(by="_gid_num", ascending=True)
            .reset_index(drop=True)
        )

    prev_dt = None
    for _, r in tracked_hist.iterrows():
        raw_dt_str = str(r.get("date", "")).strip()
        fixed_dt_str = raw_dt_str
        try:
            cur_dt = datetime.strptime(raw_dt_str, "%Y-%m-%d %H:%M")
            if prev_dt is not None and cur_dt < prev_dt:
                cur_dt = cur_dt + timedelta(hours=3)
                fixed_dt_str = cur_dt.strftime("%Y-%m-%d %H:%M")
            prev_dt = cur_dt
        except Exception:
            pass

        sh = int(r["raw_shift"])
        if sh == 0:
            sh_label = "🎯 Tam Favori Geldi (0)"
        elif sh == -1:
            sh_label = "📉 1 Alt Bareme Düştü (-1)"
        elif sh == 1:
            sh_label = "📈 1 Üst Bareme Çıktı (+1)"
        elif sh < -1:
            sh_label = f"❄️ {abs(sh)} Alt Bareme Düştü ({sh})"
        else:
            sh_label = f"🔥 {sh} Üst Bareme Çıktı (+{sh})"

        gid_item = str(r.get("game_id", "")).strip()
        open_b = logged_opening_fav.get(gid_item, r["exp_barem_36"])
        close_b = r["exp_barem_36"]
        if open_b in barem_names and close_b in barem_names:
            drift_val = barem_names.index(close_b) - barem_names.index(open_b)
        else:
            drift_val = 0
        if drift_val > 0:
            drift_lbl = f"📈 Başlangıçtan +{drift_val} Üste Çıktı ({open_b}➔{close_b})"
        elif drift_val < 0:
            drift_lbl = f"📉 Başlangıçtan {abs(drift_val)} Alta Düştü ({open_b}➔{close_b})"
        else:
            drift_lbl = f"⚖️ Başlangıçla Aynı Kaldı ({close_b})"

        recent_chain.append(
            {
                "game_id": gid_item,
                "date": fixed_dt_str,
                "home_team": r.get("home_team", ""),
                "away_team": r.get("away_team", ""),
                "home_total": int(r.get("home_total", 0) or 0),
                "away_total": int(r.get("away_total", 0) or 0),
                "s36": int(r["s36"]),
                "q4_tot": int(r["q4_tot"]),
                "total_score": int(r["total_score"]),
                "opening_barem": open_b,
                "exp_barem_36": close_b,
                "exp_odd_36": float(r.get("exp_odd_36", 0.0) or 0.0),
                "drift": drift_val,
                "drift_label": drift_lbl,
                "act_barem": r["act_barem"],
                "raw_shift": sh,
                "wave_code": int(r["wave_code"]),
                "shift_label": sh_label,
                "source": r["source"],
            }
        )

    all_wave_codes = list(hist["wave_code"])
    all_raw_shifts = list(hist["raw_shift"])
    tracked_wave_codes = [x["wave_code"] for x in recent_chain]
    tracked_raw_shifts = [x["raw_shift"] for x in recent_chain]
    target_seq = list(custom_seq) if custom_seq is not None else (tracked_wave_codes[-4:] if tracked_wave_codes else [])

    # If user selected -2 or +2 in custom_seq, match against raw_shifts; otherwise match wave_codes
    use_raw = any(abs(int(x)) >= 2 for x in target_seq)
    search_space = all_raw_shifts if use_raw else all_wave_codes

    # Find historical matches following the exact 4-match or 3-match wave pattern
    next_raw_shifts = []
    seq_len = len(target_seq)
    if seq_len >= 1 and len(search_space) > seq_len:
        for i in range(len(search_space) - seq_len):
            if search_space[i : i + seq_len] == target_seq:
                next_raw_shifts.append(all_raw_shifts[i + seq_len])

    # Fallback to last 3 if 4-match pattern has < 15 occurrences
    if len(next_raw_shifts) < 15 and len(target_seq) >= 3:
        short_seq = target_seq[-3:]
        next_raw_shifts = []
        for i in range(len(search_space) - 3):
            if search_space[i : i + 3] == short_seq:
                next_raw_shifts.append(all_raw_shifts[i + 3])

    # Also boost real tracked chain transitions (weight x8) so real 1xBet closing-odd behavior dominates!
    if len(tracked_wave_codes) >= 4 and target_seq:
        last_w = target_seq[-1]
        real_next = []
        for i in range(len(tracked_wave_codes) - 1):
            cur_match_code = tracked_raw_shifts[i] if use_raw else tracked_wave_codes[i]
            if cur_match_code == last_w:
                real_next.append(tracked_raw_shifts[i + 1])
        if real_next:
            next_raw_shifts = list(next_raw_shifts) + (real_next * 12)

    if not next_raw_shifts:
        next_raw_shifts = all_raw_shifts

    n_pat = len(next_raw_shifts)
    s_arr = np.array(next_raw_shifts)
    p_0 = round(float((s_arr == 0).mean()) * 100.0, 1)
    p_m1 = round(float((s_arr == -1).mean()) * 100.0, 1)
    p_p1 = round(float((s_arr == 1).mean()) * 100.0, 1)
    p_m2 = round(float((s_arr <= -2).mean()) * 100.0, 1)
    p_p2 = round(float((s_arr >= 2).mean()) * 100.0, 1)

    # Also compute directional probabilities (0 vs All Down vs All Up)
    p_down_all = round(p_m1 + p_m2, 1)
    p_up_all = round(p_p1 + p_p2, 1)

    base_b = current_fav_barem if (current_fav_barem in barem_names) else def_b
    b_0 = base_b
    b_m1 = _shift_b(base_b, -1)
    b_p1 = _shift_b(base_b, 1)
    b_m2 = _shift_b(base_b, -2)
    b_p2 = _shift_b(base_b, 2)

    pair_down_prob = round(p_0 + p_m1, 1)
    pair_up_prob = round(p_0 + p_p1, 1)

    # Combine live adjacent wing odds (primary) with calibrated projection & Markov shifts (v5.5)
    _om_raw = odds_map or {}
    def _open_o(_b):
        try:
            _v = float(_om_raw.get(_b, 99.0) or 99.0)
            return _v if 1.001 <= _v < 85.0 else 99.0
        except Exception:
            return 99.0

    # Fix boundary barems (when b_0 is lowest or highest barem, shifted neighbor equals b_0!)
    om1_w = _open_o(b_m1) if (b_m1 != "-" and b_m1 != b_0) else 99.0
    op1_w = _open_o(b_p1) if (b_p1 != "-" and b_p1 != b_0) else 99.0
    ob0_w = _open_o(b_0)

    chosen_dir = None
    if om1_w < 85.0 and op1_w >= 85.0:
        chosen_dir = "down"
    elif op1_w < 85.0 and om1_w >= 85.0:
        chosen_dir = "up"
    elif om1_w < 85.0 and op1_w < 85.0:
        if om1_w < op1_w - 0.01:
            chosen_dir = "down"
        elif op1_w < om1_w - 0.01:
            chosen_dir = "up"
    if chosen_dir is None and proj_tot is not None and proj_tot > 0:
        _pb_w = get_range_from_score(proj_tot, league=league)
        if _pb_w in barem_names and b_0 in barem_names:
            if barem_names.index(_pb_w) < barem_names.index(b_0) and b_m1 != b_0:
                chosen_dir = "down"
            elif barem_names.index(_pb_w) > barem_names.index(b_0) and b_p1 != b_0:
                chosen_dir = "up"
    if chosen_dir is None:
        if b_m1 == b_0:
            chosen_dir = "up"
        elif b_p1 == b_0:
            chosen_dir = "down"
        else:
            chosen_dir = "down" if pair_down_prob >= pair_up_prob else "up"

    if chosen_dir == "down" and b_m1 != "-" and b_m1 != b_0:
        pair_a, pair_b = b_m1, b_0
        rec_pair_prob = pair_down_prob
        wing_odd_sel = om1_w
        wave_dir = f"📉 Favori ({b_0}) veya 1 Alt Barem ({b_m1}) Eğilimi (%{pair_down_prob})"
    else:
        _alt_nb = b_p1 if (b_p1 != "-" and b_p1 != b_0) else b_m1
        pair_a, pair_b = b_0, _alt_nb
        rec_pair_prob = pair_up_prob
        wing_odd_sel = op1_w if _alt_nb == b_p1 else om1_w
        wave_dir = f"📈 Favori ({b_0}) veya 1 Üst Barem ({pair_b}) Eğilimi (%{pair_up_prob})"

    # If chosen wing has no open odd, fall back cleanly to best open non-b_0 barem
    if wing_odd_sel >= 85.0 and _om_raw:
        _open_others = sorted(
            [(bn, _open_o(bn)) for bn in barem_names if bn != b_0 and _open_o(bn) < 85.0],
            key=lambda x: x[1],
        )
        if _open_others:
            pair_a, pair_b = b_0, _open_others[0][0]
            wing_odd_sel = _open_others[0][1]

    eff_wave_odd = 0.0
    if 1.01 < ob0_w < 85.0 and 1.01 < wing_odd_sel < 85.0 and pair_a != pair_b:
        eff_wave_odd = round(1.0 / ((1.0 / ob0_w) + (1.0 / wing_odd_sel)), 2)

    rec_pair = f"{pair_a} & {pair_b}" + (
        f" (Ortak: {eff_wave_odd:.2f})" if eff_wave_odd > 0 else ""
    )

    last_5_codes = [x["raw_shift"] for x in recent_chain[-5:]]
    def _fmt_code(c: int) -> str:
        if c == 0:
            return "Favori Geldi"
        elif c < 0:
            return f"{abs(c)} Alt Geldi"
        else:
            return f"{c} Üst Geldi"
    last_5_fmt = " ➔ ".join([_fmt_code(c) for c in last_5_codes])

    return {
        "recent_chain": recent_chain,
        "last_5_codes": last_5_codes,
        "last_5_labels": last_5_fmt,
        "active_pattern": target_seq,
        "pattern_matches_count": n_pat,
        "prob_0": p_0,
        "prob_minus_1": p_m1,
        "prob_plus_1": p_p1,
        "prob_minus_2": p_m2,
        "prob_plus_2": p_p2,
        "prob_down_total": p_down_all,
        "prob_up_total": p_up_all,
        "wave_direction": wave_dir,
        "target_0_barem": b_0,
        "target_minus_1_barem": b_m1,
        "target_plus_1_barem": b_p1,
        "target_minus_2_barem": b_m2,
        "target_plus_2_barem": b_p2,
        "recommended_pair": rec_pair,
        "recommended_pair_barems": [pair_a, pair_b],
        "recommended_pair_eff_odd": eff_wave_odd,
        "recommended_pair_prob": rec_pair_prob,
    }


def _shift_barem_name(base_b: str, delta: int, league: str = "nba") -> str:
    names = [b[0] for b in get_league_config(league)["barem_ranges"]]
    if base_b not in names:
        return base_b
    return names[max(0, min(len(names) - 1, names.index(base_b) + delta))]


def analyze_smart_35m_decision(
    fav1_barem: str,
    fav1_odd: float,
    fav2_barem: str,
    fav2_odd: float,
    game_id: str,
    df_log: pd.DataFrame,
    recent_chain: list,
    score_gap: int = 0,
    cur_tot: int = 0,
    elapsed_mins: float = None,
    odds_map: dict = None,
    league: str = "nba",
    proj_tot: float = None,
) -> dict:
    """
    Deep Symmetry & Closing-Minute Smart Decision Engine (supports both NBA 35m and Euro 29m).
    """
    cfg = get_league_config(league)
    dec_min = int(cfg["decision_min"])
    close_min_int = int(cfg["close_min_int"])
    if elapsed_mins is None:
        elapsed_mins = float(dec_min)
    barem_names = [b[0] for b in cfg["barem_ranges"]]
    barem_bounds = {b[0]: (b[1], b[2]) for b in cfg["barem_ranges"]}
    base_b = fav1_barem if fav1_barem in barem_names else cfg["default_barem"]
    b_0 = base_b
    b_m1 = _shift_barem_name(base_b, -1, league=league)
    b_m2 = _shift_barem_name(base_b, -2, league=league)
    b_p1 = _shift_barem_name(base_b, 1, league=league)

    idx_fav = barem_names.index(base_b)
    idx_sec = barem_names.index(fav2_barem) if fav2_barem in barem_names else max(0, idx_fav - 1)
    sec_dir = idx_sec - idx_fav
    sec_b_clean = fav2_barem if (fav2_barem in barem_names and fav2_barem != base_b) else b_m1

    if sec_dir >= 1:
        sec_dir_badge = f"📈 2. Favori Bir Üst Baremde ({sec_b_clean})"
        sec_dir_tip = (
            f"Şu an 1. Favori {base_b} iken, en düşük oranlı 2. Favori bir üst barem olan {sec_b_clean} aralığında duruyor. "
            f"Bu durumda maçların yukarı kayma oranı belirgin şekilde artar (~%39); ancak sadece oranı yüksek diye körlemesine 1. Favoriye yüklenilmez."
        )
    else:
        sec_dir_badge = f"📉 2. Favori Bir Alt Baremde ({sec_b_clean})"
        sec_dir_tip = (
            f"Şu an 1. Favori {base_b} iken, en düşük oranlı 2. Favori bir alt barem olan {sec_b_clean} aralığında duruyor. "
            f"175 NBA maçında bu yapıda maçların %51.4'ü aşağı kaymış, üste çıkma ise sadece %9.5 olmuştur."
        )

    # Calculate in-game drift (Q1 -> Now), V-Dip / Ters-V Trajectory, Late Pre-Close Pendulum, and Odds Drop Velocity
    open_b = base_b
    drift = 0
    max_up = 0
    max_dn = 0
    late_jump = 0
    b_30m = base_b
    odd_prev_ref = float(fav1_odd or 0.0)
    odd_prev_min = int(float(elapsed_mins or 0))
    barem_to_col_local = dict(cfg["barem_odd_cols"])
    barem_to_col_local.update({"160-": "odds_160_or_less", "131-": "odds_130_or_less", "170+": "odds_171_plus"})
    pre_lo, pre_hi = int(cfg["q3_pre_low"]), int(cfg["q3_pre_high"])
    if df_log is not None and not df_log.empty and game_id and "game_id" in df_log.columns:
        g_sub = df_log[df_log["game_id"].astype(str).str.strip() == str(game_id).strip()].copy()
        if not g_sub.empty:
            first_b = str(g_sub.iloc[0].get("highest_prob_range", "")).strip()
            if first_b in barem_names:
                open_b = first_b
                open_idx_tr = barem_names.index(open_b)
                drift = barem_names.index(base_b) - open_idx_tr
                traj_idxs = [
                    barem_names.index(str(b_val).strip())
                    for b_val in g_sub.get("highest_prob_range", [])
                    if str(b_val).strip() in barem_names
                ]
                if traj_idxs:
                    max_up = max(traj_idxs) - open_idx_tr
                    max_dn = open_idx_tr - min(traj_idxs)
            g_sub["_m"] = (
                g_sub["check_time"].astype(str).str.extract(r"Minute\s+(\d+)", expand=False).astype(float).fillna(0)
            )
            c30 = g_sub[(g_sub["_m"] >= pre_lo) & (g_sub["_m"] <= pre_hi)]
            if not c30.empty:
                b30_cand = str(c30.iloc[-1].get("highest_prob_range", "")).strip()
                if b30_cand in barem_names:
                    b_30m = b30_cand
                    late_jump = barem_names.index(base_b) - barem_names.index(b_30m)
            # Odds velocity for current favorite barem over the last 3-10 minutes
            col_fav_now = barem_to_col_local.get(base_b)
            if col_fav_now and col_fav_now in g_sub.columns:
                cur_m_val = float(elapsed_mins or g_sub["_m"].max())
                prev_rows = g_sub[(g_sub["_m"] <= max(0.0, cur_m_val - 2.0)) & (g_sub["_m"] >= max(0.0, cur_m_val - 12.0))]
                if prev_rows.empty and len(g_sub) >= 2:
                    prev_rows = g_sub.iloc[:-1]
                for _, pr in prev_rows.iloc[::-1].iterrows():
                    try:
                        po = float(pr.get(col_fav_now, 99.0) or 99.0)
                        if 1.01 < po < 85.0:
                            odd_prev_ref = po
                            odd_prev_min = int(pr.get("_m", 0))
                            break
                    except Exception:
                        pass

    cur_o_now = float(fav1_odd or 99.0)
    odd_delta = round(odd_prev_ref - cur_o_now, 2) if (1.01 < odd_prev_ref < 85.0 and 1.01 < cur_o_now < 85.0) else 0.0
    is_euro_lg = str(league or "nba").lower() == "euro"
    if odd_delta >= 0.18:
        odd_trend_badge = f"⚠️ Oran Hızla Düştü: {odd_prev_ref:.2f} ➔ {cur_o_now:.2f} (Dikkatli Ol)"
        odd_trend_tip = f"{base_b} bareminin oranı {odd_prev_min}. dakikada {odd_prev_ref:.2f} iken şu an {cur_o_now:.2f} seviyesine düştü. Son dakikalarda oranı sert düşen maçlarda aşırı yüklenme/tuzak riski artabilir."
    elif odd_delta <= -0.15:
        odd_trend_badge = (
            f"🚨 1.Fav Oranı Yükseldi: {odd_prev_ref:.2f} ➔ {cur_o_now:.2f} (2.Fav {sec_b_clean} Güçlü!)"
            if not is_euro_lg
            else f"📈 Oran Yükseliyor: {odd_prev_ref:.2f} ➔ {cur_o_now:.2f} (Zayıflıyor)"
        )
        odd_trend_tip = (
            f"{base_b} bareminin oranı {odd_prev_min}. dakikada {odd_prev_ref:.2f} iken şu an {cur_o_now:.2f} seviyesine yükseldi. "
            f"NBA'de son dakikalarda 1. Favori oranı yükseldiğinde 2. Favori ({sec_b_clean}) %50 tutma ve 1.000 TL başı +534 TL kâr sağlamıştır."
        )
    else:
        odd_trend_badge = f"➖ Oran Dengeli ({cur_o_now:.2f})" if cur_o_now < 90 else "⏳ Oran Bekleniyor"
        odd_trend_tip = f"{base_b} bareminin oranı son dakikalarda sabit/dengeli ({cur_o_now:.2f}) seyrediyor."

    if drift > 0:
        drift_badge = f"📈 Maç Başında {open_b} Bekleniyordu ➔ Şimdi {base_b} Oldu (Maç Hızlandı)"
    elif drift < 0:
        drift_badge = f"📉 Maç Başında {open_b} Bekleniyordu ➔ Şimdi {base_b} Oldu (Maç Yavaşladı)"
    else:
        drift_badge = f"⚖️ Maç Başından Beri Hep {base_b} Favori"
    drift_tip = (
        f"Maç ilk başladığında favori barem {open_b} idi, şu an ise {base_b} oldu. "
        f"Maç içinde barem yukarı tırmandıysa genelde olduğu yerde kalır veya 1 geri düşer (daha üste çıkma ~%11). "
        f"Aşağı düştüyse tersine geri yukarı zıplama (%35) ihtimali doğar; baştan beri sabit kalan maçlar en dengelidir."
    )

    # Last Digit of Score (Simple explanation)
    mid_q3 = int(cfg["mid_q3_min"])
    pre_ref_m = int(cfg["q3_pre_high"]) - 1
    mod10 = int(cur_tot) % 10 if cur_tot > 0 else cfg["mod10_center_digits"][1]
    c_digs = cfg["mod10_center_digits"]
    r_digs = cfg["mod10_risk_digits"]
    c_digs_str = ", ".join(str(x) for x in c_digs[:-1]) + f" veya {c_digs[-1]}"
    r_digs_str = ", ".join(str(x) for x in r_digs[:-1]) + f" veya {r_digs[-1]}"
    exp_rem_dec_val = float(cfg["exp_rem_decision"])

    if late_jump <= -1:
        pendulum_badge = f"🔄 Son 5 Dk'da Barem Düştü ({b_30m} ➔ {base_b} — Daha Fazla Düşmez!)"
        pendulum_tip = (
            f"{pre_ref_m}. dakikada favori {b_30m} iken {dec_min}. dakikada {base_b} baremine düştü. "
            f"Son bölümde baremi zaten düşen maçlarda bir alt barem olan {b_m1} aralığına düşüş nadirdir; "
            f"çoğunlukla ya {base_b} bareminde kalır ya da eski {b_30m} baremine geri döner."
        )
    elif late_jump >= 1:
        pendulum_badge = f"🔄 Son 5 Dk'da Barem Yükseldi ({b_30m} ➔ {base_b} — Daha Fazla Çıkmaz!)"
        pendulum_tip = (
            f"{pre_ref_m}. dakikada favori {b_30m} iken {dec_min}. dakikada {base_b} baremine yükseldi. "
            f"Kapanışa doğru zaten yükseldiği için bir üst bareme ({b_p1}) gitme ihtimali düşer; {base_b} veya {b_m1} aralığında kalır."
        )
    else:
        pendulum_badge = f"⚖️ Son 5 Dk'dır Barem Sabit ({base_b})"
        pendulum_tip = (
            f"{pre_ref_m}. dakikadan {dec_min}. dakikaya favori barem ({base_b}) hiç değişmedi. "
            f"Böyle dengeli giden maçlarda direkt {base_b} baremi yüksek oranda tutmaktadır."
        )

    if cur_tot > 0 and float(elapsed_mins or dec_min) >= mid_q3 - 5:
        if mod10 in c_digs:
            mod10_badge = f"🎯 Skorun Son Rakamı Avantajlı ({cur_tot} ➔ Tam {base_b} Ortasına Düşüyor)"
            mod10_tip = (
                f"Şu anki toplam skor ({cur_tot}) '{mod10}' rakamıyla bitiyor. Kalan sürede ortalama ~{exp_rem_dec_val:.0f} sayı atıldığında maç sonu skoru "
                f"tam {base_b} bareminin ortasına (en güvenli yerine) denk geliyor! (Avantajlı rakamlar: {c_digs_str})."
            )
        elif mod10 in r_digs:
            mod10_badge = f"⚠️ Skorun Son Rakamı Sınırda ({cur_tot} ➔ {b_m1} Alt Barem Riski Var)"
            mod10_tip = (
                f"Şu anki toplam skor ({cur_tot}) '{mod10}' rakamıyla bitiyor. Kalan sürede ortalama ~{exp_rem_dec_val:.0f} sayı eklendiğinde maç sonu skoru "
                f"{base_b} bareminin en alt sınırına denk geliyor; birkaç sayı eksik atılsa bir alt barem olan {b_m1} olur! (Sınır rakamları: {r_digs_str})."
            )
        else:
            mod10_badge = f"📐 Skorun Son Rakamı Normal ({cur_tot} ➔ {base_b} Üst Kısmına Yakın)"
            mod10_tip = (
                f"Şu anki toplam skor ({cur_tot}) '{mod10}' rakamıyla bitiyor. Ortalama ~{exp_rem_dec_val:.0f} sayı eklendiğinde {base_b} bareminin orta-üst kısmına denk gelir."
            )
    else:
        mod10_badge = f"🔢 Skorun Son Rakamı: ..{mod10} ({mid_q3 - 5}. Dakikadan Sonra Devreye Girer)"
        mod10_tip = (
            f"Maç {mid_q3}-{dec_min}. dakikaya geldiğinde toplam skorun son rakamına bakarak maç sonunun {base_b} bareminin tam ortasına mı "
            f"yoksa {b_m1} alt sınırına mı düşeceğini otomatik hesaplar."
        )

    # Twin Odds & Wing Balance in Plain Turkish
    odd_val = float(fav1_odd or 0.0)
    odd2_val = float(fav2_odd or 99.0)
    odd_gap_12 = round(odd2_val - odd_val, 2) if (odd_val < 90 and odd2_val < 90) else 99.0

    valid_odds_dict = {}
    if isinstance(odds_map, dict):
        valid_odds_dict = {k: float(v) for k, v in odds_map.items() if v is not None and 1.0 < float(v) < 90.0}
    odd_m1_live = valid_odds_dict.get(b_m1, 99.0)
    odd_p1_live = valid_odds_dict.get(b_p1, 99.0)
    wing_diff = abs(odd_p1_live - odd_m1_live) if (odd_m1_live < 90 and odd_p1_live < 90) else 99.0

    if valid_odds_dict and odd_val < 90 and odd2_val < 90:
        inv_all = sum(1.0 / v for v in valid_odds_dict.values())
        top2_conc = ((1.0 / odd_val) + (1.0 / odd2_val)) / inv_all * 100.0 if inv_all > 0 else 0.0
    else:
        top2_conc = 75.0

    if odd_gap_12 <= 0.25:
        twin_badge = (
            f"👯 Tam İkiz Favori ({base_b} ≈ {sec_b_clean}) ➔ 2.Fav Değerli (+76 TL / 1.000 TL | İkili %78.8)"
            if not is_euro_lg
            else f"👯 İlk 2 Baremin Oranı Neredeyse Eşit ({base_b} ve {sec_b_clean} — İkili %82.6)"
        )
        twin_tip = (
            f"1. Favori {base_b} ({odd_val:.2f}) ile 2. Favori {sec_b_clean} ({odd2_val:.2f}) arasındaki oran farkı çok az ({odd_gap_12:.2f}). "
            f"NBA'de tam ikiz oranlarda 2. Favori ({sec_b_clean}) tek başına 1.000 TL başı +76 TL kazandırırken, ikisinden birinin tutma oranı %79-%83 seviyesindedir."
        )
    elif top2_conc >= 78.0:
        twin_badge = f"🔒 İlk 2 Barem Ağırlıklı ({base_b} veya {sec_b_clean})"
        twin_tip = (
            f"1xBet ihtimallerinin %{top2_conc:.0f}'i sadece {base_b} ve {sec_b_clean} baremlerinde toplanmış durumda. "
            f"İki barem kapatmada ana hedef {base_b} ve {sec_b_clean} ikilisidir."
        )
    else:
        twin_badge = f"⚖️ 1. ve 2. Favori Oran Farkı: {odd_gap_12:.2f} ({base_b} vs {sec_b_clean})"
        twin_tip = (
            f"1. Favori ({base_b}: {odd_val:.2f}) ile 2. Favori ({sec_b_clean}: {odd2_val:.2f}) arasındaki oran farkı {odd_gap_12:.2f}."
        )

    if wing_diff <= 0.75:
        wing_badge = f"⚖️ Alt ({b_m1}) ve Üst ({b_p1}) Oranları Dengeli (Simetrik Kanat)"
        wing_tip = (
            f"Favorinin bir altındaki {b_m1} ({odd_m1_live:.2f}) ile bir üstündeki {b_p1} ({odd_p1_live:.2f}) oranları birbirine yakın. "
            f"Bu durumda maç alt ve üst tarafa dengeli dağılır; tek kanada yüklenmek yerine merkez/ikili koruma daha sağlıklıdır."
        )
    elif odd_m1_live < 90 and odd_p1_live < 90:
        if is_euro_lg and (odd_m1_live - odd_p1_live) >= 1.0:
            wing_badge = f"📐 Üst Kanat Düşük ({b_p1}: {odd_p1_live:.2f}) Ama Euro'da Merkez ({base_b}) Kalır"
            wing_tip = (
                f"Euro League'de 1 Üst komşu ({b_p1}: {odd_p1_live:.2f}) daha düşük oranlı görünse bile 1 Üst Barem tek başına sadece %23.8 (-328 TL / 1.000 TL) tutmuştur; "
                f"maçların büyük çoğunluğu yine merkez {base_b} bareminde kalır."
            )
        else:
            wing_badge = f"📐 Komşu Oranlar: Alt {b_m1} ({odd_m1_live:.2f}) | Üst {b_p1} ({odd_p1_live:.2f})"
            wing_tip = (
                f"Favori {base_b} bareminin alt komşusu {b_m1} oranı {odd_m1_live:.2f}, üst komşusu {b_p1} oranı {odd_p1_live:.2f}. "
                f"Hangisinin oranı açıkça düşükse maçın o tarafa kayma ihtimali yükselir."
            )
    else:
        wing_badge = "📐 Komşu Baremlerden Biri Kapalı"
        wing_tip = "Alt veya üst komşu baremlerden birine şu an oran açılmamış."

    # Sequential Match History & Pendulum Wave Symmetry in Plain Turkish (calibrated to 238 NBA + 186 Euro matches)
    prev_shift = recent_chain[-1]["raw_shift"] if recent_chain else 0
    prev2_shift = recent_chain[-2]["raw_shift"] if len(recent_chain) >= 2 else 0
    is_pendulum_zig = (prev2_shift < 0 and prev_shift > 0) or (prev2_shift > 0 and prev_shift < 0)
    is_two_down = (prev2_shift < 0 and prev_shift < 0)
    if prev_shift <= -2:
        seq_sym_badge = f"🛡️ Önceki Maç -2 Alt Geldi ➔ Tekrar -2 Görülmedi (%0): {base_b} / {b_m1} Bölgesine Döner"
        seq_sym_tip = (
            f"Bir önceki maç 2+ basamak aşağı ({prev_shift}) düştü. 238 NBA ve 186 Euro maçında (toplam 424 maç) üst üste iki kez -2 hiç görülmemiştir (%0); "
            f"sıradaki maçta merkeze ({base_b} veya {b_m1}) toparlanır."
        )
    elif is_euro_lg and is_two_down:
        seq_sym_badge = f"🎯 Euro 2 Alt Simetrisi ({prev2_shift} ➔ {prev_shift}) ➔ Merkeze ({base_b}) Dönüş %75.0 (+433 TL / 1.000 TL)"
        seq_sym_tip = (
            f"Euro League'de üst üste 2 maç alt baremde bittiğinde sıradaki maçta 1 Üst Barem hiç gelmemiştir (%0); "
            f"maçların %75.0'i doğrudan 1. Favori ({base_b}) merkezinde sonuçlanmıştır (1.000 TL başı +433 TL)."
        )
    elif is_pendulum_zig:
        zig_rate = "%60.0 (1.000 TL başı +123 TL)" if is_euro_lg else "%52.9 (1.000 TL başı +49 TL)"
        seq_sym_badge = f"🔄 Sarkaç Simetrisi ({prev2_shift:+d} ➔ {prev_shift:+d}) ➔ Merkeze ({base_b}) Dönüş: {zig_rate}"
        seq_sym_tip = (
            f"Son iki maç önce bir yöne sonra ters yöne ({prev2_shift:+d} ardından {prev_shift:+d}) savruldu. "
            f"Sarkaç kuralına göre 3. maçta denge merkeze ({base_b}) döner ({zig_rate})."
        )
    elif prev_shift != 0 and prev2_shift != 0:
        rec_rate_txt = "%65.6 (1.000 TL başı +213 TL)" if is_euro_lg else "%44.9"
        seq_sym_badge = f"🔄 Son 2 Maçta Favori Gelmedi (Sıradaki Favori Oranı: {rec_rate_txt})"
        seq_sym_tip = (
            f"Bundan önceki 2 maçta üst üste kapanış favorisi tutmadı. "
            f"Güncel 424 maçlık veride bu durumda sıradaki maçta {base_b} tutma oranı {rec_rate_txt} seviyesindedir."
        )
    elif not is_euro_lg and prev_shift == 1:
        seq_sym_badge = f"⚠️ Önceki Maç +1 Üst Geldi ➔ NBA'de 1. Favori Zayıflar (%29.5 / -357 TL)"
        seq_sym_tip = (
            f"NBA'de bir önceki maç +1 üst baremde bittiğinde sıradaki maçta tek başına 1. Favori %29.5 (1.000 TL başı -357 TL) ile düşüş yaşar; "
            f"tek favoriye körlemesine yüklenmek yerine ikili veya değer baremi tercih edilmelidir."
        )
    elif not is_euro_lg and prev_shift == -1:
        seq_sym_badge = f"🎯 Önceki Maç -1 Alt Geldi ➔ NBA Merkeze ({base_b}) Dönüş %51.7 (+27 TL / 1.000 TL)"
        seq_sym_tip = (
            f"NBA'de bir önceki maç -1 alt baremde bittiğinde sıradaki maç %51.7 oranında 1. Favori ({base_b}) merkezine geri döner."
        )
    elif prev_shift == 0 and prev2_shift == 0:
        seq_sym_badge = "⚖️ Son 2 Maçta Üst Üste Favori Barem Geldi (Yan Bareme Kayma Riski)"
        seq_sym_tip = f"Son 2 maçta üst üste kapanış favorisi kazandı; 3. maçta {base_b} tek başına zayıflar, {sec_b_clean} komşu koruması önem kazanır."
    else:
        prev_desc = "Favori Barem Geldi" if prev_shift == 0 else (f"{abs(prev_shift)} Alt Barem Geldi" if prev_shift < 0 else f"{prev_shift} Üst Barem Geldi")
        seq_sym_badge = f"🌊 Bir Önceki Maç Sonucu: {prev_desc}"
        seq_sym_tip = "Bundan hemen önce biten maçın kapanış favorisine göre nasıl sonuçlandığını gösterir."

    gap = abs(int(score_gap or 0))
    gap_close_thr = 4 if is_euro_lg else 5
    gap_ctrl_thr = 8 if is_euro_lg else 10
    gap_blow_thr = 16 if is_euro_lg else 18
    if gap <= gap_close_thr:
        exp_rem_35 = float(cfg["exp_rem_clutch"])
        margin_badge = f"⚠️ Skor Çok Yakın: {gap} Sayı Fark (Faul & Uzatma Riski Var)"
        margin_tip = (
            f"İki takım arasındaki fark sadece {gap} sayı (0-{gap_close_thr} arası başa baş maç). Son dakikalarda taktik fauller ve uzatma "
            f"ihtimali yüzünden {dec_min}. dakikadan sonra ortalama {exp_rem_35:.1f} sayı atılır ve "
            f"favorinin tutma oranı düşer. Başa baş maçlara BAHİS ALMAMAK (Pas Geçmek) en doğrusudur."
        )
    elif gap >= gap_blow_thr:
        exp_rem_35 = float(cfg["exp_rem_blowout"])
        margin_badge = f"🧊 Fark Çok Açıldı: {gap} Sayı Fark (Maç Yavaşlar / Rölanti)"
        margin_tip = (
            f"Skor farkı {gap} sayıya ({gap_blow_thr}+ sayı) çıktığında takımlar son bölümde tempoyu düşürür. "
            f"{dec_min}. dakikadan sonra ortalama sadece {exp_rem_35:.1f} sayı atılır; "
            f"üst bareme ({b_p1}) kaçma ihtimali çok düşüktür ve {base_b} / {b_m1} bölgesi güçlenir."
        )
    elif gap >= gap_ctrl_thr:
        exp_rem_35 = float(cfg["exp_rem_ctrl"])
        margin_badge = f"🛡️ Fark Kontrollü: {gap} Sayı Fark (Uzatma Riski Düşük)"
        margin_tip = f"Skor farkı {gap} sayı ({gap_ctrl_thr}-{gap_blow_thr-1} arası); uzatma riski düşüktür. Kalan sürede ort. ~{exp_rem_35:.1f} sayı atılır."
    else:
        exp_rem_35 = exp_rem_dec_val
        margin_badge = f"⚖️ Normal Skor Farkı: {gap} Sayı Fark"
        margin_tip = f"Skor farkı {gap} sayı; kalan sürede normal tempo (~{exp_rem_dec_val:.1f} sayı) beklenir."

    # Barem interior edge position & "Hesap != Site" Projection Check
    fav_lo, fav_hi = barem_bounds.get(base_b, (201, 210))
    eff_lo = max(float(cfg["bar_lower_floor"]), float(fav_lo))
    eff_hi = min(float(fav_hi), float(cfg["bar_upper_cap"]))
    fav_mid = (eff_lo + eff_hi) / 2.0
    rem_mins_est = max(0.0, float(cfg["total_minutes"]) - float(elapsed_mins or dec_min))
    scaled_rem = exp_rem_35 * (rem_mins_est / float(cfg["rem_span_decision"])) if rem_mins_est > 0 else 0.0
    proj_tot_edge = float(cur_tot) + scaled_rem if cur_tot > 0 else fav_mid
    edge_offset = proj_tot_edge - fav_mid
    proj_barem = get_range_from_score(proj_tot_edge, league=league) if cur_tot > 0 else base_b
    proj_diff = (barem_names.index(proj_barem) - barem_names.index(base_b)) if (proj_barem in barem_names and base_b in barem_names) else 0

    if cur_tot > 0 and (close_min_int - 8.0) <= float(elapsed_mins or dec_min) <= (close_min_int + 0.5):
        if proj_diff != 0:
            dir_word = "ALTINDA" if proj_diff < 0 else "ÜSTÜNDE"
            edge_badge = f"💎 Hesap ≠ Site: Skor Hesabı {proj_barem} Diyor (Site {base_b} — Fark {dir_word})"
            edge_tip = (
                f"Şu anki skor ({cur_tot}) + farka göre kalan beklenti (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı ➔ {proj_barem}. "
                f"Skor hesabı sitenin favorisinden ({base_b}) ayrıştığında {proj_barem} tarafı NBA ve Euro'da kârlı değer sinyali üretmiştir."
            )
        elif edge_offset < -3.0:
            edge_badge = f"📐 Beklenen Bitiş: ~{proj_tot_edge:.0f} Sayı ({base_b} Alt Sınırında ➔ {b_m1} Riski Var)"
            edge_tip = (
                f"Şu anki skor ({cur_tot}) + kalan sürede beklenen sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı yapıyor. "
                f"Bu sayı {base_b} bareminin alt sınırında kaldığı için bir alt barem olan {b_m1} aralığına düşme riski artar."
            )
        elif edge_offset > 3.0:
            edge_badge = f"📐 Beklenen Bitiş: ~{proj_tot_edge:.0f} Sayı ({base_b} Üst Kısmında)"
            edge_tip = (
                f"Şu anki skor ({cur_tot}) + kalan sürede beklenen sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı yapıyor. "
                f"{base_b} bareminin orta-üst kısmında güçlü duruyor."
            )
        else:
            edge_badge = f"🎯 Beklenen Bitiş: ~{proj_tot_edge:.0f} Sayı (Tam {base_b} Ortasında)"
            edge_tip = (
                f"Şu anki skor ({cur_tot}) + kalan sürede beklenen sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı ile "
                f"tam {base_b} bareminin ortasına denk geliyor."
            )
    else:
        edge_badge = f"⏱️ {dec_min}.Dk'dan Sonra Ortalama {exp_rem_dec_val:.1f} Sayı Atılır (Bu Maçta Beklenen: ~{exp_rem_35:.1f} Sayı)"
        edge_tip = (
            f"{dec_min}. dakikadan maç bitimine kadar ortalama {exp_rem_dec_val:.1f} sayı atılır. "
            f"Maç {mid_q3}-{dec_min}. dakikaya geldiğinde burada tahmini maç sonu toplamı görünür."
        )

    prev_shift = recent_chain[-1]["raw_shift"] if recent_chain else 0
    odd_val = float(fav1_odd or 0.0)
    reasons = []
    proj_odd_now = valid_odds_dict.get(proj_barem, 99.0)

    # Smart 2-Barem Pair & Weighted Allocation (v5.5: preserves +14.2/+36.9 TL/100 1. Tercih AND upgrades 2. Tercih!)
    if gap >= (20 if not is_euro_lg else 16):
        dutch_b1, dutch_b2 = b_0, b_m1
    else:
        dutch_b1, dutch_b2 = b_0, sec_b_clean
    split_mode = "weighted_55_45"

    dutch_short = f"{dutch_b1} ve {dutch_b2}"
    d_o1 = valid_odds_dict.get(dutch_b1, odd_val if dutch_b1 == b_0 and odd_val < 90 else 99.0)
    d_o2 = valid_odds_dict.get(dutch_b2, odd2_val if dutch_b2 == sec_b_clean and odd2_val < 90 else 99.0)

    has_two_valid_odds = (dutch_b1 != dutch_b2) and (1.01 < d_o1 < 85.0) and (1.01 < d_o2 < 85.0)
    eff_d_odd = round(1.0 / ((1.0 / d_o1) + (1.0 / d_o2)), 2) if has_two_valid_odds else 0.0

    if has_two_valid_odds:
        if not is_euro_lg and 1.20 <= odd_val < 2.50 and gap >= 8 and d_o2 > d_o1:
            # 55% Fav2 + 45% Fav1 turns equal-Dutch loss positive (+4.3 TL/100 across 237 NBA games!)
            w1, w2 = 0.45, 0.55
            st_1, st_2 = 450, 550
        elif not is_euro_lg and 1.20 <= odd_val < 2.50 and gap >= 8 and d_o1 > d_o2:
            w1, w2 = 0.55, 0.45
            st_1, st_2 = 550, 450
        else:
            inv_d = (1.0 / d_o1) + (1.0 / d_o2)
            st_1 = int(round(1000.0 * (1.0 / d_o1) / inv_d))
            st_2 = 1000 - st_1
            w1 = round(st_1 / 1000.0, 4)
            w2 = round(1.0 - w1, 4)
    else:
        st_1, st_2 = 500, 500
        w1, w2 = 0.5, 0.5

    prev2_shift = recent_chain[-2]["raw_shift"] if len(recent_chain) >= 2 else 0
    two_fav_failed = (prev_shift != 0 and prev2_shift != 0)
    mc_pb = get_range_from_score(proj_tot, league=league) if (proj_tot is not None and proj_tot > 0) else proj_barem
    mc_diff = (barem_names.index(mc_pb) - barem_names.index(b_0)) if (mc_pb in barem_names and b_0 in barem_names) else 0
    mc_odd_now = valid_odds_dict.get(mc_pb, 99.0)

    primary_barems = []
    primary_weights = []
    _lg_key = str(league or "nba").lower()
    is_2k27_lg = _lg_key == "nba2k27"
    is_two_up = (prev2_shift > 0 and prev_shift > 0)
    is_ters_v = (max_up >= 1 and drift <= 0)
    is_v_dip = (max_dn >= 1 and drift >= 0)
    upper_wing_low = (odd_m1_live < 85.0 and odd_p1_live < 85.0 and (odd_m1_live - odd_p1_live) >= 1.0)
    lower_wing_low = (odd_m1_live < 85.0 and odd_p1_live < 85.0 and (odd_p1_live - odd_m1_live) >= 1.0)

    # League-Separated Plain-Turkish Decision Engine (v7.0 rebuilt on 358 NBA 2K26 + 59 NBA 2K27 + 345 Euro = 762 matches)
    if is_euro_lg and odd_val > 1.0 and odd_val < 1.20:
        zone_badge = f"⛔ ÇOK DÜŞÜK ORAN PAS GEÇ ({b_0}: {odd_val:.2f} < 1.20)"
        zone_type = "PASS_LOW_ODD"
        stake_badge = "0 TL (Pas Geç — 1.20 Altı Uzun Vadede Kayıp Yazar)"
        primary_pick = f"⛔ Bahis Alma ({b_0} oranı {odd_val:.2f} — Çok Düşük)"
        conf_pct = 45.0
        reasons.append(
            f"Euro'da 1. Favori oranı 1.20 altında ({odd_val:.2f}) kaldığında bir tek sürpriz tüm kazancı siliyor (1.000 TL başı -234 TL zarar bölgesi — pas geçilir)"
        )
    elif gap <= gap_close_thr and float(elapsed_mins or dec_min) >= mid_q3:
        zone_badge = f"⛔ BAHİS ALMA / PAS GEÇ (Skor Çok Yakın: {gap} Sayı Fark)"
        zone_type = "PASS_CLUTCH"
        stake_badge = "0 TL (Pas Geç — Uzatma ve Faul Riski)"
        primary_pick = f"⛔ Bahis Alma (Başa Baş Maç — {gap} Sayı Fark)"
        conf_pct = 32.0
        reasons.append(
            f"Maç başa baş gidiyor (fark sadece {gap} sayı). Son bölümde fauller ve uzatma nedeniyle kalan sayı artar (ort. {exp_rem_35:.1f}); "
            f"358 NBA maçında %31.1 (1.000 TL başı -313 TL) ile ağır zarar bölgesi olduğu için pas geçilir"
        )
    elif is_euro_lg and 2.50 <= odd_val < 90.0:
        zone_badge = f"⛔ ZAYIF FAVORİ PAS GEÇ ({b_0} Oranı: {odd_val:.2f} ≥ 2.50)"
        zone_type = "PASS_WEAK"
        stake_badge = "0 TL (Pas Geç — 2.50+ Oran Çok Belirsiz)"
        primary_pick = f"⛔ Bahis Alma ({b_0} oranı {odd_val:.2f} — Çok Belirsiz)"
        conf_pct = 25.0
        reasons.append("Euro maçlarında 1. Favori oranı 2.50 ve üzerine çıktığında belirsizlik çok yüksektir; pas geçilir")
    elif is_2k27_lg:
        # ── NBA 2K27 v7.0 ENGINE (59 matches: H1+H2 verified) ──
        if (odd_val >= 2.50 or prev_shift <= -2) and 1.01 < odd2_val < 85.0:
            zone_badge = f"🔥 NBA 2K27 SÜRPRİZ 2. FAVORİ FIRSATI: {sec_b_clean} ({odd2_val:.2f})"
            zone_type = "TRAP_EXTREME"
            stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
            primary_pick = f"🥈 {sec_b_clean} Baremini Oyna (2. Favori — Oran: {odd2_val:.2f})"
            primary_barems = [sec_b_clean]
            primary_weights = [1.0]
            conf_pct = 78.0
            reasons.append(
                f"NBA 2K27'de 2.50+ oranda veya -2 alt sonrasında 2. Favori ({sec_b_clean}) %66.7 tutma ve 1.000 TL başı +945 TL ile +1.540 TL net kazanç sağlamıştır"
            )
        elif lower_wing_low or is_v_dip or is_two_down or prev_shift >= 2 or (1.20 <= odd_val < 1.50):
            why_27_f1 = (
                "Alt Kanat Düşük (+157 TL)" if lower_wing_low
                else "V-Dip Simetrisi (+119 TL)" if is_v_dip
                else "Son 2 Maç Alt (+396 TL)" if is_two_down
                else "Önceki Maç 2+ Üst (+637 TL)" if prev_shift >= 2
                else "1.20–1.50 Güçlü Favori (+310 TL)"
            )
            zone_badge = f"💎 NBA 2K27 MERKEZ 1. FAVORİ ({why_27_f1}): {b_0} ({odd_val:.2f})"
            zone_type = "GOLDEN"
            stake_badge = f"1000 TL (Direkt {b_0} Baremine Oyna)"
            primary_pick = f"🎯 Direkt {b_0} Baremini Oyna ({why_27_f1} — Oran: {odd_val:.2f})"
            primary_barems = [b_0]
            primary_weights = [1.0]
            conf_pct = 81.0
            reasons.append(
                f"NBA 2K27'de '{why_27_f1}' sinyali aktifken doğrudan 1. Favori ({b_0}) hem ilk yarı hem son maçlarda istikrarlı net kâr üretmiştir"
            )
        elif (drift < 0 or odd_gap_12 <= 0.45) and 1.01 < odd2_val < 85.0:
            why_27_f2 = "Barem Aşağı Düştü (+385 TL)" if drift < 0 else f"Yakın/İkiz Oran Farkı {odd_gap_12:.2f} (+173 TL)"
            zone_badge = f"🔥 NBA 2K27 2. FAVORİ SİMETRİSİ ({why_27_f2}): {sec_b_clean}"
            zone_type = "TRAP_LOW"
            stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
            primary_pick = f"🥈 {sec_b_clean} Baremini Oyna ({why_27_f2} — Oran: {odd2_val:.2f})"
            primary_barems = [sec_b_clean]
            primary_weights = [1.0]
            conf_pct = 79.0
            reasons.append(
                f"NBA 2K27'de '{why_27_f2}' olduğunda 2. Favori ({sec_b_clean}) hem ilk yarı hem ikinci yarı maçlarında %46.7 tutma ile 1.000 TL başı +173 TL ile +385 TL kazandırmıştır"
            )
        elif 2.30 <= odd_val < 2.50:
            zone_badge = f"💎 NBA 2K27 2.30–2.50 MERKEZ FAVORİ: {b_0} ({odd_val:.2f})"
            zone_type = "GOLDEN"
            stake_badge = f"1000 TL (Direkt {b_0} Baremine Oyna)"
            primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
            primary_barems = [b_0]
            primary_weights = [1.0]
            conf_pct = 76.0
            reasons.append("NBA 2K27'de 2.30–2.50 aralığında 1. Favori (%43.8 / 1.000 TL başı +44 TL) her iki yarıda artıdadır")
        else:
            zone_badge = f"⛔ NBA 2K27 SEÇİCİ PAS GEÇ ({b_0}: {odd_val:.2f})" if odd_val < 90 else "⏳ Oran Bekleniyor"
            zone_type = "PASS_SELECTIVE"
            stake_badge = "0 TL (Pas Geç — 1.85–2.10 Üst Kanat Riskli)" if odd_val < 90 else "Oran Bekleniyor"
            primary_pick = f"⛔ Bahis Alma (Net Sinyal Yok — Takip: {dutch_short})" if odd_val < 90 else "⏳ Oran Bekleniyor"
            conf_pct = 42.0
            reasons.append("NBA 2K27'de 1.85–2.10 aralığında net simetri sinyali yoksa 1. Favori zayıf kalmaktadır; pas geçilir")
    elif not is_euro_lg:
        # ── NBA 2K26 v7.0 ENGINE (358 matches: H1+H2 verified) ──
        if (odd_val < 1.50 or odd_val >= 2.50) and 1.01 < odd2_val < 85.0:
            why_ext = f"Kısa Oran Tuzağı ({odd_val:.2f} < 1.50)" if odd_val < 1.50 else f"2.50+ Kararsız Favori ({odd_val:.2f})"
            zone_badge = f"🔥 NBA 2. FAVORİ FIRSATI ({why_ext}): {sec_b_clean}"
            zone_type = "TRAP_EXTREME"
            stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
            primary_pick = f"🥈 {sec_b_clean} Baremini Oyna (2. Favori — Oran: {odd2_val:.2f})"
            primary_barems = [sec_b_clean]
            primary_weights = [1.0]
            conf_pct = 80.0
            reasons.append(
                f"358 NBA 2K26 maçında 1. Favori oranı 1.50 altında (+338 TL) veya 2.50 üzerinde (+407 TL) olduğunda 2. Favori ({sec_b_clean}) hem eski hem yeni maçlarda yüksek net kâr üretmiştir"
            )
        elif 1.85 <= odd_val < 2.10:
            if prev_shift == 1 and 1.01 < odd2_val < 85.0:
                zone_badge = f"⚠️ ÖNCEKİ MAÇ +1 ÜST SONRASI ➔ 2. Fav ({sec_b_clean})"
                zone_type = "TRAP_LOW"
                stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
                primary_pick = f"🥈 {sec_b_clean} Baremini Oyna (+1 Üst Sonrası 2.Fav — Oran: {odd2_val:.2f})"
                primary_barems = [sec_b_clean]
                primary_weights = [1.0]
                conf_pct = 76.0
                reasons.append(
                    f"NBA 2K26'da bir önceki maç +1 üst bittiğinde 1. Favori zayıflarken 2. Favori ({sec_b_clean}) 79 maçta %41.8 tutma ve 1.000 TL başı +265 TL kazandırmıştır"
                )
            else:
                zone_badge = f"💎 NBA ALTIN TEK FAVORİ: {b_0} ({odd_val:.2f})"
                zone_type = "GOLDEN"
                stake_badge = f"1000 TL (Direkt {b_0} Baremine Oyna)"
                primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
                primary_barems = [b_0]
                primary_weights = [1.0]
                conf_pct = 84.0
                reasons.append(
                    f"358 NBA 2K26 maçında 1.85–2.10 arası 1. Favori (önceki maç +1 değilken) %60.3 tutma ve 1.000 TL başı +217 TL (eski +286 TL, yeni +103 TL) net kazanç üretmiştir"
                )
        elif 1.50 <= odd_val < 1.85 and 6 <= gap <= 20 and prev_shift != 1:
            zone_badge = f"💎 NBA KONTROLLÜ TEK FAVORİ: {b_0} ({odd_val:.2f})"
            zone_type = "GOLDEN"
            stake_badge = f"1000 TL (Direkt {b_0} Baremine Oyna)"
            primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
            primary_barems = [b_0]
            primary_weights = [1.0]
            conf_pct = 78.0
            reasons.append("NBA 2K26'da 1.50–2.10 aralığı ve 6–20 sayı farkta 1. Favori (%60.5 / 1.000 TL başı +211 TL) hem eski hem yeni maçlarda artıdadır")
        elif 2.10 <= odd_val < 2.30 and 1.01 < odd2_val < 85.0:
            zone_badge = f"🔥 NBA 2.10–2.30 2. FAVORİ ÜSTÜNLÜĞÜ: {sec_b_clean} ({odd2_val:.2f})"
            zone_type = "TRAP_LOW"
            stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
            primary_pick = f"🥈 {sec_b_clean} Baremini Oyna (2. Favori — Oran: {odd2_val:.2f})"
            primary_barems = [sec_b_clean]
            primary_weights = [1.0]
            conf_pct = 79.0
            reasons.append(
                f"358 NBA 2K26 maçında 2.10–2.30 bandında (fark 6+ iken) 1. Favori -287 TL zarar yazarken, 2. Favori ({sec_b_clean}) 82 maçta %40.2 tutma ve 1.000 TL başı +162 TL (eski +151 TL, yeni +179 TL) net kâr sağlamıştır"
            )
        elif 2.30 <= odd_val < 2.50:
            if (prev_shift == 1 or drift == 0 or upper_wing_low or odd_gap_12 <= 0.25 or is_v_dip or (10 <= gap <= 15)) and 1.01 < odd2_val < 85.0:
                why_230 = (
                    "Önceki Maç +1 Üst (+230 TL)" if prev_shift == 1
                    else "Barem Sabit Kaldı (+285 TL)" if drift == 0
                    else "Tam İkiz Oran (+128 TL)" if odd_gap_12 <= 0.25
                    else "V-Dip Simetrisi (+213 TL)" if is_v_dip
                    else "10–15 Fark (+286 TL)" if (10 <= gap <= 15)
                    else "Üst Kanat Düşük (+101 TL)"
                )
                zone_badge = f"🔥 NBA 2.30–2.50 2. FAVORİ SİMETRİSİ ({why_230}): {sec_b_clean}"
                zone_type = "TRAP_LOW"
                stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
                primary_pick = f"🥈 {sec_b_clean} Baremini Oyna ({why_230} — Oran: {odd2_val:.2f})"
                primary_barems = [sec_b_clean]
                primary_weights = [1.0]
                conf_pct = 77.0
                reasons.append(f"NBA 2K26'da 2.30–2.50 aralığında '{why_230}' olduğunda 2. Favori ({sec_b_clean}) hem eski hem yeni maçlarda yüksek kâr üretmektedir")
            else:
                zone_badge = f"💎 NBA 2.30–2.50 DEĞER FAVORİSİ: {b_0} ({odd_val:.2f})"
                zone_type = "GOLDEN"
                stake_badge = f"1000 TL (Direkt {b_0} Baremine Oyna)"
                primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
                primary_barems = [b_0]
                primary_weights = [1.0]
                conf_pct = 75.0
                reasons.append("NBA 2K26'da 2.30–2.50 aralığında barem hareketliyken 1. Favori (%45.6 / 1.000 TL başı +78 TL; eski +63 TL, yeni +113 TL) kazandırmaktadır")
        else:
            zone_badge = f"⛔ SEÇİCİ PAS GEÇ (Net Üstünlük Yok — {b_0}: {odd_val:.2f})" if odd_val < 90 else "⏳ Oran Bekleniyor"
            zone_type = "PASS_SELECTIVE"
            stake_badge = "0 TL (Pas Geç — Bu Oran/Farkta Değer Yok)" if odd_val < 90 else "Oran Bekleniyor"
            primary_pick = f"⛔ Bahis Alma (Net Fırsat Yok — Takip: {dutch_short})" if odd_val < 90 else "⏳ Oran Bekleniyor"
            conf_pct = 42.0
            reasons.append(f"Bu oran ({odd_val:.2f}) ve skor farkında ({gap}) net üstünlük yok; kasa korumak için pas geçilir")
    else:
        # ── EURO LEAGUE v7.0 ENGINE (345 matches: H1+H2 verified) ──
        if gap >= 21 and 1.01 < odd2_val < 85.0:
            zone_badge = f"🧊 EURO 21+ FARK RÖLANTİ SİMETRİSİ ➔ 2. Fav ({sec_b_clean}: {odd2_val:.2f})"
            zone_type = "TRAP_LOW"
            stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
            primary_pick = f"🥈 {sec_b_clean} Baremini Oyna (21+ Fark Rölantisi — Oran: {odd2_val:.2f})"
            primary_barems = [sec_b_clean]
            primary_weights = [1.0]
            conf_pct = 81.0
            reasons.append(
                f"345 Euro maçında 29. dakikada fark 21+ sayıya çıktığında maç yavaşladığı için 1. Favori -225 TL zarar yazarken, 2. Favori ({sec_b_clean}) 53 maçta %39.6 tutma ve 1.000 TL başı +270 TL (eski +103 TL, yeni +471 TL) net kâr sağlamıştır"
            )
        elif 2.10 <= odd_val < 2.50:
            if prev_shift == 1 or odd_gap_12 <= 0.25 or is_pendulum_zig or is_two_up:
                why_reb = (
                    "Önceki Maç +1 Üst (+445 TL)" if prev_shift == 1
                    else "Tam İkiz Oran (+313 TL)" if odd_gap_12 <= 0.25
                    else "Son 2 Maç Üst (+356 TL)" if is_two_up
                    else "Sarkaç Simetrisi (+180 TL)"
                )
                zone_badge = f"🔄 EURO MERKEZE DÖNÜŞ SİMETRİSİ ({why_reb}): {b_0} ({odd_val:.2f})"
                zone_type = "GOLDEN_REBOUND"
                stake_badge = f"1000 TL ({b_0} Merkez Baremine Oyna)"
                primary_pick = f"🎯 Direkt {b_0} Baremini Oyna ({why_reb} — Oran: {odd_val:.2f})"
                primary_barems = [b_0]
                primary_weights = [1.0]
                conf_pct = 83.0
                reasons.append(
                    f"Euro League'de 2.10–2.50 aralığında '{why_reb}' aktifken 1. Favori ({b_0}) hem eski hem yeni maçlarda %61–%69 tutma ve 1.000 TL başı +180 TL ile +445 TL net kazanç sağlamıştır"
                )
            elif gap >= 6 and 1.01 < odd2_val < 85.0:
                why_eu_f2 = "Üst Kanat Açık (+433 TL)" if upper_wing_low else "2.10–2.50 2.Fav Üstünlüğü (+163 TL)"
                zone_badge = f"🔥 EURO 2.10–2.50 2. FAVORİ FIRSATI ({why_eu_f2}): {sec_b_clean}"
                zone_type = "TRAP_LOW"
                stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
                primary_pick = f"🥈 {sec_b_clean} Baremini Oyna ({why_eu_f2} — Oran: {odd2_val:.2f})"
                primary_barems = [sec_b_clean]
                primary_weights = [1.0]
                conf_pct = 79.0
                reasons.append(
                    f"Euro League'de 2.10–2.50 aralığında (fark 6–20 iken) 2. Favori ({sec_b_clean}) 54 maçta %44.4 tutma ve 1.000 TL başı +163 TL (üst kanat düşükken %53.1 / +433 TL) net kâr sağlamıştır"
                )
            else:
                zone_badge = f"⛔ EURO SEÇİCİ PAS GEÇ ({b_0}: {odd_val:.2f})"
                zone_type = "PASS_SELECTIVE"
                stake_badge = "0 TL (Pas Geç)"
                primary_pick = f"⛔ Bahis Alma (Takip: {b_0})"
                conf_pct = 44.0
                reasons.append("Euro 2.10+ ve 5 sayı farkta net sinyal yok; pas geçilir")
        elif 1.50 <= odd_val < 2.10:
            is_best_euro = (drift < 0 or prev_shift == 1 or is_pendulum_zig or two_fav_failed or is_ters_v or odd_gap_12 <= 0.45)
            zone_badge = (
                f"💎 EURO ALTIN SİMETRİ FAVORİSİ: {b_0} ({odd_val:.2f})"
                if is_best_euro
                else f"💎 EURO GÜÇLÜ MERKEZ FAVORİ: {b_0} ({odd_val:.2f})"
            )
            zone_type = "GOLDEN"
            stake_badge = f"1000 TL (Direkt {b_0} Baremine Oyna)"
            primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
            primary_barems = [b_0]
            primary_weights = [1.0]
            conf_pct = 86.0 if is_best_euro else 79.0
            reasons.append(
                f"345 Euro maçında 1.50–2.10 ve 5–20 fark aralığında 1. Favori ({b_0}) 119 maçta %58.8 tutma ve 1.000 TL başı +132 TL (simetri sinyaliyle +%62 / +163 TL+) net kazanç sağlamıştır"
            )
        elif 1.20 <= odd_val < 1.50 and (prev_shift == 1 or is_pendulum_zig or two_fav_failed or drift < 0 or odd_gap_12 <= 0.45):
            zone_badge = f"💎 EURO DÜŞÜK ORAN GÜVENLİ MERKEZ: {b_0} ({odd_val:.2f})"
            zone_type = "GOLDEN"
            stake_badge = f"1000 TL (Direkt {b_0} Baremine Oyna)"
            primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
            primary_barems = [b_0]
            primary_weights = [1.0]
            conf_pct = 80.0
            reasons.append(f"Euro'da 1.20–1.50 aralığında merkeze dönüş sinyali aktifken 1. Favori ({b_0}) %78+ isabetle tutmaktadır")
        else:
            zone_badge = f"⛔ EURO SEÇİCİ PAS GEÇ ({b_0}: {odd_val:.2f})" if odd_val < 90 else "⏳ Oran Bekleniyor"
            zone_type = "PASS_SELECTIVE"
            stake_badge = "0 TL (Pas Geç — 1.20–1.50 Tepki Sinyalsiz Riskli)" if odd_val < 90 else "Oran Bekleniyor"
            primary_pick = f"⛔ Bahis Alma (Net Sinyal Yok — Takip: {b_0})" if odd_val < 90 else "⏳ Oran Bekleniyor"
            conf_pct = 44.0
            reasons.append("Euro League'de 1.20–1.50 aralığında merkeze dönüş sinyali yoksa getiri/risk oranı düşüktür; pas geçilir")

    # 2. Tercih (Ağırlıklı İkili Kapat) v7.0 — H1+H2 VERIFIED POSITIVE ACROSS ALL 3 LEAGUES!
    bak_b1, bak_b2 = b_0, sec_b_clean
    if is_euro_lg and gap >= 16:
        bak_b1, bak_b2 = b_0, b_m1
    bak_o1 = valid_odds_dict.get(bak_b1, odd_val if bak_b1 == b_0 and odd_val < 90 else 99.0)
    bak_o2 = valid_odds_dict.get(bak_b2, odd2_val if bak_b2 == sec_b_clean and odd2_val < 90 else 99.0)
    bak_valid_odds = (bak_b1 != bak_b2) and (1.01 < bak_o1 < 85.0) and (1.01 < bak_o2 < 85.0)
    bak_eff_odd = round(1.0 / ((1.0 / bak_o1) + (1.0 / bak_o2)), 2) if bak_valid_odds else 0.0
    bak_is_mc_proj = False

    if is_euro_lg:
        # Euro 2. Tercih (345 matches: %84.0 | +193 TL/1000 | H1 +249 TL | H2 +122 TL!)
        inv_d = ((1.0 / bak_o1) + (1.0 / bak_o2)) if bak_valid_odds else 1.0
        bak_st1 = int(round(1000.0 * (1.0 / bak_o1) / inv_d)) if bak_valid_odds else 500
        bak_st2 = 1000 - bak_st1
        bak_w1, bak_w2 = round(bak_st1 / 1000.0, 4), round(bak_st2 / 1000.0, 4)
        euro_ok_2t = (1.20 <= odd_val < 2.30) and (gap >= 6) and (bak_eff_odd >= 1.26)
        backup_playable = bool(bak_valid_odds and zone_type not in ("PASS_CLUTCH", "PASS_WEAK", "PASS_LOW_ODD") and euro_ok_2t)
    elif is_2k27_lg:
        # NBA 2K27 2. Tercih: 650/350 on Alt Kanat Dusuk & sym_gap<=0.65 (+70 TL, %85.7); 450/550 on drift<0 & sym_gap<=0.65 (+73 TL, %87.5) or 1.50-1.85 or >=2.50
        if (lower_wing_low and odd_gap_12 <= 0.65) or odd_val < 1.50:
            bak_w1, bak_w2 = 0.65, 0.35
            bak_st1, bak_st2 = 650, 350
        else:
            bak_w1, bak_w2 = 0.45, 0.55
            bak_st1, bak_st2 = 450, 550
        nba27_ok_2t = (gap >= 6) and (
            (lower_wing_low and odd_gap_12 <= 0.65)
            or (drift < 0 and odd_gap_12 <= 0.65)
            or (1.20 <= odd_val < 1.85)
            or (odd_val >= 2.50)
        )
        backup_playable = bool(bak_valid_odds and zone_type not in ("PASS_CLUTCH", "PASS_WEAK", "PASS_LOW_ODD") and nba27_ok_2t)
    else:
        # NBA 2K26 2. Tercih (358 matches: H1+H2 verified positive!):
        # - 1.85-2.10 & gap>=6 & prev_shift!=1 -> 650/350 (%84.5 / +49 TL; H1 +74, H2 +9)
        # - 2.30-2.50 & gap>=6 -> 650/350 (%80.7 / +51 TL; H1 +17, H2 +132)
        # - sym_gap<=0.45 & gap>=6 -> 650/350 (%86.3 / +47 TL; H1 +4, H2 +120)
        # - o1 < 1.50 -> 450/550 (%90.9 / +159 TL)
        # - 10 <= gap <= 20 & 1.85 <= o1 < 2.50 -> 450/550 (%80.5 / +34 TL; H1 +48, H2 +3)
        if (1.85 <= odd_val < 2.10 and prev_shift != 1) or (2.30 <= odd_val < 2.50) or (odd_gap_12 <= 0.45 and odd_val >= 1.85):
            bak_w1, bak_w2 = 0.65, 0.35
            bak_st1, bak_st2 = 650, 350
        else:
            bak_w1, bak_w2 = 0.45, 0.55
            bak_st1, bak_st2 = 450, 550
        nba_ok_2t = (gap >= 6) and (
            odd_val < 1.50
            or (1.85 <= odd_val < 2.10 and prev_shift != 1)
            or (2.30 <= odd_val < 2.50)
            or (odd_gap_12 <= 0.45 and 1.85 <= odd_val < 2.50)
            or (10 <= gap <= 20 and 1.85 <= odd_val < 2.50)
        )
        backup_playable = bool(bak_valid_odds and zone_type not in ("PASS_CLUTCH", "PASS_WEAK") and nba_ok_2t)

    if backup_playable:
        ret1_tl = int(round(bak_st1 * bak_o1))
        ret2_tl = int(round(bak_st2 * bak_o2))
        net1_tl = ret1_tl - 1000
        net2_tl = ret2_tl - 1000
        if bak_is_mc_proj:
            backup_pick = f"💎 Akıllı Ağırlıklı İkili: {bak_b1} ({bak_st1} TL) + {bak_b2} ({bak_st2} TL)"
        elif bak_st1 != bak_st2 and abs(bak_st1 - 450) <= 50 and abs(bak_st2 - 550) <= 50:
            backup_pick = f"🛡️ Güçlü Ağırlıklı İkili: {bak_b2} ({bak_st2} TL) + {bak_b1} ({bak_st1} TL)"
        else:
            backup_pick = f"🛡️ {bak_b1} ile {bak_b2} İkisini Birden Kapat"
        dutch_split_badge = (
            f"💸 1000 TL Akıllı Bölüşüm: <b>{bak_b1}</b> ({bak_o1:.2f}) ➔ <b>{bak_st1} TL</b> (Dönüş: {ret1_tl} TL / {net1_tl:+d} TL) + "
            f"<b>{bak_b2}</b> ({bak_o2:.2f}) ➔ <b>{bak_st2} TL</b> (Dönüş: {ret2_tl} TL / {net2_tl:+d} TL)"
        )
        dutch_split_tip = (
            f"Ağırlıklı/seçici ikili kapatma komisyon erimesini önler: {bak_b1} ({bak_st1} TL ➔ {net1_tl:+d} TL), "
            f"{bak_b2} ({bak_st2} TL ➔ {net2_tl:+d} TL)."
        )
        backup_barems = [bak_b1, bak_b2]
        backup_weights = [bak_w1, bak_w2]
    else:
        why_no_pair = (
            "Maç Başa Baş (Pas)" if zone_type == "PASS_CLUTCH"
            else "Oran 1.20 Altı (Pas)" if zone_type == "PASS_LOW_ODD"
            else "Zayıf Favori (Pas)" if zone_type == "PASS_WEAK"
            else f"Fark < 10 Sayı ({gap} Fark)" if (not is_euro_lg and gap < 10)
            else f"Ortak Oran Çok Düşük ({bak_eff_odd:.2f})" if bak_eff_odd > 0
            else "Canlı Oran Yetersiz"
        )
        backup_pick = f"⛔ İkili Kapatma Uygun Değil ({why_no_pair})"
        dutch_split_badge = (
            f"⛔ Bu maçta garanti ikili açılmaz (<b>{bak_b1} ve {bak_b2}</b> ortak oran {bak_eff_odd:.2f} — şartlar sağlanmıyor)"
            if bak_valid_odds
            else f"💸 İki Barem Takibi: {dutch_short} (Oran bekleniyor)"
        )
        dutch_split_tip = "Sadece kârlı olduğu doğrulanan fark/oran şartlarında garanti ikili kapatma açılır."
        backup_barems = []
        backup_weights = []

    if drift > 0:
        reasons.append(f"Maç başında {open_b} bekleniyordu, {base_b} baremine yükseldi (daha üste çıkma ~%11, ya kalır ya 1 geri çekilir)")
    elif drift < 0:
        reasons.append(f"Maç başında {open_b} bekleniyordu, {base_b} baremine düştü (tersine yukarı sıçrama riski var)")

    if prev_shift <= -2:
        reasons.append(f"Önceki maç {prev_shift} geldi; 424 maçta üst üste iki kez -2 hiç görülmedi (%0), {b_0} / {b_m1} bölgesine toparlar")
    elif is_euro_lg and is_two_down:
        reasons.append(f"Son 2 maç üst üste alt bitti; Euro'da 3. maç %75.0 oranında (1.000 TL başı +433 TL) merkeze ({b_0}) döner")
    elif is_pendulum_zig:
        reasons.append(f"Sarkaç simetrisi ({prev2_shift:+d} ➔ {prev_shift:+d}): zikzak sonrası maçlar yüksek oranda {b_0} merkezine döner")
    elif not is_euro_lg and prev_shift == 1:
        reasons.append(f"Bir önceki maç +1 üst bitti; NBA'de +1 sonrası tek 1. Favori zayıflar (%29.5 / 1.000 TL başı -357 TL)")
    elif not is_euro_lg and prev_shift == -1:
        reasons.append(f"Bir önceki maç -1 alt bitti; NBA'de -1 sonrası %51.7 oranında {b_0} merkezine dönüş görülür")

    return {
        "zone_badge": zone_badge,
        "zone_type": zone_type,
        "stake_badge": stake_badge,
        "sec_dir": sec_dir,
        "sec_dir_badge": sec_dir_badge,
        "sec_dir_tip": sec_dir_tip,
        "margin_badge": margin_badge,
        "margin_tip": margin_tip,
        "edge_badge": edge_badge,
        "edge_tip": edge_tip,
        "proj_barem": proj_barem,
        "proj_diff": proj_diff,
        "mod10_badge": mod10_badge,
        "mod10_tip": mod10_tip,
        "twin_badge": twin_badge,
        "twin_tip": twin_tip,
        "wing_badge": wing_badge,
        "wing_tip": wing_tip,
        "pendulum_badge": pendulum_badge,
        "pendulum_tip": pendulum_tip,
        "seq_sym_badge": seq_sym_badge,
        "seq_sym_tip": seq_sym_tip,
        "odd_trend_badge": odd_trend_badge,
        "odd_trend_tip": odd_trend_tip,
        "dutch_split_badge": dutch_split_badge,
        "dutch_split_tip": dutch_split_tip,
        "drift": drift,
        "open_barem": open_b,
        "drift_badge": drift_badge,
        "drift_tip": drift_tip,
        "primary_pick": primary_pick,
        "primary_barems": primary_barems,
        "primary_weights": primary_weights,
        "backup_pick": backup_pick,
        "backup_playable": backup_playable,
        "backup_barems": backup_barems,
        "backup_weights": backup_weights,
        "confidence_pct": round(conf_pct, 1),
        "reasoning": " • ".join(reasons),
    }


def analyze_matchup_live_summary(df_matches: pd.DataFrame, team_home: str, team_away: str, league: str = "nba") -> dict:
    """
    Compute concise H2H & 4th-quarter DNA summary for a live match card.
    """
    cfg = get_league_config(league)
    avg_q4_lig = float(cfg["avg_q4_sum"])
    res = {
        "last5_str": "Veri bekleniyor",
        "top_cluster_str": "-",
        "q4_pace_badge": f"⚖️ 4. Çeyrek Ortalaması: ~{avg_q4_lig:.1f} Sayı",
        "q4_pace_tip": "İki takımın 4. çeyrek sayı ortalaması.",
    }
    if df_matches is None or df_matches.empty or not team_home or not team_away:
        return res

    df_m = df_matches.copy()
    for col in ["home_q4", "away_q4", "total_score", "is_ot"]:
        if col in df_m.columns:
            df_m[col] = pd.to_numeric(df_m[col], errors="coerce")

    th = str(team_home).strip()
    ta = str(team_away).strip()

    # 1) H2H games (either direction)
    h2h = df_m[
        ((df_m["home_team"] == th) & (df_m["away_team"] == ta))
        | ((df_m["home_team"] == ta) & (df_m["away_team"] == th))
    ].dropna(subset=["total_score"])

    if not h2h.empty:
        last5_scores = [str(int(s)) for s in h2h["total_score"].tail(5).tolist()][::-1]
        avg_h2h = float(h2h["total_score"].mean())
        res["last5_str"] = f"{' • '.join(last5_scores)} (Ort: {avg_h2h:.1f})"

        best_c = None
        best_cp = -1.0
        for c_name, c_lo, c_hi, _ in cfg["cluster_ranges"]:
            p = float(((h2h["total_score"] >= c_lo) & (h2h["total_score"] <= c_hi)).mean() * 100.0)
            if p > best_cp:
                best_cp = p
                best_c = c_name
        if best_c:
            res["top_cluster_str"] = f"{best_c} (%{best_cp:.0f} — {len(h2h)} Maç)"

    no_ot = df_m[df_m.get("is_ot", 0) == 0] if "is_ot" in df_m.columns else df_m
    hq4_s = pd.concat([
        no_ot[no_ot["home_team"] == th]["home_q4"],
        no_ot[no_ot["away_team"] == th]["away_q4"],
    ]).dropna()
    aq4_s = pd.concat([
        no_ot[no_ot["home_team"] == ta]["home_q4"],
        no_ot[no_ot["away_team"] == ta]["away_q4"],
    ]).dropna()

    def_q4_half = float(cfg["q4_half_default"])
    q4_h_avg = float(hq4_s.mean()) if len(hq4_s) >= 5 else def_q4_half
    q4_a_avg = float(aq4_s.mean()) if len(aq4_s) >= 5 else def_q4_half
    q4_sum = q4_h_avg + q4_a_avg

    if q4_sum <= float(cfg["q4_slow_max"]):
        res["q4_pace_badge"] = f"🐢 Yavaş 4. Çeyrek ({q4_sum:.1f} Sayı ➔ Favori/Alt Barem Güçlü)"
        res["q4_pace_tip"] = (
            f"Bu iki takımın 4. çeyrek ortalamaları toplamı {q4_sum:.1f} sayıdır (lig ortalaması {avg_q4_lig:.1f}'ten düşük). "
            f"Yavaş 4. çeyrek takımlarının maçlarında son periyotta sayı patlaması yaşanmaz!"
        )
    elif q4_sum >= float(cfg["q4_fast_min"]):
        res["q4_pace_badge"] = f"⚡ Hızlı 4. Çeyrek ({q4_sum:.1f} Sayı ➔ Yüksek Son Periyot Temposu)"
        res["q4_pace_tip"] = (
            f"Bu iki takımın 4. çeyrek ortalamaları toplamı {q4_sum:.1f} sayıdır (lig ortalaması {avg_q4_lig:.1f}'ten yüksek). "
            f"Son periyotta hızlı hücum ettikleri için alt bareme düşme ihtimali azalır."
        )
    else:
        res["q4_pace_badge"] = f"⚖️ Dengeli 4. Çeyrek (Ort. {q4_sum:.1f} Sayı)"
        res["q4_pace_tip"] = f"Bu iki takımın 4. çeyrek sayı ortalamaları toplamı {q4_sum:.1f} sayı ile tam lig ortalamasındadır."

    return res


def simulate_35m_strategies(df_log: pd.DataFrame, league: str = "nba") -> dict:
    """
    Simulate closing-minute betting strategies (1000 TL flat stake + tiered 0/1000/2000 TL bankroll)
    across all COMPLETED matches in log CSV, and build real analytical tables.
    """
    cfg = get_league_config(league)
    barem_names = [b[0] for b in cfg["barem_ranges"]]
    barem_bounds = {b[0]: (b[1], b[2]) for b in cfg["barem_ranges"]}
    b_cols = cfg["barem_odd_cols"]
    p_cols = cfg["barem_prob_cols"]
    dec_min = int(cfg["decision_min"])
    closing_start = int(cfg["closing_start_min"])
    pre_lo, pre_hi = int(cfg["q3_pre_low"]), int(cfg["q3_pre_high"])
    min_clean_gid = int(cfg["min_clean_gid"])
    exp_rem_dec = float(cfg["exp_rem_decision"])
    c_digs = cfg["mod10_center_digits"]
    m_digs = cfg["mod10_mid_digits"]
    r_digs = cfg["mod10_risk_digits"]

    empty_res = {
        "total_matches": 0,
        "strategies": [],
        "ledger": [],
        "odds_brackets": [],
        "sec_dir_brackets": [],
        "margin_brackets": [],
        "edge_brackets": [],
        "mod10_brackets": [],
        "twin_conc_brackets": [],
        "pendulum_brackets": [],
        "minute_windows": [],
        "drift_brackets": [],
        "transitions": [],
    }

    if df_log is None or df_log.empty or "status" not in df_log.columns:
        return empty_res

    comp = df_log[df_log["status"] == "COMPLETED"].copy()
    if comp.empty:
        return empty_res

    comp["_gid_num"] = comp["game_id"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)
    comp["_min"] = comp["check_time"].astype(str).str.extract(r"Minute\s+(\d+)")[0].fillna(-1).astype(int)
    comp = comp[comp["_gid_num"] >= min_clean_gid].sort_values(by=["_gid_num", "_min"], ascending=[True, True], kind="stable")

    def _parse_sc(s_val):
        try:
            pts = str(s_val).strip().split("-")
            return int(pts[0]), int(pts[1])
        except Exception:
            return None, None

    stake = 1000.0
    ledger = []

    s_v6_pri_ret, s_v6_pri_cost, s_v6_pri_wins, s_v6_pri_cnt = 0.0, 0.0, 0, 0
    s_v6_solo_ret, s_v6_solo_cost, s_v6_solo_wins, s_v6_solo_cnt = 0.0, 0.0, 0, 0
    s_v6_bak_ret, s_v6_bak_cost, s_v6_bak_wins, s_v6_bak_cnt = 0.0, 0.0, 0, 0
    s_v6_val_ret, s_v6_val_cost, s_v6_val_wins, s_v6_val_cnt = 0.0, 0.0, 0, 0
    s_dutch_sel_ret, s_dutch_sel_cost, s_dutch_sel_wins, s_dutch_sel_cnt = 0.0, 0.0, 0, 0
    s_0_ret, s_0_wins = 0.0, 0
    s_dutch_v2_ret, s_dutch_v2_wins = 0.0, 0
    s_dutch_ret, s_dutch_wins = 0.0, 0

    bounds_mid_sim = {}
    for bn, blo, bhi in cfg["barem_ranges"]:
        eff_blo = max(float(cfg["bar_lower_floor"]), float(blo))
        eff_bhi = min(float(bhi), float(cfg["bar_upper_cap"]))
        bounds_mid_sim[bn] = (eff_blo + eff_bhi) / 2.0

    raw_records = []
    chain_sim = []
    for gid, grp in comp.groupby("game_id", sort=False):
        first_r = grp.iloc[0]
        cands = grp[grp["_min"] >= closing_start]
        if cands.empty:
            continue
        last_r = cands.iloc[-1]
        for _, cand_row in cands.iloc[::-1].iterrows():
            if any(1.0 < float(cand_row.get(b_cols[bn], 0.0) or 0.0) < 90.0 for bn in barem_names):
                last_r = cand_row
                break
        cands30 = grp[(grp["_min"] >= pre_lo) & (grp["_min"] <= pre_hi)]
        r30 = cands30.iloc[-1] if not cands30.empty else first_r
        o30_raw = {b_n: float(r30.get(b_cols[b_n], 99.0) or 99.0) for b_n in barem_names}
        p30_raw = {b_n: float(r30.get(p_cols[b_n], 0.0) or 0.0) for b_n in barem_names}
        eff30, _ = resolve_effective_barem_odds(o30_raw, p30_raw, league=league)
        b30 = min(eff30, key=eff30.get) if eff30 else str(r30.get("highest_prob_range", "")).strip()

        act_b = str(last_r.get("actual_range", "")).strip()
        o_open_raw = {b_n: float(first_r.get(b_cols[b_n], 99.0) or 99.0) for b_n in barem_names}
        p_open_raw = {b_n: float(first_r.get(p_cols[b_n], 0.0) or 0.0) for b_n in barem_names}
        eff_open, _ = resolve_effective_barem_odds(o_open_raw, p_open_raw, league=league)
        open_b = min(eff_open, key=eff_open.get) if eff_open else str(first_r.get("highest_prob_range", "")).strip()
        if act_b not in barem_names:
            continue

        o_last_raw = {b_n: float(last_r.get(b_cols[b_n], 99.0) or 99.0) for b_n in barem_names}
        p_last_raw = {b_n: float(last_r.get(p_cols[b_n], 0.0) or 0.0) for b_n in barem_names}
        odds_map, locked_fav_b = resolve_effective_barem_odds(o_last_raw, p_last_raw, league=league)
        if not odds_map:
            continue

        if locked_fav_b:
            for _, cr in cands.iloc[::-1].iterrows():
                co = float(cr.get(b_cols[locked_fav_b], 99.0) or 99.0)
                if 1.001 <= co < 85.0:
                    odds_map[locked_fav_b] = co
                    break

        sorted_odds = sorted(odds_map.items(), key=lambda x: x[1])
        fav_b, odd_0 = sorted_odds[0]
        sec_b, sec_odd = sorted_odds[1] if len(sorted_odds) > 1 else sorted_odds[0]

        idx_fav = barem_names.index(fav_b)
        idx_act = barem_names.index(act_b)
        idx_sec = barem_names.index(sec_b)
        shift = idx_act - idx_fav
        sec_dir = idx_sec - idx_fav
        drift = (idx_fav - barem_names.index(open_b)) if open_b in barem_names else 0
        late_jump = (idx_fav - barem_names.index(b30)) if b30 in barem_names else 0
        m1_b = barem_names[max(0, idx_fav - 1)]
        p1_b = barem_names[min(len(barem_names) - 1, idx_fav + 1)]
        odd_m1 = odds_map.get(m1_b, 3.20)
        odd_p1 = odds_map.get(p1_b, 99.0)
        odd_gap_12 = round(sec_odd - odd_0, 2)
        inv_all = sum(1.0 / v for v in odds_map.values())
        top2_conc = ((1.0 / odd_0) + (1.0 / sec_odd)) / inv_all * 100.0 if inv_all > 0 else 75.0
        wing_diff = abs(odd_p1 - odd_m1) if (odd_m1 < 90 and odd_p1 < 90) else 99.0

        s1, s2 = _parse_sc(last_r.get("current_score", ""))
        a1, a2 = _parse_sc(last_r.get("actual_score", ""))
        margin35 = abs(s1 - s2) if s1 is not None else 8
        cur_tot35 = (s1 + s2) if s1 is not None else 0
        final_tot = (a1 + a2) if a1 is not None else 0
        rem_pts = (final_tot - cur_tot35) if (s1 is not None and a1 is not None) else None
        min_val = int(last_r.get("_min", dec_min))
        mod10 = cur_tot35 % 10

        fav_lo, fav_hi = barem_bounds[fav_b]
        eff_lo = max(float(cfg["bar_lower_floor"]), float(fav_lo))
        eff_hi = min(float(fav_hi), float(cfg["bar_upper_cap"]))
        fav_mid = (eff_lo + eff_hi) / 2.0
        # Gap-adjusted remaining points expectation from v7.0 calibration (762 matches)
        _lg_sim = str(league or "nba").lower()
        if _lg_sim == "nba2k27":
            _g_sl_sim, _g_int_sim = -0.225, 56.01
        elif _lg_sim != "euro":
            _g_sl_sim, _g_int_sim = -0.358, 58.81
        else:
            _g_sl_sim, _g_int_sim = -0.345, 42.30
        _rem_gap_est = (_g_int_sim + _g_sl_sim * min(30.0, float(margin35))) * (
            max(0.5, float(cfg["total_minutes"]) - (float(min_val) + 0.5)) / float(cfg["rem_span_decision"] - 0.5)
        )
        proj_35 = (float(cur_tot35) + _rem_gap_est) if cur_tot35 > 0 else fav_mid
        edge_offset = proj_35 - fav_mid

        # Run actual v6.1 engines (analyze_barem_symmetry & analyze_smart_35m_decision)
        sym_sim = analyze_barem_symmetry(
            odds_map, p_last_raw, league=league, proj_tot=proj_35, score_gap=margin35, drift=drift
        )
        sm_sim = analyze_smart_35m_decision(
            fav1_barem=fav_b,
            fav1_odd=odd_0,
            fav2_barem=sec_b,
            fav2_odd=sec_odd,
            game_id=str(gid),
            df_log=grp,
            recent_chain=chain_sim[-10:],
            score_gap=margin35,
            cur_tot=cur_tot35,
            elapsed_mins=float(min_val) + 0.5,
            odds_map=odds_map,
            league=league,
            proj_tot=proj_35,
        )

        # 1) v6.1 1. Tercih (Ana Öneri — Seçici OYNA)
        is_v6_pass = str(sm_sim.get("zone_type", "")).startswith("PASS_")
        p_bs = sm_sim.get("primary_barems") or [fav_b]
        p_ws = sm_sim.get("primary_weights") or [1.0]
        pri_ret_1000 = 0.0
        pri_win = False
        if not is_v6_pass and p_bs and all(1.001 <= odds_map.get(b, 99.0) < 85.0 for b in p_bs):
            s_v6_pri_cnt += 1
            s_v6_pri_cost += stake
            pri_win = act_b in p_bs
            for k_b, b_item in enumerate(p_bs):
                if b_item == act_b:
                    w_item = p_ws[k_b] if k_b < len(p_ws) else (1.0 / len(p_bs))
                    pri_ret_1000 += round(stake * w_item * odds_map[b_item], 1)
            if pri_win:
                s_v6_pri_wins += 1
                s_v6_pri_ret += pri_ret_1000
            if len(p_bs) == 1:
                s_v6_solo_cnt += 1
                s_v6_solo_cost += stake
                if pri_win:
                    s_v6_solo_wins += 1
                    s_v6_solo_ret += pri_ret_1000

        # 2) v6.1 2. Tercih (Akıllı Ağırlıklı / Seçici İkili Kapat)
        bak_ok = bool(sm_sim.get("backup_playable", False))
        b_bs = sm_sim.get("backup_barems") or []
        b_ws = sm_sim.get("backup_weights") or []
        bak_ret_1000 = 0.0
        bak_win = False
        if bak_ok and len(b_bs) >= 2 and all(1.001 <= odds_map.get(b, 99.0) < 85.0 for b in b_bs[:2]):
            s_v6_bak_cnt += 1
            s_v6_bak_cost += stake
            bak_win = act_b in b_bs[:2]
            for k_b in range(2):
                if b_bs[k_b] == act_b:
                    bak_ret_1000 += round(stake * b_ws[k_b] * odds_map[b_bs[k_b]], 1)
            if bak_win:
                s_v6_bak_wins += 1
                s_v6_bak_ret += bak_ret_1000

        # 3) v6.1 Seçici Akıllı Değer (Value)
        val_has_edge = bool(sym_sim.get("has_value_edge", False))
        val_sel_b = sym_sim.get("selective_value_range") or ""
        if val_has_edge and val_sel_b in odds_map and 1.001 <= odds_map[val_sel_b] < 85.0:
            s_v6_val_cnt += 1
            s_v6_val_cost += stake
            if act_b == val_sel_b:
                s_v6_val_wins += 1
                s_v6_val_ret += round(stake * odds_map[val_sel_b], 1)

        # 4) Pure 0 (1. Favori Tek Başına)
        hit_0 = act_b == fav_b
        ret_0 = round(stake * odd_0, 1) if hit_0 else 0.0
        s_0_ret += ret_0
        s_0_wins += int(hit_0)

        # 5) Fixed Dutching (0 & -1)
        hit_dutch = act_b in (fav_b, m1_b)
        dutch_odd = round(1.0 / ((1.0 / odd_0) + (1.0 / odd_m1)), 2) if m1_b != fav_b else odd_0
        ret_dutch = round(stake * dutch_odd, 1) if hit_dutch else 0.0
        s_dutch_ret += ret_dutch
        s_dutch_wins += int(hit_dutch)

        # 6) All-Match Top-2 Equal Dutching
        d2_partner = m1_b if margin35 >= 20 else sec_b
        d2_p_odd = odd_m1 if d2_partner == m1_b else sec_odd
        if d2_partner != fav_b and d2_p_odd > 1.0:
            dutch_v2_odd = round(1.0 / ((1.0 / odd_0) + (1.0 / d2_p_odd)), 2)
            hit_dutch_v2 = act_b in (fav_b, d2_partner)
        else:
            dutch_v2_odd = odd_0
            hit_dutch_v2 = act_b == fav_b
        ret_dutch_v2 = round(stake * dutch_v2_odd, 1) if hit_dutch_v2 else 0.0
        s_dutch_v2_ret += ret_dutch_v2
        s_dutch_v2_wins += int(hit_dutch_v2)

        is_clutch_pass = margin35 <= (4 if str(league or "nba").lower() == "euro" else 5)
        if not is_clutch_pass:
            s_dutch_sel_cnt += 1
            s_dutch_sel_cost += stake
            if hit_dutch_v2:
                s_dutch_sel_wins += 1
                s_dutch_sel_ret += ret_dutch_v2

        raw_records.append(
            {
                "game_id": gid,
                "game_name": str(last_r.get("game_name", "")).replace(" (cyber)", ""),
                "check_time": str(last_r.get("check_time", "")),
                "min35": min_val,
                "margin35": margin35,
                "cur_tot35": cur_tot35,
                "mod10": mod10,
                "rem_pts": rem_pts,
                "edge_offset": edge_offset,
                "open_b": open_b,
                "fav_b": fav_b,
                "odd_0": odd_0,
                "sec_b": sec_b,
                "sec_odd": sec_odd,
                "sec_dir": sec_dir,
                "m1_b": m1_b,
                "odd_m1": odd_m1,
                "odd_p1": odd_p1,
                "odd_gap_12": odd_gap_12,
                "top2_conc": top2_conc,
                "wing_diff": wing_diff,
                "late_jump": late_jump,
                "act_b": act_b,
                "act_score": str(last_r.get("actual_score", "")),
                "shift": shift,
                "drift": drift,
            }
        )
        chain_sim.append({"raw_shift": shift})

        sec_arrow = f"📈 Üstte ({sec_b})" if sec_dir >= 1 else f"📉 Altta ({sec_b})"
        if is_v6_pass:
            pri_status = f"⛔ Pas Geçildi ({sm_sim.get('zone_badge', 'Pas')})"
        else:
            pri_status = f"✅ +{pri_ret_1000 - stake:.0f} TL" if pri_win else "❌ -1000 TL"

        if not bak_ok:
            bak_status = "⛔ İkili Açılmadı (Riskli/Düşük Oran)"
        else:
            bak_status = f"✅ +{bak_ret_1000 - stake:.0f} TL" if bak_win else "❌ -1000 TL"

        fav_disp_lbl = f"{fav_b} (🔒 Kilit / {odd_0:.2f})" if locked_fav_b else f"{fav_b} ({odd_0:.2f})"
        ledger.append(
            {
                "Maç": str(last_r.get("game_name", "")).replace(" (cyber)", ""),
                "Kapanış Dk & Fark": f"{last_r.get('check_time', '')} ({margin35} Sayı Fark | Sonu ..{mod10})",
                f"{dec_min}.Dk 1. Favori": fav_disp_lbl,
                "2. Favori Nerede?": f"{sec_arrow} ({sec_odd:.2f})",
                "🎯 1. Tercih (Ana Öneri)": sm_sim.get("primary_pick", "-"),
                "🛡️ 2. Tercih (İkili Kapat)": sm_sim.get("backup_pick", "-"),
                "Maçın Bittiği Barem": f"{act_b} ({last_r.get('actual_score', '')})",
                "1. Tercih Sonucu (1.000 TL)": pri_status,
                "2. Tercih Sonucu (1.000 TL)": bak_status,
            }
        )

    tot_m = len(raw_records)
    if tot_m == 0:
        return empty_res

    tot_cost = tot_m * stake
    strategies = [
        {
            "name": "👑 Yeni v6.1 — 1. Tercih (Ana Öneri: Sadece OYNA Denilen Maçlar, Pas Hariç)",
            "wins": s_v6_pri_wins,
            "total": s_v6_pri_cnt,
            "win_rate": round(s_v6_pri_wins / max(1, s_v6_pri_cnt) * 100.0, 1),
            "cost": round(s_v6_pri_cost, 0),
            "return": round(s_v6_pri_ret, 0),
            "net_profit": round(s_v6_pri_ret - s_v6_pri_cost, 0),
            "roi_pct": round((s_v6_pri_ret / max(1.0, s_v6_pri_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🎯 Yeni v6.1 — 1. Tercih (Sadece Tek Barem Önerilen Maçlar)",
            "wins": s_v6_solo_wins,
            "total": s_v6_solo_cnt,
            "win_rate": round(s_v6_solo_wins / max(1, s_v6_solo_cnt) * 100.0, 1),
            "cost": round(s_v6_solo_cost, 0),
            "return": round(s_v6_solo_ret, 0),
            "net_profit": round(s_v6_solo_ret - s_v6_solo_cost, 0),
            "roi_pct": round((s_v6_solo_ret / max(1.0, s_v6_solo_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🔥 Yeni v6.1 — Seçici Akıllı Değer (Sadece Değer Fırsatı Yakalanan Maçlar)",
            "wins": s_v6_val_wins,
            "total": s_v6_val_cnt,
            "win_rate": round(s_v6_val_wins / max(1, s_v6_val_cnt) * 100.0, 1),
            "cost": round(s_v6_val_cost, 0),
            "return": round(s_v6_val_ret, 0),
            "net_profit": round(s_v6_val_ret - s_v6_val_cost, 0),
            "roi_pct": round((s_v6_val_ret / max(1.0, s_v6_val_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🛡️ Yeni v6.1 — 2. Tercih (Akıllı Ağırlıklı / Seçici Garanti İkili Kapat)",
            "wins": s_v6_bak_wins,
            "total": s_v6_bak_cnt,
            "win_rate": round(s_v6_bak_wins / max(1, s_v6_bak_cnt) * 100.0, 1),
            "cost": round(s_v6_bak_cost, 0),
            "return": round(s_v6_bak_ret, 0),
            "net_profit": round(s_v6_bak_ret - s_v6_bak_cost, 0),
            "roi_pct": round((s_v6_bak_ret / max(1.0, s_v6_bak_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🛡️ Klasik Seçici İki Barem (0-5 Fark Pas + Eşit Bölüşüm 1. & 2. Favori)",
            "wins": s_dutch_sel_wins,
            "total": s_dutch_sel_cnt,
            "win_rate": round(s_dutch_sel_wins / max(1, s_dutch_sel_cnt) * 100.0, 1),
            "cost": round(s_dutch_sel_cost, 0),
            "return": round(s_dutch_sel_ret, 0),
            "net_profit": round(s_dutch_sel_ret - s_dutch_sel_cost, 0),
            "roi_pct": round((s_dutch_sel_ret / max(1.0, s_dutch_sel_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🛡️ Tüm Maçlara Eşit İki Barem Kapatma (Filtresiz 1. & 2. Favori İkilisi)",
            "wins": s_dutch_v2_wins,
            "total": tot_m,
            "win_rate": round(s_dutch_v2_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_dutch_v2_ret, 0),
            "net_profit": round(s_dutch_v2_ret - tot_cost, 0),
            "roi_pct": round((s_dutch_v2_ret / tot_cost - 1.0) * 100.0, 1),
        },
        {
            "name": "🎯 Her Maç Sadece 1. Favori Baremi Oynamak (Hiç Kural Kullanmadan)",
            "wins": s_0_wins,
            "total": tot_m,
            "win_rate": round(s_0_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_0_ret, 0),
            "net_profit": round(s_0_ret - tot_cost, 0),
            "roi_pct": round((s_0_ret / tot_cost - 1.0) * 100.0, 1),
        },
    ]

    # NEW SYMMETRY TABLE A: Last Digit of Score (1.000 TL basis)
    c_digs_lbl = " - ".join(str(x) for x in c_digs)
    m_digs_lbl = " - ".join(str(x) for x in m_digs)
    r_digs_lbl = " - ".join(str(x) for x in r_digs)
    mod10_brackets = []
    for digits, lbl, advice in [
        (c_digs, f"🎯 Skorun Son Rakamı {c_digs_lbl} (Tam Barem Ortasına Düşer)", f"💎 Kalan sürede ~{exp_rem_dec:.0f} sayı atılınca tam barem ortasına denk gelir"),
        (m_digs, f"📐 Skorun Son Rakamı {m_digs_lbl} (Barem Üst Kısmına Yakın)", f"⚖️ Kalan sürede ~{exp_rem_dec:.0f} sayı atılınca baremin orta-üst kısmına denk gelir"),
        (r_digs, f"⚠️ Skorun Son Rakamı {r_digs_lbl} (Alt Sınırda Kalır / 1 Alt Barem Riski)", f"📉 Kalan sürede ~{exp_rem_dec:.0f} sayı atılınca baremin en alt sınırında kalır: 1 Alt Barem riski yükselir"),
    ]:
        sub = [x for x in raw_records if x["mod10"] in digits]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_down = sum(1 for x in sub if x["shift"] <= -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            mod10_brackets.append(
                {
                    f"{dec_min}. Dk Toplam Skorun Son Rakamı": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "📉 Alt Bareme Düşme": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Anlama Geliyor?": advice,
                }
            )

    # NEW SYMMETRY TABLE B: Twin Odds & Top-2 Concentration (dynamic 1.000 TL metrics)
    twin_conc_brackets = []
    for cond_fn, lbl, advice in [
        (lambda x: x["odd_gap_12"] <= 0.25, "👯 Tam İkiz Favori (1. ve 2. Fav Oran Farkı <= 0.25)", "⚖️ İki barem neredeyse eşittir; NBA'de 2. Favori (+76 TL / 1.000 TL), genel olarak ikili koruma güçlüdür"),
        (lambda x: x["odd_gap_12"] <= 0.65, "🛡️ Yakın Simetri (1. ve 2. Fav Oran Farkı <= 0.65)", "🛡️ Maçların ~%78-%85'i bu iki baremden birinde biter; 550/450 TL ağırlıklı ikili için en uygun bölgedir"),
        (lambda x: x["wing_diff"] <= 0.75, "⚖️ Alt ve Üst Komşu Barem Oranları Dengeli (Fark <= 0.75)", "🎯 Alt ve üst kanatlar simetrik dağılır; merkez ve komşu dengelidir"),
        (lambda x: x["top2_conc"] < 70.0, "⚠️ Oranlar 3 Bareme Dağılmış / Kararsız Maç (İlk 2 İhtimali < %70)", "⚠️ Maç kararsızdır; tek barem riski yüksektir"),
    ]:
        sub = [x for x in raw_records if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_sec = sum(1 for x in sub if x["act_b"] == x["sec_b"])
            c_top2 = sum(1 for x in sub if x["act_b"] in (x["fav_b"], x["sec_b"]))
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            net_1000_2 = ((sum(x["sec_odd"] for x in sub if x["act_b"] == x["sec_b"] and x["sec_odd"] < 85) - n_s) / n_s) * 1000.0
            twin_conc_brackets.append(
                {
                    "Oranların Simetri Durumu": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "🥈 2. Favori (1.000 TL Başı)": f"%{c_sec/n_s*100:.1f} ({c_sec}/{n_s}) | {net_1000_2:+.0f} TL",
                    "🛡️ 1. veya 2. Favori Toplam": f"%{c_top2/n_s*100:.1f} ({c_top2}/{n_s})",
                    "💡 Ne İşe Yarar?": advice,
                }
            )

    # NEW SYMMETRY TABLE C: Late 5-Min (30m -> 35m) Change
    pendulum_brackets = []
    for cond_fn, lbl, advice in [
        (lambda x: x["late_jump"] == 0, f"⚖️ Son 5 Dakikada ({pre_hi-1}➔{dec_min}.Dk) Favori Barem Değişmedi", "🎯 Barem sabit kaldıysa merkez 1. Favori en dengeli halindedir"),
        (lambda x: x["late_jump"] <= -1, f"🔄 Son 5 Dakikada ({pre_hi-1}➔{dec_min}.Dk) Favori Barem Aşağı Düştü", "🚫 Kapanışta zaten düştüğü için daha da alta düşmesi nadirdir; yeni Favoride kalır veya eski baremine döner"),
        (lambda x: x["late_jump"] >= 1, f"🔄 Son 5 Dakikada ({pre_hi-1}➔{dec_min}.Dk) Favori Barem Yukarı Çıktı", "🧊 Kapanışta zaten yükseldiği için daha da üste kaçma ihtimali azalır; yeni Favori veya 1 Alt Baremde kalır"),
    ]:
        sub = [x for x in raw_records if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_m1 = sum(1 for x in sub if x["shift"] == -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            pendulum_brackets.append(
                {
                    "Son 5 Dakikadaki Barem Değişimi": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "📉 1 Alt Bareme Düşme": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Yapmalı?": advice,
                }
            )

    # 1. Odds Bracket Breakdown (v6.1 calibrated on 424 matches, 1.000 TL basis!)
    is_euro_sim = str(league or "nba").lower() == "euro"
    odds_brackets = []
    for cond_fn, lbl, advice in [
        (
            lambda x: x["odd_0"] < 1.20,
            "⛔ Çok Düşük Oran (< 1.20 — Pas Bölgesi)",
            "⛔ PAS GEÇ: Tutma oranı yüksek görünse de tek bir sürpriz tüm kazancı siler (1.000 TL başı NBA -349 TL, Euro -197 TL)",
        ),
        (
            lambda x: 1.20 <= x["odd_0"] < 1.50,
            "🚨 Kısa Oran Tuzağı (1.20 - 1.49 Arası)",
            "🥈 NBA'de 2. Favoriyi Oyna (+966 TL / 1.000 TL) veya Ağırlıklı İkili Kapat (%100 Tutar, +333 TL)" if not is_euro_sim else "⛔ Euro'da 1.20-1.49 tek başına düşük getiri verir; seçici pas geçilir",
        ),
        (
            lambda x: 1.50 <= x["odd_0"] < 1.85,
            "⚖️ Orta-Kısa Oran Aralığı (1.50 - 1.84 Arası)",
            "💎 NBA'de sadece Maç Sonu Tahmini siteden farklıysa oyna; aksi halde pas geç" if not is_euro_sim else "🎯 Euro'da başa baş olmayan maçlarda Direkt 1. Favori (%58.6) oynanır",
        ),
        (
            lambda x: 1.85 <= x["odd_0"] < 2.10,
            "💎 ALTIN TEK FAVORİ ARALIĞI (1.85 - 2.09)",
            "🎯 EN KÂRLI BÖLGE: Başa baş olmayan maçlarda Direkt 1. Favoriyi Oyna (NBA +264 TL / Euro +137 TL — her iki dönemde de artı!)",
        ),
        (
            lambda x: 2.10 <= x["odd_0"] < 2.30,
            "⚠️ 2.10 - 2.29 Oran Aralığı (Kritik Ayrım Bölgesi)",
            "🚨 NBA'de 1. Favori Tuzağı (-256 TL) ➔ Fark 10-20 ise 550/450 TL İkili, düşüş varsa 2. Favori!" if not is_euro_sim else "🔄 Euro'da sadece Tepki (Son 2 maç favori yattıysa veya barem düştüyse) varsa 1. Favori oyna",
        ),
        (
            lambda x: 2.30 <= x["odd_0"] < 2.50,
            "🛡️ 2.30 - 2.49 Yüksek Oran Aralığı",
            "🛡️ NBA'de Fark 10-20 ise 550/450 TL Ağırlıklı İkili veya Tahmin Fırsatı" if not is_euro_sim else "⛔ Euro'da 2.30 ve üzeri 1. Favoriler ağır zarar yazar (-731 TL) ➔ Pas Geç!",
        ),
        (
            lambda x: x["odd_0"] >= 2.50,
            "⛔ 2.50 ve Üzeri Belirsiz Favori (Pas Bölgesi)",
            "⛔ PAS GEÇ: 1. Favori çok belirsizdir (1.000 TL başı -572 TL zarar bölgesi)",
        ),
    ]:
        sub = [x for x in raw_records if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_sec = sum(1 for x in sub if x["act_b"] == x["sec_b"])
            c_top2 = sum(1 for x in sub if x["act_b"] in (x["fav_b"], x["sec_b"]))
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            net_1000_2 = ((sum(x["sec_odd"] for x in sub if x["act_b"] == x["sec_b"] and x["sec_odd"] < 85) - n_s) / n_s) * 1000.0
            odds_brackets.append(
                {
                    f"{dec_min}. Dk 1. Favori Oran Aralığı": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "🥈 2. Favori (1.000 TL Başı)": f"%{c_sec/n_s*100:.1f} ({c_sec}/{n_s}) | {net_1000_2:+.0f} TL",
                    "🛡️ 1. veya 2. Fav İkili Tutma": f"%{c_top2/n_s*100:.1f} ({c_top2}/{n_s})",
                    "💡 Ne Yapmalı? (v6.1 Kuralı)": advice,
                }
            )

    # 2. 2nd Favorite Direction Breakdown (-1 Lower vs +1 Upper)
    sec_dir_brackets = []
    for cond_fn, lbl, advice in [
        (lambda x: x["sec_dir"] >= 1 and 1.85 <= x["odd_0"] < 2.10, "💎 2. Favori Üstte VE Oran 1.85 - 2.09 (Altın Bölge)", "🎯 Direkt 1. Favori çok güçlüdür; maç kolay kolay alt bareme düşmez"),
        (lambda x: x["sec_dir"] <= -1 and 1.85 <= x["odd_0"] < 2.10, "💎 2. Favori Altta VE Oran 1.85 - 2.09 (Altın Bölge)", "🎯 Direkt 1. Favori veya Euro Alt Sapma varsa 1 Alt Barem değer taşır"),
        (lambda x: x["sec_dir"] >= 1 and x["odd_0"] >= 2.10, "📈 2. Favori Üstte VE Oran >= 2.10", "⚠️ Eskiden tek favori sanılırdı ama 2.10+ oranlarda tek 1. Favori risklidir; NBA'de 10-20 farkta 550/450 İkili daha güvenlidir"),
        (lambda x: x["sec_dir"] <= -1 and x["odd_0"] >= 2.10, "📉 2. Favori Altta VE Oran >= 2.10", "📉 Alt kanat baskındır; NBA'de düşüş varsa 2. Favori, Euro'da alt sapma varsa 1 Alt Barem öne çıkar"),
    ]:
        sub = [x for x in raw_records if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_m1 = sum(1 for x in sub if x["shift"] == -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            net_1000_m1 = ((sum(x["odd_m1"] for x in sub if x["shift"] == -1 and x["odd_m1"] < 85) - n_s) / n_s) * 1000.0
            sec_dir_brackets.append(
                {
                    "2. Favori Hangi Baremde Duruyor?": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "📉 1 Alt Barem (1.000 TL Başı)": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s}) | {net_1000_m1:+.0f} TL",
                    "📈 1 veya 2 Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Oynanmalı?": advice,
                }
            )

    # 3. Score Margin Breakdown (1.000 TL basis)
    margin_brackets = []
    for lo_m, hi_m, lbl, advice in [
        (0, 6, "⛔ 0 - 5 Sayı Fark (Başa Baş Maç / Uzatma ve Faul Riski)", "⛔ KESİN PAS GEÇ! Kalan sürede fauller ve uzatma yüzünden sayı şişer (NBA 1.000 TL başı -345 TL zarar)"),
        (6, 10, "⚖️ 6 - 9 Sayı Fark (Normal Skor Farkı)", "🎯 1.85-2.10 Altın Favori veya Tahmin Fırsatı varsa Tek Barem oyna; NBA'de bu farkta ikili kapatma açılmaz"),
        (10, 21, "🛡️ 10 - 20 Sayı Fark (Kontrollü Fark — En İyi İkili Bölgesi)", "🛡️ Uzatma riski düşüktür; hem 1.85-2.10 Tek Favori hem de 550/450 TL Ağırlıklı İkili (%80+ tutma) yüksek kâr verir"),
        (21, 999, "🧊 21+ Sayı Fark (Fark Çok Açıldı / Rölanti)", "🧊 Maç yavaşlar (son çeyrek rölanti); üst bareme kaçma ihtimali düşer, Favori ve 1 Alt Barem güçlenir"),
    ]:
        sub = [x for x in raw_records if lo_m <= x["margin35"] < hi_m]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_down = sum(1 for x in sub if x["shift"] <= -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            rems = [x["rem_pts"] for x in sub if x["min35"] == dec_min and x["rem_pts"] is not None]
            avg_rem = f"{sum(rems)/len(rems):.1f} Sayı" if rems else f"~{exp_rem_dec:.0f} Sayı"
            margin_brackets.append(
                {
                    f"{dec_min}. Dk Skor Farkı": lbl,
                    "Maç Sayısı": n_s,
                    "Kalan Sürede Ort. Atılan Sayı": avg_rem,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "📉 Alt Bareme Düşme": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Oynanmalı?": advice,
                }
            )

    # 4. Barem Interior Edge Position
    edge_brackets = []
    sub_35only = [x for x in raw_records if x["min35"] >= dec_min - 1]
    for cond_fn, lbl, advice in [
        (lambda x: x["edge_offset"] < -3.0, "📐 Baremin Alt Sınırında Kalıyor (Tahmin < Favori Ortası - 3 Sayı)", "📉 1 Alt Bareme düşme eğilimi yüksektir; Euro'da +356 TL / 1.000 TL alt sapma fırsatı üretir"),
        (lambda x: -3.0 <= x["edge_offset"] <= 3.0, "🎯 Baremin Tam Ortasında (Tahmin ≈ Favori Ortası)", "💎 Güvenli Merkez: Maç sonu beklentisi tam 1. Favorinin göbeğindedir"),
        (lambda x: x["edge_offset"] > 3.0, "📐 Baremin Üst Kısmında (Tahmin > Favori Ortası + 3 Sayı)", "📈 NBA'de üst barem değer fırsatı (+191 TL / 1.000 TL) üretir; Euro'da ise merkezde kalır"),
    ]:
        sub = [x for x in sub_35only if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_down = sum(1 for x in sub if x["shift"] <= -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            edge_brackets.append(
                {
                    "Tahmini Maç Sonu Baremde Nereye Düşüyor?": lbl,
                    "Maç Sayısı": n_s,
                    "📉 Alt Bareme Düşme": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne İşe Yarar?": advice,
                }
            )

    # 5. Minute-by-Minute Optimal Entry Window (1.000 TL basis)
    m_early = int(cfg["early_q2_min"])
    m_half = int(cfg["halftime_min"])
    m_midq3 = int(cfg["mid_q3_min"])
    m_trap = int(cfg["trap_min"])
    minute_windows = []
    for m_target, m_lbl, m_advice in [
        (m_early, f"⚡ {m_early}. Dakika (2. Çeyrek Ortası — Erken Bahis)", "⏳ Henüz erken; sadece gidişat takibi yapılır"),
        (m_half, f"⏸️ {m_half}. Dakika (Devre Arası)", "⚖️ İlk yarı / ikinci yarı simetrisi şekillenir"),
        (m_midq3, f"🕒 {m_midq3}. Dakika (3. Çeyrek Ortası)", "📐 Oranlar daralmaya başlar"),
        (m_trap, f"🚨 {m_trap - 1} - {m_trap}. Dakika (Kapanış Öncesi Geçiş)", "⚠️ Son barem kayması burada yaşanır"),
        (dec_min, f"🔥 {dec_min - 1} - {dec_min}. Dakika (Kapanış — ANA KARAR ANI)", "🎯 Tüm v6.1 kuralları ve en yüksek isabet burada çalışır"),
    ]:
        m_hits0, m_ret0, m_hits_d, m_ret_d, m_n = 0, 0.0, 0, 0.0, 0
        for gid, grp in comp.groupby("game_id", sort=False):
            cand = grp[(grp["_min"] >= m_target - 1) & (grp["_min"] <= m_target + 1)]
            if cand.empty:
                continue
            r_m = cand.iloc[-1]
            a_b = str(r_m.get("actual_range", "")).strip()
            if a_b not in barem_names:
                continue
            om_raw = {b_n: float(r_m.get(b_cols[b_n], 0.0) or 0.0) for b_n in barem_names}
            pm_raw = {b_n: float(r_m.get(p_cols[b_n], 0.0) or 0.0) for b_n in barem_names}
            om, _ = resolve_effective_barem_odds(om_raw, pm_raw, league=league)
            if not om:
                continue
            sb = sorted(om.items(), key=lambda x: x[1])
            f_b, f_o = sb[0]
            s_b, s_o = sb[1] if len(sb) > 1 else sb[0]
            m_n += 1
            if a_b == f_b:
                m_hits0 += 1
                m_ret0 += f_o
            if a_b in (f_b, s_b):
                m_hits_d += 1
                m_ret_d += (1.0 / (1.0 / f_o + 1.0 / s_o)) if s_b != f_b else f_o
        if m_n > 0:
            minute_windows.append(
                {
                    "Bahis Alınan Dakika": m_lbl,
                    "Maç Sayısı": m_n,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{m_hits0/m_n*100:.1f} ({m_hits0}/{m_n}) | {((m_ret0-m_n)/m_n)*1000:+.0f} TL",
                    "🛡️ 1. ve 2. Fav İkili (1.000 TL Başı)": f"%{m_hits_d/m_n*100:.1f} ({m_hits_d}/{m_n}) | {((m_ret_d-m_n)/m_n)*1000:+.0f} TL",
                    "💡 Hangi Dakika Daha İyi?": m_advice,
                }
            )

    # 6. Drift (Opening -> Closing Barem Change) Breakdown
    drift_brackets = []
    for cond_fn, lbl, advice in [
        (lambda d: d >= 1, f"📈 Maç Başından {dec_min}. Dakikaya Barem Yükseldi", "🎯 Maç hızlı gitti; üstten daha da üste kaçma nadirdir, merkez veya 1 Alt dengelenir"),
        (lambda d: d == 0, f"⚖️ Maç Başından {dec_min}. Dakikaya Barem Aynı Kaldı", "⚖️ Baştan sona dengeli tempo; 1.85-2.10 aralığındaysa 1. Favori çok güçlüdür"),
        (lambda d: d == -1, f"📉 Maç Başından {dec_min}. Dakikaya Barem 1 Kademe Düştü", "🔄 Euro'da 1. Favori tepkisi güçlüdür (+128 TL / 1.000 TL); NBA'de ise 2.10-2.30 aralığında 2. Favoriye kayar"),
        (lambda d: d <= -2, f"❄️ Maç Başından {dec_min}. Dakikaya Barem Çok Düştü (2+ Kademe)", "🛡️ Zaten çok düştüğü için daha fazla düşmez; yeni Favori veya eski üst komşusunda dengelenir"),
    ]:
        sub = [x for x in raw_records if cond_fn(x["drift"])]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_m1 = sum(1 for x in sub if x["shift"] == -1)
            c_m2 = sum(1 for x in sub if x["shift"] <= -2)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            net_1000_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 1000.0
            drift_brackets.append(
                {
                    f"Maç Başından {dec_min}. Dakikaya Barem Değişimi": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "📉 1 Alt Barem": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ 2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 1 veya 2 Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Anlama Geliyor?": advice,
                }
            )

    # 7. Sequential Match-to-Match & Pendulum Wave Symmetry Transitions (Real Tracked Chain)
    transitions = []
    transition_rules = [
        (lambda i: raw_records[i]["shift"] == 0, "🎯 Önceki Maçta 1. Favori (0) Geldiyse ➔ Sıradaki Maç"),
        (lambda i: raw_records[i]["shift"] == -1, "📉 Önceki Maç 1 Alt (-1) Geldiyse ➔ Sıradaki Maç (NBA Merkeze Dönüş %51.7)"),
        (lambda i: raw_records[i]["shift"] <= -2, "🛡️ Önceki Maç 2 Alt (-2) Geldiyse ➔ Sıradaki Maç (Tekrar -2 Gelme: %0.0!)"),
        (lambda i: raw_records[i]["shift"] == 1, "📈 Önceki Maç 1 Üst (+1) Geldiyse ➔ Sıradaki Maç (NBA'de Tek Fav Zayıflar)"),
        (
            lambda i: i >= 1 and ((raw_records[i - 1]["shift"] < 0 and raw_records[i]["shift"] > 0) or (raw_records[i - 1]["shift"] > 0 and raw_records[i]["shift"] < 0)),
            "🔄 Sarkaç (Zikzak: Alt➔Üst veya Üst➔Alt) Sonrası ➔ Sıradaki Maç (Merkeze Dönüş!)",
        ),
        (
            lambda i: i >= 1 and raw_records[i - 1]["shift"] < 0 and raw_records[i]["shift"] < 0,
            "❄️ Üst Üste 2 Maç Alt Geldiyse ➔ Sıradaki Maç (Euro Merkez %75.0 / +433 TL)",
        ),
        (
            lambda i: i >= 1 and raw_records[i - 1]["shift"] != 0 and raw_records[i]["shift"] != 0,
            "🔄 Üst Üste 2 Maç Favori Yattıysa ➔ Sıradaki Maç (Euro Merkez %65.6 / +213 TL)",
        ),
    ]
    for cond_idx_fn, lbl in transition_rules:
        nxt = [raw_records[i + 1] for i in range(len(raw_records) - 1) if cond_idx_fn(i)]
        if nxt:
            n_s = len(nxt)
            c_0 = sum(1 for x in nxt if x["shift"] == 0)
            c_m1 = sum(1 for x in nxt if x["shift"] == -1)
            c_m2 = sum(1 for x in nxt if x["shift"] <= -2)
            c_up = sum(1 for x in nxt if x["shift"] >= 1)
            net_1000_0 = ((sum(x["odd_0"] for x in nxt if x["shift"] == 0) - n_s) / n_s) * 1000.0
            transitions.append(
                {
                    "Ardışık Maç Dalga & Sarkaç Simetrisi": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 Sıradaki 1. Favori (1.000 TL Başı)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | {net_1000_0:+.0f} TL",
                    "📉 Sıradaki 1 Alt Barem": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ Sıradaki 2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 Sıradaki Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "🛡️ Merkez 3'lü (-1 / 0 / +1) Tutma": f"%{(n_s - c_m2 - sum(1 for x in nxt if x['shift'] >= 2))/n_s*100:.1f}",
                }
            )

    return {
        "total_matches": tot_m,
        "strategies": strategies,
        "ledger": ledger[::-1],
        "odds_brackets": odds_brackets,
        "sec_dir_brackets": sec_dir_brackets,
        "margin_brackets": margin_brackets,
        "edge_brackets": edge_brackets,
        "mod10_brackets": mod10_brackets,
        "twin_conc_brackets": twin_conc_brackets,
        "pendulum_brackets": pendulum_brackets,
        "minute_windows": minute_windows,
        "drift_brackets": drift_brackets,
        "transitions": transitions,
    }


def analyze_archive_4500_symmetry(df_hist: pd.DataFrame, league: str = "nba") -> dict:
    """
    Analyzes completed historical archive matches for deep Mod-10 & Half-by-Half symmetries in plain Turkish.
    """
    if df_hist is None or df_hist.empty:
        return {"total_archive": 0, "mod10_36m": [], "half_symmetry": []}

    cfg = get_league_config(league)
    barem_names = [b[0] for b in cfg["barem_ranges"]]
    rem_c = int(cfg["exp_rem_close"])
    close_m = int(cfg["close_min_int"])
    h1_slow = int(cfg["archive_h1_slow_max"])
    h1_fast = int(cfg["archive_h1_fast_min"])

    df_h = df_hist.copy()
    for col in ["home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"]:
        if col in df_h.columns:
            df_h[col] = pd.to_numeric(df_h[col], errors="coerce").fillna(0)
        else:
            return {"total_archive": 0, "mod10_36m": [], "half_symmetry": []}

    df_h["Q1"] = df_h["home_q1"] + df_h["away_q1"]
    df_h["Q2"] = df_h["home_q2"] + df_h["away_q2"]
    df_h["Q3"] = df_h["home_q3"] + df_h["away_q3"]
    df_h["Q4"] = df_h["home_q4"] + df_h["away_q4"]
    df_h["H1"] = df_h["Q1"] + df_h["Q2"]
    df_h["H2"] = df_h["Q3"] + df_h["Q4"]
    df_h["S36"] = df_h["H1"] + df_h["Q3"]
    df_h = df_h[
        (df_h["total_score"] >= int(cfg["archive_min_total"]))
        & (df_h["S36"] >= int(cfg["archive_min_s3"]))
    ].copy()
    if df_h.empty:
        return {"total_archive": 0, "mod10_36m": [], "half_symmetry": []}

    df_h["s36_mod10"] = (df_h["S36"].astype(int)) % 10
    df_h["act_b_idx"] = df_h["total_score"].apply(lambda x: barem_names.index(get_range_from_score(x, league=league)))
    mod10_rows = []
    if str(league or "nba").lower() != "euro":
        digit_notes = {
            0: "⚖️ +52 sayı eklenince sonu ..2 olur (Baremin alt-orta kısmı): Beklenen Barem ve 1 Alt Barem dengeli",
            1: "📉 +52 sayı eklenince sonu ..3 olur (Alt sınıra yakın): 1 Alt Barem ve Beklenen Barem dengeli",
            2: "⚖️ +52 sayı eklenince sonu ..4 olur (Barem ortasına yakın)",
            3: "🎯 EN GÜVENLİ SON RAKAM (+52 eklenince sonu ..5 tam barem ortası olur): Beklenen Barem oranı en yüksek!",
            4: "🎯 +52 sayı eklenince sonu ..6 olur (Barem ortası)",
            5: "📈 +52 sayı eklenince sonu ..7 olur (Orta-üst kısım)",
            6: "📈 +52 sayı eklenince sonu ..8 olur (Üst sınıra yakın)",
            7: "🔥 +52 sayı eklenince sonu ..9 olur (Üst sınırda): Üst bareme geçiş güçlü!",
            8: "🔥 ÜST BAREME GEÇİŞ (+52 eklenince sonu ..0 barem tavanı olur)",
            9: "❄️ ALT BAREME DÜŞÜŞ (+52 eklenince sonu ..1 barem tabanı olur)",
        }
    else:
        digit_notes = {
            0: "🎯 +37 sayı eklenince sonu ..7 olur (Orta-üst kısım): Beklenen Barem çok güçlü",
            1: "📈 +37 sayı eklenince sonu ..8 olur (Üst sınıra yakın): 1 Üst Barem şansı yükselir",
            2: "📈 +37 sayı eklenince sonu ..9 olur (Üst sınırda): Üst bareme taşma eğilimi yüksektir",
            3: "🔥 ÜST BAREME GEÇİŞ (+37 eklenince sonu ..0 barem tavanı olur): En yüksek +1 Üst çıkış oranı!",
            4: "❄️ ALT BAREME DÜŞÜŞ (+37 eklenince sonu ..1 barem tabanı olur): 1 Alt Bareme düşme oranı en yüksek!",
            5: "📉 +37 sayı eklenince sonu ..2 olur (Alt sınıra yakın): 1 Alt Barem riski sürer",
            6: "⚖️ +37 sayı eklenince sonu ..3 olur (Alt-orta): 1 Alt ve Merkez dengeli",
            7: "🎯 +37 sayı eklenince sonu ..4 olur (Barem ortası): Beklenen Barem %53+ tutar",
            8: "🎯 EN GÜVENLİ SON RAKAM (+37 eklenince sonu ..5 tam barem ortası olur): En yüksek Merkez tutma oranı (~%56)!",
            9: "🎯 +37 sayı eklenince sonu ..6 olur (Barem ortası): Güvenli bölge",
        }
    ex_prefix = "15" if str(league or "nba").lower() != "euro" else "10"
    for d in range(10):
        sub = df_h[df_h["s36_mod10"] == d]
        if sub.empty:
            continue
        exp_tot = sub["S36"] + rem_c
        exp_b_idx = exp_tot.apply(lambda x: barem_names.index(get_range_from_score(x, league=league)))
        same_b = (sub["act_b_idx"] == exp_b_idx).mean() * 100.0
        down_b = (sub["act_b_idx"] < exp_b_idx).mean() * 100.0
        up_b = (sub["act_b_idx"] > exp_b_idx).mean() * 100.0
        mod10_rows.append(
            {
                f"3. Çeyrek Sonu ({close_m}.Dk) Skorun Son Rakamı": f"Sonu ..{d} ile bitenler (Örn: {ex_prefix}{d})",
                "Maç Sayısı": len(sub),
                f"Maç Sonu Beklenen Son Rakam (+{rem_c} Sayı)": f"..{(d + rem_c) % 10}",
                "📉 Alt Bareme Düşme": f"%{down_b:.1f}",
                "🎯 Beklenen Baremde Kalma": f"%{same_b:.1f}",
                "📈 Üst Bareme Çıkma": f"%{up_b:.1f}",
                f"💡 {len(df_h)} Maçlık İstatistik Özeti": digit_notes.get(d, ""),
            }
        )

    half_rows = []
    for lbl, cond, note in [
        (f"❄️ İlk Yarı Az Sayı Çıktı ({h1_slow} Sayı ve Altı)", df_h["H1"] <= h1_slow, "📈 İkinci Yarı Hızlanır: Düşük ilk yarı sonrası 2. yarı daha açık oynanır!"),
        (f"⚖️ İlk Yarı Normal Bitti ({h1_slow + 1} - {h1_fast - 1} Sayı Arası)", (df_h["H1"] > h1_slow) & (df_h["H1"] < h1_fast), "🪞 İki Yarı Dengeli Gider: 1. Yarı ile 2. Yarı yakın sayıda biter"),
        (f"🔥 İlk Yarı Çok Sayı Çıktı ({h1_fast} Sayı ve Üstü)", df_h["H1"] >= h1_fast, "🧊 İkinci Yarı Yavaşlar: Çok yüksek ilk yarı sonrası 2. yarı tempo biraz dengelenir!"),
    ]:
        sub = df_h[cond]
        if sub.empty:
            continue
        half_rows.append(
            {
                "İlk Yarı (1. + 2. Çeyrek) Toplam Sayı": lbl,
                "Maç Sayısı": len(sub),
                "1. Yarı Ortalaması": f"{sub['H1'].mean():.1f} Sayı",
                "2. Yarı Ortalaması": f"{sub['H2'].mean():.1f} Sayı",
                "Son Çeyrek (4. Çeyrek) Ortalaması": f"{sub['Q4'].mean():.1f} Sayı",
                "Maç Sonu Toplam Ort.": f"{sub['total_score'].mean():.1f} Sayı",
                "💡 İkinci Yarıda Ne Olur?": note,
            }
        )

    return {
        "total_archive": len(df_h),
        "mod10_36m": mod10_rows,
        "half_symmetry": half_rows,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TEAM-PAIR (H2H) HISTORY ENGINE (LEAGUE-AWARE)
# ═════════════════════════════════════════════════════════════════════════════
def analyze_team_pair_history(
    df_matches: pd.DataFrame, team_home: str, team_away: str, last_n: int = 8, league: str = "nba"
) -> dict:
    """
    Pair-based history: every finished game between the two teams (either direction).
    Returns counts, average total, barem distribution, tempo vs league and a plain-Turkish hint.
    """
    res = {
        "n": 0,
        "avg_total": 0.0,
        "std_total": 0.0,
        "league_avg": 0.0,
        "diff_vs_league": 0.0,
        "top_barem": "-",
        "top_barem_pct": 0.0,
        "second_barem": "-",
        "second_barem_pct": 0.0,
        "barem_counts": {},
        "avg_margin": 0.0,
        "ot_rate": 0.0,
        "avg_q4": 0.0,
        "last_games": [],
        "tempo_label": "⏳ Veri Bekleniyor",
        "hint": "Bu iki takım daha önce karşılaşmamış; çift bazlı sinyal yok.",
        "confidence": "none",
    }
    if df_matches is None or df_matches.empty or not team_home or not team_away:
        return res
    df_m = df_matches.copy()
    for col in ["total_score", "margin", "is_ot", "home_q4", "away_q4", "home_total", "away_total"]:
        if col in df_m.columns:
            df_m[col] = pd.to_numeric(df_m[col], errors="coerce")
    df_m = df_m.dropna(subset=["total_score"])
    if df_m.empty:
        return res
    res["league_avg"] = round(float(df_m["total_score"].mean()), 1)

    th = str(team_home).strip()
    ta = str(team_away).strip()
    h2h = df_m[
        ((df_m["home_team"] == th) & (df_m["away_team"] == ta))
        | ((df_m["home_team"] == ta) & (df_m["away_team"] == th))
    ].copy()
    if h2h.empty:
        return res
    if "date" in h2h.columns:
        h2h = h2h.sort_values("date", kind="stable")

    n = len(h2h)
    tots = h2h["total_score"].astype(float)
    avg_t = float(tots.mean())
    res["n"] = n
    res["avg_total"] = round(avg_t, 1)
    res["std_total"] = round(float(tots.std()) if n > 1 else 0.0, 1)
    res["diff_vs_league"] = round(avg_t - res["league_avg"], 1)

    barems = tots.apply(lambda s: get_range_from_score(s, league=league))
    counts = barems.value_counts()
    res["barem_counts"] = {str(k): int(v) for k, v in counts.items()}
    res["top_barem"] = str(counts.index[0])
    res["top_barem_pct"] = round(float(counts.iloc[0]) / n * 100.0, 1)
    if len(counts) > 1:
        res["second_barem"] = str(counts.index[1])
        res["second_barem_pct"] = round(float(counts.iloc[1]) / n * 100.0, 1)

    if "margin" in h2h.columns:
        res["avg_margin"] = round(float(h2h["margin"].abs().mean()), 1)
    if "is_ot" in h2h.columns:
        res["ot_rate"] = round(float((h2h["is_ot"].fillna(0) > 0).mean()) * 100.0, 1)
    if "home_q4" in h2h.columns and "away_q4" in h2h.columns:
        res["avg_q4"] = round(float((h2h["home_q4"] + h2h["away_q4"]).mean()), 1)

    last_games = []
    for _, r in h2h.tail(last_n).iloc[::-1].iterrows():
        tot = int(r["total_score"])
        last_games.append(
            {
                "Tarih": str(r.get("date", "")),
                "Maç": f"{str(r.get('home_team', '')).replace(' (cyber)', '')} vs {str(r.get('away_team', '')).replace(' (cyber)', '')}",
                "Skor": f"{int(r.get('home_total', 0) or 0)}-{int(r.get('away_total', 0) or 0)}",
                "Toplam": tot,
                "Barem": get_range_from_score(tot, league=league),
            }
        )
    res["last_games"] = last_games

    diff = res["diff_vs_league"]
    thr_diff = 8.0 if str(league or "nba").lower() != "euro" else 6.0
    if diff >= thr_diff:
        res["tempo_label"] = f"🔥 Yüksek Skorlu Çift (Lig Ort. +{diff:.1f})"
    elif diff <= -thr_diff:
        res["tempo_label"] = f"🧊 Düşük Skorlu Çift (Lig Ort. {diff:.1f})"
    else:
        res["tempo_label"] = f"⚖️ Lig Ortalamasına Yakın ({diff:+.1f})"

    if n >= 5:
        res["confidence"] = "high"
    elif n >= 3:
        res["confidence"] = "medium"
    else:
        res["confidence"] = "low"

    conf_txt = {
        "high": "güvenilir örneklem",
        "medium": "orta örneklem",
        "low": "az örneklem — sadece ipucu olarak bakın",
    }[res["confidence"]]
    res["hint"] = (
        f"{n} karşılaşmada ortalama {avg_t:.1f} sayı; en sık biten barem {res['top_barem']} "
        f"(%{res['top_barem_pct']:.0f}). {res['tempo_label']} • {conf_txt}."
    )
    return res


def build_team_pair_table(df_matches: pd.DataFrame, min_games: int = 2, league: str = "nba") -> pd.DataFrame:
    """One row per unordered team pair with >= min_games finished games."""
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()
    df_m = df_matches.copy()
    df_m["total_score"] = pd.to_numeric(df_m["total_score"], errors="coerce")
    df_m = df_m.dropna(subset=["total_score", "home_team", "away_team"])
    if df_m.empty:
        return pd.DataFrame()
    league_avg = float(df_m["total_score"].mean())
    df_m["_pair"] = df_m.apply(
        lambda r: " vs ".join(sorted([str(r["home_team"]).replace(" (cyber)", ""), str(r["away_team"]).replace(" (cyber)", "")])),
        axis=1,
    )
    rows = []
    for pair, g in df_m.groupby("_pair"):
        if len(g) < min_games:
            continue
        tots = g["total_score"].astype(float)
        bar = tots.apply(lambda s: get_range_from_score(s, league=league)).value_counts()
        rows.append(
            {
                "Takım Çifti": pair,
                "Maç": len(g),
                "Ort. Toplam": round(float(tots.mean()), 1),
                "Lig Farkı": f"{float(tots.mean()) - league_avg:+.1f}",
                "En Sık Barem": f"{bar.index[0]} (%{bar.iloc[0] / len(g) * 100:.0f})",
                "Min - Max": f"{int(tots.min())} - {int(tots.max())}",
                "_sort": len(g),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["_sort", "Ort. Toplam"], ascending=[False, False]).drop(columns=["_sort"])
    return out.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# DATA VALIDATION ("HATA BEKÇİSİ") (LEAGUE-AWARE)
# ═════════════════════════════════════════════════════════════════════════════
def _parse_score_pair(s):
    import re as _re

    m = _re.search(r"(\d+)\s*[-:]\s*(\d+)", str(s))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def validate_symmetry_log(df_log: pd.DataFrame, league: str = "nba") -> dict:
    """
    Scans every minute-row of the barem log and flags suspicious records.
    Severity: 🔴 Hata (veri yanlış olabilir) / 🟠 Uyarı (kontrol et) / 🔵 Bilgi.
    """
    import re as _re

    cfg = get_league_config(league)
    val_cols = [
        (b[0], cfg["barem_odd_cols"][b[0]], cfg["barem_prob_cols"][b[0]])
        for b in cfg["barem_ranges"]
    ]
    close_m_int = int(cfg["close_min_int"])
    gap_lo = close_m_int - 8
    out = {"issues": [], "n_rows": 0, "n_games": 0, "counts": {"🔴 Hata": 0, "🟠 Uyarı": 0, "🔵 Bilgi": 0}, "by_type": {}}
    if df_log is None or df_log.empty:
        return out

    df = df_log.reset_index(drop=True).copy()
    out["n_rows"] = len(df)
    out["n_games"] = int(df["game_id"].nunique()) if "game_id" in df.columns else 0

    def _add(sev, typ, row, detail, fix):
        out["issues"].append(
            {
                "Önem": sev,
                "Sorun Türü": typ,
                "Maç": str(row.get("game_name", "")).replace(" (cyber)", ""),
                "Maç Kodu": str(row.get("game_id", "")),
                "Dakika": str(row.get("check_time", "")),
                "Detay": detail,
                "Önerilen Çözüm": fix,
            }
        )
        out["counts"][sev] = out["counts"].get(sev, 0) + 1
        out["by_type"][typ] = out["by_type"].get(typ, 0) + 1

    for _, r in df.iterrows():
        o_d, p_d = {}, {}
        for b, oc, pc in val_cols:
            try:
                o_d[b] = float(r.get(oc, 99.0) or 99.0)
            except Exception:
                o_d[b] = 99.0
            try:
                p_d[b] = float(r.get(pc, 0.0) or 0.0)
            except Exception:
                p_d[b] = 0.0

        stored_fav = str(r.get("highest_prob_range", "")).strip()
        try:
            sym = analyze_barem_symmetry(o_d, p_d, league=league)
        except Exception:
            sym = {}
        new_fav = sym.get("highest_prob_range")
        if new_fav and stored_fav and new_fav != stored_fav:
            _add(
                "🔴 Hata",
                "Favori Uyuşmazlığı",
                r,
                f"Kayıtlı favori {stored_fav}, motorun hesapladığı {new_fav}",
                "Geçmiş kayıtları yeniden hesapla (otomatik onarım)",
            )

        if stored_fav in o_d:
            fo = o_d[stored_fav]
            if 9.0 <= fo < 85.0 and not sym.get("is_locked_fav", False):
                _add(
                    "🟠 Uyarı",
                    "Favori Oranı Çok Yüksek",
                    r,
                    f"Favori {stored_fav} oranı {fo:.2f}; gerçek favori kilitlenmiş olabilir",
                    "Komşu barem oranlarına ve 99.0 kilitlerine bak",
                )
            if fo < 1.0:
                _add("🔴 Hata", "Geçersiz Oran", r, f"{stored_fav} oranı {fo:.2f} (<1.00)", "Satırı sil / yeniden çek")

        open_cnt = sum(1 for v in o_d.values() if 1.0 < v < 85.0)
        mm = _re.search(r"Minute\s+(\d+)", str(r.get("check_time", "")))
        mn = int(mm.group(1)) if mm else None
        if open_cnt == 0 and mn is not None and mn < close_m_int - 6:
            _add("🟠 Uyarı", "Hiç Açık Oran Yok", r, "Tüm baremler kapalı (99.0)", "Kaynak bozuk olabilir; kaydı kontrol et")

        if str(r.get("status", "")) == "COMPLETED":
            act_sc = str(r.get("actual_score", "")).strip()
            act_rg = str(r.get("actual_range", "")).strip()
            if not act_rg or act_rg == "nan":
                _add("🔴 Hata", "Biten Maçta Sonuç Yok", r, "COMPLETED ama actual_range boş", "Sonuçları yeniden çek")
            else:
                pair = _parse_score_pair(act_sc)
                if pair:
                    exp_rg = get_range_from_score(pair[0] + pair[1], league=league)
                    if exp_rg != act_rg:
                        _add(
                            "🔴 Hata",
                            "Sonuç Barem Uyuşmazlığı",
                            r,
                            f"Skor {act_sc} = {pair[0] + pair[1]} sayı ➔ {exp_rg}, kayıt {act_rg}",
                            "actual_range değerini düzelt",
                        )

    if "game_id" in df.columns and "check_time" in df.columns:
        df["_min"] = df["check_time"].astype(str).str.extract(r"Minute\s+(\d+)")[0].astype(float)
        for gid, g in df.groupby("game_id", sort=False):
            g = g.sort_values("_min", kind="stable")
            last_row = g.iloc[-1]
            mins = g["_min"].dropna().tolist()
            for a, b in zip(mins, mins[1:]):
                if b - a > 3 and a >= gap_lo and b <= close_m_int:
                    rr = g[g["_min"] == b].iloc[0]
                    _add(
                        "🔵 Bilgi",
                        "Dakika Atlaması",
                        rr,
                        f"{int(a)}. dakikadan {int(b)}. dakikaya atlandı ({int(b - a)} dk boşluk)",
                        "Tarama gecikmesi; kritik değilse yok say",
                    )
                    break
            prev_tot = None
            for _, rr in g.iterrows():
                pr = _parse_score_pair(rr.get("current_score", ""))
                if not pr:
                    continue
                tot = pr[0] + pr[1]
                if prev_tot is not None and tot < prev_tot - 1:
                    _add(
                        "🔴 Hata",
                        "Skor Geriye Gitti",
                        rr,
                        f"Toplam skor {prev_tot} ➔ {tot} düştü",
                        "Yanlış eşleşen maç satırı olabilir; kontrol et",
                    )
                    break
                prev_tot = tot
            if "actual_score" in g.columns:
                acts = {str(x).strip() for x in g["actual_score"].dropna().tolist() if str(x).strip() not in ("", "nan")}
                if len(acts) > 1:
                    _add(
                        "🔴 Hata",
                        "Maç İçinde Farklı Final Skoru",
                        last_row,
                        f"Aynı maçta {len(acts)} farklı sonuç: {', '.join(sorted(acts))}",
                        "Doğru sonucu tek değere eşitle",
                    )

        dups = df.duplicated(subset=["game_id", "check_time"], keep=False)
        if dups.any():
            rr = df[dups].iloc[0]
            _add("🟠 Uyarı", "Tekrarlı Kayıt", rr, f"{int(dups.sum())} satır aynı maç + dakika", "Yinelenenleri temizle")

    sev_order = {"🔴 Hata": 0, "🟠 Uyarı": 1, "🔵 Bilgi": 2}
    out["issues"].sort(key=lambda x: (sev_order.get(x["Önem"], 9), x["Maç Kodu"]))
    return out
