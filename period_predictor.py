from datetime import datetime, timedelta
import numpy as np
import pandas as pd

__all__ = [
    "BAREM_RANGES",
    "CLUSTER_RANGES",
    "get_range_from_score",
    "get_cluster_from_score",
    "build_q3_q4_streak_signal",
    "predict_period_scores",
    "predict_live_inplay",
    "analyze_barem_symmetry",
    "analyze_sequential_barem_wave",
    "analyze_smart_35m_decision",
    "simulate_35m_strategies",
]

BAREM_RANGES = [
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

CLUSTER_RANGES = [
    ("161-180", 160.5, 180.5, ("161-170", "171-180")),
    ("171-190", 170.5, 190.5, ("171-180", "181-190")),
    ("181-200", 180.5, 200.5, ("181-190", "191-200")),
    ("191-210", 190.5, 210.5, ("191-200", "201-210")),
    ("201-220", 200.5, 220.5, ("201-210", "211-220")),
    ("211-230", 210.5, 230.5, ("211-220", "221-230")),
    ("221-240", 220.5, 240.5, ("221-230", "231+")),
]


def get_range_from_score(total: float) -> str:
    total_int = int(round(float(total)))
    if total_int <= 160:
        return "160 Or Less"
    elif total_int <= 170:
        return "161-170"
    elif total_int <= 180:
        return "171-180"
    elif total_int <= 190:
        return "181-190"
    elif total_int <= 200:
        return "191-200"
    elif total_int <= 210:
        return "201-210"
    elif total_int <= 220:
        return "211-220"
    elif total_int <= 230:
        return "221-230"
    else:
        return "231+"


def get_cluster_from_score(total: float) -> str:
    total_int = int(round(float(total)))
    if total_int <= 180:
        return "161-180"
    elif total_int <= 190:
        return "171-190"
    elif total_int <= 200:
        return "181-200"
    elif total_int <= 210:
        return "191-210"
    elif total_int <= 220:
        return "201-220"
    elif total_int <= 230:
        return "211-230"
    else:
        return "221-240"


def build_q3_q4_streak_signal(
    df,
    projected_q3_total=53,
    q3_threshold=50,
    streak_window=3,
    team_home="",
    team_away="",
    q3_score=None,
):
    """Calculate how often a Q3 total is followed by a Q4 burst or collapse."""
    if isinstance(projected_q3_total, str):
        team_home = projected_q3_total
        if isinstance(q3_threshold, str):
            team_away = q3_threshold
            q3_threshold = 50
        projected_q3_total = q3_score if q3_score is not None else 53
    elif q3_score is not None:
        projected_q3_total = q3_score

    default_resp = {
        "normal": {"probability": 0.50, "count": 0},
        "burst_positive": {"probability": 0.25, "count": 0},
        "collapse_negative": {"probability": 0.25, "count": 0},
        "signal": "⚖️ Dengeli Q4 Beklentisi",
        "detail": "Standart CyberLeague periyot dağılımı (53.0 sayı/çeyrek).",
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

    hist = df.dropna(subset=required).copy()
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
        bins=[-np.inf, 48.5, 55.5, np.inf],
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
        f"(Patlama ≥56: %{b_pct}, Normal 49-55: %{n_pct}, Çöküş ≤48: %{c_pct})."
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


def predict_period_scores(df, home_q1, away_q1, home_q2, away_q2, team_home=None, team_away=None, top_k=8):
    """Predict Q3/Q4 scoring using H2H/historical data or pure pace priors when database is empty."""
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
        # Pure pace + CyberLeague prior (53.0 pts per quarter baseline)
        prior_q = 53.0
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

    hist_raw = df.dropna(subset=required).copy()
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
    quarter_length=12.0,
    top_k=15,
):
    """
    Predict live match final score & barem distribution.
    Works 100% seamlessly both when `df` has historical matches AND when `df` is completely empty (clean slate).
    """
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
        hist_raw = df.dropna(subset=required).copy()
        for c in required:
            hist_raw[c] = hist_raw[c].astype(float)
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

    # Calculate baseline quarter score (CyberLeague average ~53.0 pts per 12m quarter = 212.0 total)
    if not hist.empty:
        avg_match_total = sum(hist[c].mean() for c in required)
        league_avg_q_score = float(avg_match_total) / 4.0
    else:
        # Also adapt baseline to completed quarters of the current match if available
        if completed_quarters:
            comp_q_avg = sum(h + a for h, a in completed_quarters) / len(completed_quarters)
            league_avg_q_score = 0.65 * comp_q_avg + 0.35 * 53.0
        else:
            league_avg_q_score = 53.0

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

    pred_home_final = sum(h_q for h_q, _ in completed_quarters) + predicted_home_q_final + sum(
        pred_subsequent.get(f"home_q{q_idx}", 0.0) for q_idx in range(current_quarter + 1, 5)
    )
    pred_away_final = sum(a_q for _, a_q in completed_quarters) + predicted_away_q_final + sum(
        pred_subsequent.get(f"away_q{q_idx}", 0.0) for q_idx in range(current_quarter + 1, 5)
    )

    total_elapsed_mins = (current_quarter - 1) * quarter_length + elapsed_in_q
    w_bayesian = float(np.clip(total_elapsed_mins / (4.0 * quarter_length), 0.15, 0.98))

    raw_pred_total = pred_home_final + pred_away_final
    bayesian_pred_total = (1.0 - w_bayesian) * hist_avg_total + w_bayesian * raw_pred_total

    if raw_pred_total > 0:
        shrinkage = bayesian_pred_total / raw_pred_total
        pred_home_final *= shrinkage
        pred_away_final *= shrinkage

    # Blowout / Rölanti Freni: 4.500 maçlık arşivde fark 28+ sayıya çıktığında Q4 ortalaması 52.5'tan 44-46'ya düşer
    cur_tot_h = sum(h_q for h_q, _ in completed_quarters) + float(home_live_score)
    cur_tot_a = sum(a_q for _, a_q in completed_quarters) + float(away_live_score)
    live_gap = abs(cur_tot_h - cur_tot_a)
    if current_quarter >= 3 and live_gap >= 28:
        brake_pts = min(9.5, 4.0 + (live_gap - 28.0) * 0.22) * ((48.0 - min(47.0, total_elapsed_mins)) / 16.0)
        tot_now = pred_home_final + pred_away_final
        if tot_now > 0:
            brake_ratio = max(0.92, (tot_now - brake_pts) / tot_now)
            pred_home_final *= brake_ratio
            pred_away_final *= brake_ratio

    # Dynamic standard deviation shrinks as remaining match minutes decrease!
    total_rem_mins = max(0.5, (4.0 * quarter_length) - total_elapsed_mins)
    rem_fraction = float(np.clip(total_rem_mins / (4.0 * quarter_length), 0.08, 1.0))
    # Full game std dev per team is ~8.5 pts (total game std dev ~12.0 pts), shrinking with sqrt(rem_fraction)
    team_std_dev = max(2.2, 8.5 * np.sqrt(rem_fraction))

    rng = np.random.default_rng(int( abs(pred_home_final * 100 + pred_away_final * 10) ) % 100000)
    sim_size = 10000
    sim_home_totals = rng.normal(pred_home_final, team_std_dev, sim_size)
    sim_away_totals = rng.normal(pred_away_final, team_std_dev, sim_size)
    sim_totals = sim_home_totals + sim_away_totals

    barem_probs = {}
    for b_name, b_low, b_high in BAREM_RANGES:
        if b_name == "160 Or Less":
            prob = float((sim_totals <= b_high).mean()) * 100.0
        elif b_name == "231+":
            prob = float((sim_totals > b_low).mean()) * 100.0
        else:
            prob = float(((sim_totals > b_low) & (sim_totals <= b_high)).mean()) * 100.0
        barem_probs[b_name] = round(prob, 1)

    clusters = {}
    for c_name, c_low, c_high, _ in CLUSTER_RANGES:
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


def analyze_barem_symmetry(odds_dict: dict, barem_probs: dict) -> dict:
    """
    Analyze 1xBet Barem Odds vs Model Probabilities:
    - Identifies Bookmaker 1st & 2nd Favorites
    - Calculates Symmetry Ratio (gap between top 2 favorite barem odds)
    - Calculates Best Value Range (highest positive EV / Edge)
    - Calculates Best Dutching (Çifte Barem) combined odd
    """
    valid_odds = {k: float(v) for k, v in (odds_dict or {}).items() if v and 1.01 < float(v) < 90.0}
    sorted_by_odds = sorted(valid_odds.items(), key=lambda x: x[1])

    bookie_fav_1 = sorted_by_odds[0][0] if len(sorted_by_odds) >= 1 else None
    bookie_fav_1_odd = sorted_by_odds[0][1] if len(sorted_by_odds) >= 1 else 99.0
    bookie_fav_2 = sorted_by_odds[1][0] if len(sorted_by_odds) >= 2 else None
    bookie_fav_2_odd = sorted_by_odds[1][1] if len(sorted_by_odds) >= 2 else 99.0

    # Symmetry gap between 1st and 2nd lowest odds
    symmetry_gap = round(bookie_fav_2_odd - bookie_fav_1_odd, 3) if (bookie_fav_1 and bookie_fav_2) else 99.0
    is_symmetric_trap = symmetry_gap <= 0.65

    # Dutching (Çifte Şans) effective odd if backing top 2 adjacent barems
    dutch_odd = round(1.0 / ((1.0 / bookie_fav_1_odd) + (1.0 / bookie_fav_2_odd)), 2) if (bookie_fav_1 and bookie_fav_2) else 0.0

    # Value pick calculation: Model Prob vs Implied Prob
    best_value_range = None
    best_ev = -999.0
    for b_name, _, _ in BAREM_RANGES:
        prob_pct = float((barem_probs or {}).get(b_name, 0.0))
        odd = valid_odds.get(b_name, 99.0)
        if odd < 90.0 and prob_pct >= 12.0:
            ev = (prob_pct / 100.0) * odd - 1.0
            if ev > best_ev:
                best_ev = ev
                best_value_range = b_name

    if not best_value_range:
        if barem_probs:
            best_value_range = max(barem_probs.keys(), key=lambda k: barem_probs[k])
        elif bookie_fav_1:
            best_value_range = bookie_fav_1
        else:
            best_value_range = "201-210"

    highest_prob_range = bookie_fav_1 if bookie_fav_1 else (
        max(barem_probs.keys(), key=lambda k: barem_probs[k]) if barem_probs else "201-210"
    )

    return {
        "highest_prob_range": highest_prob_range,
        "bookie_fav_1": bookie_fav_1,
        "bookie_fav_1_odd": bookie_fav_1_odd,
        "bookie_fav_2": bookie_fav_2,
        "bookie_fav_2_odd": bookie_fav_2_odd,
        "symmetry_gap": symmetry_gap,
        "is_symmetric_trap": is_symmetric_trap,
        "dutch_odd": dutch_odd,
        "value_range": best_value_range,
        "best_ev_pct": round(best_ev * 100.0, 1) if best_ev > -900 else 0.0,
    }


def analyze_sequential_barem_wave(
    df_matches: pd.DataFrame,
    df_log: pd.DataFrame = None,
    current_fav_barem: str = None,
    custom_seq: list = None,
) -> dict:
    """
    Analyze consecutive match-to-match 36th-minute Expected Barem vs Actual Final Barem shifts:
      0  -> Expected 36th-minute favorite barem hit (Merkez Barem)
     -1  -> Shifted 1 barem bracket DOWN (-1 Alt Barem)
     +1  -> Shifted 1 barem bracket UP (+1 Üst Barem)
    Uses Markov & N-gram sequence matching across all 4,500+ consecutive historical matches.
    """
    barem_names = [b[0] for b in BAREM_RANGES]

    def _shift_barem_name(base_b: str, delta: int) -> str:
        if base_b not in barem_names:
            return "-"
        idx = max(0, min(len(barem_names) - 1, barem_names.index(base_b) + delta))
        return barem_names[idx]

    if df_matches is None or df_matches.empty:
        return {
            "recent_chain": [],
            "last_5_codes": [],
            "last_5_labels": "-",
            "pattern_matches_count": 0,
            "prob_0": 41.7,
            "prob_minus_1": 24.0,
            "prob_plus_1": 21.7,
            "prob_minus_2": 6.2,
            "prob_plus_2": 6.4,
            "wave_direction": "0 / -1 (Beklenen veya 1 Alt Barem)",
            "target_0_barem": current_fav_barem or "201-210",
            "target_minus_1_barem": _shift_barem_name(current_fav_barem or "201-210", -1),
            "target_plus_1_barem": _shift_barem_name(current_fav_barem or "201-210", 1),
            "target_minus_2_barem": _shift_barem_name(current_fav_barem or "201-210", -2),
            "target_plus_2_barem": _shift_barem_name(current_fav_barem or "201-210", 2),
            "recommended_pair": f"{_shift_barem_name(current_fav_barem or '201-210', -1)} & {current_fav_barem or '201-210'}",
            "recommended_pair_prob": 65.7,
        }

    # Map closing favorite barem AND its exact odd from live symmetry_analysis_log.csv where available
    barem_to_col = {
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
    logged_closing_fav = {}
    logged_closing_odd = {}
    logged_closing_min = {}
    logged_opening_fav = {}
    if df_log is not None and not df_log.empty and "game_id" in df_log.columns:
        for gid, grp in df_log.groupby("game_id"):
            first_r = grp.iloc[0]
            last_r = grp.iloc[-1]
            first_fav = str(first_r.get("highest_prob_range", "")).strip()
            fav_val = str(last_r.get("highest_prob_range", "")).strip()
            if fav_val in barem_names:
                gid_clean = str(gid).strip()
                logged_closing_fav[gid_clean] = fav_val
                logged_opening_fav[gid_clean] = first_fav if first_fav in barem_names else fav_val
                odd_col = barem_to_col.get(fav_val, "")
                odd_val = float(last_r.get(odd_col, 0.0) or 0.0) if odd_col else 0.0
                logged_closing_odd[gid_clean] = round(odd_val, 2) if 1.0 < odd_val < 90.0 else 0.0
                logged_closing_min[gid_clean] = str(last_r.get("check_time", "")).strip()

    req_cols = ["home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4", "total_score"]
    hist = df_matches.dropna(subset=req_cols).copy()
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
    # 1xBet 36th-minute closing expectation: Q3 end score + blended Q4 pace (53.0 baseline + current pace)
    hist["exp_tot_36"] = hist["s36"] + (0.65 * 53.0 + 0.35 * (hist["s36"] / 3.0))

    exp_barems = []
    exp_odds = []
    sources = []
    for _, r in hist.iterrows():
        gid_s = str(r.get("game_id", "")).strip()
        if gid_s in logged_closing_fav:
            exp_barems.append(logged_closing_fav[gid_s])
            o_v = logged_closing_odd.get(gid_s, 0.0)
            c_m = logged_closing_min.get(gid_s, "35.Dk")
            exp_odds.append(o_v)
            sources.append(f"✅ Gerçek 1xBet Oranı ({c_m}: {o_v:.2f})" if o_v > 0 else f"✅ Gerçek 1xBet Oranı ({c_m})")
        else:
            exp_barems.append(get_range_from_score(r["exp_tot_36"]))
            exp_odds.append(0.0)
            sources.append("🧮 36.Dk Skor Tahmini (Oran Kaydı Yok)")

    hist["exp_barem_36"] = exp_barems
    hist["exp_odd_36"] = exp_odds
    hist["source"] = sources
    hist["act_barem"] = hist["total_score"].apply(get_range_from_score)

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

    # Build recent_chain STRICTLY AND ONLY from matches recorded in symmetry_analysis_log.csv!
    # Sort strictly by 1xBet sequential game_id so chronological order is 100% immune to server timezone.
    recent_chain = []
    tracked_hist = hist[hist["game_id"].astype(str).str.strip().isin(logged_closing_fav.keys())].copy()
    if not tracked_hist.empty:
        tracked_hist["_gid_num"] = (
            tracked_hist["game_id"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)
        )
        tracked_hist = (
            tracked_hist[tracked_hist["_gid_num"] >= 770265000]
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

    base_b = current_fav_barem if (current_fav_barem in barem_names) else "201-210"
    b_0 = base_b
    b_m1 = _shift_barem_name(base_b, -1)
    b_p1 = _shift_barem_name(base_b, 1)
    b_m2 = _shift_barem_name(base_b, -2)
    b_p2 = _shift_barem_name(base_b, 2)

    pair_down_prob = round(p_0 + p_m1, 1)
    pair_up_prob = round(p_0 + p_p1, 1)

    if pair_down_prob >= pair_up_prob + 2.0:
        wave_dir = f"📉 Favori ({b_0}) veya 1 Alt Barem ({b_m1}) Eğilimi (%{pair_down_prob})"
        rec_pair = f"{b_m1} (1 Alt Barem) ile {b_0} (Favori Barem)"
        rec_pair_prob = pair_down_prob
    elif pair_up_prob >= pair_down_prob + 2.0:
        wave_dir = f"📈 Favori ({b_0}) veya 1 Üst Barem ({b_p1}) Eğilimi (%{pair_up_prob})"
        rec_pair = f"{b_0} (Favori Barem) ile {b_p1} (1 Üst Barem)"
        rec_pair_prob = pair_up_prob
    else:
        wave_dir = f"🎯 Favori ({b_0}) Odaklı Dengeli Dalga (Favori %{p_0} | 1 Alt %{p_m1} | 1 Üst %{p_p1})"
        if pair_down_prob >= pair_up_prob:
            rec_pair = f"{b_m1} (1 Alt Barem) ile {b_0} (Favori Barem)"
            rec_pair_prob = pair_down_prob
        else:
            rec_pair = f"{b_0} (Favori Barem) ile {b_p1} (1 Üst Barem)"
            rec_pair_prob = pair_up_prob

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
        "recommended_pair_prob": rec_pair_prob,
    }


def _shift_barem_name(base_b: str, delta: int) -> str:
    names = [b[0] for b in BAREM_RANGES]
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
    elapsed_mins: float = 35.0,
    odds_map: dict = None,
) -> dict:
    """
    v4.0 Deep Symmetry & 35th-Minute Smart Decision Engine (calibrated on 62+ closing matches & 4,550+ archive matches):
    1. Score Gap (Clutch vs Blowout)
    2. 2nd Favorite Direction (Market Skew: +1 Upper vs -1 Lower)
    3. Extreme Low-Odd Trap (< 1.85)
    4. Mod-10 Last Digit Barem Center Symmetry (cur_tot % 10: 1-4 Göbek vs 8-9-0 Alt Sınır Tuzağı)
    5. Twin Odds Symmetry (fav2_odd - fav1_odd <= 0.25) & Top-2 Concentration (>= 78%)
    6. Bell-Curve Wing Symmetry (|odd(+1) - odd(-1)| <= 0.75)
    7. Late 5-Min Pendulum Symmetry (30m -> 35m barem jump/drop)
    8. Sequential 2-Deviation Snapback Symmetry (Last 2 matches != 0 -> 3rd match 0 Merkez +33.6% ROI)
    """
    barem_names = [b[0] for b in BAREM_RANGES]
    barem_bounds = {b[0]: (b[1], b[2]) for b in BAREM_RANGES}
    base_b = fav1_barem if fav1_barem in barem_names else "201-210"
    b_0 = base_b
    b_m1 = _shift_barem_name(base_b, -1)
    b_m2 = _shift_barem_name(base_b, -2)
    b_p1 = _shift_barem_name(base_b, 1)

    idx_fav = barem_names.index(base_b)
    idx_sec = barem_names.index(fav2_barem) if fav2_barem in barem_names else max(0, idx_fav - 1)
    sec_dir = idx_sec - idx_fav
    sec_b_clean = fav2_barem if (fav2_barem in barem_names and fav2_barem != base_b) else b_m1

    if sec_dir >= 1:
        sec_dir_badge = f"📈 2. Favori Bir Üst Baremde ({sec_b_clean})"
        sec_dir_tip = (
            f"Şu an 1. Favori {base_b} iken, en düşük oranlı 2. Favori bir üst barem olan {sec_b_clean} aralığında duruyor. "
            f"Geçmiş 62 maçta 2. favori üst baremdeyken maçların alt bareme ({b_m1}) düşme ihtimali çok düşüktür (%7); "
            f"oran 2.10 üstüyse direkt {base_b} oynamak %61 tutturma ve +%39.9 net kâr sağlamıştır."
        )
    else:
        sec_dir_badge = f"📉 2. Favori Bir Alt Baremde ({sec_b_clean})"
        sec_dir_tip = (
            f"Şu an 1. Favori {base_b} iken, en düşük oranlı 2. Favori bir alt barem olan {sec_b_clean} aralığında duruyor. "
            f"Bu durumda maçların neredeyse yarısı (%48.5) alt bareme ({b_m1}) kaymış, üst bareme ({b_p1}) ise hiç (%0) çıkmamıştır."
        )

    # Calculate in-game drift (Q1 -> Now) and Late 5m Pendulum (30m -> 35m) for this game_id
    open_b = base_b
    drift = 0
    late_jump = 0
    b_30m = base_b
    if df_log is not None and not df_log.empty and game_id and "game_id" in df_log.columns:
        g_sub = df_log[df_log["game_id"].astype(str).str.strip() == str(game_id).strip()].copy()
        if not g_sub.empty:
            first_b = str(g_sub.iloc[0].get("highest_prob_range", "")).strip()
            if first_b in barem_names:
                open_b = first_b
                drift = barem_names.index(base_b) - barem_names.index(open_b)
            g_sub["_m"] = (
                g_sub["check_time"].astype(str).str.extract(r"Minute\s+(\d+)", expand=False).astype(float).fillna(0)
            )
            c30 = g_sub[(g_sub["_m"] >= 27) & (g_sub["_m"] <= 31)]
            if not c30.empty:
                b30_cand = str(c30.iloc[-1].get("highest_prob_range", "")).strip()
                if b30_cand in barem_names:
                    b_30m = b30_cand
                    late_jump = barem_names.index(base_b) - barem_names.index(b_30m)

    if drift > 0:
        drift_badge = f"📈 Maç Başında {open_b} Bekleniyordu ➔ Şimdi {base_b} Oldu (Maç Hızlandı)"
    elif drift < 0:
        drift_badge = f"📉 Maç Başında {open_b} Bekleniyordu ➔ Şimdi {base_b} Oldu (Maç Yavaşladı)"
    else:
        drift_badge = f"⚖️ Maç Başından Beri Hep {base_b} Favori"
    drift_tip = (
        f"Maç ilk başladığında favori barem {open_b} idi, şu an ise {base_b} oldu. "
        f"Maç içinde beklenti yükseldiyse genelde yeni ulaştığı {base_b} bareminde veya başladığı {open_b} bareminde biter. "
        f"Maç içinde beklenti zaten düştüyse daha da alt bareme oynama tuzağına düşmeyin; maçların %82'si {base_b} veya üstünde kalır."
    )

    # Late 5-Min Change (30m -> 35m)
    if late_jump <= -1:
        pendulum_badge = f"🔄 Son 5 Dk'da Barem Düştü ({b_30m} ➔ {base_b} — Daha Fazla Düşmez!)"
        pendulum_tip = (
            f"30. dakikada favori {b_30m} iken 35. dakikada {base_b} baremine düştü. "
            f"62 maçlık veride son 5 dakikada baremi düşen maçların HİÇBİRİ (%0) bir alt barem olan {b_m1} aralığına düşmemiştir! "
            f"%85.8 ihtimalle ya şu anki {base_b} bareminde kalır ya da eski {b_30m} baremine geri döner."
        )
    elif late_jump >= 1:
        pendulum_badge = f"🔄 Son 5 Dk'da Barem Yükseldi ({b_30m} ➔ {base_b} — Daha Fazla Çıkmaz!)"
        pendulum_tip = (
            f"30. dakikada favori {b_30m} iken 35. dakikada {base_b} baremine yükseldi. "
            f"Son 5 dakikada zaten yükseldiği için buradan bir üst bareme ({b_p1}) gitme ihtimali sadece %8.3'tür; maçların %75'i {base_b} veya {b_m1} aralığında kalır."
        )
    else:
        pendulum_badge = f"⚖️ Son 5 Dk'dır Barem Sabit ({base_b})"
        pendulum_tip = (
            f"30. dakikadan 35. dakikaya favori barem ({base_b}) hiç değişmedi. "
            f"Böyle dengeli giden maçlarda direkt {base_b} baremi %53.5 oranında tutmaktadır."
        )

    # Last Digit of Score (Simple explanation)
    mod10 = int(cur_tot) % 10 if cur_tot > 0 else 2
    if cur_tot > 0 and float(elapsed_mins or 35.0) >= 30.0:
        if mod10 in (1, 2, 3, 4):
            mod10_badge = f"🎯 Skorun Son Rakamı Avantajlı ({cur_tot} ➔ Tam {base_b} Ortasına Düşüyor)"
            mod10_tip = (
                f"Şu anki toplam skor ({cur_tot}) '{mod10}' rakamıyla bitiyor. Kalan sürede ortalama ~53 sayı atıldığında maç sonu skoru "
                f"tam {base_b} bareminin ortasına (en güvenli yerine) denk geliyor! "
                f"Skorun sonu 1, 2, 3 veya 4 ile biten maçlarda direkt {base_b} baremi %57.6 oranında tutmuştur."
            )
        elif mod10 in (8, 9, 0):
            mod10_badge = f"⚠️ Skorun Son Rakamı Sınırda ({cur_tot} ➔ {b_m1} Alt Barem Riski Var)"
            mod10_tip = (
                f"Şu anki toplam skor ({cur_tot}) '{mod10}' rakamıyla bitiyor. Kalan sürede ortalama ~53 sayı eklendiğinde maç sonu skoru "
                f"{base_b} bareminin en alt sınırına (bıçak sırtına) denk geliyor; sadece 2 sayı eksik atılsa bir alt barem olan {b_m1} olur! "
                f"Bu yüzden skor sonu 8, 9 veya 0 olan maçlarda alt barem ({b_m1}) %43.8 oranında gelmiştir."
            )
        else:
            mod10_badge = f"📐 Skorun Son Rakamı Normal ({cur_tot} ➔ {base_b} Üst Kısmına Yakın)"
            mod10_tip = (
                f"Şu anki toplam skor ({cur_tot}) '{mod10}' rakamıyla bitiyor. Ortalama ~53 sayı eklendiğinde {base_b} bareminin orta-üst kısmına denk gelir "
                f"({base_b} tutma oranı %53.8)."
            )
    else:
        mod10_badge = f"🔢 Skorun Son Rakamı: ..{mod10} (30. Dakikadan Sonra Devreye Girer)"
        mod10_tip = (
            f"Maç 30-35. dakikaya geldiğinde toplam skorun son rakamına bakarak maç sonunun {base_b} bareminin tam ortasına mı "
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
        twin_badge = f"👯 İlk 2 Baremin Oranı Neredeyse Eşit ({base_b} ve {sec_b_clean} — İkisinden Biri %89 Gelir)"
        twin_tip = (
            f"1. Favori {base_b} ({odd_val:.2f}) ile 2. Favori {sec_b_clean} ({odd2_val:.2f}) arasındaki oran farkı çok az ({odd_gap_12:.2f}). "
            f"Böyle maçların %44.4'ü {base_b}, %44.4'ü {sec_b_clean} bitmiştir; yani bu iki baremden birinin tutma ihtimali %88.9 (9 maçta 8)'dir!"
        )
    elif top2_conc >= 78.0:
        twin_badge = f"🔒 İlk 2 Barem Çok Güçlü ({base_b} veya {sec_b_clean} %90 Gelir)"
        twin_tip = (
            f"1xBet ihtimallerinin %{top2_conc:.0f}'i sadece {base_b} ve {sec_b_clean} baremlerinde toplanmış durumda. "
            f"Böyle maçlarda {base_b} veya {sec_b_clean} ikilisinden birinin tutma oranı %90.5 (21 maçta 19) olmuştur!"
        )
    else:
        twin_badge = f"⚖️ 1. ve 2. Favori Oran Farkı: {odd_gap_12:.2f} ({base_b} vs {sec_b_clean})"
        twin_tip = (
            f"1. Favori ({base_b}: {odd_val:.2f}) ile 2. Favori ({sec_b_clean}: {odd2_val:.2f}) arasındaki oran farkı {odd_gap_12:.2f}."
        )

    if wing_diff <= 0.75:
        wing_badge = f"⚖️ Alt ({b_m1}) ve Üst ({b_p1}) Oranları Dengeli ({base_b} Çok Güçlü)"
        wing_tip = (
            f"Favorinin bir altındaki {b_m1} ({odd_m1_live:.2f}) ile bir üstündeki {b_p1} ({odd_p1_live:.2f}) oranları birbirine çok yakın. "
            f"Bu durum beklentinin tam {base_b} ortasında toplandığını gösterir; {base_b} %53.8 tutarken alt barem ({b_m1}) sadece %7.7 gelmiştir."
        )
    elif odd_m1_live < 90 and odd_p1_live < 90:
        wing_badge = f"📐 Komşu Oranlar: Alt {b_m1} ({odd_m1_live:.2f}) | Üst {b_p1} ({odd_p1_live:.2f})"
        wing_tip = (
            f"Favori {base_b} bareminin alt komşusu {b_m1} oranı {odd_m1_live:.2f}, üst komşusu {b_p1} oranı {odd_p1_live:.2f}. "
            f"Hangisinin oranı daha düşükse maçın o tarafa kayma ihtimali daha yüksektir."
        )
    else:
        wing_badge = "📐 Komşu Baremlerden Biri Kapalı"
        wing_tip = "Alt veya üst komşu baremlerden birine şu an oran açılmamış."

    # Sequential Match History in Plain Turkish
    prev_shift = recent_chain[-1]["raw_shift"] if recent_chain else 0
    prev2_shift = recent_chain[-2]["raw_shift"] if len(recent_chain) >= 2 else 0
    if prev_shift != 0 and prev2_shift != 0:
        seq_sym_badge = f"🔄 Son 2 Maçta Favori Gelmedi ➔ Bu Maç Favori ({base_b}) Gelme İhtimali Yüksek (%57)"
        seq_sym_tip = (
            f"Bundan önceki 2 maçta üst üste 35. dakika favorisi tutmadı. "
            f"Geçmiş 62 maçta üst üste 2 kez sürpriz olduktan sonra 3. maçta direkt Favori Baremin ({base_b}) gelme oranı %57.1 olmuştur!"
        )
    elif prev_shift == 0 and prev2_shift == 0:
        seq_sym_badge = "⚖️ Son 2 Maçta Üst Üste Favori Barem Geldi"
        seq_sym_tip = f"Son 2 maçta üst üste 35. dakika favorisi kazandı; bu maçta %50 Favori ({base_b}), %25 Bir Alt Barem ({b_m1}) dengesi görülür."
    else:
        prev_desc = "Favori Barem Geldi" if prev_shift == 0 else (f"{abs(prev_shift)} Alt Barem Geldi" if prev_shift < 0 else f"{prev_shift} Üst Barem Geldi")
        seq_sym_badge = f"🌊 Bir Önceki Maç Sonucu: {prev_desc}"
        seq_sym_tip = "Bundan hemen önce biten maçın 35. dakika favorisine göre nasıl sonuçlandığını gösterir."

    gap = abs(int(score_gap or 0))
    if gap <= 5:
        exp_rem_35 = 59.0
        margin_badge = f"⚠️ Skor Çok Yakın: {gap} Sayı Fark (Faul & Uzatma Riski Var)"
        margin_tip = (
            f"İki takım arasındaki fark sadece {gap} sayı (0-5 arası başa baş maç). Son dakikalarda taktik fauller ve uzatma "
            f"ihtimali yüzünden 35. dakikadan sonra ortalama 59 sayı atılır (normalden 6 sayı fazla) ve "
            f"maçların üçte biri (%33) üst bareme ({b_p1}) kaçar. Bu yüzden başa baş maçlara BAHİS ALMAMAK (Pas Geçmek) en doğrusudur."
        )
    elif gap >= 18:
        exp_rem_35 = 47.4
        margin_badge = f"🧊 Fark Çok Açıldı: {gap} Sayı Fark (Maç Yavaşlar / Rölanti)"
        margin_tip = (
            f"Skor farkı {gap} sayıya (18+ sayı) çıktığında takımlar son çeyrekte tempoyu düşürür. "
            f"35. dakikadan sonra ortalama sadece 47.4 sayı atılır (yakın maçtan 11.6 sayı daha az!). "
            f"Üst bareme ({b_p1}) kaçma ihtimali sadece %5'tir; direkt {base_b} baremi %60 oranında tutar."
        )
    elif gap >= 10:
        exp_rem_35 = 53.5
        margin_badge = f"🛡️ Fark Kontrollü: {gap} Sayı Fark (Uzatma Riski Düşük)"
        margin_tip = f"Skor farkı {gap} sayı (10-17 arası); uzatma riski düşüktür. İki barem kapatılacaksa {base_b} ve bir alt barem {b_m1} birlikte çok güçlüdür."
    else:
        exp_rem_35 = 53.0
        margin_badge = f"⚖️ Normal Skor Farkı: {gap} Sayı Fark"
        margin_tip = f"Skor farkı {gap} sayı (6-9 arası); kalan sürede normal tempo (~53 sayı) beklenir."

    # Barem interior edge position
    fav_lo, fav_hi = barem_bounds.get(base_b, (201, 210))
    fav_mid = (fav_lo + min(fav_hi, 240)) / 2.0
    rem_mins_est = max(0.0, 48.0 - float(elapsed_mins or 35.0))
    scaled_rem = exp_rem_35 * (rem_mins_est / 13.0) if rem_mins_est > 0 else 0.0
    proj_tot_edge = float(cur_tot) + scaled_rem if cur_tot > 0 else fav_mid
    edge_offset = proj_tot_edge - fav_mid

    if cur_tot > 0 and 28.0 <= float(elapsed_mins or 35.0) <= 36.5:
        if edge_offset < -3.0:
            edge_badge = f"📐 Beklenen Bitiş: ~{proj_tot_edge:.0f} Sayı ({base_b} Alt Sınırında ➔ {b_m1} Riski Var)"
            edge_tip = (
                f"Şu anki skor ({cur_tot}) + kalan sürede beklenen sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı yapıyor. "
                f"Bu sayı {base_b} bareminin en alt sınırında kaldığı için maçların %55.6'sı bir alt barem olan {b_m1} aralığına düşmektedir."
            )
        elif edge_offset > 3.0:
            edge_badge = f"📐 Beklenen Bitiş: ~{proj_tot_edge:.0f} Sayı ({base_b} Üst Kısmında)"
            edge_tip = (
                f"Şu anki skor ({cur_tot}) + kalan sürede beklenen sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı yapıyor. "
                f"{base_b} bareminin orta-üst kısmında güçlü duruyor (%56 ihtimalle {base_b} tutar)."
            )
        else:
            edge_badge = f"🎯 Beklenen Bitiş: ~{proj_tot_edge:.0f} Sayı (Tam {base_b} Ortasında)"
            edge_tip = (
                f"Şu anki skor ({cur_tot}) + kalan sürede beklenen sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı ile "
                f"tam {base_b} bareminin ortasına denk geliyor (%55.6 ihtimalle {base_b} tutar)."
            )
    else:
        edge_badge = f"⏱️ 35.Dk'dan Sonra Ortalama 53 Sayı Atılır (Bu Maçta Beklenen: ~{exp_rem_35:.0f} Sayı)"
        edge_tip = (
            "Geçmiş 62 maçta 35. dakikadan maç bitimine kadar ortalama 53.3 sayı atılmıştır "
            "(yakın maçlarda 59 sayı, fark açık maçlarda 47.4 sayı). Maç 30-35. dakikaya geldiğinde burada tahmini maç sonu toplamı görünür."
        )

    prev_shift = recent_chain[-1]["raw_shift"] if recent_chain else 0
    odd_val = float(fav1_odd or 0.0)
    reasons = []

    # Two-barem cover recommendation
    if gap >= 10:
        dutch_pick = f"🛡️ {b_0} ile {b_m1} İkisini Birden Kapat"
        dutch_short = f"{b_0} ve {b_m1}"
    else:
        dutch_pick = f"🛡️ {b_0} ile {sec_b_clean} İkisini Birden Kapat"
        dutch_short = f"{b_0} ve {sec_b_clean}"

    # Plain-Turkish Decision Engine
    if gap <= 5 and float(elapsed_mins or 35.0) >= 30.0:
        zone_badge = f"⛔ BAHİS ALMA / PAS GEÇ (Skor Çok Yakın: {gap} Sayı Fark)"
        zone_type = "PASS_CLUTCH"
        stake_badge = "0 TL (Pas Geç — Uzatma ve Faul Riski)"
        primary_pick = f"⛔ Bahis Alma (İlla Oynanacaksa: {b_0} veya {b_p1})"
        backup_pick = dutch_pick
        conf_pct = 38.0
        reasons.append(
            f"Maç başa baş gidiyor (fark sadece {gap} sayı). Son dakikalarda taktik fauller ve uzatma yüzünden normalden çok fazla sayı (ort. 59 sayı) atılıyor ve maçların %33'ü üst barem olan {b_p1} aralığına kaçıyor. Bu yüzden bu maça bahis almamak en güvenli karardır"
        )
    elif odd_val > 1.0 and odd_val < 1.85:
        zone_badge = f"⚠️ 1. FAVORİ ORANI ÇOK DÜŞÜK TUZAĞI ({b_0} Oranı: {odd_val:.2f})"
        zone_type = "TRAP_EXTREME"
        stake_badge = f"1000 TL ({sec_b_clean} Baremine Oyna)"
        primary_pick = f"🥈 {sec_b_clean} Baremini Oyna (2. Favori — Oran: {fav2_odd:.2f})"
        backup_pick = f"🛡️ {b_0} ile {sec_b_clean} İkisini Birden Kapat"
        conf_pct = 74.0
        reasons.append(
            f"Şu an 1. Favori olan {b_0} bareminin oranı 1.85'in altında ({odd_val:.2f}). Geçmiş maçlarda bu kadar düşük oranlı favoriler %37 zarar ettirirken, onun yerine 2. Favori olan {sec_b_clean} baremini oynamak +%41.8 net kâr sağladı"
        )
    elif odd_val >= 2.10 and sec_dir >= 1:
        zone_badge = f"🔥 ÇOK GÜÇLÜ SİNYAL: {b_0} BAREMİ ÖNDE (Oran: {odd_val:.2f})"
        zone_type = "SUPER_SIGNAL"
        stake_badge = f"2000 TL (Güçlü Sinyal — {b_0} Baremine)"
        primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
        backup_pick = dutch_pick
        conf_pct = 88.0
        reasons.append(
            f"Şu an en güçlü aday {b_0} baremi (Oran: {odd_val:.2f}) ve 2. favori üst barem olan {sec_b_clean} aralığında duruyor. Geçmişte bu durumdaki maçların %61'i direkt {b_0} aralığında bitti ve hiçbiri alt bareme ({b_m1}) düşmedi. İki barem kapatmak isterseniz {dutch_short} ikilisini seçebilirsiniz"
        )
    elif gap >= 18 and odd_val >= 1.85:
        zone_badge = f"🔥 FARK AÇIK / MAÇ YAVAŞLADI SİNYALİ ({b_0} — Fark: {gap} Sayı)"
        zone_type = "SUPER_SIGNAL"
        stake_badge = f"2000 TL (Güçlü Sinyal — {b_0} Baremine)"
        primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
        backup_pick = f"🛡️ {b_0} ile {b_m1} İkisini Birden Kapat (%95 Tutar)"
        conf_pct = 86.0
        reasons.append(
            f"Skor farkı {gap} sayıya çıktığı için takımlar tempoyu düşürdü (kalan sürede ort. 47 sayı atılıyor). Bu durumdaki maçların %60'ı direkt {b_0} aralığında, %95'i ise {b_0} veya {b_m1} aralığında bitiyor"
        )
    elif odd_val > 1.0 and odd_val < 2.10 and sec_dir <= -1 and late_jump >= 0:
        zone_badge = f"⚠️ ALT BAREME KAYMA EĞİLİMİ ({b_m1} Daha Avantajlı)"
        zone_type = "TRAP_LOW"
        stake_badge = f"1000 TL ({b_m1} Baremine Oyna)"
        primary_pick = f"📉 Bir Alt Barem Olan {b_m1} Aralığını Oyna ({b_0} Yerine)"
        backup_pick = f"🛡️ {b_0} ile {b_m1} İkisini Birden Kapat"
        conf_pct = 76.0
        reasons.append(
            f"1. Favori olan {b_0} oranı 2.10'un altında ({odd_val:.2f}) ve 2. Favori alt baremde ({b_m1}) duruyor. Böyle maçlarda {b_0} zarar ettirirken, bir alt barem olan {b_m1} aralığını oynamak düzenli kâr (+%16.6) sağladı"
        )
    elif cur_tot > 0 and float(elapsed_mins or 35.0) >= 30.0 and (mod10 in (8, 9, 0)) and sec_dir <= -1 and odd_val < 2.25 and late_jump >= 0:
        zone_badge = f"📉 SKOR SINIRDA: {b_m1} ALT BAREME YAKIN (Skor: {cur_tot})"
        zone_type = "TRAP_LOW"
        stake_badge = f"1000 TL ({b_m1} Baremine Oyna)"
        primary_pick = f"📉 Bir Alt Barem Olan {b_m1} Aralığını Oyna"
        backup_pick = f"🛡️ {b_0} ile {b_m1} İkisini Birden Kapat"
        conf_pct = 75.0
        reasons.append(
            f"Şu anki toplam skor ({cur_tot}) son rakamı nedeniyle kalan ~53 sayı eklendiğinde {b_0} bareminin en alt sınırında kalıyor ve piyasa da {b_m1} alt baremini işaret ediyor (%44 ihtimalle {b_m1} aralığına kayar)"
        )
    elif 1.85 <= odd_val <= 2.79:
        zone_badge = f"💎 İDEAL ORAN: {b_0} BAREMİ ÖNDE (Oran: {odd_val:.2f})"
        zone_type = "GOLDEN"
        stake_badge = f"1000 TL ({b_0} Baremine Oyna)"
        primary_pick = f"🎯 Direkt {b_0} Baremini Oyna (Oran: {odd_val:.2f})"
        backup_pick = dutch_pick
        conf_pct = 82.0 if mod10 in (1, 2, 3, 4) else 79.0
        reasons.append(
            f"Şu an en güçlü aday {b_0} baremi (Oran: {odd_val:.2f}). Benzer maçların %55'i direkt {b_0} aralığında bitti. Garantiye almak için iki barem kapatmak isterseniz {dutch_short} ikilisini seçin (10 maçın 8'inde bu ikisinden biri tuttu)"
        )
    else:
        zone_badge = f"⚖️ YÜKSEK ORAN / KARARSIZ MAÇ ({odd_val:.2f})" if odd_val < 90 else "⏳ Oran Bekleniyor"
        zone_type = "WIDE"
        stake_badge = f"1000 TL ({dutch_short} İkisini Kapat)"
        primary_pick = f"🎯 {b_0} Baremi"
        backup_pick = dutch_pick
        conf_pct = 68.0
        reasons.append(f"Oranlar yüksek ({odd_val:.2f}) ve maç kararsız; tek barem yerine {dutch_short} ikilisini birden kapatmak çok daha güvenlidir")

    if drift > 0:
        reasons.append(f"Maç başında {open_b} bekleniyordu, maç hızlı gittiği için beklenti {base_b} baremine yükseldi (genelde {base_b} veya {open_b} aralığında biter)")
    elif drift < 0:
        reasons.append(f"Maç başında {open_b} bekleniyordu, maç yavaşladığı için beklenti {base_b} baremine düştü")

    if prev_shift != 0 and prev2_shift != 0:
        reasons.append(f"Son 2 maçta üst üste sürpriz olduğu için bu maçta direkt favori baremin ({b_0}) gelme ihtimali yüksektir (%57)")
    elif prev_shift >= 1:
        reasons.append(f"Bir önceki maç üst baremde bittiği için bu maçta %69 ihtimalle {b_0} veya bir alt barem olan {b_m1} gelir")

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
        "drift": drift,
        "open_barem": open_b,
        "drift_badge": drift_badge,
        "drift_tip": drift_tip,
        "primary_pick": primary_pick,
        "backup_pick": backup_pick,
        "confidence_pct": round(conf_pct, 1),
        "reasoning": " • ".join(reasons),
    }


def simulate_35m_strategies(df_log: pd.DataFrame) -> dict:
    """
    Simulate 35th-minute betting strategies (1000 TL flat stake + tiered 0/1000/2000 TL bankroll)
    across all COMPLETED matches in symmetry_analysis_log.csv, and build real analytical tables:
    - Strategies comparison (v3 Selective, Tiered Bankroll, v3 All-Match, v2 Legacy, Pure 0, Smart Dutching v2, Fixed Dutching)
    - Odds brackets (<1.85, 1.85-2.09, 2.10-2.34, >=2.35)
    - 2nd Favorite direction brackets (-1 Lower vs +1 Upper)
    - Score Margin brackets (0-5 Clutch/OT, 6-11 Normal, 12-17 High, 18+ Blowout)
    - Barem interior edge brackets (Lower edge, Center, Upper edge)
    - Minute-by-minute timing window ROI (18m, 24m, 30m, 32-33m Trap, 34-35m Closing)
    - Drift (Q1 -> 35m) & Sequential Transitions
    """
    barem_names = [b[0] for b in BAREM_RANGES]
    barem_bounds = {b[0]: (b[1], b[2]) for b in BAREM_RANGES}
    b_cols = {
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
    comp = comp[comp["_gid_num"] >= 770265000].sort_values(by=["_gid_num", "_min"], ascending=[True, True], kind="stable")

    def _parse_sc(s_val):
        try:
            pts = str(s_val).strip().split("-")
            return int(pts[0]), int(pts[1])
        except Exception:
            return None, None

    stake = 1000.0
    ledger = []

    s_v4_sel_ret, s_v4_sel_cost, s_v4_sel_wins, s_v4_sel_cnt = 0.0, 0.0, 0, 0
    s_v3_sel_ret, s_v3_sel_cost, s_v3_sel_wins, s_v3_sel_cnt = 0.0, 0.0, 0, 0
    s_tiered_ret, s_tiered_cost, s_tiered_wins, s_tiered_cnt = 0.0, 0.0, 0, 0
    s_v3_all_ret, s_v3_all_wins = 0.0, 0
    s_smart_ret, s_smart_wins = 0.0, 0
    s_0_ret, s_0_wins = 0.0, 0
    s_dutch_v2_ret, s_dutch_v2_wins = 0.0, 0
    s_dutch_ret, s_dutch_wins = 0.0, 0

    raw_records = []
    for gid, grp in comp.groupby("game_id", sort=False):
        first_r = grp.iloc[0]
        cands = grp[grp["_min"] >= 32]
        if cands.empty:
            continue
        last_r = cands.iloc[-1]
        cands30 = grp[(grp["_min"] >= 27) & (grp["_min"] <= 31)]
        r30 = cands30.iloc[-1] if not cands30.empty else first_r
        b30 = str(r30.get("highest_prob_range", "")).strip()

        act_b = str(last_r.get("actual_range", "")).strip()
        open_b = str(first_r.get("highest_prob_range", "")).strip()
        if act_b not in barem_names:
            continue

        odds_map = {}
        for b_n in barem_names:
            v = float(last_r.get(b_cols[b_n], 0.0) or 0.0)
            if 1.0 < v < 90.0:
                odds_map[b_n] = v
        if not odds_map:
            continue

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
        min_val = int(last_r.get("_min", 35))
        mod10 = cur_tot35 % 10

        fav_lo, fav_hi = barem_bounds[fav_b]
        fav_mid = (fav_lo + min(fav_hi, 240)) / 2.0
        proj_35 = cur_tot35 + 53.3 if cur_tot35 > 0 else fav_mid
        edge_offset = proj_35 - fav_mid

        # 1) Legacy v2 rule (<2.10 -> -1 Alt, else 0 Merkez)
        if odd_0 < 2.10 or drift <= -3:
            v2_b = m1_b
            v2_odd = odd_m1
        else:
            v2_b = fav_b
            v2_odd = odd_0
        hit_v2 = act_b == v2_b
        ret_v2 = round(stake * v2_odd, 1) if hit_v2 else 0.0
        s_smart_ret += ret_v2
        s_smart_wins += int(hit_v2)

        # 2) v3 rule (<1.85 -> 2nd Fav | <2.10 & sec_dir==-1 -> -1 Alt | else 0 Merkez)
        if odd_0 < 1.85:
            v3_b = sec_b
            v3_odd = sec_odd
            v3_rule_lbl = "⚠️ Oran <1.85 Tuzağı ➔ 2. Favori"
        elif odd_0 < 2.10 and sec_dir <= -1:
            v3_b = m1_b
            v3_odd = odd_m1
            v3_rule_lbl = "📉 Oran <2.10 & 2.Fav Altta ➔ 1 Alt Barem"
        else:
            v3_b = fav_b
            v3_odd = odd_0
            if sec_dir >= 1 and odd_0 >= 2.10:
                v3_rule_lbl = "🔥 Güçlü Sinyal (2.Fav Üstte) ➔ 1. Favori"
            elif margin35 >= 18:
                v3_rule_lbl = "🧊 18+ Fark (Maç Yavaşladı) ➔ 1. Favori"
            else:
                v3_rule_lbl = "💎 İdeal Oran ➔ 1. Favori"

        hit_v3_all = act_b == v3_b
        ret_v3_all = round(stake * v3_odd, 1) if hit_v3_all else 0.0
        s_v3_all_ret += ret_v3_all
        s_v3_all_wins += int(hit_v3_all)

        # 2b) v4 Deep Symmetry rule (adds Mod-10 Alt Sınır [8,9,0] + Late-Jump Pendulum guard)
        if odd_0 < 1.85:
            v4_b, v4_odd = sec_b, sec_odd
        elif odd_0 < 2.10 and sec_dir <= -1 and late_jump >= 0:
            v4_b, v4_odd = m1_b, odd_m1
        elif (mod10 in (8, 9, 0)) and sec_dir <= -1 and odd_0 < 2.25 and late_jump >= 0:
            v4_b, v4_odd = m1_b, odd_m1
            v3_rule_lbl = "📉 Skor Sınırda (..8/9/0) ➔ 1 Alt Barem"
        else:
            v4_b, v4_odd = fav_b, odd_0
        hit_v4 = act_b == v4_b
        ret_v4 = round(stake * v4_odd, 1) if hit_v4 else 0.0

        # 3) Selective v3 & v4 (Pass when margin35 <= 5 Clutch/OT risk)
        is_clutch_pass = margin35 <= 5
        if not is_clutch_pass:
            s_v3_sel_cnt += 1
            s_v3_sel_cost += stake
            if hit_v3_all:
                s_v3_sel_wins += 1
                s_v3_sel_ret += ret_v3_all

            s_v4_sel_cnt += 1
            s_v4_sel_cost += stake
            if hit_v4:
                s_v4_sel_wins += 1
                s_v4_sel_ret += ret_v4

        # 4) Tiered Bankroll (0 TL on Clutch <=5 | 2000 TL on Super Signal | 1000 TL on Normal)
        is_super = (not is_clutch_pass) and ((sec_dir >= 1 and odd_0 >= 2.10) or (margin35 >= 18 and odd_0 >= 1.85))
        t_stake = 0.0 if is_clutch_pass else (2000.0 if is_super else 1000.0)
        if t_stake > 0:
            s_tiered_cnt += 1
            s_tiered_cost += t_stake
            t_ret = round(t_stake * v3_odd, 1) if hit_v3_all else 0.0
            if hit_v3_all:
                s_tiered_wins += 1
                s_tiered_ret += t_ret
        else:
            t_ret = 0.0

        # 5) Pure 0 (Merkez)
        hit_0 = act_b == fav_b
        ret_0 = round(stake * odd_0, 1) if hit_0 else 0.0
        s_0_ret += ret_0
        s_0_wins += int(hit_0)

        # 6) Fixed Dutching (0 & -1)
        hit_dutch = act_b in (fav_b, m1_b)
        dutch_odd = round(1.0 / ((1.0 / odd_0) + (1.0 / odd_m1)), 2) if m1_b != fav_b else odd_0
        ret_dutch = round(stake * dutch_odd, 1) if hit_dutch else 0.0
        s_dutch_ret += ret_dutch
        s_dutch_wins += int(hit_dutch)

        # 7) Smart Dutching v2 (margin >= 10 -> 0 & -1; else -> 0 & 2ndFav)
        d2_partner = m1_b if margin35 >= 10 else sec_b
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

        sec_arrow = f"📈 Üstte ({sec_b})" if sec_dir >= 1 else f"📉 Altta ({sec_b})"
        if is_clutch_pass:
            sel_status = f"⛔ Pas Geçildi (Fark {margin35})"
            tiered_status = "⛔ 0 TL (Pas Geçildi)"
        else:
            sel_status = f"✅ +{ret_v4 - stake:.0f} TL" if hit_v4 else "❌ -1000 TL"
            tiered_status = (
                f"✅ +{t_ret - t_stake:.0f} TL ({t_stake:.0f} Yatırım)"
                if hit_v3_all
                else f"❌ -{t_stake:.0f} TL"
            )

        ledger.append(
            {
                "Maç": str(last_r.get("game_name", "")).replace(" (cyber)", ""),
                "Kapanış Dk & Fark": f"{last_r.get('check_time', '')} ({margin35} Sayı Fark | Sonu ..{mod10})",
                "35.Dk 1. Favori": f"{fav_b} ({odd_0:.2f})",
                "2. Favori Nerede?": f"{sec_arrow} ({sec_odd:.2f})",
                "🤖 Sistemin Önerdiği Barem": f"{v4_b} ({v4_odd:.2f}) [{v3_rule_lbl}]",
                "Maçın Bittiği Barem": f"{act_b} ({last_r.get('actual_score', '')})",
                "En İyi Seçici Strateji (1000 TL / Pas)": sel_status,
                "Kademeli Kasa (0/1000/2000)": tiered_status,
                "İki Baremi Birden Kapatma": f"✅ +{ret_dutch_v2 - stake:.0f} TL ({dutch_v2_odd:.2f})" if hit_dutch_v2 else "❌ -1000 TL",
            }
        )

    tot_m = len(raw_records)
    if tot_m == 0:
        return empty_res

    tot_cost = tot_m * stake
    strategies = [
        {
            "name": "👑 En Gelişmiş Seçici Strateji (0-5 Sayı Farklı Yakın Maçları Pas Geç + Skor & Oran Kuralı)",
            "wins": s_v4_sel_wins,
            "total": s_v4_sel_cnt,
            "win_rate": round(s_v4_sel_wins / max(1, s_v4_sel_cnt) * 100.0, 1),
            "cost": round(s_v4_sel_cost, 0),
            "return": round(s_v4_sel_ret, 0),
            "net_profit": round(s_v4_sel_ret - s_v4_sel_cost, 0),
            "roi_pct": round((s_v4_sel_ret / max(1.0, s_v4_sel_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🔥 Seçici Akıllı Strateji (0-5 Sayı Farklı Yakın Maçları Pas Geç + Oran Kuralı)",
            "wins": s_v3_sel_wins,
            "total": s_v3_sel_cnt,
            "win_rate": round(s_v3_sel_wins / max(1, s_v3_sel_cnt) * 100.0, 1),
            "cost": round(s_v3_sel_cost, 0),
            "return": round(s_v3_sel_ret, 0),
            "net_profit": round(s_v3_sel_ret - s_v3_sel_cost, 0),
            "roi_pct": round((s_v3_sel_ret / max(1.0, s_v3_sel_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "💰 Kademeli Kasa (Yakın Maç: 0 TL Pas | Normal Maç: 1000 TL | Güçlü Sinyal: 2000 TL)",
            "wins": s_tiered_wins,
            "total": s_tiered_cnt,
            "win_rate": round(s_tiered_wins / max(1, s_tiered_cnt) * 100.0, 1),
            "cost": round(s_tiered_cost, 0),
            "return": round(s_tiered_ret, 0),
            "net_profit": round(s_tiered_ret - s_tiered_cost, 0),
            "roi_pct": round((s_tiered_ret / max(1.0, s_tiered_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🤖 Tüm Maçlara Akıllı Tek Barem (Oran <1.85➔2.Fav | Oran <2.10&Alt➔1 Alt | Diğer➔1.Fav)",
            "wins": s_v3_all_wins,
            "total": tot_m,
            "win_rate": round(s_v3_all_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_v3_all_ret, 0),
            "net_profit": round(s_v3_all_ret - tot_cost, 0),
            "roi_pct": round((s_v3_all_ret / tot_cost - 1.0) * 100.0, 1),
        },
        {
            "name": "🤖 Eski Basit Kural (Oran <2.10 ise 1 Alt Barem / Oran >=2.10 ise 1. Favori Barem)",
            "wins": s_smart_wins,
            "total": tot_m,
            "win_rate": round(s_smart_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_smart_ret, 0),
            "net_profit": round(s_smart_ret - tot_cost, 0),
            "roi_pct": round((s_smart_ret / tot_cost - 1.0) * 100.0, 1),
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
        {
            "name": "🛡️ Akıllı İki Barem Kapatma (Fark 10+ ise Favori & 1 Alt / Diğerlerinde 1. & 2. Favori)",
            "wins": s_dutch_v2_wins,
            "total": tot_m,
            "win_rate": round(s_dutch_v2_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_dutch_v2_ret, 0),
            "net_profit": round(s_dutch_v2_ret - tot_cost, 0),
            "roi_pct": round((s_dutch_v2_ret / tot_cost - 1.0) * 100.0, 1),
        },
        {
            "name": "🛡️ Sabit İki Barem Kapatma (Her Maç 1. Favori + 1 Alt Baremi Kapatmak)",
            "wins": s_dutch_wins,
            "total": tot_m,
            "win_rate": round(s_dutch_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_dutch_ret, 0),
            "net_profit": round(s_dutch_ret - tot_cost, 0),
            "roi_pct": round((s_dutch_wins and s_dutch_ret / tot_cost - 1.0) * 100.0, 1),
        },
    ]

    # NEW SYMMETRY TABLE A: Last Digit of Score
    mod10_brackets = []
    for digits, lbl, advice in [
        ((1, 2, 3, 4), "🎯 Skorun Son Rakamı 1 - 2 - 3 - 4 (Tam Barem Ortasına Düşer)", "💎 Kalan sürede ~53 sayı atılınca tam barem ortasına denk gelir: 1. Favori %57.6 tutar (+%22 net kâr)"),
        ((5, 6, 7), "📐 Skorun Son Rakamı 5 - 6 - 7 (Barem Üst Kısmına Yakın)", "⚖️ Kalan sürede ~53 sayı atılınca baremin orta-üst kısmına denk gelir: 1. Favori %53.8 tutar (+%10.5 net kâr)"),
        ((8, 9, 0), "⚠️ Skorun Son Rakamı 8 - 9 - 0 (Alt Sınırda Kalır / 1 Alt Barem Riski)", "📉 Kalan sürede ~53 sayı atılınca baremin en alt sınırında kalır (2 sayı az atılsa 1 Alt Barem olur!): 1 Alt Barem %43.8 gelir"),
    ]:
        sub = [x for x in raw_records if x["mod10"] in digits]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_down = sum(1 for x in sub if x["shift"] <= -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            roi_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 100.0
            mod10_brackets.append(
                {
                    "35. Dk Toplam Skorun Son Rakamı": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori Tutma & Kâr": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | Kâr: %{roi_0:+.1f}",
                    "📉 Alt Bareme Düşme": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Anlama Geliyor?": advice,
                }
            )

    # NEW SYMMETRY TABLE B: Twin Odds & Top-2 Concentration
    twin_conc_brackets = []
    for cond_fn, lbl, advice in [
        (lambda x: x["odd_gap_12"] <= 0.25, "👯 1. ve 2. Favori Oranları Neredeyse Eşit (Fark <= 0.25)", "⚖️ 1. Favori (%44.4) ve 2. Favori (%44.4) eşit sıklıkta gelir! İkisinden birinin tutma oranı %88.9 (9 maçta 8)"),
        (lambda x: x["top2_conc"] >= 78.0, "🔒 İlk 2 Barem Çok Güçlü (Toplam İhtimal >= %78)", "🛡️ Maç neredeyse kesin ilk 2 baremden birinde biter: İkisinden birinin tutma oranı %90.5 (21 maçta 19)!"),
        (lambda x: x["wing_diff"] <= 0.75, "⚖️ Alt ve Üst Komşu Barem Oranları Birbirine Eşit (Fark <= 0.75)", "🎯 Beklenti tam 1. Favorinin ortasındadır: 1. Favori %53.8 tutar, 1 Alt Barem sadece %7.7 gelir"),
        (lambda x: x["top2_conc"] < 70.0, "⚠️ Oranlar 3 Bareme Dağılmış / Kararsız Maç (İlk 2 İhtimali < %70)", "⚠️ Maç kararsızdır; ilk 2 baremin tutma oranı %69.2'ye düşer, dikkatli olunmalı"),
    ]:
        sub = [x for x in raw_records if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_sec = sum(1 for x in sub if x["act_b"] == x["sec_b"])
            c_top2 = sum(1 for x in sub if x["act_b"] in (x["fav_b"], x["sec_b"]))
            roi_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 100.0
            twin_conc_brackets.append(
                {
                    "Oranların Durumu": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori Tutma & Kâr": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "🥈 2. Favori Tutma": f"%{c_sec/n_s*100:.1f} ({c_sec}/{n_s})",
                    "🛡️ 1. veya 2. Favori Toplam": f"%{c_top2/n_s*100:.1f} ({c_top2}/{n_s})",
                    "💡 Ne İşe Yarar?": advice,
                }
            )

    # NEW SYMMETRY TABLE C: Late 5-Min (30m -> 35m) Change
    pendulum_brackets = []
    for cond_fn, lbl, advice in [
        (lambda x: x["late_jump"] == 0, "⚖️ Son 5 Dakikada (30➔35.Dk) Favori Barem Değişmedi", "🎯 Barem sabit kaldıysa 1. Favori %53.5 tutar ve +%13.2 net kâr bırakır"),
        (lambda x: x["late_jump"] <= -1, "🔄 Son 5 Dakikada (30➔35.Dk) Favori Barem Aşağı Düştü", "🚫 Hiçbir maç (%0) bir alt bareme daha düşmedi! %85.8 ihtimalle yeni Favoride kalır veya eski üst baremine döner"),
        (lambda x: x["late_jump"] >= 1, "🔄 Son 5 Dakikada (30➔35.Dk) Favori Barem Yukarı Çıktı", "🧊 Daha da üst bareme kaçma ihtimali %8.3'e düşer; %75 ihtimalle yeni Favoride veya 1 Alt Baremde kalır"),
    ]:
        sub = [x for x in raw_records if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_m1 = sum(1 for x in sub if x["shift"] == -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            roi_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 100.0
            pendulum_brackets.append(
                {
                    "Son 5 Dakikadaki (30➔35.Dk) Barem Değişimi": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori Tutma & Kâr": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "📉 1 Alt Bareme Düşme": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Yapmalı?": advice,
                }
            )

    # 1. Odds Bracket Breakdown (with <1.85 separated!)
    odds_brackets = []
    for low_o, high_o, lbl, advice in [
        (1.0, 1.85, "🚨 Oran Çok Düşük Tuzağı (1.85 Altı)", "🥈 1. Favori Zarar Ettirir (-%37) ➔ Onun Yerine 2. Favoriyi Oyna (+%41.8 Kâr)"),
        (1.85, 2.10, "⚠️ Düşük Oran Aralığı (1.85 - 2.09)", "📉 2. Favori Alttaysa 1 Alt Baremi Oyna (+%22.5 Kâr), Üstteyse 1. Favoriyi Oyna"),
        (2.10, 2.35, "💎 İdeal Oran Aralığı (2.10 - 2.34)", "🎯 Direkt 1. Favoriyi Oyna (%52 Tutar, +%15.6 Kâr; 2. Favori Üstteyse +%39.9 Kâr)"),
        (2.35, 99.0, "⚖️ Yüksek Oran Aralığı (2.35 ve Üstü)", "🎯 1. Favoriyi Oyna (+%21.6 Kâr) veya 1. ve 2. Favori İkisini Birden Kapat"),
    ]:
        sub = [x for x in raw_records if low_o <= x["odd_0"] < high_o]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_m1 = sum(1 for x in sub if x["shift"] == -1)
            c_m2 = sum(1 for x in sub if x["shift"] <= -2)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            roi_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 100.0
            odds_brackets.append(
                {
                    "35. Dk 1. Favori Oran Aralığı": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori Tutma & Kâr": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | Kâr: %{roi_0:+.1f}",
                    "📉 1 Alt Barem": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ 2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 1 veya 2 Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Yapmalı?": advice,
                }
            )

    # 2. 2nd Favorite Direction Breakdown (-1 Lower vs +1 Upper)
    sec_dir_brackets = []
    for cond_fn, lbl, advice in [
        (lambda x: x["sec_dir"] >= 1 and x["odd_0"] >= 2.10, "🔥 2. Favori Üst Baremde VE Oran >= 2.10", "🎯 ÇOK GÜÇLÜ SİNYAL: 1. Favori %61.1 Tutar (+%39.9 Kâr), 1 Alt Barem Hiç Gelmez (%0)!"),
        (lambda x: x["sec_dir"] >= 1 and x["odd_0"] < 2.10, "📈 2. Favori Üst Baremde VE Oran < 2.10", "🎯 1. Favori (%55.6 Tutar) veya 1. Favori ile 1 Üst Baremi Kapat"),
        (lambda x: x["sec_dir"] <= -1 and x["odd_0"] < 2.10, "📉 2. Favori Alt Baremde VE Oran < 2.10", "📉 1. Favori Zarar Ettirir (-%16.6) ➔ 1 Alt Baremi Oyna (+%16.6 Kâr)!"),
        (lambda x: x["sec_dir"] <= -1 and x["odd_0"] >= 2.10, "⚖️ 2. Favori Alt Baremde VE Oran >= 2.10", "🛡️ Yarı Yarıya 1. Favori ve 1 Alt Barem Gelir (Üst Barem %0!) ➔ İkisini Birden Kapat"),
    ]:
        sub = [x for x in raw_records if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_m1 = sum(1 for x in sub if x["shift"] == -1)
            c_m2 = sum(1 for x in sub if x["shift"] <= -2)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            roi_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 100.0
            roi_m1 = ((sum(x["odd_m1"] for x in sub if x["shift"] == -1) - n_s) / n_s) * 100.0
            sec_dir_brackets.append(
                {
                    "2. Favori Hangi Baremde Duruyor?": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori Tutma & Kâr": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "📉 1 Alt Barem Tutma & Kâr": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s}) | %{roi_m1:+.1f}",
                    "❄️ 2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 1 veya 2 Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Oynanmalı?": advice,
                }
            )

    # 3. Score Margin at 35m Breakdown
    margin_brackets = []
    for lo_m, hi_m, lbl, advice in [
        (0, 6, "⛔ 0 - 5 Sayı Fark (Başa Baş Maç / Uzatma ve Faul Riski)", "⛔ PAS GEÇ (Bahis Alma)! Kalan sürede ort. 59 sayı atılır, maçların %33'ü üst bareme kaçar"),
        (6, 12, "⚖️ 6 - 11 Sayı Fark (Normal Skor Farkı)", "🛡️ Normal Tempo (~53 sayı) ➔ 1. ve 2. Favori İkisini Kapat veya Sistemin Önerisini Oyna"),
        (12, 18, "🛡️ 12 - 17 Sayı Fark (Rahat Fark)", "🛡️ Uzatma Riski Yok ➔ 1. Favori Oyna veya 1. Favori ile 1 Alt Baremi Kapat"),
        (18, 999, "🧊 18+ Sayı Fark (Fark Çok Açıldı / Maç Yavaşlar)", "🔥 GÜÇLÜ FAVORİ: Kalan sürede ort. sadece 47.4 sayı atılır; 1. Favori %60 Tutar (+%27.5 Kâr)!"),
    ]:
        sub = [x for x in raw_records if lo_m <= x["margin35"] < hi_m]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_down = sum(1 for x in sub if x["shift"] <= -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            roi_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 100.0
            rems = [x["rem_pts"] for x in sub if x["min35"] == 35 and x["rem_pts"] is not None]
            avg_rem = f"{sum(rems)/len(rems):.1f} Sayı" if rems else "~53 Sayı"
            margin_brackets.append(
                {
                    "35. Dk Skor Farkı": lbl,
                    "Maç Sayısı": n_s,
                    "Kalan Sürede Ort. Atılan Sayı": avg_rem,
                    "🎯 1. Favori Tutma & Kâr": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "📉 Alt Bareme Düşme": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Oynanmalı?": advice,
                }
            )

    # 4. Barem Interior Edge Position
    edge_brackets = []
    sub_35only = [x for x in raw_records if x["min35"] == 35]
    for cond_fn, lbl, advice in [
        (lambda x: x["edge_offset"] < -3.0, "📐 Baremin Alt Sınırında Kalıyor (Skor + 53 Sayı = Alt Sınırda)", "📉 DİKKAT: Maçların %55.6'sı 1 Alt Bareme Düşer (1. Favori -%24.6 Zarar Ettirir!)"),
        (lambda x: -3.0 <= x["edge_offset"] <= 3.0, "🎯 Baremin Tam Ortasında (Skor + 53 Sayı = Tam Ortada)", "💎 GÜVENLİ BÖLGE: Maçların %55.6'sı Direkt 1. Favoride Biter (+%17.8 Kâr)"),
        (lambda x: x["edge_offset"] > 3.0, "📐 Baremin Üst Kısmında (Skor + 53 Sayı = Üst Sınıra Yakın)", "📈 GÜÇLÜ FAVORİ: %56.3 Oranında 1. Favori Tutar (+%22.2 Kâr), %31.3 Üst Bareme Çıkar"),
    ]:
        sub = [x for x in sub_35only if cond_fn(x)]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_down = sum(1 for x in sub if x["shift"] <= -1)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            roi_0 = ((sum(x["odd_0"] for x in sub if x["shift"] == 0) - n_s) / n_s) * 100.0
            edge_brackets.append(
                {
                    "Tahmini Maç Sonu Baremde Nereye Düşüyor?": lbl,
                    "Maç Sayısı": n_s,
                    "📉 Alt Bareme Düşme": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "🎯 1. Favoride Kalma & Kâr": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "📈 Üst Bareme Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne İşe Yarar?": advice,
                }
            )

    # 5. Minute-by-Minute Optimal Entry Window
    minute_windows = []
    for m_target, m_lbl, m_advice in [
        (18, "⚡ 18. Dakika (2. Çeyrek Ortası — Erken Bahis)", "💎 Erken bahiste en kârlı dakika: 1. Favori (+%16.8 Kâr) ve İlk 2 Favori İkilisi (+%14.3 Kâr)"),
        (24, "⏸️ 24. Dakika (Devre Arası)", "⚖️ 1. Favori kârlı (+%11.4 Kâr), devre arası tempo analizi için uygun"),
        (30, "🕒 30. Dakika (3. Çeyrek Ortası)", "⚖️ Oranlar şekillenmeye başlar (%45.9 1. Favori tutma, +%5.8 Kâr)"),
        (33, "🚨 32 - 33. Dakika (3. Çeyrek Sonu — TUZAK DAKİKA)", "⛔ BAHİS ALMA! Barem değişim anı olduğu için en çok yanılma ve zarar (-%17.1) bu dakikadadır"),
        (35, "🔥 34 - 35. Dakika (Kapanış — EN YÜKSEK İSABET)", "🎯 Ana Karar Anı: 1. Favori (%50.8 Tutar), İlk 2 Favori İkilisi (%83.6 Tutar)"),
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
            om = {b_n: float(r_m.get(b_cols[b_n], 0.0) or 0.0) for b_n in barem_names}
            om = {k: v for k, v in om.items() if 1.0 < v < 90.0}
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
                    "🎯 1. Favori Tutma & Kâr": f"%{m_hits0/m_n*100:.1f} ({m_hits0}/{m_n}) | Kâr: %{(m_ret0-m_n)/m_n*100:+.1f}",
                    "🛡️ 1. ve 2. Favori İkisini Kapatma": f"%{m_hits_d/m_n*100:.1f} ({m_hits_d}/{m_n}) | Kâr: %{(m_ret_d-m_n)/m_n*100:+.1f}",
                    "💡 Hangi Dakika Daha İyi?": m_advice,
                }
            )

    # 6. Drift (Opening -> 35m Barem Change) Breakdown
    drift_brackets = []
    for cond_fn, lbl, advice in [
        (lambda d: d >= 1, "📈 Maç Başından 35. Dakikaya Barem Yükseldi", "🎯 Maç Hızlı Gitti: 35. Dk Favorisi %58 Tutar (+%26.1 Kâr), Alt Bareme Nadiren Düşer"),
        (lambda d: d == 0, "⚖️ Maç Başından 35. Dakikaya Barem Aynı Kaldı", "📉 Maçların %43'ü Alt Bareme Kayar ➔ 1. Favori ile 1 Alt Baremi Birlikte Kapatmak Güvenlidir"),
        (lambda d: d == -1, "📉 Maç Başından 35. Dakikaya Barem 1 Kademe Düştü", "⚖️ %43 Yeni Favoride Kalır, %43 Tekrar Eski Üst Baremine Çıkar"),
        (lambda d: d <= -2, "❄️ Maç Başından 35. Dakikaya Barem Çok Düştü (2-3 Kademe)", "🎯 Zaten Çok Düştüğü İçin Daha Fazla Düşmez; Yeni Favoride Dengelenir (+%14.1 Kâr)"),
    ]:
        sub = [x for x in raw_records if cond_fn(x["drift"])]
        if sub:
            n_s = len(sub)
            c_0 = sum(1 for x in sub if x["shift"] == 0)
            c_m1 = sum(1 for x in sub if x["shift"] == -1)
            c_m2 = sum(1 for x in sub if x["shift"] <= -2)
            c_up = sum(1 for x in sub if x["shift"] >= 1)
            drift_brackets.append(
                {
                    "Maç Başından 35. Dakikaya Barem Değişimi": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 1. Favori Barem": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s})",
                    "📉 1 Alt Barem": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ 2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 1 veya 2 Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Anlama Geliyor?": advice,
                }
            )

    # 7. Sequential Match-to-Match Transitions (Real Tracked Chain)
    transitions = []
    for cond_fn, lbl in [
        (lambda s: s == 0, "🎯 Önceki Maçta 1. Favori Barem Tuttuysa ➔ Sıradaki Maç"),
        (lambda s: s <= -1, "📉 Önceki Maç Alt Baremde Bittiyse ➔ Sıradaki Maç"),
        (lambda s: s >= 1, "📈 Önceki Maç Üst Baremde Bittiyse ➔ Sıradaki Maç"),
    ]:
        nxt = [raw_records[i + 1] for i in range(len(raw_records) - 1) if cond_fn(raw_records[i]["shift"])]
        if nxt:
            n_s = len(nxt)
            c_0 = sum(1 for x in nxt if x["shift"] == 0)
            c_m1 = sum(1 for x in nxt if x["shift"] == -1)
            c_m2 = sum(1 for x in nxt if x["shift"] <= -2)
            c_up = sum(1 for x in nxt if x["shift"] >= 1)
            transitions.append(
                {
                    "Önceki Maçın Sonucu": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 Sıradaki Maç 1. Favori": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s})",
                    "📉 Sıradaki Maç 1 Alt Barem": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ Sıradaki Maç 2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 Sıradaki Maç Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "🛡️ Favori veya 1 Alt Gelme Oranı": f"%{(c_0+c_m1)/n_s*100:.1f} ({c_0+c_m1}/{n_s})",
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


def analyze_archive_4500_symmetry(df_hist: pd.DataFrame) -> dict:
    """
    Analyzes 4,550+ completed CyberLeague matches for deep mathematical symmetries in plain Turkish.
    """
    if df_hist is None or df_hist.empty:
        return {"total_archive": 0, "mod10_36m": [], "half_symmetry": []}

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
    df_h = df_h[(df_h["total_score"] >= 140) & (df_h["S36"] >= 100)].copy()
    if df_h.empty:
        return {"total_archive": 0, "mod10_36m": [], "half_symmetry": []}

    df_h["s36_mod10"] = (df_h["S36"].astype(int)) % 10
    mod10_rows = []
    digit_notes = {
        0: "⚖️ +52 sayı eklenince sonu ..2 olur (Baremin alt-orta kısmı): Beklenen Barem %40.3, 1 Alt Barem %39.7 başa baş",
        1: "📉 +52 sayı eklenince sonu ..3 olur (Alt sınıra yakın): 1 Alt Barem (%38.7) ve Beklenen Barem (%36.9) dengeli",
        2: "⚖️ +52 sayı eklenince sonu ..4 olur (Barem ortasına yakın): Üç yön dengeli (%36 Beklenen / %31 Alt / %32 Üst)",
        3: "🎯 EN GÜVENLİ SON RAKAM (+52 eklenince sonu ..5 tam barem ortası olur): Beklenen Barem %45.3 (En Yüksek!), Alt %27.6 = Üst %27.1 eşit!",
        4: "🎯 +52 sayı eklenince sonu ..6 olur (Barem ortası): Beklenen Barem %40.0, Üst Barem %33.8",
        5: "📈 +52 sayı eklenince sonu ..7 olur (Orta-üst kısım): Beklenen Barem %40.7, Üst Barem %37.3",
        6: "📈 +52 sayı eklenince sonu ..8 olur (Üst sınıra yakın): Beklenen Barem %41.1 ve Üst Barem %40.9 eşit!",
        7: "🔥 +52 sayı eklenince sonu ..9 olur (Üst sınırda): 1 Üst Barem %46.0 ile öne geçer!",
        8: "🔥 ÜST BAREME GEÇİŞ (+52 eklenince sonu ..0 barem tavanı olur): 1 sayı bile fazla atılsa Üst Bareme geçer (%51.6 Üst Barem!)",
        9: "❄️ ALT BAREME DÜŞÜŞ (+52 eklenince sonu ..1 barem tabanı olur): 1 sayı bile az atılsa Alt Bareme düşer (%48.5 Alt Barem!)",
    }
    for d in range(10):
        sub = df_h[df_h["s36_mod10"] == d]
        if sub.empty:
            continue
        exp_tot = sub["S36"] + 52
        act_b_idx = (sub["total_score"] - 1) // 10
        exp_b_idx = (exp_tot - 1) // 10
        same_b = (act_b_idx == exp_b_idx).mean() * 100.0
        down_b = (act_b_idx < exp_b_idx).mean() * 100.0
        up_b = (act_b_idx > exp_b_idx).mean() * 100.0
        mod10_rows.append(
            {
                "3. Çeyrek Sonu (36.Dk) Skorun Son Rakamı": f"Sonu ..{d} ile bitenler (Örn: 15{d})",
                "Maç Sayısı": len(sub),
                "Maç Sonu Beklenen Son Rakam (+52 Sayı)": f"..{(d + 2) % 10}",
                "📉 Alt Bareme Düşme": f"%{down_b:.1f}",
                "🎯 Beklenen Baremde Kalma": f"%{same_b:.1f}",
                "📈 Üst Bareme Çıkma": f"%{up_b:.1f}",
                "💡 4.550 Maçlık İstatistik Özeti": digit_notes.get(d, ""),
            }
        )

    half_rows = []
    for lbl, cond, note in [
        ("❄️ İlk Yarı Az Sayı Çıktı (98 Sayı ve Altı)", df_h["H1"] <= 98, "📈 İkinci Yarı Hızlanır: 2. yarıda oyun açılır ve ilk yarıdan ortalama +12.6 sayı DAHA FAZLA atılır!"),
        ("⚖️ İlk Yarı Normal Bitti (99 - 107 Sayı Arası)", (df_h["H1"] >= 99) & (df_h["H1"] <= 107), "🪞 İki Yarı Eşit Gider: 1. Yarı (103.0) ile 2. Yarı (105.4) neredeyse birebir aynı sayıda biter"),
        ("🔥 İlk Yarı Çok Sayı Çıktı (108 Sayı ve Üstü)", df_h["H1"] >= 108, "🧊 İkinci Yarı Yavaşlar: 2. yarıda tempo düşer ve ilk yarıdan ortalama -8.4 sayı DAHA AZ atılır!"),
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
