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
    if df_log is not None and not df_log.empty and "game_id" in df_log.columns:
        for gid, grp in df_log.groupby("game_id"):
            last_r = grp.iloc[-1]
            fav_val = str(last_r.get("highest_prob_range", "")).strip()
            if fav_val in barem_names:
                gid_clean = str(gid).strip()
                logged_closing_fav[gid_clean] = fav_val
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
    # Never include morning/past matches that finished before the system was opened.
    recent_chain = []
    tracked_hist = hist[hist["game_id"].astype(str).str.strip().isin(logged_closing_fav.keys())].copy()
    for _, r in tracked_hist.iterrows():
        sh = int(r["raw_shift"])
        if sh == 0:
            sh_label = "🎯 0 (Beklenen)"
        elif sh == -1:
            sh_label = "📉 -1 (1 Alt Barem)"
        elif sh == 1:
            sh_label = "📈 +1 (1 Üst Barem)"
        elif sh < -1:
            sh_label = f"❄️ {sh} ({abs(sh)} Alt Barem)"
        else:
            sh_label = f"🔥 +{sh} ({sh} Üst Barem)"

        recent_chain.append(
            {
                "game_id": r.get("game_id", ""),
                "date": r.get("date", ""),
                "home_team": r.get("home_team", ""),
                "away_team": r.get("away_team", ""),
                "home_total": int(r.get("home_total", 0) or 0),
                "away_total": int(r.get("away_total", 0) or 0),
                "s36": int(r["s36"]),
                "q4_tot": int(r["q4_tot"]),
                "total_score": int(r["total_score"]),
                "exp_barem_36": r["exp_barem_36"],
                "exp_odd_36": float(r.get("exp_odd_36", 0.0) or 0.0),
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
        wave_dir = f"📉 Merkez (0) veya -1 Alt Barem Eğilimi (%{pair_down_prob})"
        rec_pair = f"{b_m1} & {b_0} (Beklenen + 1 Alt Barem)"
        rec_pair_prob = pair_down_prob
    elif pair_up_prob >= pair_down_prob + 2.0:
        wave_dir = f"📈 Merkez (0) veya +1 Üst Barem Eğilimi (%{pair_up_prob})"
        rec_pair = f"{b_0} & {b_p1} (Beklenen + 1 Üst Barem)"
        rec_pair_prob = pair_up_prob
    else:
        wave_dir = f"🎯 Merkez (0) Odaklı Dengeli Dalga (Merkez %{p_0} | -1 Alt %{p_m1} | +1 Üst %{p_p1})"
        if pair_down_prob >= pair_up_prob:
            rec_pair = f"{b_m1} & {b_0} (Beklenen + 1 Alt Barem)"
            rec_pair_prob = pair_down_prob
        else:
            rec_pair = f"{b_0} & {b_p1} (Beklenen + 1 Üst Barem)"
            rec_pair_prob = pair_up_prob

    last_5_codes = [x["raw_shift"] for x in recent_chain[-5:]]
    last_5_fmt = " ➔ ".join(
        [("0 (Merkez)" if c == 0 else f"{c:+d}. Barem") for c in last_5_codes]
    )

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
