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
) -> dict:
    """
    v3.1 35th-Minute Smart Decision & Trap Detector (calibrated on 62+ real 1xBet closing matches):
    1. Score Gap (Clutch vs Blowout):
       - 0-5 pts gap: Clutch/OT Trap (avg 59.0 pts in last 13m, 33% jump to +1/+2 -> PASS / NO BET).
       - 18+ pts gap: Blowout Brake (avg 47.4 pts in last 13m, 95% stay in 0 or -1, 0 hits 60% / +27.5% ROI).
    2. 2nd Favorite Direction (Market Skew: +1 Upper vs -1 Lower):
       - 2nd Fav at +1 Upper & Fav Odd >= 2.10: Super Signal! 0 (Merkez) hits 61.1% (+39.9% ROI), -1 hits 0%.
       - 2nd Fav at -1 Lower & Fav Odd < 2.10: -1 Alt Barem hits +16.6% ROI while 0 loses -16.6%.
    3. Extreme Low-Odd Trap (< 1.85):
       - 1st Fav loses -37.1% ROI; 2nd Favorite wins +41.8% ROI.
    4. Position Within Barem (Score + Expected Remaining Points vs Barem Midpoint):
       - Detects whether projected total sits on the lower edge (-1 risk 56%), center, or upper edge.
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
        sec_dir_badge = f"📈 2.Favori Üst Baremi Gösteriyor ({sec_b_clean})"
        sec_dir_tip = (
            "1xBet'in en düşük oranlı 2. seçeneği şu anki favorinin BİR ÜST bareminde duruyor. "
            "62 maçlık veride 2. favori üstteyken maçların alt bareme düşme ihtimali çok düşüktür (%7); "
            "oran 2.10 üstüyse direkt 1. Favori Barem %61.1 isabet ve +%39.9 kâr (ROI) üretmiştir."
        )
    else:
        sec_dir_badge = f"📉 2.Favori Alt Baremi Gösteriyor ({sec_b_clean})"
        sec_dir_tip = (
            "1xBet'in 2. favorisi şu anki favorinin BİR ALT bareminde duruyor. "
            "Bu durumda maçların %48.5'i alt baremlere kaymış, üst bareme ise hiç (%0) gitmemiştir."
        )

    # Calculate in-game drift for this game_id
    open_b = base_b
    drift = 0
    if df_log is not None and not df_log.empty and game_id and "game_id" in df_log.columns:
        g_sub = df_log[df_log["game_id"].astype(str).str.strip() == str(game_id).strip()]
        if not g_sub.empty:
            first_b = str(g_sub.iloc[0].get("highest_prob_range", "")).strip()
            if first_b in barem_names:
                open_b = first_b
                drift = barem_names.index(base_b) - barem_names.index(open_b)

    if drift > 0:
        drift_badge = f"📈 Maç Başından Beri +{drift} Üste Çıktı ({open_b}➔{base_b})"
    elif drift < 0:
        drift_badge = f"📉 Maç Başından Beri {abs(drift)} Alta Düştü ({open_b}➔{base_b})"
    else:
        drift_badge = f"⚖️ Maç Başıyla Aynı Barem ({base_b})"
    drift_tip = (
        f"Maçın 1. çeyrek başındaki favori baremi ({open_b}) ile şu anki favori baremi ({base_b}) arasındaki değişim. "
        "Üste çıkması maçın hızlı gittiğini, alta düşmesi ise maçın yavaşladığını gösterir."
    )

    gap = abs(int(score_gap or 0))
    if gap <= 5:
        exp_rem_35 = 59.0
        margin_badge = f"⚠️ {gap} Sayı Fark (Yakın Maç / Uzatma Riski)"
        margin_tip = (
            "Fark 0-5 sayı arasındayken maç başa baş gider. Son dakikalarda taktik fauller ve uzatma (OT) "
            "ihtimali yüzünden 35. dakikadan sonra ortalama 59.0 sayı atılır (normalden +6 sayı fazla) ve "
            "maçların %33'ü üst bareme patlar. Bu yüzden yakın maçları PAS GEÇMEK kârı +%17.6'dan +%29.1'e çıkarır."
        )
    elif gap >= 18:
        exp_rem_35 = 47.4
        margin_badge = f"🧊 {gap} Sayı Fark (Fark Açık / Rölanti Freni)"
        margin_tip = (
            "Fark 18+ sayı açıldığında takımlar son çeyrekte tempoyu düşürür (rölanti). "
            "35. dakikadan sonra ortalama sadece 47.4 sayı atılır (yakın maçtan 11.6 sayı daha az!). "
            "Üst bareme kaçma ihtimali %5'tir; 1. Favori (0 Merkez) %60 isabetle +%27.5 ROI kazandırır."
        )
    elif gap >= 10:
        exp_rem_35 = 53.5
        margin_badge = f"🛡️ {gap} Sayı Fark (Kontrollü Fark)"
        margin_tip = "Fark 10-17 sayı bandında; uzatma riski düşük, İkili Korumada 0 Merkez + -1 Alt Barem birlikte çalışır."
    else:
        exp_rem_35 = 53.0
        margin_badge = f"⚖️ {gap} Sayı Fark (Normal Fark)"
        margin_tip = "Fark 6-9 sayı bandında; standart son çeyrek temposu (~53 sayı) beklenir."

    # Barem interior edge position (when match is near 30-36m or anytime using remaining pace)
    fav_lo, fav_hi = barem_bounds.get(base_b, (201, 210))
    fav_mid = (fav_lo + min(fav_hi, 240)) / 2.0
    rem_mins_est = max(0.0, 48.0 - float(elapsed_mins or 35.0))
    scaled_rem = exp_rem_35 * (rem_mins_est / 13.0) if rem_mins_est > 0 else 0.0
    proj_tot_edge = float(cur_tot) + scaled_rem if cur_tot > 0 else fav_mid
    edge_offset = proj_tot_edge - fav_mid

    if cur_tot > 0 and 28.0 <= float(elapsed_mins or 35.0) <= 36.5:
        if edge_offset < -3.0:
            edge_badge = f"📐 Alt Kenarda (Tahmini: ~{proj_tot_edge:.0f} | {base_b} Alt Sınırı)"
            edge_tip = (
                f"Mevcut skor ({cur_tot}) + farka göre beklenen kalan sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı yapıyor. "
                f"Bu değer {base_b} bareminin alt sınırına çok yakın olduğu için maçların %55.6'sı bir alt bareme (-1) kaymaktadır."
            )
        elif edge_offset > 3.0:
            edge_badge = f"📐 Üst Kenarda (Tahmini: ~{proj_tot_edge:.0f} | {base_b} Üst Sınırı)"
            edge_tip = (
                f"Mevcut skor ({cur_tot}) + beklenen kalan sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı yapıyor. "
                f"{base_b} bareminin orta-üst kısmında güçlü duruyor (%56 Merkez isabeti, +%22.2 ROI)."
            )
        else:
            edge_badge = f"🎯 Barem Tam Ortasında (Tahmini: ~{proj_tot_edge:.0f})"
            edge_tip = (
                f"Mevcut skor ({cur_tot}) + beklenen kalan sayı (~{scaled_rem:.0f}) = ~{proj_tot_edge:.0f} sayı ile "
                f"tam {base_b} bareminin merkezinde (%55.6 Merkez isabeti, +%17.8 ROI)."
            )
    else:
        edge_badge = f"⏱️ 35.Dk Kalan Sayı Standardı: Ort. 53.3 Sayı (Bu farkta ~{exp_rem_35:.0f})"
        edge_tip = (
            "62 maçlık gerçek veride 35. dakikadan maç sonuna kadar ortalama 53.3 sayı atılmıştır "
            "(0-5 farkta 59.0 sayı, 18+ farkta 47.4 sayı). 30-35. dakikaya gelindiğinde burası tam kenar hesabını gösterir."
        )

    prev_shift = recent_chain[-1]["raw_shift"] if recent_chain else 0
    odd_val = float(fav1_odd or 0.0)
    reasons = []

    # Smart Dutching v2 recommendation: 10+ gap -> 0 & -1; otherwise -> 0 & 2ndFav
    if gap >= 10:
        dutch_pick = f"🛡️ {b_0} (Favori) ile {b_m1} (1 Alt Barem) İkisini Kapat"
    else:
        dutch_pick = f"🛡️ {b_0} (1. Favori) ile {sec_b_clean} (2. Favori) İkisini Kapat"

    # V3.1 Decision Engine
    if gap <= 5 and float(elapsed_mins or 35.0) >= 30.0:
        zone_badge = f"⛔ PAS GEÇ / YAKIN MAÇ TUZAĞI (Fark: {gap} Sayı)"
        zone_type = "PASS_CLUTCH"
        stake_badge = "0 TL (PAS GEÇ — Uzatma/Faul Riski)"
        primary_pick = f"⛔ Bahis Alma (İlla Oynanacaksa: {b_0} veya {b_p1})"
        backup_pick = dutch_pick
        conf_pct = 38.0
        reasons.append(
            f"35.dk civarı fark {gap} sayı (0-5 arası) olduğunda son bölümde ort. 59 sayı atılıyor ve maçların %33'ü üst bareme patlıyor (Favori Barem ROI: -%10.8)"
        )
    elif odd_val > 1.0 and odd_val < 1.85:
        zone_badge = f"⚠️ <1.85 AŞIRI DÜŞÜK ORAN TUZAĞI ({odd_val:.2f})"
        zone_type = "TRAP_EXTREME"
        stake_badge = "1000 TL (Standart Bahis — 2. Favoriye)"
        primary_pick = f"🥈 2. Favori Baremi Oyna: {sec_b_clean} (Oran: {fav2_odd:.2f})"
        backup_pick = f"🎯 1. Favori Barem: {b_0}"
        conf_pct = 74.0
        reasons.append(
            f"1. Favori oranı 1.85 altındayken ({odd_val:.2f}) 1. Favori -%37.1 zarar yazıyor; doğrudan 2. Favori ({sec_b_clean}) oynamak +%41.8 ROI kazandırdı"
        )
    elif odd_val >= 2.10 and sec_dir >= 1:
        zone_badge = f"🔥 SÜPER SİNYAL: 2.FAVORİ ÜSTTE + ORAN {odd_val:.2f}"
        zone_type = "SUPER_SIGNAL"
        stake_badge = "2000 TL (Kademeli Kasa: Güçlü Sinyal)"
        primary_pick = f"🎯 Direkt 1. Favori Baremi Oyna: {b_0} (Oran: {odd_val:.2f})"
        backup_pick = dutch_pick
        conf_pct = 88.0
        reasons.append(
            f"2. Favori üstte ({sec_b_clean}) ve oran >=2.10 ({odd_val:.2f}): 62 maçta %61.1 doğrudan 1. Favori ({b_0}) tuttu (+%39.9 ROI) ve hiç alt bareme düşmedi (%0)"
        )
    elif gap >= 18 and odd_val >= 1.85:
        zone_badge = f"🔥 FARK AÇIK (RÖLANTİ) SİNYALİ (Fark: {gap} | Oran: {odd_val:.2f})"
        zone_type = "SUPER_SIGNAL"
        stake_badge = "2000 TL (Kademeli Kasa: Rölanti Sinyali)"
        primary_pick = f"🎯 Direkt 1. Favori Baremi Oyna: {b_0} (Oran: {odd_val:.2f})"
        backup_pick = f"🛡️ {b_0} (Favori) ile {b_m1} (1 Alt) İkisini Kapat (%95 Tutma)"
        conf_pct = 86.0
        reasons.append(
            f"18+ sayı farkta ({gap}) son çeyrek rölantiye girer (ort. 47.4 sayı); maçların %95'i Favori ({b_0}) veya 1 Alt ({b_m1}) biter, 1. Favori tek başına %60 isabet ve +%27.5 ROI üretir"
        )
    elif odd_val > 1.0 and odd_val < 2.10 and sec_dir <= -1:
        zone_badge = f"⚠️ DÜŞÜK ORAN + ALT BAREM BASKISI ({odd_val:.2f})"
        zone_type = "TRAP_LOW"
        stake_badge = "1000 TL (Standart Bahis — 1 Alt Bareme)"
        primary_pick = f"📉 1 Alt Baremi Oyna: {b_m1} (Favori {b_0} Yerine)"
        backup_pick = f"🎯 1. Favori Barem: {b_0}"
        conf_pct = 76.0
        reasons.append(
            f"Oran <2.10 ({odd_val:.2f}) ve 2. Favori altta ({b_m1}) olduğunda 1. Favori ({b_0}) -%16.6 zarar ederken 1 Alt Barem ({b_m1}) +%16.6 ROI sağlıyor"
        )
    elif 1.85 <= odd_val <= 2.79:
        zone_badge = f"💎 İDEAL ORAN BANDI ({odd_val:.2f})"
        zone_type = "GOLDEN"
        stake_badge = "1000 TL (Standart Bahis — 1. Favoriye)"
        primary_pick = f"🎯 Direkt 1. Favori Baremi Oyna: {b_0}"
        backup_pick = dutch_pick
        conf_pct = 80.0
        reasons.append(
            f"Oran {odd_val:.2f} dengeli bölgede; 1. Favori ({b_0}) doğrudan ~%52 isabet (+%16-21 ROI), İkili Koruma ({dutch_pick}) %82+ isabet sağlıyor"
        )
    else:
        zone_badge = f"⚖️ YÜKSEK ORAN / KARARSIZ MAÇ ({odd_val:.2f})" if odd_val < 90 else "⏳ Oran Bekleniyor"
        zone_type = "WIDE"
        stake_badge = "1000 TL (İkili Koruma Önerilir)"
        primary_pick = f"🎯 1. Favori Barem: {b_0}"
        backup_pick = dutch_pick
        conf_pct = 68.0
        reasons.append("Oran >=2.80 kararsız bölgede; tek barem yerine 1. ve 2. Favori İkili Koruma (Çifte Şans) daha güvenlidir")

    if prev_shift >= 1:
        reasons.append(f"Önceki maç {prev_shift} Üst Baremde bitti (Üst bittikten sonra sıradaki maçta %69 ihtimalle Favori veya 1 Alt Barem gelir)")

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
        cands = grp[grp["_min"] >= 30]
        last_r = cands.iloc[-1] if not cands.empty else grp.iloc[-1]
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
        m1_b = barem_names[max(0, idx_fav - 1)]
        odd_m1 = odds_map.get(m1_b, 3.20)

        s1, s2 = _parse_sc(last_r.get("current_score", ""))
        a1, a2 = _parse_sc(last_r.get("actual_score", ""))
        margin35 = abs(s1 - s2) if s1 is not None else 8
        cur_tot35 = (s1 + s2) if s1 is not None else 0
        final_tot = (a1 + a2) if a1 is not None else 0
        rem_pts = (final_tot - cur_tot35) if (s1 is not None and a1 is not None) else None
        min_val = int(last_r.get("_min", 35))

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

        # 2) New v3 rule (<1.85 -> 2nd Fav | <2.10 & sec_dir==-1 -> -1 Alt | else 0 Merkez)
        if odd_0 < 1.85:
            v3_b = sec_b
            v3_odd = sec_odd
            v3_rule_lbl = "⚠️ <1.85 Tuzağı ➔ 2.Fav"
        elif odd_0 < 2.10 and sec_dir <= -1:
            v3_b = m1_b
            v3_odd = odd_m1
            v3_rule_lbl = "📉 <2.10 & 2.Fav Altta ➔ -1 Alt"
        else:
            v3_b = fav_b
            v3_odd = odd_0
            if sec_dir >= 1 and odd_0 >= 2.10:
                v3_rule_lbl = "🔥 Süper Sinyal (2.Fav Üstte) ➔ 0 Merkez"
            elif margin35 >= 18:
                v3_rule_lbl = "🧊 18+ Fark Rölanti ➔ 0 Merkez"
            else:
                v3_rule_lbl = "💎 Standart ➔ 0 Merkez"

        hit_v3_all = act_b == v3_b
        ret_v3_all = round(stake * v3_odd, 1) if hit_v3_all else 0.0
        s_v3_all_ret += ret_v3_all
        s_v3_all_wins += int(hit_v3_all)

        # 3) Selective v3 (Pass when margin35 <= 5 Clutch/OT risk)
        is_clutch_pass = margin35 <= 5
        if not is_clutch_pass:
            s_v3_sel_cnt += 1
            s_v3_sel_cost += stake
            if hit_v3_all:
                s_v3_sel_wins += 1
                s_v3_sel_ret += ret_v3_all

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
                "act_b": act_b,
                "act_score": str(last_r.get("actual_score", "")),
                "shift": shift,
                "drift": drift,
            }
        )

        sec_arrow = f"📈 Üst ({sec_b})" if sec_dir >= 1 else f"📉 Alt ({sec_b})"
        if is_clutch_pass:
            sel_status = f"⛔ PAS (Fark {margin35})"
            tiered_status = "⛔ 0 TL (Pas Geçildi)"
        else:
            sel_status = f"✅ +{ret_v3_all - stake:.0f} TL" if hit_v3_all else "❌ -1000 TL"
            tiered_status = (
                f"✅ +{t_ret - t_stake:.0f} TL ({t_stake:.0f} Yatırım)"
                if hit_v3_all
                else f"❌ -{t_stake:.0f} TL"
            )

        ledger.append(
            {
                "Maç": str(last_r.get("game_name", "")).replace(" (cyber)", ""),
                "Kapanış Dk & Fark": f"{last_r.get('check_time', '')} ({margin35} Sayı Fark)",
                "35.Dk Favori (0)": f"{fav_b} ({odd_0:.2f})",
                "2. Favori Yönü": f"{sec_arrow} ({sec_odd:.2f})",
                "🤖 v3 Akıllı Seçim": f"{v3_b} ({v3_odd:.2f}) [{v3_rule_lbl}]",
                "Biten Barem (Skor)": f"{act_b} ({last_r.get('actual_score', '')})",
                "v3 Seçici (1000 TL / Pas)": sel_status,
                "Kademeli Kasa (0/1000/2000)": tiered_status,
                "Akıllı İkili Koruma": f"✅ +{ret_dutch_v2 - stake:.0f} TL ({dutch_v2_odd:.2f})" if hit_dutch_v2 else "❌ -1000 TL",
            }
        )

    tot_m = len(raw_records)
    if tot_m == 0:
        return empty_res

    tot_cost = tot_m * stake
    strategies = [
        {
            "name": "🔥 v3 Seçici Akıllı Strateji (0-5 Fark Yakın Maçları Pas Geç + v3 Kural)",
            "wins": s_v3_sel_wins,
            "total": s_v3_sel_cnt,
            "win_rate": round(s_v3_sel_wins / max(1, s_v3_sel_cnt) * 100.0, 1),
            "cost": round(s_v3_sel_cost, 0),
            "return": round(s_v3_sel_ret, 0),
            "net_profit": round(s_v3_sel_ret - s_v3_sel_cost, 0),
            "roi_pct": round((s_v3_sel_ret / max(1.0, s_v3_sel_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "💰 Kademeli Güven Kasası (Yakın Maç: 0 TL | Normal: 1000 TL | Süper Sinyal: 2000 TL)",
            "wins": s_tiered_wins,
            "total": s_tiered_cnt,
            "win_rate": round(s_tiered_wins / max(1, s_tiered_cnt) * 100.0, 1),
            "cost": round(s_tiered_cost, 0),
            "return": round(s_tiered_ret, 0),
            "net_profit": round(s_tiered_ret - s_tiered_cost, 0),
            "roi_pct": round((s_tiered_ret / max(1.0, s_tiered_cost) - 1.0) * 100.0, 1),
        },
        {
            "name": "🤖 v3 Akıllı Hibrit (Tüm Maçlar: <1.85➔2.Fav | <2.10&Alt➔-1 | Diğer➔0 Merkez)",
            "wins": s_v3_all_wins,
            "total": tot_m,
            "win_rate": round(s_v3_all_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_v3_all_ret, 0),
            "net_profit": round(s_v3_all_ret - tot_cost, 0),
            "roi_pct": round((s_v3_all_ret / tot_cost - 1.0) * 100.0, 1),
        },
        {
            "name": "🤖 v2 Eski Akıllı Hibrit (<2.10 ise -1 Alt / >=2.10 ise 0 Merkez)",
            "wins": s_smart_wins,
            "total": tot_m,
            "win_rate": round(s_smart_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_smart_ret, 0),
            "net_profit": round(s_smart_ret - tot_cost, 0),
            "roi_pct": round((s_smart_ret / tot_cost - 1.0) * 100.0, 1),
        },
        {
            "name": "🎯 Sabit 0 (Her Maç Sadece 35.Dk 1. Favori Merkez)",
            "wins": s_0_wins,
            "total": tot_m,
            "win_rate": round(s_0_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_0_ret, 0),
            "net_profit": round(s_0_ret - tot_cost, 0),
            "roi_pct": round((s_0_ret / tot_cost - 1.0) * 100.0, 1),
        },
        {
            "name": "🛡️ Akıllı İkili Koruma (Dutching v2: 10+ Farkta 0&-1 / Diğerlerinde 0&2.Fav)",
            "wins": s_dutch_v2_wins,
            "total": tot_m,
            "win_rate": round(s_dutch_v2_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_dutch_v2_ret, 0),
            "net_profit": round(s_dutch_v2_ret - tot_cost, 0),
            "roi_pct": round((s_dutch_v2_ret / tot_cost - 1.0) * 100.0, 1),
        },
        {
            "name": "🛡️ Sabit İkili Koruma (Eski Dutching: Her Maç 0 Merkez + -1 Alt)",
            "wins": s_dutch_wins,
            "total": tot_m,
            "win_rate": round(s_dutch_wins / tot_m * 100.0, 1),
            "cost": round(tot_cost, 0),
            "return": round(s_dutch_ret, 0),
            "net_profit": round(s_dutch_ret - tot_cost, 0),
            "roi_pct": round((s_dutch_wins and s_dutch_ret / tot_cost - 1.0) * 100.0, 1),
        },
    ]

    # 1. Odds Bracket Breakdown (with <1.85 separated!)
    odds_brackets = []
    for low_o, high_o, lbl, advice in [
        (1.0, 1.85, "🚨 Aşırı Düşük Oran Tuzağı (< 1.85)", "🥈 1. Favori Zarar Yazar (-%37 ROI) ➔ Doğrudan 2. Favori Oyna (+%41.8 ROI)"),
        (1.85, 2.10, "⚠️ Düşük Oran Bandı (1.85 - 2.09)", "📉 2.Favori Alttaysa -1 Alt (+%22.5 ROI), Üstteyse 0 Merkez Oyna"),
        (2.10, 2.35, "💎 Altın Oran Bandı (2.10 - 2.34)", "🎯 0 Merkez Oyna (%52 İsabet, +%15.6 ROI; 2.Fav Üstteyse +%39.9 ROI)"),
        (2.35, 99.0, "⚖️ Yüksek Oran Bandı (>= 2.35)", "🎯 2.35-2.79 Arası 0 Merkez (+%21.6 ROI) veya 1.&2. Favori İkili Koruma"),
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
                    "35. Dk 1. Favori Oran Bandı": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 0 (Merkez) İsabet & ROI": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | ROI: %{roi_0:+.1f}",
                    "📉 -1 Alt Barem": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ -2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 +1/+2 Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne Yapmalı? (Kural Özeti)": advice,
                }
            )

    # 2. 2nd Favorite Direction Breakdown (-1 Lower vs +1 Upper)
    sec_dir_brackets = []
    for cond_fn, lbl, advice in [
        (lambda x: x["sec_dir"] >= 1 and x["odd_0"] >= 2.10, "🔥 2. Favori Üstte (+1) VE Oran >= 2.10", "🎯 SÜPER SİNYAL: 0 Merkez %61.1 Tutar (+%39.9 ROI), -1 Alt Hiç Gelmez (%0)!"),
        (lambda x: x["sec_dir"] >= 1 and x["odd_0"] < 2.10, "📈 2. Favori Üstte (+1) VE Oran < 2.10", "🎯 0 Merkez (%55.6 İsabet) veya 0 & +1 İkili Koruma"),
        (lambda x: x["sec_dir"] <= -1 and x["odd_0"] < 2.10, "📉 2. Favori Altta (-1) VE Oran < 2.10", "📉 0 Merkez Zarar Eder (-%16.6) ➔ -1 Alt Barem Oyna (+%16.6 ROI)!"),
        (lambda x: x["sec_dir"] <= -1 and x["odd_0"] >= 2.10, "⚖️ 2. Favori Altta (-1) VE Oran >= 2.10", "🛡️ Yarı Yarıya 0 Merkez ve -1 Alt Gelir (+1 Üst %0!) ➔ 0 & -1 İkili Koruma"),
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
                    "2. Favorinin Konumu (Piyasa Eğilimi)": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 0 Merkez (ROI)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "📉 -1 Alt (ROI)": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s}) | %{roi_m1:+.1f}",
                    "❄️ -2 Alt": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 +1/+2 Üst": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne İşe Yarar / Ne Oynanır?": advice,
                }
            )

    # 3. Score Margin at 35m Breakdown (Clutch vs Blowout)
    margin_brackets = []
    for lo_m, hi_m, lbl, advice in [
        (0, 6, "⛔ 0 - 5 Sayı Fark (Baş Başa / Uzatma & Faul Tuzağı)", "⛔ PAS GEÇ! Son 13 dk'da ort. 59 sayı atılır, %33 üst bareme patlar (Zarar Bölgesi)"),
        (6, 12, "⚖️ 6 - 11 Sayı Fark (Normal Rekabet)", "🛡️ Standart Tempo (~53 sayı) ➔ 1. ve 2. Favori İkili Koruma veya v3 Kuralı"),
        (12, 18, "🛡️ 12 - 17 Sayı Fark (Açık Fark)", "🛡️ Uzatma Riski Yok ➔ 0 Merkez veya 0 & -1 İkili Koruma"),
        (18, 999, "🧊 18+ Sayı Fark (Blowout / Rölanti Freni)", "🔥 GÜÇLÜ MERKEZ: Son bölümde ort. 47.4 sayı atılır; 0 Merkez %60 Tutar (+%27.5 ROI)!"),
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
                    "35. Dk Skor Farkı Bandı": lbl,
                    "Maç Sayısı": n_s,
                    "Son 13 Dk Ort. Atılan Sayı": avg_rem,
                    "🎯 0 Merkez (ROI)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "📉 -1/-2 Alt Barem": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "📈 +1/+2 Üst Barem": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne İşe Yarar / Ne Oynanır?": advice,
                }
            )

    # 4. Barem Interior Edge Position (Score at 35m + 53.3 vs Barem Midpoint)
    edge_brackets = []
    sub_35only = [x for x in raw_records if x["min35"] == 35]
    for cond_fn, lbl, advice in [
        (lambda x: x["edge_offset"] < -3.0, "📐 Alt Kenarda (35.Dk Skor + 53.3 < Barem Ortası - 3)", "📉 DİKKAT: Maçların %55.6'sı -1/-2 Alt Bareme Düşer (0 Merkez -%24.6 Zarar Eder!)"),
        (lambda x: -3.0 <= x["edge_offset"] <= 3.0, "🎯 Tam Merkezde (35.Dk Skor + 53.3 ≈ Barem Ortası)", "💎 GÜVENLİ MERKEZ: Maçların %55.6'sı Tam 0 Merkezde Kalır (+%17.8 ROI)"),
        (lambda x: x["edge_offset"] > 3.0, "📐 Üst Kenarda (35.Dk Skor + 53.3 > Barem Ortası + 3)", "📈 GÜÇLÜ MERKEZ / ÜST: %56.3 Merkez (+%22.2 ROI), %31.3 +1 Üst Bareme Çıkar"),
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
                    "35. Dk Skorun Barem İçindeki Yeri": lbl,
                    "Maç Sayısı": n_s,
                    "📉 -1/-2 Alta Kayma": f"%{c_down/n_s*100:.1f} ({c_down}/{n_s})",
                    "🎯 0 Merkezde Kalma (ROI)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s}) | %{roi_0:+.1f}",
                    "📈 +1/+2 Üste Çıkma": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Ne İşe Yarar?": advice,
                }
            )

    # 5. Minute-by-Minute Optimal Entry Window
    minute_windows = []
    for m_target, m_lbl, m_advice in [
        (18, "⚡ 18. Dakika (2. Çeyrek Ortası — Erken Değer)", "💎 Erken girişte en kârlı pencere: Tekli 0 (+%16.8 ROI) ve 1.&2.Fav İkili (+%14.3 ROI)"),
        (24, "⏸️ 24. Dakika (Devre Arası)", "⚖️ Tekli 0 Merkez pozitif (+%11.4 ROI), devre arası tempo analizi için uygun"),
        (30, "🕒 30. Dakika (3. Çeyrek Ortası)", "⚖️ Oranlar oturmaya başlar (%45.9 Merkez isabeti, +%5.8 ROI)"),
        (33, "🚨 32 - 33. Dakika (Q3 Sonu Öncesi — TUZAK PENCERE)", "⛔ BAHİS ALMA! Barem geçiş anı olduğu için en yüksek yanılma ve zarar (-%17.1 ROI) buradadır"),
        (35, "🔥 34 - 35. Dakika (Kapanış — EN YÜKSEK İSABET)", "🎯 Ana Karar Anı: Tekli 0 (%50.8 İsabet), 1.&2.Fav İkili Koruma (%83.6 İsabet, +%1.5 ROI)"),
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
                    "Bahis Giriş Dakikası": m_lbl,
                    "Maç Sayısı": m_n,
                    "🎯 1. Favori İsabet & ROI": f"%{m_hits0/m_n*100:.1f} ({m_hits0}/{m_n}) | ROI: %{(m_ret0-m_n)/m_n*100:+.1f}",
                    "🛡️ 1. & 2. Favori İkili Koruma": f"%{m_hits_d/m_n*100:.1f} ({m_hits_d}/{m_n}) | ROI: %{(m_ret_d-m_n)/m_n*100:+.1f}",
                    "💡 Zamanlama Tavsiyesi": m_advice,
                }
            )

    # 6. Drift (Q1 -> 35m Barem Migration) Breakdown
    drift_brackets = []
    for cond_fn, lbl, advice in [
        (lambda d: d >= 1, "📈 Barem Yükseldi (Q1 ➔ 35.Dk +1/+3)", "🎯 Maç Hızlandı: 0 Merkez %58 Tutar (+%26.1 ROI), Alta Nadiren Düşer"),
        (lambda d: d == 0, "⚖️ Barem Sabit Kaldı (Q1 ➔ 35.Dk 0)", "📉 Maçların %43'ü -1/-2 Alta Kayar ➔ 0 & -1 İkili Koruma Önerilir"),
        (lambda d: d == -1, "📉 Barem 1 Kademe Düştü (Q1 ➔ 35.Dk -1)", "⚖️ %43 Merkezde Kalır, %43 Tekrar +1 Üste Tepki Verir"),
        (lambda d: d <= -2, "❄️ Barem Çöktü (Q1 ➔ 35.Dk -2/-3)", "🎯 Fazla Düştüğü İçin Yeni Merkezde (0) Dengelenir (+%14.1 ROI)"),
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
                    "Maç İçi Barem Göçü (Q1 ➔ 35. Dk)": lbl,
                    "Maç Sayısı": n_s,
                    "🎯 0 (Merkez)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s})",
                    "📉 -1 Alt Barem": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ -2 Alt Barem": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 +1/+2 Üst": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "💡 Davranış Özeti": advice,
                }
            )

    # 7. Sequential Match-to-Match Transitions (Real Tracked Chain)
    transitions = []
    for cond_fn, lbl in [
        (lambda s: s == 0, "🎯 Önceki Maç 0 (Merkez) Bittiğinde ➔ Sıradaki Maç"),
        (lambda s: s <= -1, "📉 Önceki Maç -1 / -2 (Alt Barem) Bittiğinde ➔ Sıradaki Maç"),
        (lambda s: s >= 1, "📈 Önceki Maç +1 / +2 (Üst Barem) Bittiğinde ➔ Sıradaki Maç"),
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
                    "Ardışık Maç Geçiş Durumu": lbl,
                    "Örneklem": n_s,
                    "🎯 Sıradaki 0 (Merkez)": f"%{c_0/n_s*100:.1f} ({c_0}/{n_s})",
                    "📉 Sıradaki -1 Alt": f"%{c_m1/n_s*100:.1f} ({c_m1}/{n_s})",
                    "❄️ Sıradaki -2 Alt": f"%{c_m2/n_s*100:.1f} ({c_m2}/{n_s})",
                    "📈 Sıradaki +1/+2 Üst": f"%{c_up/n_s*100:.1f} ({c_up}/{n_s})",
                    "🛡️ (0 veya -1) Oranı": f"%{(c_0+c_m1)/n_s*100:.1f} ({c_0+c_m1}/{n_s})",
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
        "minute_windows": minute_windows,
        "drift_brackets": drift_brackets,
        "transitions": transitions,
    }
