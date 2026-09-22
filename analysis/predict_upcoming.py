"""
Predicts outcomes for upcoming, not-yet-played VCT matches (e.g. Champions
2026) using the model trained by train_model.py.

IMPORTANT — read this before trusting any output here: the model this
script uses was evaluated at ROC-AUC ~0.59 and did NOT beat a simple
baseline on accuracy (see analysis/README.md for the full, honest
write-up). What follows are genuine outputs of a real, tested pipeline —
not reliable forecasts. Treat every prediction here as "this is what a
simple statistical model says," not "this is what will happen."

This script does NOT reuse build_match_features (which is for building
training data from matches that already happened, and deliberately
excludes each match's own outcome). Instead it uses get_current_team_stats
and get_h2h_winrate — functions built specifically to answer "what does
each team's form look like RIGHT NOW, for predicting a match that hasn't
been played yet."
"""
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from joblib import load
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(__file__))
from features import get_current_team_stats, get_h2h_winrate

load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# Adjust this if your tournament is named differently in your data —
# the script prints what it actually finds so you can tell whether this
# filter needs changing.
TOURNAMENT_NAME_FILTER = "champion"


def get_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", 5432)
    dbname = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")


def load_all_matches(engine) -> pd.DataFrame:
    query = """
        SELECT m.*, t.tournament_name, t.serie_full_name
        FROM analytics.fct_matches m
        LEFT JOIN analytics.dim_tournaments t ON m.tournament_id = t.tournament_id
    """
    # Bypasses pandas' SQLAlchemy-connectable detection entirely, which
    # behaves inconsistently across SQLAlchemy versions (this environment
    # runs SQLAlchemy 1.4, pinned by Airflow itself, vs 2.0 locally).
    # raw_connection() hands pandas the actual underlying psycopg2
    # connection, which always has the .cursor() method pandas' fallback
    # code path needs, regardless of SQLAlchemy version.
    raw_conn = engine.raw_connection()
    try:
        df = pd.read_sql(query, raw_conn)
    finally:
        raw_conn.close()

    for col in ("end_at", "begin_at", "scheduled_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def get_match_date(row) -> pd.Timestamp:
    """Upcoming matches have no end_at yet, so use whichever of
    scheduled_at / begin_at is available instead."""
    for col in ("scheduled_at", "begin_at", "end_at"):
        if col in row and pd.notna(row.get(col)):
            return row[col]
    return pd.NaT


def main(engine=None):
    if engine is None:
        engine = get_engine()
    all_matches = load_all_matches(engine)
    finished = all_matches[all_matches["match_status"] == "finished"].copy()
    upcoming = all_matches[all_matches["match_status"] != "finished"].copy()

    print(f"{len(finished)} finished matches, {len(upcoming)} not-finished matches in the database\n")

    if upcoming.empty:
        print("No upcoming matches found at all. Nothing to predict yet — "
              "check that the pipeline has run recently and the tournament "
              "has been scheduled on PandaScore.")
        return

    # Show what's actually there, so a wrong filter is obvious rather than
    # silently producing zero results. serie_full_name is the overall event
    # name (e.g. "VCT Champions 2026") — league_name is just "VCT" for
    # everything, and tournament_name is the specific stage (e.g. "Group A").
    if "serie_full_name" in upcoming.columns:
        print("Series (events) among upcoming matches:")
        print(upcoming["serie_full_name"].value_counts().to_string())
        print()

    if "serie_full_name" in upcoming.columns:
        target = upcoming[upcoming["serie_full_name"].str.contains(TOURNAMENT_NAME_FILTER, case=False, na=False)]
    else:
        target = upcoming

    if target.empty:
        print(f"No upcoming matches matched the filter '{TOURNAMENT_NAME_FILTER}'. "
              f"Check the list above and adjust TOURNAMENT_NAME_FILTER at the top of this script.")
        return

    print(f"Found {len(target)} upcoming match(es) matching '{TOURNAMENT_NAME_FILTER}'\n")

    saved = load(MODEL_PATH)
    model, feature_columns = saved["model"], saved["feature_columns"]
    current_stats = get_current_team_stats(finished).set_index("team_id")

    print("=" * 70)
    print("PREDICTIONS — read the disclaimer at the top of this file first.")
    print("This model's real, measured accuracy did not beat a simple")
    print("baseline (see analysis/README.md). Treat these as illustrative")
    print("model output, not genuine forecasts.")
    print("=" * 70)

    bracket_tbd_count = 0

    for _, match in target.sort_values(by=list(target.columns[:1])).iterrows():
        a_id, b_id = match["team_a_id"], match["team_b_id"]
        a_name, b_name = match["team_a_name"], match["team_b_name"]
        match_date = get_match_date(match)

        # Playoff bracket slots ("Winner of Group A vs ...") have no teams
        # assigned yet until earlier rounds finish — not missing data, just
        # genuinely not determined yet. Counted and summarized once at the
        # end instead of printed 20+ times.
        if pd.isna(a_id) or pd.isna(b_id):
            bracket_tbd_count += 1
            continue

        if a_id not in current_stats.index or b_id not in current_stats.index:
            print(f"\n{a_name} vs {b_name}: skipped — no prior match history "
                  f"for one or both teams in this database yet.")
            continue

        a_stats = current_stats.loc[a_id]
        b_stats = current_stats.loc[b_id]
        h2h = get_h2h_winrate(a_id, b_id, finished)

        # NaN here genuinely means "these two teams have never played each
        # other under this league before" — very plausible for cross-region
        # matchups that only happen at Masters/Champions. Rather than skip
        # the whole match over this, fall back to a neutral 0.5 (an even,
        # uninformative prior) and say so explicitly in the output — this
        # is an honest assumption, not a hidden one.
        h2h_is_unknown = pd.isna(h2h)
        if h2h_is_unknown:
            h2h = 0.5

        a_rest = (match_date - a_stats["last_match_date"]).days if pd.notna(match_date) else float("nan")
        b_rest = (match_date - b_stats["last_match_date"]).days if pd.notna(match_date) else float("nan")

        row = pd.DataFrame([{
            "team_a_rolling_winrate": a_stats["rolling_winrate"],
            "team_b_rolling_winrate": b_stats["rolling_winrate"],
            "team_a_matches_played": a_stats["matches_played"],
            "team_b_matches_played": b_stats["matches_played"],
            "team_a_days_rest": a_rest,
            "team_b_days_rest": b_rest,
            "h2h_team_a_winrate": h2h,
        }])[feature_columns]

        if row.isna().any(axis=None):
            print(f"\n{a_name} vs {b_name}: skipped — missing rest-day data "
                  f"(the match's scheduled date could not be determined).")
            continue

        prob_a_wins = model.predict_proba(row)[0][1]
        predicted_winner = a_name if prob_a_wins >= 0.5 else b_name
        confidence = max(prob_a_wins, 1 - prob_a_wins)

        print(f"\n{a_name} vs {b_name}")
        if h2h_is_unknown:
            print(f"  (first meeting between these two teams in this dataset — head-to-head assumed even)")
        print(f"  Predicted winner: {predicted_winner} ({confidence:.0%} model confidence)")
        print(f"  P({a_name} wins) = {prob_a_wins:.2f}   P({b_name} wins) = {1 - prob_a_wins:.2f}")
        print(f"  Recent form: {a_name} {a_stats['rolling_winrate']:.2f} vs {b_name} {b_stats['rolling_winrate']:.2f}")

    if bracket_tbd_count:
        print(f"\n({bracket_tbd_count} playoff bracket slot(s) skipped — teams not yet "
              f"determined; they depend on Group Stage results still to come.)")


if __name__ == "__main__":
    main()