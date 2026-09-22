"""
Feature engineering for VCT match outcome prediction.

The single most important rule followed throughout this file: every feature
for a given match is computed using ONLY matches that happened strictly
before it (based on `end_at`). This is what makes the features usable for
real prediction — a model can only use information that would actually be
available before the match is played. Violating this ("leakage") produces
a model that looks great in testing and is useless in production, because
in reality you can never know a team's *future* win rate when predicting
today's match.

Input: a DataFrame shaped like analytics.fct_matches, with at least these
columns (matching the dbt mart's actual output):
    match_id, end_at, team_a_id, team_a_name, team_a_score,
    team_b_id, team_b_name, team_b_score, winner_id, tournament_id,
    match_status

Output: the same matches, each with added feature columns, ready to hand
to a model.
"""
import pandas as pd

ROLLING_WINDOW = 5  # number of past matches used for each team's "recent form"


def _build_team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes match-level data (one row per match, two teams as columns)
    into team-level data (one row per team per match they played in).

    This "long" format is necessary because a team's win rate depends on
    ALL of their matches, regardless of whether they were "team_a" or
    "team_b" in any given match record.
    """
    a_side = matches[
        ["match_id", "end_at", "team_a_id", "team_b_id", "winner_id"]
    ].rename(columns={"team_a_id": "team_id", "team_b_id": "opponent_id"})
    a_side["won"] = (a_side["winner_id"] == a_side["team_id"]).astype(int)

    b_side = matches[
        ["match_id", "end_at", "team_b_id", "team_a_id", "winner_id"]
    ].rename(columns={"team_b_id": "team_id", "team_a_id": "opponent_id"})
    b_side["won"] = (b_side["winner_id"] == b_side["team_id"]).astype(int)

    history = pd.concat([a_side, b_side], ignore_index=True)
    history = history.sort_values(["team_id", "end_at"]).reset_index(drop=True)
    return history


def _add_leakage_safe_team_stats(history: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-match row, computes stats using only that team's PRIOR
    matches. The `.shift(1)` before every rolling/expanding calculation is
    what enforces this: it excludes the current row before computing
    anything, so a match's features never include its own outcome.
    """
    grouped = history.groupby("team_id", group_keys=False)

    history["rolling_winrate"] = grouped["won"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    history["career_matches_played"] = grouped.cumcount()  # count of PRIOR matches
    history["days_since_last_match"] = grouped["end_at"].apply(
        lambda s: s.diff().dt.days
    )
    return history


def _add_head_to_head_feature(matches: pd.DataFrame) -> pd.Series:
    """
    For each match, computes team_a's historical win rate against this
    specific opponent, using only matches between this pair that happened
    before the current one. Returns a Series aligned to `matches.index`.
    """
    matches = matches.sort_values("end_at")
    h2h_winrate = pd.Series(index=matches.index, dtype=float)

    # pair_key is order-independent so "A vs B" and "B vs A" share history
    pair_key = matches.apply(
        lambda r: tuple(sorted([r["team_a_id"], r["team_b_id"]])), axis=1
    )

    seen_wins = {}   # pair_key -> {team_id: prior_win_count}
    seen_total = {}  # pair_key -> prior_match_count

    for idx, row in matches.iterrows():
        key = pair_key[idx]
        total = seen_total.get(key, 0)
        wins_for_a = seen_wins.get(key, {}).get(row["team_a_id"], 0)
        h2h_winrate[idx] = (wins_for_a / total) if total > 0 else float("nan")

        seen_total[key] = total + 1
        wins_dict = seen_wins.setdefault(key, {})
        wins_dict[row["winner_id"]] = wins_dict.get(row["winner_id"], 0) + 1

    return h2h_winrate.reindex(matches.index)


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Takes match-level data and returns it with added,
    leakage-safe feature columns:

        team_a_rolling_winrate, team_b_rolling_winrate  — recent form
        team_a_days_rest, team_b_days_rest              — fatigue/rest proxy
        team_a_matches_played, team_b_matches_played    — experience so far
        h2h_team_a_winrate                              — pairwise history
        team_a_won                                      — the prediction target
    """
    matches = matches[matches["match_status"] == "finished"].copy()
    matches["end_at"] = pd.to_datetime(matches["end_at"])
    matches = matches.sort_values("end_at").reset_index(drop=True)

    history = _build_team_match_history(matches)
    history = _add_leakage_safe_team_stats(history)

    # Pull each team's pre-match stats back onto the match row, keyed by
    # (match_id, team_id) so team_a and team_b each get their own columns.
    a_stats = history.rename(
        columns={
            "rolling_winrate": "team_a_rolling_winrate",
            "career_matches_played": "team_a_matches_played",
            "days_since_last_match": "team_a_days_rest",
        }
    )[["match_id", "team_id", "team_a_rolling_winrate", "team_a_matches_played", "team_a_days_rest"]]

    b_stats = history.rename(
        columns={
            "rolling_winrate": "team_b_rolling_winrate",
            "career_matches_played": "team_b_matches_played",
            "days_since_last_match": "team_b_days_rest",
        }
    )[["match_id", "team_id", "team_b_rolling_winrate", "team_b_matches_played", "team_b_days_rest"]]

    features = matches.merge(
        a_stats, left_on=["match_id", "team_a_id"], right_on=["match_id", "team_id"], how="left"
    ).drop(columns="team_id")

    features = features.merge(
        b_stats, left_on=["match_id", "team_b_id"], right_on=["match_id", "team_id"], how="left"
    ).drop(columns="team_id")

    features["h2h_team_a_winrate"] = _add_head_to_head_feature(matches)
    features["team_a_won"] = (features["winner_id"] == features["team_a_id"]).astype(int)

    return features

def get_current_team_stats(finished_matches: pd.DataFrame) -> pd.DataFrame:
    """
    For predicting an UPCOMING match (not for building historical training
    data). Returns each team's stats as of right now — their rolling win
    rate over their last (up to) 5 finished matches, total matches played,
    and the date of their most recent match.

    This is intentionally different from build_match_features: that
    function excludes each match's own outcome via .shift(1), because it's
    building features for matches that already happened. Here there is no
    "current match" to exclude — we want each team's full known history
    up to today, to predict a match that hasn't been played yet.
    """
    finished_matches = finished_matches[finished_matches["match_status"] == "finished"].copy()
    finished_matches["end_at"] = pd.to_datetime(finished_matches["end_at"])

    history = _build_team_match_history(finished_matches).sort_values(["team_id", "end_at"])
    grouped = history.groupby("team_id")

    stats = grouped.agg(
        matches_played=("won", "count"),
        last_match_date=("end_at", "max"),
    ).reset_index()

    rolling = grouped["won"].apply(lambda s: s.tail(ROLLING_WINDOW).mean())
    rolling = rolling.reset_index(name="rolling_winrate")

    return stats.merge(rolling, on="team_id")


def get_h2h_winrate(team_a_id, team_b_id, finished_matches: pd.DataFrame) -> float:
    """
    team_a_id's historical win rate against team_b_id, across every
    finished meeting between them to date. Returns NaN if they have never
    played each other — an honest "unknown," not a misleading 0 or 0.5.
    """
    finished_matches = finished_matches[finished_matches["match_status"] == "finished"]
    is_this_pair = (
        ((finished_matches["team_a_id"] == team_a_id) & (finished_matches["team_b_id"] == team_b_id))
        | ((finished_matches["team_a_id"] == team_b_id) & (finished_matches["team_b_id"] == team_a_id))
    )
    meetings = finished_matches[is_this_pair]
    if len(meetings) == 0:
        return float("nan")
    return (meetings["winner_id"] == team_a_id).sum() / len(meetings)