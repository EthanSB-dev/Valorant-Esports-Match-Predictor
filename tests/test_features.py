"""
Unit tests for analysis/features.py.

These tests exist specifically to lock in leakage-safe behavior: every
assertion below was hand-calculated first, then checked against the
function's actual output. If a future change to features.py breaks the
leakage-safety guarantee, these tests are what catch it.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

from features import build_match_features  # noqa: E402

MATCH_COLUMNS = [
    "match_id", "end_at", "team_a_id", "team_a_name", "team_a_score",
    "team_b_id", "team_b_name", "team_b_score", "winner_id",
    "tournament_id", "match_status",
]


def test_first_ever_match_has_no_prior_stats():
    """A team's very first match has no history to draw on — every
    feature should be NaN, not zero (zero would falsely imply "known
    to always lose")."""
    matches = pd.DataFrame(
        [(1, "2025-01-01", 100, "A", 2, 200, "B", 0, 100, 1, "finished")],
        columns=MATCH_COLUMNS,
    )
    result = build_match_features(matches)

    assert pd.isna(result.loc[0, "team_a_rolling_winrate"])
    assert pd.isna(result.loc[0, "team_a_days_rest"])
    assert pd.isna(result.loc[0, "h2h_team_a_winrate"])
    assert result.loc[0, "team_a_matches_played"] == 0


def test_rolling_winrate_excludes_current_match():
    """The core leakage check: team A's rolling win rate going into
    match 3 must reflect only matches 1 and 2 (one win, one loss —
    average 0.5), never match 3's own outcome."""
    matches = pd.DataFrame(
        [
            (1, "2025-01-01", 100, "A", 2, 200, "B", 0, 100, 1, "finished"),  # A wins
            (2, "2025-01-05", 100, "A", 0, 300, "C", 2, 300, 1, "finished"),  # A loses
            (3, "2025-01-10", 100, "A", 2, 200, "B", 1, 100, 1, "finished"),  # A wins
        ],
        columns=MATCH_COLUMNS,
    )
    result = build_match_features(matches)

    match_3 = result[result["match_id"] == 3].iloc[0]
    assert match_3["team_a_rolling_winrate"] == 0.5
    assert match_3["team_a_matches_played"] == 2


def test_days_rest_computed_correctly():
    matches = pd.DataFrame(
        [
            (1, "2025-01-01", 100, "A", 2, 200, "B", 0, 100, 1, "finished"),
            (2, "2025-01-05", 100, "A", 0, 300, "C", 2, 300, 1, "finished"),
        ],
        columns=MATCH_COLUMNS,
    )
    result = build_match_features(matches)

    match_2 = result[result["match_id"] == 2].iloc[0]
    assert match_2["team_a_days_rest"] == 4


def test_head_to_head_uses_only_prior_meetings_between_the_pair():
    """A's h2h win rate going into their second meeting with B should
    reflect only their first meeting, and must not be contaminated by
    A's matches against other opponents in between."""
    matches = pd.DataFrame(
        [
            (1, "2025-01-01", 100, "A", 2, 200, "B", 0, 100, 1, "finished"),  # A beats B
            (2, "2025-01-05", 100, "A", 0, 300, "C", 2, 300, 1, "finished"),  # unrelated match
            (3, "2025-01-10", 100, "A", 2, 200, "B", 1, 100, 1, "finished"),  # A beats B again
        ],
        columns=MATCH_COLUMNS,
    )
    result = build_match_features(matches)

    match_1 = result[result["match_id"] == 1].iloc[0]
    match_3 = result[result["match_id"] == 3].iloc[0]
    assert pd.isna(match_1["h2h_team_a_winrate"])  # no prior meeting yet
    assert match_3["h2h_team_a_winrate"] == 1.0     # won their only prior meeting


def test_rolling_window_caps_at_five_matches():
    """Team A's 7th match should reflect only matches 2-6 (the most
    recent 5), not an all-time average back to match 1."""
    rows = []
    outcomes = ["L", "L", "L", "W", "W", "W", "W"]
    for i, outcome in enumerate(outcomes):
        winner = 200 if outcome == "L" else 100
        rows.append((i + 1, f"2025-01-{i+1:02d}", 100, "A", 0, 200, "B", 2, winner, 1, "finished"))
    matches = pd.DataFrame(rows, columns=MATCH_COLUMNS)

    result = build_match_features(matches)

    match_7 = result[result["match_id"] == 7].iloc[0]
    # Matches 2-6 were L, L, W, W, W -> 3 wins out of 5
    assert match_7["team_a_rolling_winrate"] == 0.6


def test_unfinished_matches_are_excluded():
    """A match still in progress has no real outcome yet and must not
    be used to build features or appear in the output."""
    matches = pd.DataFrame(
        [
            (1, "2025-01-01", 100, "A", 2, 200, "B", 0, 100, 1, "finished"),
            (2, "2025-01-05", 100, "A", 1, 200, "B", 0, None, 1, "running"),
        ],
        columns=MATCH_COLUMNS,
    )
    result = build_match_features(matches)

    assert len(result) == 1
    assert result.iloc[0]["match_id"] == 1