"""
Trains and evaluates models predicting match winners from the features
built by analysis/features.py.

Two things this script does carefully, and why:

1. De-biases the team_a/team_b label. Which team is listed as "team_a"
   vs "team_b" in the raw data is arbitrary (an artifact of the source
   API), not meaningful. EDA showed team_a won ~56% of matches overall —
   a model could hit 56% accuracy just by always guessing "team_a wins,"
   without learning anything real. To prevent that, roughly half of all
   matches have their team_a/team_b sides randomly swapped before
   training, forcing the model to learn from the actual feature values
   rather than an arbitrary labeling quirk.

2. Splits train/test BY TIME, not randomly. This is time-series data:
   using a random split could put a March match in training and a
   January match in testing, which is backwards from how the model
   would ever really be used (predicting a future match from past
   data). A chronological split — train on the earlier matches, test
   on the most recent ones — is the only split that honestly simulates
   real deployment.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "team_a_rolling_winrate", "team_b_rolling_winrate",
    "team_a_matches_played", "team_b_matches_played",
    "team_a_days_rest", "team_b_days_rest",
    "h2h_team_a_winrate",
]


def debias_team_labels(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Randomly swaps team_a/team_b (and every feature that depends on
    which side is "A") for ~50% of matches, so the model can't exploit
    whatever arbitrary pattern caused team_a to win more often in the
    raw data than team_b.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    swap_mask = rng.random(len(df)) < 0.5

    swapped = df.loc[swap_mask].copy()
    swapped["team_a_rolling_winrate"], swapped["team_b_rolling_winrate"] = (
        df.loc[swap_mask, "team_b_rolling_winrate"], df.loc[swap_mask, "team_a_rolling_winrate"]
    )
    swapped["team_a_matches_played"], swapped["team_b_matches_played"] = (
        df.loc[swap_mask, "team_b_matches_played"], df.loc[swap_mask, "team_a_matches_played"]
    )
    swapped["team_a_days_rest"], swapped["team_b_days_rest"] = (
        df.loc[swap_mask, "team_b_days_rest"], df.loc[swap_mask, "team_a_days_rest"]
    )
    # h2h_team_a_winrate and its opponent's win rate always sum to 1
    # (every match has exactly one winner), so flipping sides is just:
    swapped["h2h_team_a_winrate"] = 1 - df.loc[swap_mask, "h2h_team_a_winrate"]
    swapped["team_a_won"] = 1 - df.loc[swap_mask, "team_a_won"]

    df.loc[swap_mask] = swapped
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Sorts by match date and takes the most recent `test_fraction` of
    matches as the test set. This is the only honest way to evaluate a
    model meant to predict FUTURE matches from PAST ones.
    """
    df = df.sort_values("end_at").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def main():
    df = pd.read_csv("data/match_features.csv", parse_dates=["end_at"])
    print(f"Loaded {len(df)} matches")

    # Drop matches missing any feature — these are teams' first-ever
    # matches, where no prior history exists yet (by design, see
    # features.py). They can't be used for training or evaluation.
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    print(f"Dropped {before - len(df)} matches with incomplete history "
          f"(a team's first match, or first meeting for h2h)")
    print(f"{len(df)} matches remain")

    df = debias_team_labels(df)

    train_df, test_df = time_based_split(df)
    print(f"\nTrain: {len(train_df)} matches ({train_df['end_at'].min().date()} to {train_df['end_at'].max().date()})")
    print(f"Test:  {len(test_df)} matches ({test_df['end_at'].min().date()} to {test_df['end_at'].max().date()})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["team_a_won"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["team_a_won"]

    # Baseline: what accuracy would "always guess the majority class"
    # get? A real model needs to beat THIS, not 50%, since after
    # de-biasing the classes should be roughly balanced anyway — this
    # baseline is the honest bar to clear.
    baseline_guess = y_train.mode()[0]
    baseline_accuracy = (y_test == baseline_guess).mean()
    print(f"\nBaseline (always guess {baseline_guess}): {baseline_accuracy:.3f} accuracy")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_SEED),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
        print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
        print(f"ROC-AUC:  {roc_auc_score(y_test, probs):.3f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, preds))
        print("\nClassification report:")
        print(classification_report(y_test, preds))

        if name == "Logistic Regression":
            print("Feature coefficients (higher = pushes toward team_a winning):")
            for feat, coef in sorted(zip(FEATURE_COLUMNS, model.coef_[0]), key=lambda x: -abs(x[1])):
                print(f"  {feat}: {coef:+.3f}")
        else:
            print("Feature importances:")
            for feat, imp in sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]):
                print(f"  {feat}: {imp:.3f}")


if __name__ == "__main__":
    main()