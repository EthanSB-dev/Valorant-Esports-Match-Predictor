# Match Outcome Prediction — Analysis

An experimental extension of the VCT pipeline: using the data already
collected and modeled by dbt to explore whether match outcomes can be
predicted from team history, and to be honest about how well that
actually works.

## Approach

1. **Feature engineering** (`features.py`) — builds leakage-safe features
   for each match: each team's rolling win rate over their last 5
   matches, days of rest since their last match, matches played, and
   head-to-head win rate against the specific opponent. "Leakage-safe"
   means every feature is computed using only matches that happened
   strictly before the one being predicted — never the match's own
   outcome or any future data. This is validated by unit tests in
   `tests/test_features.py`.

2. **Exploratory analysis** (`eda.ipynb`) — checks whether these features
   actually relate to who wins before attempting to model anything.

3. **Modeling** (`train_model.py`) — trains and evaluates a Logistic
   Regression and a Random Forest, using:
   - A **de-biasing step**: which team is labeled "team_a" vs "team_b"
     in the raw data is arbitrary, not meaningful — but team_a won
     ~56% of matches in the raw data, a labeling artifact a model could
     exploit instead of learning anything real. Roughly half of all
     matches have their sides randomly swapped before training to
     remove this.
   - A **time-based train/test split**: since this is time-series data,
     splitting randomly would let the model "see the future" during
     training. The model is trained on the earliest ~80% of matches and
     evaluated only on the most recent ~20%, honestly simulating how it
     would actually be used.

## Results

| Metric | Logistic Regression | Random Forest | Baseline |
|---|---:|---:|---:|
| Accuracy | 0.524 | 0.524 | 0.524 |
| ROC-AUC | 0.588 | 0.556 | 0.500 |

The baseline is "always guess the more common outcome" after de-biasing.

## Honest conclusion

**These features alone are not sufficient to reliably predict VCT match
outcomes.** Both models matched the accuracy baseline exactly, meaning
neither improved on the simplest possible guess. The ROC-AUC scores
(above 0.5 for both models) show the features do carry *some* real
signal — recent form and head-to-head history are not meaningless, and
the model's learned coefficients point in intuitively correct
directions (a team's own recent form pushes toward them winning; a
stronger opponent's recent form pushes against it) — but that signal is
too weak on its own to translate into meaningfully better predictions.

This is a reasonable outcome, not a failed experiment. Professional VCT
matches are competitive and volatile, and several real factors this
analysis doesn't capture likely matter more than what's modeled here:

- **Roster changes.** VCT teams change players between tournaments;
  a team's "recent form" from two months ago may reflect a materially
  different roster than the one playing today.
- **Map-level performance.** A team's overall win rate averages across
  maps with very different team-specific strengths and weaknesses.
- **Tournament context.** A Kickoff match and a Champions grand final
  are different competitive situations even between the same two teams,
  and this model treats them identically.

A stronger model would likely need player-level and map-level data,
neither of which the current pipeline collects.

## Reproducing this analysis

```bash
cd analysis
python3.12 -m venv venv          # a Python version with mature Jupyter/ML wheel support
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook eda.ipynb       # run top to bottom; exports data/match_features.csv
python train_model.py            # reads that CSV, trains and evaluates both models
```