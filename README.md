# Valorant Esports Match Predictor

[![CI](https://github.com/EthanSB-dev/Valorant-Esports-Match-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/EthanSB-dev/Valorant-Esports-Match-Predictor/actions/workflows/ci.yml)

**Can recent form and head-to-head history predict who wins a professional Valorant match?**

This is a data analytics and data science project: it asks a real question about Valorant Champions Tour (VCT) esports matches, explores the data honestly, builds a predictive model, and evaluates it rigorously — including reporting the result even when it isn't the exciting one. It also applies the model to real, currently-scheduled matches (VCT Champions 2026), producing live predictions on an automated schedule.

Underneath the analysis sits a small amount of data engineering, built specifically to support it: an automated pipeline that collects, cleans, and models the underlying match data so the analytics layer has something trustworthy to work with. That infrastructure is documented at the bottom of this README; the analysis is the main story.

**The short answer to the question above:** there's real, measurable signal in team form and history (ROC-AUC ≈ 0.59), but it isn't enough on its own to reliably beat a naive baseline — and the analysis below explains exactly why, rather than overselling the result.

## The Question, and the Approach

Predicting competitive match outcomes is a genuinely hard problem — that's part of what makes it an interesting analytics question. Rather than reach for the most complex model available, this project follows a deliberately rigorous data science process:

1. **Get clean, reliable data.** A small ELT pipeline (the data engineering piece) so the analysis has something trustworthy to work with (see [Supporting Data Infrastructure](#supporting-data-infrastructure)).
2. **Engineer features without cheating.** Every feature is built to be *leakage-safe* — computed only from information that would genuinely be available before a match is played (see [Feature Engineering](#feature-engineering)).
3. **Explore before modeling.** Check whether the data shows any real relationship at all before building anything predictive (see [Exploratory Analysis](#exploratory-analysis)).
4. **Evaluate honestly.** Use a train/test split that respects time, compare against a real baseline, and report the result even when it's not the exciting one (see [Modeling & Results](#modeling--results)).
5. **Put the model to work.** Apply it to real, currently-scheduled matches, not just historical ones (see [Live Predictions for Upcoming Matches](#live-predictions-for-upcoming-matches)).

## Feature Engineering

[`analysis/features.py`](analysis/features.py) builds, for every match:

- **Rolling win rate** — each team's win rate over their last 5 matches
- **Days of rest** — time since each team's previous match
- **Matches played** — each team's experience level at that point
- **Head-to-head win rate** — this specific pair's history against each other

The single rule enforced throughout: every feature for a given match uses **only matches that happened strictly before it**. A team's "rolling win rate" going into a match can never include that match's own outcome — this is what makes the features usable for genuine prediction rather than accidentally already knowing the answer. This is validated by unit tests in [`tests/test_features.py`](tests/test_features.py), which check the exact leakage-safety behavior against hand-calculated examples, and run automatically in CI on every change.

The same module also provides `get_current_team_stats()` and `get_h2h_winrate()` — functions built specifically for predicting matches that *haven't happened yet*, as opposed to building historical training data. They intentionally include a team's full history up to today, rather than excluding a "current match" that doesn't exist yet.

## Exploratory Analysis

Before modeling anything, [`analysis/eda.ipynb`](analysis/eda.ipynb) checks whether the engineered features show any real relationship to match outcomes.

**A finding worth calling out:** the raw data showed `team_a` winning ~56% of matches — not because "team_a" means anything (it's just whichever team the source API happened to list first), but as a labeling artifact a model could exploit instead of learning anything real. This got corrected before modeling (see below) — catching it here, rather than after training a suspiciously accurate model, is the actual point of doing EDA first.

Beyond that, the exploration showed a real but modest relationship: teams in better recent form and with stronger head-to-head records against their opponent do win somewhat more often, but the effect is weak — consistent with how close professional Valorant matches typically are.

## Modeling & Results

[`analysis/train_model.py`](analysis/train_model.py) trains and evaluates a Logistic Regression and a Random Forest, with two methodological choices that matter:

- **De-biasing the team labels.** Roughly half of all matches have their team_a/team_b sides randomly swapped before training, removing the labeling artifact found in EDA so the model has to learn from actual feature values, not label position.
- **A time-based train/test split.** The model trains on the earliest ~80% of matches and is evaluated only on the most recent ~20% — never randomly split, since a random split on time-series data would let the model implicitly "see the future" during training.

| Metric | Logistic Regression | Random Forest | Baseline |
|---|---:|---:|---:|
| Accuracy | 0.524 | 0.524 | 0.524 |
| ROC-AUC | 0.588 | 0.556 | 0.500 |

The baseline is "always guess the more common outcome" after de-biasing.

**Honest conclusion: these features alone are not sufficient to reliably predict VCT match outcomes.** Both models matched the accuracy baseline exactly, meaning neither improved on the simplest possible guess. The ROC-AUC scores (above 0.5 for both models) show the features do carry *some* real signal — recent form and head-to-head history are not meaningless, and the model's learned coefficients point in intuitively correct directions (a team's own recent form pushes toward them winning; a stronger opponent's recent form pushes against it) — but that signal is too weak on its own to translate into meaningfully better predictions.

This is a reasonable outcome, not a failed experiment. Professional VCT matches are competitive and volatile, and several real factors this analysis doesn't capture likely matter more than what's modeled here:

- **Roster changes.** VCT teams change players between tournaments; a team's "recent form" from two months ago may reflect a materially different roster than the one playing today.
- **Map-level performance.** A team's overall win rate averages across maps with very different team-specific strengths and weaknesses.
- **Tournament context.** A Kickoff match and a Champions grand final are different competitive situations even between the same two teams, and this model treats them identically.

The trained Logistic Regression model (the one that beat baseline on ROC-AUC) is saved to `analysis/model.joblib` and reused directly by the live prediction system below — training happens once, and both the evaluation numbers above and the live predictions come from the exact same model artifact.

The full write-up, including reproduction steps, is in [`analysis/README.md`](analysis/README.md).

## Live Predictions for Upcoming Matches

The same trained model runs against currently-scheduled matches, not just historical ones. [`extract/extract_upcoming_matches.py`](extract/extract_upcoming_matches.py) pulls scheduled (not-yet-played) matches from PandaScore, and [`analysis/predict_upcoming.py`](analysis/predict_upcoming.py) predicts each one using each team's real, current form — automatically, as part of the same Airflow DAG that runs every six hours.

This surfaced a real bug during development: the extraction client only ever called PandaScore's `/matches/past` endpoint, so upcoming matches never entered the pipeline at all until a second endpoint (`/matches/upcoming`) was added. Fixing it required no changes to the checkpointed historical-extraction logic, since `load_matches.py`'s upsert-by-`match_id` design means a placeholder "not yet played" row simply gets overwritten with the real result once a match concludes.

Sample output, predicting VCT Champions 2026 group-stage matchups:

```text
G2 Esports vs TYLOO
  Predicted winner: G2 Esports (77% model confidence)
  P(G2 Esports wins) = 0.77   P(TYLOO wins) = 0.23
  Recent form: G2 Esports 0.80 vs TYLOO 0.80
```

As with all model output in this project, these are illustrative results from a real, tested pipeline — not reliable forecasts. See [Modeling & Results](#modeling--results) above for the honest accuracy numbers behind them.

## Interactive Dashboard

A Streamlit app in [`dashboard/app.py`](dashboard/app.py) reads directly from the analytics warehouse and renders team win rates, tournament summaries, and recent match results — a way to explore the underlying data interactively, separate from the modeling work above.

![Recent matches view](docs/images/recent_matches.png)

```text
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Supporting Data Infrastructure

None of the analysis above is possible without reliable, tested, up-to-date data — this is the data engineering layer that makes the analytics and data science work possible.

### Architecture

```text
PandaScore API (past matches + upcoming matches)
    ↓
Python extraction with VCT filtering and checkpoints
    ↓
PostgreSQL: public.raw_matches
    ↓
dbt: staging → intermediate → marts (star schema)
    ↓
PostgreSQL analytics views  →  analysis/ (features, EDA, modeling, live predictions)
    ↓
Apache Airflow orchestration every 6 hours
```

The pipeline follows an ELT pattern: raw PandaScore match payloads are loaded without pre-transformation, then dbt parses and models the data in PostgreSQL — giving the analysis layer a clean, well-tested warehouse to work from rather than raw API responses.

### Tech Stack

| Category | Tools |
|---|---|
| Data source | PandaScore API |
| Extraction and loading | Python |
| Orchestration | Apache Airflow 2.9.3 |
| Storage | PostgreSQL 16 |
| Transformation and testing | dbt Core with dbt-postgres |
| Analysis and modeling | pandas, scikit-learn, joblib, Jupyter |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions, pytest |
| Containerization | Docker and Docker Compose |

### Data Model

| Layer | Relation | Purpose |
|---|---|---|
| Raw | `public.raw_matches` | One row per PandaScore match ID with its unmodified JSON payload |
| Staging | `analytics.stg_matches` | Flattens raw JSON into typed match fields |
| Intermediate | `analytics.int_match_teams` | Match-to-team relationships |
| Mart | `analytics.dim_teams` | De-duplicated team dimension |
| Mart | `analytics.dim_tournaments` | De-duplicated tournament dimension |
| Mart | `analytics.fct_matches` | Final match-level fact view — the table the analysis layer reads from |

### Orchestration

The Airflow DAG `vct_extraction` runs every six hours (`0 */6 * * *`):

```text
[run_extraction, run_extraction_upcoming] → run_load → dbt_run → dbt_test → run_prediction → log_summary
```

| Task | Responsibility |
|---|---|
| `run_extraction` | Retrieves new, finished flagship VCT matches from PandaScore |
| `run_extraction_upcoming` | Retrieves currently-scheduled (not yet played) flagship VCT matches |
| `run_load` | Loads all extracted raw JSON into `public.raw_matches` (upserts by match ID) |
| `dbt_run` | Refreshes dbt staging, intermediate, and mart views |
| `dbt_test` | Runs source and model data-quality tests |
| `run_prediction` | Runs the trained model against currently-scheduled matches |
| `log_summary` | Logs completion timestamps |

`run_extraction` and `run_extraction_upcoming` run independently in parallel — one has no dependency on the other. `run_prediction` runs only after `dbt_test` succeeds, guaranteeing predictions are always made against freshly-verified data.

### Continuous Integration

Every push and pull request to `main` runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job | What it validates |
|---|---|
| `python-tests` | Unit tests for extraction filtering, checkpointing, and leakage-safe feature engineering |
| `dbt-build` | A real Postgres service container, seeded with fixture data, running the actual `dbt build` — all models and 13 data-quality tests |

### Verified Pipeline Results

| Metric | Result |
|---|---:|
| Raw match records | 1,621 |
| Final fact-match records | 1,621 |
| Teams | 73+ |
| dbt data tests passed | 13 |
| dbt build result | 18 passed, 0 warnings, 0 errors |

### Local Setup

**Prerequisites:** Docker Desktop with Docker Compose, a PandaScore API key, Git, Python 3.12+ for local dbt/analysis work.

```bash
git clone https://github.com/EthanSB-dev/Valorant-Esports-Match-Predictor.git
cd Valorant-Esports-Match-Predictor
cp .env.example .env   # fill in your credentials
docker compose up -d --build
```

Open `http://localhost:8080`, sign in with your configured Airflow admin credentials, unpause the `vct_extraction` DAG, and trigger a run. A successful run turns all seven tasks green.

**Running dbt locally** (outside Airflow):
```bash
cd transform/vct_dbt
dbt debug
dbt build
```

**Running the analysis** (see [`analysis/README.md`](analysis/README.md) for full detail):
```bash
cd analysis
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook eda.ipynb   # exports data/match_features.csv
python train_model.py        # trains and saves model.joblib
python predict_upcoming.py   # predicts currently-scheduled matches
```

## Repository Structure

```text
.
├── .github/workflows/ci.yml       # pytest + dbt build on every PR
├── dags/vct_extraction_dag.py     # Airflow orchestration (7 tasks)
├── extract/
│   ├── pandascore_client.py       # PandaScore API client (past + upcoming endpoints)
│   ├── extract_matches.py         # Checkpointed extraction of finished matches
│   ├── extract_upcoming_matches.py # Extraction of currently-scheduled matches
│   ├── vct_filters.py             # VCT/flagship filtering rules
│   └── checkpoint.py              # Extraction checkpoint management
├── load/load_matches.py           # Raw JSON → PostgreSQL (upsert by match_id)
├── transform/vct_dbt/             # dbt: staging, intermediate, marts, tests
├── analysis/
│   ├── features.py                # Leakage-safe feature engineering + live-stats functions
│   ├── eda.ipynb                  # Exploratory analysis
│   ├── train_model.py             # Model training and evaluation
│   ├── predict_upcoming.py        # Live predictions for scheduled matches
│   ├── model.joblib                # Trained model artifact
│   ├── README.md                  # Full analysis write-up
│   └── data/match_features.csv    # Exported features (reproducible without a live DB)
├── dashboard/app.py                # Streamlit analytics dashboard
├── tests/                          # Unit tests (filters, checkpointing, features)
├── docs/images/                    # Screenshots
├── docker-compose.yml
├── Dockerfile
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## Future Directions

- Player-level and map-level data, likely necessary for materially better prediction accuracy
- Roster-change tracking, so "recent form" reflects the actual current lineup
- Tournament-stage weighting (group stage vs. playoffs vs. finals)
- Automated model retraining (currently a manual, checked-in `model.joblib`)
- dbt source-freshness checks and pipeline failure alerting

## Security

- Credentials live in `.env` and Airflow connection variables — never committed
- `.env.example` contains only placeholder values

## License

This project is intended for portfolio use.