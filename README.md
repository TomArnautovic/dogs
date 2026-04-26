# Greyhounds ANN Lab

This repository now contains a modern MVP that replaces the old MATLAB prototype with a Python pipeline for:

- storing greyhound racing data in Supabase/Postgres,
- ingesting historical and upcoming race rows,
- training an ANN to predict finishing order,
- monitoring training runs through a small dashboard,
- scoring future races and storing predictions.

The original MATLAB files are still kept in [`MATLAB m-files`](/Users/tomarnautovic/dev/greyhounds/MATLAB%20m-files) as reference material. The current trainer is now intentionally closer to that workflow: it trains on sampled race-order permutations and predicts by scoring candidate finish orders directly.

## Stack

- Database: Supabase Postgres or a local SQLite fallback for development
- ETL: Python CLI importers for CSV, GBGB JSON, plus a RapidAPI racecard ingester for daily cards
- Model: PyTorch feedforward permutation scorer aligned with the original MATLAB approach
- UI: Streamlit dashboard for kicking off training and reviewing recent runs

## Project layout

- [`supabase/schema.sql`](/Users/tomarnautovic/dev/greyhounds/supabase/schema.sql) defines the database tables.
- [`src/greyhounds/db.py`](/Users/tomarnautovic/dev/greyhounds/src/greyhounds/db.py) contains ORM models and session helpers.
- [`src/greyhounds/ingest.py`](/Users/tomarnautovic/dev/greyhounds/src/greyhounds/ingest.py) loads runner-level race data into the database.
- [`src/greyhounds/ml.py`](/Users/tomarnautovic/dev/greyhounds/src/greyhounds/ml.py) builds race features, trains the ANN, and predicts upcoming races.
- [`src/greyhounds/cli.py`](/Users/tomarnautovic/dev/greyhounds/src/greyhounds/cli.py) exposes the command line workflow.
- [`apps/training_dashboard.py`](/Users/tomarnautovic/dev/greyhounds/apps/training_dashboard.py) is the monitoring and tuning interface.

## Python version

Use Python `3.12.x` for the cleanest install path. The current machine is on `3.14`, but parts of the ML ecosystem are still catching up, so I pinned the project for `>=3.12,<3.14`.

## Quick start

1. Create a Python 3.12 environment.
2. Install the project:

```bash
pip install -e .
```

3. Copy [`.env.example`](/Users/tomarnautovic/dev/greyhounds/.env.example) to `.env` and point `DATABASE_URL` at Supabase.
4. Paste [`supabase/schema.sql`](/Users/tomarnautovic/dev/greyhounds/supabase/schema.sql) into the Supabase SQL editor.
5. Initialise a local dev database if you want to test without Supabase:

```bash
greyhounds init-db
```

6. Import runner-level data from CSV:

```bash
greyhounds ingest-csv data/results.csv --source manual
```

7. Train a model:

```bash
greyhounds train --epochs 60 --batch-size 32 --learning-rate 0.001 --hidden-size-1 128 --hidden-size-2 64
```

8. Predict upcoming races:

```bash
greyhounds predict
```

9. Start the dashboard:

```bash
streamlit run apps/training_dashboard.py
```

## Daily racecard ingestion

Put your RapidAPI key in [`.env.example`](/Users/tomarnautovic/dev/greyhounds/.env.example) as `RAPIDAPI_KEY` after copying it to `.env`.

To import the upcoming racecards for a given day:

```bash
greyhounds ingest-rapidapi-racecards --date 2026-04-13
```

To import only selected tracks from that day:

```bash
greyhounds ingest-rapidapi-racecards --date 2026-04-13 --track "Central Park" --track "Romford"
```

By default that skips races already marked finished or canceled, which keeps the prediction queue focused on still-to-run races.

If you want to refresh the whole day and pull finished races as well:

```bash
greyhounds ingest-rapidapi-racecards --date 2026-04-13 --include-finished
```

Then score the races that are now in the database:

```bash
greyhounds predict
```

The Streamlit dashboard also includes a racecard import panel where you can choose the date, preview the returned cards, and select the tracks to import before running predictions.

## Runner-level CSV format

Each row represents one dog in one race. The importer is tolerant of missing optional columns.

Required practical columns:

- `track_name`
- `scheduled_start`
- `trap_number`
- `dog_name`

Strongly recommended columns:

- `source`
- `source_race_id`
- `source_track_id`
- `source_dog_id`
- `race_number`
- `distance_m`
- `grade`
- `trainer_name`
- `owner_name`
- `weight_kg`
- `finish_position`
- `official_time_s`
- `sectional_s`
- `sp_text`
- `sp_decimal`

Upcoming race files can omit result fields like `finish_position` and `official_time_s`.

The RapidAPI ingester maps daily racecards into this same runner-level shape, including trap numbers, trainers, non-runners, starting price, and any returned finishing positions for later refresh runs.

## Notes on the model

The training pipeline builds one race example at a time and computes runner features only from races that happened before that race. That prevents look-ahead leakage.

The ANN is trained in a MATLAB-style permutation setup: for each completed race with a complete 1..N finishing order, it samples candidate finishing orders, learns a low score for orders close to the true result, and then predicts future races by evaluating candidate permutations and choosing the lowest-scoring order. Races with missing, duplicated, or out-of-range `finish_position` values are skipped for both training targets and historical form features.

## Data source caution

The optional GBGB importer is included as a technical integration point, but you should only automate external sources after checking the relevant terms and any permissions you need for your intended use.
