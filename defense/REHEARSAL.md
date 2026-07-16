# Defense 1 Rehearsal Runbook

Use `develop` for rehearsals. Use `main` only after the team approves the final
release pull request.

## Team Speaking Order

1. Fateh: architecture, ingestion, validation, statistics, and alerts.
2. Yassine: model training, evaluation, Airflow, and Docker infrastructure.
3. Ashish: FastAPI, PostgreSQL, prediction persistence, and prediction DAG.
4. Akash: Streamlit single/batch prediction and prediction history.

Every member should also be able to explain the complete flow:

```text
CSV -> ingestion DAG -> validation -> good/bad split -> prediction DAG
    -> FastAPI -> model -> PostgreSQL -> Streamlit history
```

## Preflight

Run this at least 30 minutes before the defense:

```bash
git switch develop
git pull --ff-only origin develop
docker compose up -d --build
docker compose ps
```

Confirm these pages open:

- Streamlit: http://127.0.0.1:8501
- FastAPI docs: http://127.0.0.1:8000/docs
- Airflow: http://127.0.0.1:8080

Open Data Docs at http://127.0.0.1:8081 after the first ingestion run. A fresh
environment can return HTTP 403 while the report directory is still empty.

Read the generated Airflow password:

```bash
docker compose exec airflow-api-server \
  cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Run the quality checks:

```bash
python -m flake8 .
python -m pytest -q
docker compose config --quiet
```

Keep the five files in `demo_data/defense1`. Do not copy them all into
`data/raw_data` before the demonstration.

## Demo Sequence

### 1. Architecture and containers - Fateh and Yassine

Show `docker compose ps`. Explain that Docker Compose starts PostgreSQL,
FastAPI, Streamlit, Airflow, Data Docs, and the supporting services with one
command.

### 2. Manual prediction - Akash and Ashish

1. Open Streamlit.
2. Make one single prediction.
3. Upload a feature-only CSV in the Batch prediction tab.
4. Open Past Predictions and show the `Manual predictions` source.
5. Open FastAPI `/docs` and explain that Streamlit calls `/predict` and
   `/past-predictions`; it does not load the model or database directly.

### 3. High-criticality ingestion - Fateh

Copy only the invalid file:

```bash
cp demo_data/defense1/01_all_invalid.csv data/raw_data/
```

Unpause or trigger `bike_data_ingestion` in Airflow. Show all five tasks:

1. `read_data`
2. `validate_data`
3. `save_statistics`
4. `send_alerts`
5. `split_and_save_data`

Expected evidence:

- 10 total rows, 0 valid, 10 invalid, high criticality
- a Teams alert
- `01_all_invalid_bad.csv` in `data/bad_data`
- the Great Expectations report at http://127.0.0.1:8081

### 4. Mixed and clean ingestion - Fateh

Copy `05_mixed.csv`, run ingestion, and show 7 good plus 3 bad rows with a
medium alert. Then process `02_clean.csv`, `03_clean.csv`, and `04_clean.csv`
one at a time. Each clean file should produce 10 good rows and no alert.

### 5. Scheduled predictions - Ashish

Unpause or trigger `bike_demand_prediction`. The first run should combine all
new good rows into one FastAPI request and store scheduled predictions. The
prepared files produce 37 valid rows in total.

Trigger the DAG again without adding data. Show `check_for_new_data` and
`make_predictions` as skipped. Explain that the state file prevents duplicate
processing while good-data files remain available.

### 6. Close - All members

Open `defense1_slides.pdf`. State that no known Defense 1 feature failed in the
final rehearsal, then explain the four-person ownership slide.

## Likely Questions

**Why use `hour.csv` instead of `day.csv`?**

Hourly data supports the required hourly demand prediction and includes the
`hr` feature. The daily file cannot model within-day demand changes.

**Why is the split chronological?**

The model trains on earlier rows and tests on later rows, which better matches
future prediction and avoids evaluating earlier data with later information.

**Why Great Expectations and custom validation?**

Great Expectations creates auditable expectations and Data Docs. The custom
validation also produces row-level good/bad outputs and domain-specific error
counts for routing and alerts.

**When is a Teams alert sent?**

Only medium- or high-criticality validation results send an alert. None and
low results do not create alert noise.

**Why save ingestion statistics?**

They provide persistent evidence of file name, row counts, criticality,
validation errors, and Great Expectations results for monitoring and defense.

**Why does Streamlit not load the model directly?**

FastAPI owns model inference and database writes. This gives the web app and
Airflow one prediction contract and avoids duplicated serving logic.

**Why does the prediction DAG leave good files in place?**

The state file records processed paths, so files remain auditable while later
runs process only new files.

**Why use Docker Compose?**

It gives every team member the same service versions, network names,
environment variables, volumes, and startup command.

**Why are Airflow API and JWT secrets shared?**

Airflow 3 signs task-execution tokens. The API server and scheduler must use
the same secrets or task requests are rejected.

**What is not part of Defense 1?**

Grafana dashboards, continuous retraining, and advanced MLflow workflows are
future work. The Defense 1 model is trained by script and loaded by FastAPI.

## Fast Recovery

Check service state and logs:

```bash
docker compose ps
docker compose logs --tail=100 api streamlit airflow-scheduler
```

Restart one service:

```bash
docker compose restart api
```

Rebuild after a code change:

```bash
docker compose up -d --build
```

Use a full reset only before a rehearsal when deleting all database and
Airflow state is acceptable:

```bash
docker compose down -v --remove-orphans
docker compose up -d --build
```

Never display or commit `.env`, database passwords, Airflow secrets, or the
real Teams webhook URL.
