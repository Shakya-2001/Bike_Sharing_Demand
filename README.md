# Hourly Bike Sharing Demand Prediction

This EPITA Data Science in Production project predicts hourly bike-rental
demand and demonstrates a complete Defense 1 data pipeline and serving
workflow.

## Defense 1 Components

- HistGradientBoosting regression model loaded from a local file
- FastAPI service for single and batch predictions
- PostgreSQL storage for predictions and ingestion statistics
- Streamlit multipage application for predictions and prediction history
- Great Expectations validation, Checkpoint, and Data Docs
- Airflow 3 ingestion DAG scheduled every minute
- Airflow 3 prediction DAG scheduled every two minutes
- Teams notifications for medium- and high-criticality data problems
- Docker Compose orchestration
- GitHub Actions with separate Flake8 and pytest steps

## Dataset and Model

The project uses the
[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
and its `hour.csv` file. The target is `cnt`, the hourly number of bike
rentals. The model uses a chronological 80/20 train/test split to avoid
evaluating earlier observations with knowledge of later observations.

## Architecture

The Streamlit application and scheduled prediction DAG send feature data to
FastAPI. FastAPI loads the model once at startup, makes predictions, and saves
the predictions and their features through SQLAlchemy. The ingestion DAG reads
one file from `data/raw_data`, validates it with Great Expectations, saves
summary statistics with Airflow's PostgresHook, sends alerts when required,
and routes rows to `good_data` or `bad_data`. The prediction DAG sends all new
good rows to FastAPI in one batch request.

## Prerequisites

- Git
- Docker Desktop with Docker Compose v2
- Python 3.12 for dataset and test scripts
- A Microsoft Teams workflow webhook for alert delivery

Confirm the required versions:

```bash
docker --version
docker compose version
python3.12 --version
```

## Clone and Configure

```bash
git clone https://github.com/fatehcherrad/dsp-bike-sharing-demand.git
cd dsp-bike-sharing-demand
cp .env.docker.example .env
```

The final defense must run from the default `main` branch. Team members testing
work before the final release should explicitly switch to `develop`.

Edit `.env` and replace both `change-me` values with local passwords. Keep the
password in `DATABASE_URL` identical to `POSTGRES_PASSWORD`. Add the Teams
workflow URL locally:

```env
POSTGRES_PASSWORD=your-local-postgres-password
DATABASE_URL=postgresql://dsp:your-local-postgres-password@postgres:5432/dsp
AIRFLOW_API_SECRET_KEY=your-long-random-airflow-api-secret
AIRFLOW_JWT_SECRET=your-long-random-airflow-jwt-secret
GRAFANA_ADMIN_PASSWORD=your-local-grafana-password
TEAMS_WEBHOOK_URL=https://your-private-teams-workflow-url
```

The two Airflow secrets must be shared by every Airflow container. Without
them, Airflow 3 rejects task-execution tokens and DAG tasks fail before their
Python functions start.

Never commit `.env` or the real webhook URL.

## Start the Complete Application

Make sure Docker Desktop is running, then execute:

```bash
docker compose up -d --build
docker compose ps
```

Wait until PostgreSQL, FastAPI, Streamlit, and the Airflow API server report
healthy. The first build can take several minutes.

## Service URLs

| Service | URL | Login |
| --- | --- | --- |
| Streamlit | http://127.0.0.1:8501 | None |
| FastAPI docs | http://127.0.0.1:8000/docs | None |
| Airflow 3 | http://127.0.0.1:8080 | User: `admin` |
| Data Docs | http://127.0.0.1:8081 | None |
| Grafana | http://127.0.0.1:3000 | Values from `.env` |

Airflow's Simple Auth Manager generates the administrator password at startup.
Read it with:

```bash
docker compose exec airflow-api-server \
  cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

MLflow is containerized for the complete project architecture but is not used
for Defense 1 model serving and is not exposed on a host port.

## Local Python Environment

The Docker application does not require a host Python environment. Create one
when running dataset scripts or tests locally:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset Scripts

Download and extract the UCI dataset:

```bash
python -m scripts.download_dataset
```

Create 30 ingestion files with exactly 10 rows each:

```bash
python -m scripts.split_dataset data/raw/hour.csv data/raw_data 30
```

Create a reproducible corrupted file:

```bash
python -m scripts.generate_data_errors \
  data/raw_data/batch_001.csv \
  data/raw_data/corrupted_batch.csv \
  --null-probability 0.2 \
  --range-probability 0.2 \
  --category-probability 0.2 \
  --type-probability 0.2 \
  --duplicate-probability 0.2 \
  --inconsistency-probability 0.2 \
  --outlier-probability 0.2 \
  --schema-probability 0 \
  --seed 42
```

Use `--drop-column` to force a high-criticality missing-column scenario.

## Defense Demo Files

Five prepared files are stored in `demo_data/defense1`. Every CSV contains
exactly 10 rows. Regenerate them from `hour.csv` with:

```bash
python -m scripts.prepare_defense_demo
```

Keep the files outside `data/raw_data` until their demo step. Copy only one
file at a time:

```bash
cp demo_data/defense1/01_all_invalid.csv data/raw_data/
```

The expected outcomes are recorded in
`demo_data/defense1/manifest.json`:

- `01_all_invalid.csv`: high criticality, 10 bad rows, Teams alert
- `02_clean.csv`: 10 good rows, no alert
- `03_clean.csv`: 10 good rows, no alert
- `04_clean.csv`: 10 good rows, no alert
- `05_mixed.csv`: medium criticality, 7 good and 3 bad rows, Teams alert

## Airflow DAGs

The DAGs are paused after a fresh installation. Unpause them in the Airflow UI
or run:

```bash
docker compose exec airflow-scheduler \
  airflow dags unpause bike_data_ingestion
docker compose exec airflow-scheduler \
  airflow dags unpause bike_demand_prediction
```

The ingestion DAG runs every minute and contains:

1. `read_data`
2. `validate_data`
3. `save_statistics`
4. `send_alerts`
5. `split_and_save_data`

The last three tasks run in parallel after validation. Teams notifications are
sent only for medium or high criticality.

The prediction DAG runs every two minutes and contains:

1. `check_for_new_data`
2. `make_predictions`

It combines all new good-data rows into one FastAPI batch request. It does not
move or delete files from `good_data`.

## Defense 1 Demonstration Order

1. Show all Docker containers running with `docker compose ps`.
2. Make a single prediction in Streamlit.
3. Upload a feature-only CSV for a batch prediction.
4. Show both records on the Past Predictions page.
5. Copy `01_all_invalid.csv` into `raw_data` and trigger ingestion.
6. Show high criticality, the Teams alert, database statistics, rendered Data
   Docs, and all rows in `bad_data`.
7. Copy `05_mixed.csv` and show the seven/three good/bad split.
8. Copy the three clean files and run ingestion for each one.
9. Trigger the prediction DAG and show scheduled predictions in Streamlit.
10. Trigger it again without new files and show that no data is reprocessed.

Data Docs are generated by Great Expectations, not handwritten HTML. Open the
report through http://127.0.0.1:8081 after an ingestion run.

## Tests and Continuous Integration

Run the same checks used by GitHub Actions:

```bash
python -m flake8 .
python -m pytest -q
docker compose config --quiet
```

GitHub Actions runs linting and tests as separate steps for pull requests into
`develop`.

## Git Workflow

- `main`: stable defense releases only
- `develop`: integrated and tested development code
- `feature/*`: one feature created from the latest `develop`

Do not commit directly to `main` or `develop`. Open a pull request into
`develop`, wait for CI, squash-merge, and delete the completed feature branch.
Create the `develop` to `main` pull request only after the final defense
rehearsal.

## Stop or Reset

Stop containers while preserving database and service volumes:

```bash
docker compose down
```

Completely reset all containers and named volumes:

```bash
docker compose down -v --remove-orphans
```

The reset command deletes PostgreSQL, Airflow, Grafana, and MLflow state. Use it
only when a clean environment is intended. If PostgreSQL credentials change,
perform a volume reset before restarting.

## Troubleshooting

Check service status and logs:

```bash
docker compose ps
docker compose logs --tail=100 api
docker compose logs --tail=100 airflow-api-server
docker compose logs --tail=100 airflow-scheduler
```

Confirm FastAPI readiness:

```bash
curl http://127.0.0.1:8000/health
```

List loaded DAGs:

```bash
docker compose exec airflow-scheduler airflow dags list
```

If a port is already occupied, stop the application using that port before
starting this Compose project. If a container was built before a code change,
run `docker compose up -d --build` again.

## Team Responsibilities

- Fateh Cherrad: data splitting, corruption, validation, ingestion lifecycle,
  statistics, Teams alerts, and Data Docs
- Yassine: baseline model, chronological model evaluation, Docker Compose, and
  Airflow infrastructure
- Ashish: FastAPI, SQLAlchemy, PostgreSQL, prediction storage, and prediction
  DAG integration
- Akash: Streamlit single and batch prediction workflows and prediction
  history

Every team member must be able to explain Docker and Airflow during the
defense.
