# Hourly Bike Sharing Demand Prediction

This EPITA Data Science in Production project predicts hourly bicycle rentals
from calendar and weather features. It validates incoming CSVs, serves and
records predictions, retrains the model in Airflow, and monitors data quality
and model drift in Grafana.

The model uses the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
Its target is `cnt`. The chronological 80/20 split reserves the final 20% for
evaluation. The labeled demo batches replay observations from the training
portion.

## Architecture

1. Airflow reads CSVs from `data/raw_data`. Great Expectations validates each
   batch, PostgreSQL stores the results, and nginx serves the generated reports.
2. Streamlit and the prediction DAG call FastAPI. FastAPI serves the MLflow
   champion and records inputs, predictions, and the model version in PostgreSQL.
3. The training DAG trains and evaluates a candidate on labeled data. When the
   candidate passes, Airflow updates the champion alias, reloads FastAPI, and
   archives the processed batches.
4. Grafana queries ingestion statistics, prediction history, and training
   baselines in PostgreSQL. Teams receives ingestion, rejection, and Grafana
   alerts in their respective channels.

## Start a fresh installation

Use Git and Docker Desktop with Docker Compose v2. Run these PowerShell commands
from the project root. For the final defense, check out the released `main`
branch.

```powershell
if (!(Test-Path .env)) {
    Copy-Item .env.docker.example .env
}
```

Set local PostgreSQL and Grafana passwords in `.env`. Use the same PostgreSQL
password in `POSTGRES_PASSWORD` and `DATABASE_URL`. Set
`AIRFLOW_API_SECRET_KEY` and `AIRFLOW_JWT_SECRET` to long random values shared
by the Airflow services. Use `TEAMS_WEBHOOK_URL` for **Data Quality Alerts**
and add `TEAMS_ML_WEBHOOK_URL` for **ML Alerts**. Keep `.env` private.

| Setting | Default or purpose |
| --- | --- |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` inside Compose |
| `MLFLOW_MODEL_NAME` | `BikeDemand` |
| `TRAINING_MIN_NEW_ROWS` | 100 new labeled rows per training attempt |
| `TRAINING_MAX_INFERENCE_MS_PER_ROW` | 10 ms per row |
| `TRAINING_SCHEDULE` | `*/10 * * * *` |
| `DATA_DOCS_BASE_URL` | `http://127.0.0.1:8081` for Teams report links |

Start PostgreSQL and MLflow first. FastAPI loads the registered champion at
startup, so register the original model before starting the API.

```powershell
docker compose config --quiet
docker compose up -d postgres mlflow
```

Wait for MLflow at http://127.0.0.1:5000, then prepare the data and register
the original model:

```powershell
docker compose build api
docker compose run --rm --no-deps -v "${PWD}:/workspace" -w /workspace api python -m scripts.download_dataset
docker compose run --rm --no-deps -v "${PWD}:/workspace" -w /workspace api python -m scripts.prepare_training_data --number-of-files 30
docker compose run --rm --no-deps -v "${PWD}:/workspace:ro" -w /workspace api python -m scripts.register_baseline
```

The registration script checks that the downloaded model produces the original predictions and
assigns the `champion` alias. Data preparation creates the historical training
set, 30 ten-row labeled batches, and the held-out test set. Run data preparation
once per training directory.

If upgrading an existing Defense 1 database, add the version column before
starting the updated API. This command uses the example database and user
names.

```powershell
docker compose exec postgres psql -U dsp -d dsp -c "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_version VARCHAR(64);"
```

For **fresh Grafana volumes** with both webhook URLs configured, install the
exported alert rules and contact points before starting Grafana.

```powershell
New-Item -ItemType Directory -Force .\grafana\provisioning\alerting | Out-Null
Copy-Item -Path .\grafana\alerting\*.yml -Destination .\grafana\provisioning\alerting
```

Grafana loads both dashboard JSON files through the dashboard provider. If
your existing Grafana volume already has the rules and contact points created
in the UI, continue using those. Provisioning on top of those records can
conflict with their IDs.

```powershell
docker compose up -d --build
docker compose ps -a
```

Once the API is healthy, initialize the training statistics for the original
champion and give Airflow write access to its training directories.

```powershell
docker compose run --rm --no-deps -v "${PWD}:/workspace:ro" -w /workspace api python -m scripts.initialize_training_baseline
docker compose exec --user root airflow-scheduler sh -c 'mkdir -p /opt/airflow/data/training_runs && chown -R airflow:root /opt/airflow/data/training /opt/airflow/data/training_runs'
```

Expect 13,903 original champion training rows. This initializer is for the
original baseline model. The training DAG saves statistics for subsequent
versions.

## Local service URLs

| Service | URL | Login |
| --- | --- | --- |
| Streamlit | http://127.0.0.1:8501 | None |
| FastAPI docs | http://127.0.0.1:8000/docs | None |
| Airflow | http://127.0.0.1:8080 | `admin` |
| Data Docs | http://127.0.0.1:8081 | None |
| Grafana | http://127.0.0.1:3000 | Grafana credentials in `.env` |
| MLflow | http://127.0.0.1:5000 | None |

Read Airflow's generated administrator password from the container.

```powershell
docker compose exec airflow-api-server cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Data Docs can return HTTP 403 until the first validation report is generated.
The default Teams report URLs use localhost and open from the demo computer.

## Airflow and model lifecycle

| DAG | Default schedule | Behavior |
| --- | --- | --- |
| `bike_data_ingestion` | Every minute | Validates one raw file, stores statistics, routes good and bad rows, and alerts Teams for medium and high criticality. |
| `bike_demand_prediction` | Every two minutes | Sends new good files to FastAPI as one batch. Tracks processed files. |
| `bike_demand_training` | Every ten minutes | Requires 100 new labeled rows, trains and evaluates a candidate, and branches on the promotion decision. |

Enable the DAGs in Airflow when their inputs are ready. FastAPI records the
served model version with each prediction. The training DAG snapshots its
inputs and compares the candidate and champion on the same held-out set.
Promotion requires RMSE less than or equal to the champion's and mean inference
time below the configured limit. The default limit is 10 ms per row.

On promotion, Airflow updates the MLflow `champion` alias, reloads the API, and
archives the exact batches after scheduled predictions have processed them.
On rejection, the DAG logs the comparison in MLflow and alerts **ML Alerts**.
When no new files or too few new labels are available, Airflow marks the
affected tasks skipped and displays the overall run as success.

## Defense walkthrough

Create the input directory if it is missing.

```powershell
New-Item -ItemType Directory -Force .\data\raw_data | Out-Null
```

1. Make one single and one batch prediction in Streamlit. Check **Past
   Predictions** for the model version and the stored results.
2. Queue `01_all_invalid.csv` from `demo_data/defense1` in `data/raw_data`.
   Show the ingestion run, the 10 rejected rows in `bad_data`, and the
   Teams alert. Open the linked Great Expectations report.
3. Queue `05_mixed.csv` and show 7 valid rows and 3 invalid rows. The three
   clean demo files each produce 10 valid rows. Show the scheduled results.
4. Run scheduled prediction again without a new file and show the skipped
   tasks.
5. Show the training DAG's promotion, reload, and archive tasks. The verified
   replay started with version 1 and used 130 labeled rows from batches
   `001, 004, 005, 007, 008, 011, 017, 018, 020, 027, 028, 029, 030`. Use the
   corresponding files in `data/training/incoming_batches` for a fresh replay.
   Keep the training DAG paused while ingesting those batches and completing
   scheduled prediction. Then enable training and inspect the successful run,
   MLflow's evaluation metrics, and the new `champion` alias.
6. Show a rejected candidate's comparison and Teams message. Show the
   insufficient-data run and its skip reason.
7. Open **Bike Sharing Data Quality** and **Bike Sharing Drift and
   Predictions** in Grafana. Use a time range that includes the test data.
   Show the invalid-rate and missing-column alerts in **Data Quality
   Alerts** and the temperature and weather drift alerts in **ML Alerts**.

## Monitoring

**Bike Sharing Data Quality** shows the invalid-row rate, validation errors by
category, valid and invalid row counts, and failed Great Expectations checks.
**Bike Sharing Drift and Predictions** shows temperature drift, weather
distribution drift, median predicted demand, and the P90–P10 prediction spread.
The drift queries match predictions to the training statistics for the model
version that served them.

| Alert | Condition over the last 10 minutes | Teams channel |
| --- | --- | --- |
| High invalid-row rate | A batch exceeds 50% invalid rows | Data Quality Alerts |
| Missing required columns | A batch has a missing required column | Data Quality Alerts |
| High temperature drift | Serving mean differs from training mean by more than 0.15 on the normalized scale | ML Alerts |
| High weather distribution drift | Weather distribution distance exceeds 40% | ML Alerts |

Grafana evaluates the rules once per minute. Both drift rules require at least
10 predictions for a model version in the evaluation window.

## Dataset tools

With Python 3.12 and the packages from `requirements.txt`, run these from the
project root:

```powershell
python -m scripts.download_dataset
python -m scripts.split_dataset data/raw/hour.csv data/raw_data 30
python -m scripts.generate_data_errors --help
python -m scripts.prepare_defense_demo
```

The splitter produces ten-row CSVs. The error generator accepts configurable
probabilities and a seed. Use `--drop-column` to demonstrate schema failure.
Keep the prepared demo files outside `raw_data` until their demo step.

## Tests and release

```powershell
docker compose config --quiet
docker compose run --rm --no-deps -v "${PWD}:/verification:ro" -w /tmp api sh -c 'cp -a /verification /tmp/project && cd /tmp/project && unset API_BASE_URL DATABASE_URL && python -m flake8 . && python -m pytest -q -p no:cacheprovider'
```

GitHub Actions runs lint and tests in separate steps. Open feature PRs into
`develop`. The release workflow creates a versioned GitHub Release with
generated notes when a same-repository PR from `develop` is merged into
`main`. It starts with `v1.0.0` and increments the patch version on subsequent
releases.

Local validation passed all 65 tests and Flake8. The first GitHub Release run
is pending verification after the workflow is merged into `main`.

## Troubleshooting and shutdown

```powershell
docker compose ps -a
docker compose logs --tail=60 api airflow-scheduler grafana
docker compose exec airflow-scheduler airflow dags list-import-errors
Invoke-RestMethod http://127.0.0.1:8000/health
```

`airflow-init` exits with code 0 after initializing the metadata database.
Check the failing task's execution log in Airflow if a DAG run fails.
Use `docker compose stop` to stop the services and preserve their data.

## Defense 1 contributors

- Fateh Cherrad implemented data preparation, ingestion, validation, statistics,
  alerts, and Data Docs.
- Yassine implemented the original model, evaluation, Docker Compose, and
  Airflow infrastructure.
- Ashish implemented FastAPI, PostgreSQL prediction storage, and scheduled
  prediction integration.
- Akash implemented the Streamlit prediction and history pages.
