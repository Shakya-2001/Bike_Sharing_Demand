# Hourly Bike Sharing Demand Prediction

This project predicts the number of bikes rented in a given hour using weather and calendar data.

The project is built for the EPITA Data Science in Production practical work. The main goal is not to build the most complex model, but to demonstrate a complete production-style ML workflow with:

- FastAPI model serving
- Streamlit prediction UI
- PostgreSQL storage
- Airflow ingestion and prediction DAGs
- Great Expectations data validation
- Docker Compose
- CI with linting and tests

## Dataset

Dataset: UCI Bike Sharing Dataset

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Main file used: `hour.csv`

Target column: `cnt`

## Why `hour.csv`

The dataset ZIP contains `day.csv`, `hour.csv`, and `Readme.txt`.

This project uses `hour.csv` because it has one row per hour and enough rows for the required ingestion workflow. The Defense 1 instructions require splitting data into files of exactly 10 rows and running scheduled ingestion/prediction jobs. The daily file is too small and less useful for that demo.

## ML Problem

Regression problem:

> Predict hourly bike rental demand from weather, season, weekday, holiday, and time information.

## Columns

Target:

- `cnt`: total bike rentals for the hour

Model features:

- `season`
- `yr`
- `mnth`
- `hr`
- `holiday`
- `weekday`
- `workingday`
- `weathersit`
- `temp`
- `atemp`
- `hum`
- `windspeed`

Columns not used as model input:

- `instant`: row identifier
- `dteday`: date, kept for monitoring and file tracking when useful
- `casual`: data leakage, because `casual + registered = cnt`
- `registered`: data leakage, because `casual + registered = cnt`
- `cnt`: target column

## Current Status

Step 1: project structure created.

Step 2: dataset download script and baseline training script created.

Baseline model:

- Model: Hist Gradient Boosting Regressor
- Rows: 17,379
- MAE: around 24 rentals
- RMSE: around 39 rentals
- R2: around 0.952

Next step: build the FastAPI `/health` and `/predict` endpoints.

Step 3: FastAPI endpoints created and tested.

Available API endpoints:

- `GET /health`
- `POST /predict`
- `GET /past-predictions`

Predictions are saved through SQLAlchemy. Set `DATABASE_URL` to use PostgreSQL. When `DATABASE_URL` is not set, the project uses a local SQLite database only for development.

Next step: build the Streamlit UI with a prediction page and a past predictions page.

Step 4: Streamlit UI created and verified.

Run the API:

```bash
.venv/bin/uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Run the Streamlit app:

```bash
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/streamlit run app/streamlit_app/Prediction.py
```

Step 5: data splitting script created.

Generate ingestion files:

```bash
.venv/bin/python scripts/split_dataset.py data/raw/hour.csv data/raw_data 30
```

Each generated file contains exactly 10 data rows.

Next step: build the data error generation script for Great Expectations demos.

Step 6: data error generation script created.

Generate a corrupted demo file:

```bash
.venv/bin/python scripts/generate_data_errors.py \
  data/raw_data/batch_001.csv \
  data/raw_data/demo_corrupted.csv \
  --null-probability 0.3 \
  --range-probability 0.3 \
  --category-probability 0.3 \
  --type-probability 0.3 \
  --duplicate-probability 0.2 \
  --inconsistency-probability 0.3 \
  --outlier-probability 0.3 \
  --drop-column
```

Implemented error types:

- Null value in required column
- Value outside valid range
- Invalid categorical value
- Missing required column
- Wrong data type
- Duplicate primary key
- Cross-field inconsistency
- Statistical outlier

Next step: prepare Docker Compose with FastAPI, Streamlit, and PostgreSQL.

Step 7: Docker Compose files created for the core services.

Services:

- `postgres`: PostgreSQL database
- `api`: FastAPI model service
- `streamlit`: Streamlit webapp

Run with Docker:

```bash
docker compose up --build
```

Open:

- FastAPI docs: http://127.0.0.1:8000/docs
- Streamlit app: http://127.0.0.1:8501

Note: Docker must be installed and running on the machine.

Step 8: validation and ingestion pipeline created.

Validate one file:

```bash
.venv/bin/python scripts/validate_data.py data/raw_data/demo_corrupted.csv
```

Ingest one random raw file:

```bash
.venv/bin/python scripts/ingest_one_file.py
```

The ingestion flow:

- randomly selects one CSV from `data/raw_data`
- validates it with Great Expectations
- computes error statistics and criticality
- saves ingestion statistics
- moves clean files to `data/good_data`
- moves fully invalid files to `data/bad_data`
- splits mixed files into good and bad outputs

Step 9: Airflow DAGs created and verified.

Run all services:

```bash
docker compose -f docker-compose.yml -f docker-compose.airflow.yml up --build
```

Open:

- FastAPI docs: http://127.0.0.1:8000/docs
- Streamlit app: http://127.0.0.1:8501
- Airflow UI: http://127.0.0.1:8080

Airflow DAGs:

- `bike_data_ingestion`: reads one raw 10-row CSV file, validates it with Great Expectations, stores data quality statistics, emits alert information for medium/high criticality, and writes valid/invalid outputs.
- `bike_scheduled_predictions`: checks new files in `data/good_data`, calls the FastAPI `/predict` endpoint in one batch request, and stores scheduled predictions in PostgreSQL.

Step 10: CI created.

The GitHub Actions workflow in `.github/workflows/ci.yml` installs dependencies and runs `pytest` on pushes and pull requests to `main`.

## Defense 1 Checklist

- Docker Compose: implemented for PostgreSQL, FastAPI, Streamlit, and Airflow.
- Git and CI: GitHub Actions workflow added for automated tests.
- Database: PostgreSQL stores predictions and ingestion statistics.
- API: `/health`, `/predict`, and `/past-predictions` implemented.
- Webapp: Streamlit prediction and past prediction pages call the API.
- Data splitting: script generates files with exactly 10 data rows.
- Data error generation: script creates null, range, category, type, duplicate, inconsistency, outlier, and missing-column errors.
- Ingestion DAG: validates with Great Expectations, records statistics, splits good/bad data, and emits alert information.
- Prediction DAG: reads `good_data`, sends one batch API call, and does not move or delete processed files.
- Tests: API, splitting, error generation, validation, and ingestion pipeline tests pass.
