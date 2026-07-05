# Hourly Bike Sharing Demand Prediction

This project is built for the EPITA Data Science in Production practical work.

The goal is to predict hourly bike rental demand from weather and calendar features, then expose the model through a production-style workflow with an API, webapp, database, Airflow jobs, data validation, Docker, and CI.

## Dataset

Dataset: UCI Bike Sharing Dataset

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Main file planned for the project: `hour.csv`

Target column: `cnt`

We use `hour.csv` instead of `day.csv` because the hourly file has more rows and is better suited for the required ingestion workflow with small 10-row batch files.

## Planned Architecture

- FastAPI model service for predictions and prediction history
- Streamlit webapp for manual and batch predictions
- PostgreSQL database for predictions and ingestion statistics
- Airflow DAG for data ingestion and validation
- Airflow DAG for scheduled batch predictions
- Great Expectations for data quality validation
- Docker Compose for local orchestration
- GitHub Actions for linting and tests

## Branch Workflow

The project follows the course Git workflow:

- `main`: stable branch used only for defenses
- `develop`: integration branch
- `feature/*`: feature branches created from `develop`

Rules:

- Do not commit directly to `main` or `develop`.
- Create feature branches from `develop`.
- Open Pull Requests from feature branches into `develop`.
- Squash merge feature branches when possible.
- Delete finished feature branches after merge.
- Merge `develop` into `main` only before the defense.

## Team Feature Split

- `feature/api-database`: FastAPI service, SQLAlchemy models, prediction storage
- `feature/streamlit-webapp`: Streamlit prediction UI and past predictions page
- `feature/data-validation`: data splitting, corrupted data generation, Great Expectations validation
- `feature/docker-airflow`: Docker Compose, Airflow ingestion DAG, Airflow prediction DAG

## Local Setup

Create a local environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

The project is intended to run with Docker Compose once the feature branches are merged.

## Defense 1 Scope

Defense 1 focuses on:

- Git workflow and CI
- Docker
- FastAPI model API
- Streamlit webapp
- PostgreSQL storage
- Airflow ingestion and prediction jobs
- Great Expectations validation
- data splitting and data error generation

Each team member must understand and be able to explain the feature they contributed.
