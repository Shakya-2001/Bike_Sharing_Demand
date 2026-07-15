from __future__ import annotations

import ast
import stat
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"


def read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_required_docker_files_exist() -> None:
    assert COMPOSE_PATH.exists()
    assert (PROJECT_ROOT / "docker/api.Dockerfile").exists()
    assert (PROJECT_ROOT / "docker/airflow.Dockerfile").exists()
    assert (PROJECT_ROOT / "docker/postgres/init-databases.sh").exists()


def test_compose_contains_all_required_services() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    required_services = [
        "postgres:",
        "api:",
        "streamlit:",
        "airflow-init:",
        "airflow-api-server:",
        "airflow-scheduler:",
        "airflow-dag-processor:",
        "mlflow:",
        "grafana:",
        "data-docs:",
    ]
    for service in required_services:
        assert f"  {service}" in compose


def test_compose_uses_health_checks_and_persistent_volumes() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert compose.count("healthcheck:") >= 2
    assert "condition: service_healthy" in compose
    assert "postgres_data:" in compose
    assert "airflow_logs:" in compose
    assert "mlflow_data:" in compose
    assert "grafana_data:" in compose
    assert "restart: unless-stopped" in compose
    assert "version:" not in compose


def test_services_use_docker_names_instead_of_localhost() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "API_BASE_URL: http://api:8000" in compose
    assert "PREDICTION_API_URL: http://api:8000" in compose
    assert "@postgres:5432" in compose


def test_api_image_uses_python_312_and_healthcheck_dependency() -> None:
    dockerfile = read("docker/api.Dockerfile")
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM python:3.12-slim")
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "curl" in dockerfile
    assert "http://localhost:8000/health" in compose


def test_airflow_uses_version_3_and_mounts_dags() -> None:
    dockerfile = read("docker/airflow.Dockerfile")
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "FROM apache/airflow:3." in dockerfile
    assert "./dags:/opt/airflow/dags" in compose
    assert "schedule" not in dockerfile
    assert 'command: ["airflow", "api-server"]' in compose
    assert 'command: ["airflow", "scheduler"]' in compose


def test_secrets_are_referenced_through_environment_variables() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "env_file: .env" in compose
    assert "${POSTGRES_PASSWORD}" in compose
    assert "${GRAFANA_ADMIN_PASSWORD}" in compose
    assert "AIRFLOW__API__SECRET_KEY: ${AIRFLOW_API_SECRET_KEY}" in compose
    assert "AIRFLOW__API_AUTH__JWT_SECRET: ${AIRFLOW_JWT_SECRET}" in compose
    assert "TEAMS_WEBHOOK_URL=https://" not in compose


def test_postgres_initializer_is_executable() -> None:
    initializer = PROJECT_ROOT / "docker/postgres/init-databases.sh"
    assert initializer.stat().st_mode & stat.S_IXUSR


def test_postgres_initializer_uses_unix_line_endings() -> None:
    attributes = read(".gitattributes")
    initializer = PROJECT_ROOT / "docker/postgres/init-databases.sh"

    assert "*.sh text eol=lf" in attributes.splitlines()
    assert b"\r\n" not in initializer.read_bytes()


def test_airflow_database_connections_use_configured_postgres_user() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    assert (
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: "
        "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}"
        "@postgres:5432/airflow"
    ) in compose
    assert (
        "AIRFLOW_CONN_DSP_POSTGRES: "
        "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}"
        "@postgres:5432/${POSTGRES_DB}"
    ) in compose
    assert "postgresql+psycopg2://airflow:" not in compose


def test_ingestion_dag_avoids_api_database_imports_at_module_load() -> None:
    tree = ast.parse(read("dags/ingestion_dag.py"))
    database_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("app.database")
    ]
    assert database_imports == []


def test_nginx_serves_the_generated_great_expectations_data_docs() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    expected_mount = (
        "./data/gx/gx/uncommitted/data_docs/local_site:"
        "/usr/share/nginx/html:ro"
    )
    assert expected_mount in compose
    assert "./data/gx/uncommitted/data_docs/local_site:" not in compose
