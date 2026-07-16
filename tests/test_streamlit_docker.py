from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "docker" / "streamlit.Dockerfile"


def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_streamlit_dockerfile_exists() -> None:
    assert DOCKERFILE.exists()


def test_streamlit_image_uses_python_312() -> None:
    assert dockerfile_text().startswith("FROM python:3.12-slim")


def test_streamlit_image_installs_project_requirements() -> None:
    content = dockerfile_text()
    assert "COPY requirements.txt" in content
    assert "pip install --no-cache-dir -r requirements.txt" in content


def test_streamlit_image_exposes_and_starts_port_8501() -> None:
    content = dockerfile_text()
    assert "EXPOSE 8501" in content
    assert '"streamlit", "run", "app/web/Home.py"' in content
    assert '"--server.address=0.0.0.0"' in content
    assert '"--server.port=8501"' in content


def test_streamlit_image_has_healthcheck() -> None:
    content = dockerfile_text()
    assert "HEALTHCHECK" in content
    assert "http://localhost:8501/_stcore/health" in content
    assert "curl" in content


def test_streamlit_image_does_not_copy_model_or_database() -> None:
    content = dockerfile_text()
    assert "COPY models" not in content
    assert "COPY data" not in content
    assert "DATABASE_URL" not in content
    assert "TEAMS_WEBHOOK_URL" not in content


def test_api_url_is_supplied_by_compose_environment() -> None:
    content = dockerfile_text()
    assert "API_BASE_URL=" not in content
