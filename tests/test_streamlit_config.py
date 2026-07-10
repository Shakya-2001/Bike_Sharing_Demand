from __future__ import annotations

from pathlib import Path

from app.web import api_client


def test_streamlit_files_exist() -> None:
    assert Path("app/web/Home.py").exists()
    assert Path("app/web/api_client.py").exists()


def test_api_client_defaults_to_local_fastapi() -> None:
    assert api_client.API_BASE_URL == "http://127.0.0.1:8000"
    assert api_client.REQUEST_TIMEOUT == 10
