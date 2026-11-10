from __future__ import annotations

from app.training.registry import is_missing_champion


def test_missing_alias_error_from_mlflow_means_no_champion() -> None:
    message = (
        "INVALID_PARAMETER_VALUE: Registered model alias champion "
        "not found."
    )

    assert is_missing_champion("INVALID_PARAMETER_VALUE", message) is True


def test_missing_model_error_means_no_champion() -> None:
    message = "RESOURCE_DOES_NOT_EXIST: Registered Model not found"

    assert is_missing_champion("RESOURCE_DOES_NOT_EXIST", message) is True


def test_other_mlflow_errors_are_not_treated_as_no_champion() -> None:
    message = "INTERNAL_ERROR: database is locked"

    assert is_missing_champion("INTERNAL_ERROR", message) is False
