from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
        nullable=False,
        index=True,
    )


class IngestionStat(Base):
    __tablename__ = "ingestion_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    valid_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    invalid_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    invalid_percent: Mapped[float] = mapped_column(Float, nullable=False)
    criticality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )
    missing_columns: Mapped[list] = mapped_column(JSON, nullable=False)
    error_counts: Mapped[dict] = mapped_column(JSON, nullable=False)
    gx_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gx_failed_expectations: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    gx_report_path: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
        nullable=False,
        index=True,
    )


class TrainingStat(Base):
    __tablename__ = "training_stats"
    __table_args__ = (
        UniqueConstraint(
            "model_name",
            "model_version",
            name="uq_training_stats_model_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    mlflow_run_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    training_rows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    numerical_stats: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )
    categorical_stats: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
        nullable=False,
    )
