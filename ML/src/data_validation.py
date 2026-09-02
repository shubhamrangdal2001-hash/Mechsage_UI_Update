"""Data validation utilities for raw and feature-engineered C-MAPSS frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Structured data validation results.

    Attributes:
        dataset_name: Name of the dataset/split being validated.
        row_count: Number of rows.
        column_count: Number of columns.
        missing_values: Missing value counts per column.
        duplicate_rows: Number of duplicated rows.
        dtype_map: Column dtype mapping.
        outlier_counts: IQR-based outlier counts per numeric column.
        errors: Validation errors that should be inspected.
        warnings: Non-blocking validation warnings.
    """

    dataset_name: str
    row_count: int
    column_count: int
    missing_values: dict[str, int]
    duplicate_rows: int
    dtype_map: dict[str, str]
    outlier_counts: dict[str, int]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "dtype_map": self.dtype_map,
            "outlier_counts": self.outlier_counts,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _outlier_counts(df: pd.DataFrame, numeric_columns: Iterable[str]) -> dict[str, int]:
    """Compute IQR outlier counts for numeric columns."""
    counts: dict[str, int] = {}
    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            counts[column] = 0
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if np.isclose(iqr, 0.0):
            counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        counts[column] = int(((series < lower) | (series > upper)).sum())
    return counts


def validate_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: Optional[Iterable[str]] = None,
    target_columns: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Validate a dataframe for common data quality issues.

    Args:
        df: DataFrame to validate.
        dataset_name: Human-readable split or dataset name.
        required_columns: Columns that must be present.
        target_columns: Target columns where missing values are hard errors.

    Returns:
        Structured validation report.
    """
    errors: list[str] = []
    warnings: list[str] = []
    required = list(required_columns or [])
    targets = list(target_columns or [])

    missing_required = sorted(set(required) - set(df.columns))
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")

    missing_values = {column: int(value) for column, value in df.isna().sum().items()}
    for column in targets:
        if column in missing_values and missing_values[column] > 0:
            errors.append(f"Target column {column} has {missing_values[column]} missing values")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} duplicate rows")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = _outlier_counts(df, numeric_columns)
    high_outlier_columns = [column for column, count in outliers.items() if count > 0]
    if high_outlier_columns:
        warnings.append(f"IQR outliers detected in {len(high_outlier_columns)} numeric columns")

    if "RUL" in df.columns and (df["RUL"] < 0).any():
        errors.append("RUL contains negative values")
    if "anomaly" in df.columns and not set(df["anomaly"].dropna().unique()).issubset({0, 1}):
        errors.append("anomaly column must contain only 0/1 labels")

    return ValidationReport(
        dataset_name=dataset_name,
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        dtype_map={column: str(dtype) for column, dtype in df.dtypes.items()},
        outlier_counts=outliers,
        errors=errors,
        warnings=warnings,
    )
