
import numpy as np
import pandas as pd
import html
import math
from typing import Any, Union, Optional

def is_numeric_series(s: pd.Series) -> bool:
    """Check if a pandas Series is numeric."""
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return False
    return pd.api.types.is_numeric_dtype(s)

def safe_series(df: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    """Safely get a series from a DataFrame."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)

def non_na_numeric(series: pd.Series) -> pd.Series:
    """Clean a series to only include finite numeric values."""
    if series is None or not isinstance(series, pd.Series):
        return pd.Series([], dtype=float)
    if not pd.api.types.is_numeric_dtype(series):
        series = pd.to_numeric(series, errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()

def safe_text(series: pd.Series, fallback_prefix: str = "Item") -> pd.Series:
    """Safely convert a series to text, handling missing values."""
    if series is None or series.isna().all():
        return pd.Series([f"{fallback_prefix} {i+1}" for i in range(len(series) if series is not None else 0)])
    return series.astype(str).fillna(fallback_prefix)

def to_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float with error handling."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default

def to_int(value: Any, default: int = 0) -> int:
    """Convert value to int with error handling."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default

def safe_html(text: Any, default: str = "") -> str:
    """Convert text to HTML-safe string."""
    if text is None:
        return default
    return html.escape(str(text))

def get_priority_bucket(score: float) -> tuple[str, str]:
    """Get priority bucket and label from score."""
    if score >= 0.7:
        return "high", "High Priority"
    if score >= 0.4:
        return "medium", "Medium Priority"
    return "low", "Low Priority"