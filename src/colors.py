"""Shared color palette for app charts and forecast plots."""

from src.globals import TYPE_FORECASTS

# Core market colors
DA_COLOR = "#1f77b4"
RT_COLOR = "#ff7f0e"

# Revenue and decision-series colors
REVENUE_COLOR = "#a74ea7"
PERFECT_REVENUE_COLOR = "#59a14f"

# Neutral/supporting colors
HISTORICAL_COLOR = "#000000"
SOC_COLOR = "#000000"
DEFAULT_FORECAST_COLOR = "#4e4949"
MISMATCH_FILL_RGBA = "rgba(220, 20, 60, 0.10)"

# Forecast-method palette (single-market plots)
FORECAST_METHOD_COLORS: dict[TYPE_FORECASTS, str] = {
    "persistence": RT_COLOR,
    "perfect": "#2ca02c",
    "xgboost": "#d62728",
    "regression": "#9467bd",
}
