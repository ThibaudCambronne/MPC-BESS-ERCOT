"""Shared color palette for app charts and forecast plots."""

from src.globals import TYPE_FORECASTS

# Core market colors
DA_COLOR = "#1f77b4"
DA_STAGE_DA_FORECAST_COLOR = "#6995b4"
RT_COLOR = "#ff7f0e"
DA_STAGE_RT_FORECAST_COLOR = "#fcad69"

# Revenue and decision-series colors
REVENUE_COLOR = "#512d7e"
DA_STAGE_REVENUE_COLOR = "#b369c0"
PERFECT_REVENUE_COLOR = "#2ca02c"

# Neutral/supporting colors
HISTORICAL_COLOR = "#000000"
SOC_COLOR = "#000000"
DEFAULT_FORECAST_COLOR = "#4e4949"
MISMATCH_FILL_RGBA = "rgba(220, 20, 60, 0.10)"

# Forecast-method palette (single-market plots)
FORECAST_METHOD_COLORS: dict[TYPE_FORECASTS, str] = {
    "persistence": "#a0772c",
    "perfect": PERFECT_REVENUE_COLOR,
    "xgboost": "#d62728",
    "regression": "#7867bd",
}
