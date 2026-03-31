from typing import cast

import pandas as pd
import streamlit as st

from src.globals import TYPE_FORECASTS

ALGO_PYOMO = "Pyomo"
ALGO_CVXPY = "CVXPY"


def render_operating_day_selector(
    datetime_index: pd.DatetimeIndex,
    label: str = "Choose operating day",
) -> pd.Timestamp:
    available_dates = pd.Index(datetime_index.date).unique()
    min_date = pd.Timestamp(available_dates.min()).date()
    max_date = pd.Timestamp(available_dates.max()).date()

    selected_date = st.date_input(
        label,
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )
    return pd.Timestamp(selected_date)


def render_month_selector(
    datetime_index: pd.DatetimeIndex,
    label: str = "Choose month",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    months = pd.PeriodIndex(datetime_index.to_period("M").unique()).sort_values()
    selected_month = st.selectbox(label, options=months, index=len(months) - 1)

    month_period = pd.Period(selected_month, freq="M")
    month_start = month_period.start_time
    month_end = month_period.end_time.normalize()
    return month_start, month_end


def render_forecast_method_selector(
    label: str = "Forecast model",
    default_index: int = 0,
) -> TYPE_FORECASTS:
    methods: list[TYPE_FORECASTS] = [
        "persistence",
        "perfect",
        "xgboost",
        "regression",
    ]
    selected = st.selectbox(label, options=methods, index=default_index)
    return cast(TYPE_FORECASTS, selected)


def render_algorithm_selector(
    label: str = "Scheduling algorithm",
    default_index: int = 0,
) -> str:
    return str(
        st.selectbox(
            label,
            options=[ALGO_PYOMO, ALGO_CVXPY],
            index=default_index,
        )
    )
