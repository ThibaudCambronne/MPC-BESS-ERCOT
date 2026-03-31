import pandas as pd
import streamlit as st

from app.utils_app.data import get_cached_ercot_data
from src.forecasts.build_forecast_vs_actual_plotly_figure import (
    build_forecast_vs_actual_plotly_figure,
)
from src.forecasts.forecaster import get_forecast
from src.globals import PRICE_NODE, TYPE_FORECASTS

st.title("Price Forecasts")
st.caption("Forecast comparison across all methods")

data = get_cached_ercot_data()

datetime_index = pd.DatetimeIndex(data.index)

col_date, col_market, col_horizon = st.columns(3)
with col_date:
    available_dates = pd.Index(datetime_index.date).unique()
    min_date = pd.Timestamp(available_dates.min()).date()
    max_date = pd.Timestamp(available_dates.max()).date()

    selected_date = st.date_input(
        "Choose a date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )
    current_time = pd.Timestamp(selected_date)

with col_market:
    selected_market = st.selectbox("Choose a market", options=["DA", "RT"], index=0)

with col_horizon:
    horizon_hours = int(
        st.number_input(
            "Horizon (hours)",
            min_value=1,
            max_value=72,
            value=24,
            step=1,
        )
    )

price_col = f"{PRICE_NODE}_{selected_market}M"

methods: list[TYPE_FORECASTS] = [
    "persistence",
    "perfect",
    "xgboost",
    "regression",
]
forecasts: dict[str, pd.Series] = {}

for method in methods:
    try:
        forecasts[method] = get_forecast(
            data=data,
            current_time=current_time,
            horizon_hours=horizon_hours,
            market=selected_market,  # type: ignore
            method=method,
            price_node=PRICE_NODE,
            verbose=False,
        )
    except ValueError as exc:
        st.warning(f"Could not generate {method} forecast: {exc}")

if not forecasts:
    st.error("No forecast method could be computed for this date and market.")
    st.stop()

first_forecast = next(iter(forecasts.values()))
if not first_forecast.index.isin(data.index).all():
    missing_count = int((~first_forecast.index.isin(data.index)).sum())
    st.warning(
        f"Actual prices are missing for {missing_count} forecast timesteps. "
        "The chart will show gaps where data is unavailable."
    )

figure = build_forecast_vs_actual_plotly_figure(
    current_time=current_time,
    data=data,
    forecasts=forecasts,
    market=selected_market,  # type: ignore
    price_col=price_col,
)

st.plotly_chart(figure, width="stretch")

forecast_end = first_forecast.index.max()

st.write(
    f"Using node: **{PRICE_NODE}**, methods: **{', '.join(forecasts.keys())}**, "
    f"horizon: **{horizon_hours}h**, window: **{current_time:%Y-%m-%d %H:%M} -> {forecast_end:%Y-%m-%d %H:%M}**"
)
