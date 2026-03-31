import pandas as pd
import plotly.graph_objects as go


def _get_shared_forecast_index(forecasts: dict[str, pd.Series]) -> pd.DatetimeIndex:
    if not forecasts:
        raise ValueError("At least one forecast series must be provided.")

    first_forecast = next(iter(forecasts.values()))
    forecast_index = pd.DatetimeIndex(first_forecast.index)

    for method_name, forecast in forecasts.items():
        if not pd.DatetimeIndex(forecast.index).equals(forecast_index):
            raise ValueError(
                "All forecast series must share the same index. "
                f"Mismatched series: {method_name}"
            )

    return forecast_index


def _add_market_traces(
    fig: go.Figure,
    forecast_index: pd.DatetimeIndex,
    historical: pd.Series,
    actual: pd.Series,
    forecast_series: dict[str, pd.Series],
    historical_name: str,
    actual_name: str,
    forecast_name_template: str,
    historical_color: str,
    actual_color: str,
    forecast_colors: dict[str, str],
    historical_width: int = 1,
    historical_opacity: float = 1.0,
):
    fig.add_trace(
        go.Scatter(
            x=list(historical.index),
            y=list(historical.values),
            mode="lines",
            name=historical_name,
            line={"color": historical_color, "width": historical_width},
            opacity=historical_opacity,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(actual.index),
            y=list(actual.values),
            mode="lines",
            name=actual_name,
            line={"color": actual_color, "width": 2},
        )
    )

    for method_name, forecast in forecast_series.items():
        forecast_aligned = forecast.reindex(forecast_index)
        fig.add_trace(
            go.Scatter(
                x=list(forecast_aligned.index),
                y=list(forecast_aligned.values),
                mode="lines",
                name=forecast_name_template.format(method=method_name.capitalize()),
                line={
                    "color": forecast_colors.get(method_name.lower(), "#4e4949"),
                    "width": 2,
                    "dash": "dash",
                },
            )
        )


def _add_market_order_mismatch_rectangles(
    fig: go.Figure,
    forecast_index: pd.DatetimeIndex,
    da_forecast: pd.Series,
    rt_forecast: pd.Series,
    actual_da: pd.Series,
    actual_rt: pd.Series,
):
    mismatch = ((da_forecast > rt_forecast) != (actual_da > actual_rt)).fillna(False)
    if len(forecast_index) > 1:
        step = forecast_index[1] - forecast_index[0]
    else:
        step = pd.Timedelta(minutes=15)

    active_start = None
    for ts, is_mismatch in mismatch.items():
        if is_mismatch and active_start is None:
            active_start = ts
        elif (not is_mismatch) and active_start is not None:
            fig.add_vrect(
                x0=active_start,
                x1=ts,
                fillcolor="rgba(220, 20, 60, 0.10)",
                line_width=0,
                layer="below",
            )
            active_start = None

    if active_start is not None:
        fig.add_vrect(
            x0=active_start,
            x1=forecast_index.max() + step,
            fillcolor="rgba(220, 20, 60, 0.10)",
            line_width=0,
            layer="below",
        )


def build_forecast_vs_actual_plotly_figure(
    current_time: pd.Timestamp,
    data: pd.DataFrame,
    forecasts: dict[str, pd.Series],
    market: str,
    price_col: str,
    rt_forecasts: dict[str, pd.Series] | None = None,
    rt_price_col: str | None = None,
    highlight_market_order_mismatch: bool = False,
    historical_days: int = 5,
    visible_history_hours: int = 8,
):
    """Build a Plotly figure comparing one or more forecasts against actual prices."""
    forecast_index = _get_shared_forecast_index(forecasts)

    historical_start = current_time - pd.Timedelta(days=historical_days)
    fig = go.Figure()

    actual_da = data.loc[forecast_index, price_col]
    historical_da = data.loc[
        (data.index >= historical_start) & (data.index < current_time),
        price_col,
    ]

    if rt_forecasts is None:
        color_map = {
            "actual": "#1f77b4",
            "persistence": "#ff7f0e",
            "perfect": "#2ca02c",
            "xgboost": "#d62728",
            "regression": "#9467bd",
        }

        _add_market_traces(
            fig=fig,
            forecast_index=forecast_index,
            historical=historical_da,
            actual=actual_da,
            forecast_series=forecasts,
            historical_name=f"Historical ({historical_days}d)",
            actual_name="Actual",
            forecast_name_template="{method} Forecast",
            historical_color="#000000",
            actual_color=color_map["actual"],
            forecast_colors=color_map,
            historical_width=2,
            historical_opacity=1.0,
        )
    else:
        if rt_price_col is None:
            raise ValueError("rt_price_col is required when rt_forecasts is provided.")

        rt_forecast_index = _get_shared_forecast_index(rt_forecasts)
        if not rt_forecast_index.equals(forecast_index):
            raise ValueError("RT forecast index must match DA forecast index.")

        if set(rt_forecasts.keys()) != set(forecasts.keys()):
            raise ValueError(
                "forecasts and rt_forecasts must use matching method keys."
            )

        actual_rt = data.loc[forecast_index, rt_price_col]
        historical_rt = data.loc[
            (data.index >= historical_start) & (data.index < current_time),
            rt_price_col,
        ]

        market_colors = {
            "DA": "#1f77b4",
            "RT": "#ff7f0e",
        }

        _add_market_traces(
            fig=fig,
            forecast_index=forecast_index,
            historical=historical_da,
            actual=actual_da,
            forecast_series=forecasts,
            historical_name=f"DA Historical ({historical_days}d)",
            actual_name="DA Actual",
            forecast_name_template="DA {method} Forecast",
            historical_color=market_colors["DA"],
            actual_color=market_colors["DA"],
            forecast_colors={key: market_colors["DA"] for key in forecasts},
            historical_width=1,
            historical_opacity=0.4,
        )

        _add_market_traces(
            fig=fig,
            forecast_index=forecast_index,
            historical=historical_rt,
            actual=actual_rt,
            forecast_series=rt_forecasts,
            historical_name=f"RT Historical ({historical_days}d)",
            actual_name="RT Actual",
            forecast_name_template="RT {method} Forecast",
            historical_color=market_colors["RT"],
            actual_color=market_colors["RT"],
            forecast_colors={key: market_colors["RT"] for key in rt_forecasts},
            historical_width=1,
            historical_opacity=0.4,
        )

        if highlight_market_order_mismatch:
            first_method = next(iter(forecasts.keys()))
            _add_market_order_mismatch_rectangles(
                fig=fig,
                forecast_index=forecast_index,
                da_forecast=forecasts[first_method],
                rt_forecast=rt_forecasts[first_method],
                actual_da=actual_da,
                actual_rt=actual_rt,
            )

    fig.add_vline(
        x=current_time,
        line_width=1,
        line_dash="dash",
        line_color="black",
    )
    fig.update_layout(
        title=f"{market} Price Forecast Comparison",
        xaxis_title="Time",
        yaxis_title="Price",
        legend_title="Series",
        template="plotly_white",
        xaxis={
            "range": [
                current_time - pd.Timedelta(hours=visible_history_hours),
                forecast_index.max(),
            ]
        },
        # margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig
