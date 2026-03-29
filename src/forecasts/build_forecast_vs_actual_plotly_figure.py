import pandas as pd
import plotly.graph_objects as go


def build_forecast_vs_actual_plotly_figure(
    current_time: pd.Timestamp,
    data: pd.DataFrame,
    forecasts: dict[str, pd.Series],
    market: str,
    price_col: str,
):
    """Build a Plotly figure comparing one or more forecasts against actual prices."""

    if not forecasts:
        raise ValueError("At least one forecast series must be provided.")

    first_forecast = next(iter(forecasts.values()))
    forecast_index = first_forecast.index

    for method_name, forecast in forecasts.items():
        if not forecast.index.equals(forecast_index):
            raise ValueError(
                "All forecast series must share the same index. "
                f"Mismatched series: {method_name}"
            )

    forecast_window_df = data.loc[
        (data.index >= forecast_index.min()) & (data.index <= forecast_index.max()),
        [price_col],
    ]
    actual = forecast_window_df[price_col].reindex(forecast_index)

    historical_start = current_time - pd.Timedelta(days=5)
    historical = data.loc[
        (data.index >= historical_start) & (data.index < current_time),
        price_col,
    ]

    color_map = {
        "actual": "#1f77b4",
        "persistence": "#ff7f0e",
        "perfect": "#2ca02c",
        "xgboost": "#d62728",
        "regression": "#9467bd",
    }

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(historical.index),
            y=list(historical.values),
            mode="lines",
            name="Historical (5d)",
            line={"color": "#000000", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(actual.index),
            y=list(actual.values),
            mode="lines",
            name="Actual",
            line={"color": color_map["actual"], "width": 2},
        )
    )
    for method_name, forecast in forecasts.items():
        trace_name = f"{method_name.capitalize()} Forecast"
        fig.add_trace(
            go.Scatter(
                x=list(forecast.index),
                y=list(forecast.values),
                mode="lines",
                name=trace_name,
                line={
                    "color": color_map.get(method_name.lower(), "#4e4949"),
                    "width": 2,
                    "dash": "dash",
                },
            )
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
                current_time - pd.Timedelta(hours=8),
                forecast_index.max(),
            ]
        },
        # margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig
