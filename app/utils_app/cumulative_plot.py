from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.colors import (
    DA_COLOR,
    DA_STAGE_REVENUE_COLOR,
    PERFECT_REVENUE_COLOR,
    REVENUE_COLOR,
    RT_COLOR,
)
from src.globals import FREQUENCY


def _add_line_traces(
    fig: go.Figure,
    x_values: Iterable[Any],
    trace_specs: Sequence[dict[str, Any]],
    row: int,
    col: int,
    legendgroup: str,
) -> None:
    for spec in trace_specs:
        line: dict[str, Any] = {"color": spec["color"], "width": 2}
        if spec.get("dash"):
            line["dash"] = spec["dash"]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=spec["y"],
                mode="lines",
                name=spec["name"],
                line=line,
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )


def _add_tb_marker_traces(
    fig: go.Figure,
    tb_marker_series: Sequence[dict[str, Any]] | None,
    row: int,
    col: int,
    legendgroup: str,
    opacity: float,
) -> None:
    if not tb_marker_series:
        return

    for marker_trace in tb_marker_series:
        name = str(marker_trace["name"])
        color = DA_COLOR if "DA" in name else RT_COLOR
        symbol = "circle" if "TB2" in name else "diamond"
        fig.add_trace(
            go.Scatter(
                x=marker_trace["x"],
                y=marker_trace["y"],
                mode="markers",
                name=name,
                marker={"size": 8, "color": color, "symbol": symbol},
                legendgroup=legendgroup,
                opacity=opacity,
            ),
            row=row,
            col=col,
        )


def add_bid_soc_panel_traces(
    fig: go.Figure,
    operating_index: Iterable[Any],
    operating_day_start: pd.Timestamp,
    bid_traces: Sequence[dict[str, Any]],
    soc_values: np.ndarray,
    soc_name: str,
    soc_color: str,
    row: int,
    col: int = 1,
    legendgroup: str = "subplot2",
) -> None:
    for bid_trace in bid_traces:
        fig.add_trace(
            go.Bar(
                x=operating_index,
                y=np.round(np.asarray(bid_trace["y"]), 1),
                name=str(bid_trace["name"]),
                marker_color=str(bid_trace["color"]),
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )

    soc_index = pd.date_range(
        start=operating_day_start,
        periods=len(soc_values),
        freq=FREQUENCY,
    )
    fig.add_trace(
        go.Scatter(
            x=soc_index,
            y=np.round(soc_values * 100, 0),
            mode="lines",
            name=soc_name,
            line={"color": soc_color, "width": 2, "dash": "dot"},
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
        secondary_y=True,
    )


def add_cumulative_revenue_traces(
    fig: go.Figure,
    x_values: Iterable[Any],
    planned_cumulative: np.ndarray,
    realized_cumulative: np.ndarray,
    perfect_cumulative: np.ndarray,
    row: int,
    col: int,
    legendgroup: str = "subplot1",
    tb_marker_series: Sequence[dict[str, Any]] | None = None,
) -> None:
    _add_line_traces(
        fig=fig,
        x_values=x_values,
        trace_specs=[
            {
                "y": planned_cumulative,
                "name": "Planned Cumulative",
                "color": REVENUE_COLOR,
                "dash": "dash",
            },
            {
                "y": realized_cumulative,
                "name": "Realized Cumulative",
                "color": REVENUE_COLOR,
            },
            {
                "y": perfect_cumulative,
                "name": "Perfect Cumulative",
                "color": PERFECT_REVENUE_COLOR,
            },
        ],
        row=row,
        col=col,
        legendgroup=legendgroup,
    )

    _add_tb_marker_traces(
        fig=fig,
        tb_marker_series=tb_marker_series,
        row=row,
        col=col,
        legendgroup=legendgroup,
        opacity=0.8,
    )


def add_schedule_comparison_traces(
    fig: go.Figure,
    x_values: Iterable[Any],
    da_only_planned: np.ndarray,
    da_only_actual: np.ndarray,
    final_planned: np.ndarray,
    final_actual: np.ndarray,
    perfect_cumulative: np.ndarray,
    row: int,
    col: int,
    legendgroup: str = "subplot1",
    tb_marker_series: Sequence[dict[str, Any]] | None = None,
) -> None:
    _add_line_traces(
        fig=fig,
        x_values=x_values,
        trace_specs=[
            {
                "y": da_only_planned,
                "name": "DA-only Forecast",
                "color": DA_STAGE_REVENUE_COLOR,
                "dash": "dash",
            },
            {
                "y": da_only_actual,
                "name": "DA-only Actual",
                "color": DA_STAGE_REVENUE_COLOR,
            },
            {
                "y": final_planned,
                "name": "Final Forecast",
                "color": REVENUE_COLOR,
                "dash": "dash",
            },
            {
                "y": final_actual,
                "name": "Final Actual",
                "color": REVENUE_COLOR,
            },
            {
                "y": perfect_cumulative,
                "name": "Perfect Forecast",
                "color": PERFECT_REVENUE_COLOR,
            },
        ],
        row=row,
        col=col,
        legendgroup=legendgroup,
    )

    _add_tb_marker_traces(
        fig=fig,
        tb_marker_series=tb_marker_series,
        row=row,
        col=col,
        legendgroup=legendgroup,
        opacity=0.85,
    )
