from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from src.colors import DA_COLOR, PERFECT_REVENUE_COLOR, REVENUE_COLOR, RT_COLOR


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
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=planned_cumulative,
            mode="lines",
            name="Planned Cumulative",
            line={"color": REVENUE_COLOR, "width": 2, "dash": "dash"},
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=realized_cumulative,
            mode="lines",
            name="Realized Cumulative",
            line={"color": REVENUE_COLOR, "width": 2},
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=perfect_cumulative,
            mode="lines",
            name="Perfect Cumulative",
            line={"color": PERFECT_REVENUE_COLOR, "width": 2},
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )

    if not tb_marker_series:
        return

    for marker_trace in tb_marker_series:
        marker_dict = {
            "size": 8,
            "color": DA_COLOR if "DA" in marker_trace["name"] else RT_COLOR,
            "symbol": "circle" if "TB2" in marker_trace["name"] else "diamond",
        }
        fig.add_trace(
            go.Scatter(
                x=marker_trace["x"],
                y=marker_trace["y"],
                mode="markers",
                name=marker_trace["name"],
                marker=marker_dict,
                legendgroup=legendgroup,
                opacity=0.8,
            ),
            row=row,
            col=col,
        )
