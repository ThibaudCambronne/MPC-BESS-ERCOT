import streamlit as st


def render_revenue_kpis(
    da_stage_actual: float,
    rt_stage_actual: float,
    perfect_total: float,
    tb2_da_total: float | None = None,
    tb4_da_total: float | None = None,
) -> None:
    show_tb = tb2_da_total is not None or tb4_da_total is not None
    kpi_tb2 = None
    kpi_tb4 = None

    if show_tb:
        kpi1, divider, kpi2, kpi_tb2, kpi_tb4, kpi3 = st.columns([1, 0.1, 1, 1, 1, 1])
    else:
        kpi1, divider, kpi2, kpi3 = st.columns([1, 0.1, 1, 1])

    kpi1.metric(
        "Realized Revenue",
        f"${rt_stage_actual:,.0f}",
    )

    with divider:
        # Add a vertical divider line
        st.markdown(
            """
            <style>
            .divider {
                border-left: 1px solid #ccc;
                height: 100px;
                margin: 0 auto;
            }
            </style>
            <div class="divider"></div>
            """,
            unsafe_allow_html=True,
        )

    kpi2.metric(
        "DA Stage Only Revenue",
        f"${da_stage_actual:,.0f}",
        delta=f"{(da_stage_actual - rt_stage_actual) / abs(rt_stage_actual):.0%}",
        delta_color="inverse",
    )

    if show_tb:
        assert kpi_tb2 is not None
        assert kpi_tb4 is not None
        kpi_tb2.metric(
            "TB2 DA",
            f"${tb2_da_total:,.0f}",
            delta=f"{((tb2_da_total or 0) - rt_stage_actual) / abs(rt_stage_actual):.0%}",
            delta_color="inverse",
            help="Top-Bottom (TB2) of the actual DA market, multiplied by the battery power (limited by the battery duration). "
            "More info about this metric: [Modoenergy](https://modoenergy.com/research/en/top-bottom-tb-price-spreads-revenue-benchmark-us-iso-explainer)",
        )
        kpi_tb4.metric(
            "TB4 DA",
            f"${tb4_da_total:,.0f}",
            delta=f"{((tb4_da_total or 0) - rt_stage_actual) / abs(rt_stage_actual):.0%}",
            delta_color="inverse",
            help="Top-Bottom (TB4) of the actual DA market, multiplied by the battery power (limited by the battery duration). "
            "More info about this metric: [Modoenergy](https://modoenergy.com/research/en/top-bottom-tb-price-spreads-revenue-benchmark-us-iso-explainer)",
        )

    kpi3.metric(
        "Perfect-Decision Revenue",
        f"${perfect_total:,.0f}",
        delta=f"{(perfect_total - rt_stage_actual) / abs(rt_stage_actual):.0%}",
        delta_color="inverse",
    )


def render_forecast_accuracy_markdown(
    metric_rows: list[tuple[str, str, str, dict[str, float]]],
    title: str = "Forecast Accuracy",
) -> None:
    st.markdown(f"#### {title}")
    st.markdown(
        "Lower is better for MAE/RMSE/MAPE.",
        help="**MAE**: Mean Absolute Error"
        "\\\n **RMSE**: Root Mean Squared Error"
        "\\\n**sMAPE**: Symmetric Mean Absolute Percentage Error. "
        "A given error is more penalized when the actual value is small, and less penalized when the actual value is large."
        "\\\n**Bias**: Average error. "
        "Positive means over-forecasting, negative means under-forecasting.",
    )
    rows = [
        "| Market Forecasted | Forecast Stage | Forecasting Method | MAE | RMSE | Bias | sMAPE | Samples |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for market, stage, method, metrics in metric_rows:
        rows.append(
            "| "
            + f"{market}"
            + " | "
            + f"{stage}"
            + " | "
            + f"{method}"
            + " | "
            + f"{metrics.get('mae', float('nan')):.2f}"
            + " | "
            + f"{metrics.get('rmse', float('nan')):.2f}"
            + " | "
            + f"{metrics.get('bias', float('nan')):.2f}"
            + " | "
            + f"{metrics.get('smape_pct', float('nan')):.2f}%"
            + " | "
            + f"{int(metrics.get('n_points', 0)):,}"
            + " |"
        )

    st.markdown("\n".join(rows))
