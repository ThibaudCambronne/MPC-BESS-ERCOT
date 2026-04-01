import streamlit as st


def render_revenue_kpis(
    planned_total: float,
    realized_total: float,
    perfect_total: float,
    tb2_da_total: float | None = None,
    tb4_da_total: float | None = None,
) -> None:
    show_tb = tb2_da_total is not None or tb4_da_total is not None
    kpi_tb2 = None
    kpi_tb4 = None

    if show_tb:
        kpi1, divider, kpi_tb2, kpi_tb4, kpi2, kpi3 = st.columns([1, 0.1, 1, 1, 1, 1])
    else:
        kpi1, divider, kpi2, kpi3 = st.columns([1, 0.1, 1, 1])

    kpi1.metric(
        "Realized Revenue",
        f"${realized_total:,.0f}",
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

    if show_tb:
        assert kpi_tb2 is not None
        assert kpi_tb4 is not None
        kpi_tb2.metric(
            "TB2 DA",
            f"${tb2_da_total:,.0f}",
            delta=f"{((tb2_da_total or 0) - realized_total) / abs(realized_total):.0%}",
        )
        kpi_tb4.metric(
            "TB4 DA",
            f"${tb4_da_total:,.0f}",
            delta=f"{((tb4_da_total or 0) - realized_total) / abs(realized_total):.0%}",
        )

    kpi2.metric(
        "Planned Revenue",
        f"${planned_total:,.0f}",
        delta=f"{(planned_total - realized_total) / abs(realized_total):.0%}",
    )
    kpi3.metric(
        "Perfect-Decision Revenue",
        f"${perfect_total:,.0f}",
        delta=f"{(perfect_total - realized_total) / abs(realized_total):.0%}",
    )
