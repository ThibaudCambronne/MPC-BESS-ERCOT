import streamlit as st


def render_revenue_kpis(
    planned_total: float,
    realized_total: float,
    perfect_total: float,
) -> None:
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Planned Revenue", f"${planned_total:,.0f}")
    kpi2.metric(
        "Realized Revenue",
        f"${realized_total:,.0f}",
        delta=f"{(realized_total - planned_total):,.0f}",
    )
    kpi3.metric("Perfect-Decision Revenue", f"${perfect_total:,.0f}")
