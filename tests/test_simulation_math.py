import math

import pandas as pd
import pytest

from app.utils_app.simulation_math import compute_forecast_accuracy_metrics


def test_compute_forecast_accuracy_metrics_basic():
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    forecast = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    actual = pd.Series([12.0, 18.0, 33.0, 39.0], index=index)

    metrics = compute_forecast_accuracy_metrics(forecast=forecast, actual=actual)

    assert metrics["n_points"] == 4
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(4.5))
    assert metrics["bias"] == pytest.approx(-0.5)
    assert metrics["smape_pct"] == pytest.approx(10.1908969445)
    assert "mape_pct" not in metrics


def test_compute_forecast_accuracy_metrics_handles_zero_actuals():
    index = pd.date_range("2025-01-01", periods=3, freq="15min")
    forecast = pd.Series([1.0, 2.0, 3.0], index=index)
    actual = pd.Series([0.0, 0.0, 0.0], index=index)

    metrics = compute_forecast_accuracy_metrics(forecast=forecast, actual=actual)

    assert metrics["n_points"] == 3
    assert metrics["smape_pct"] >= 0.0


def test_compute_forecast_accuracy_metrics_requires_overlap():
    forecast = pd.Series(
        [1.0, 2.0], index=pd.date_range("2025-01-01", periods=2, freq="15min")
    )
    actual = pd.Series(
        [1.5, 2.5], index=pd.date_range("2025-01-02", periods=2, freq="15min")
    )

    with pytest.raises(ValueError, match="empty overlap"):
        compute_forecast_accuracy_metrics(forecast=forecast, actual=actual)
