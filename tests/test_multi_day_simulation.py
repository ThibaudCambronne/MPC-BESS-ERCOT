import numpy as np
import pandas as pd
import pytest

from src.globals import FREQUENCY, TIME_STEPS_PER_HOUR
from src.multi_day_simulation import multi_day_simulation
from src.utils.battery_model import BatteryParams
from src.utils.data_classes import DaySimulationResult


def _build_day_result(
    operating_day: pd.Timestamp, base_value: float
) -> DaySimulationResult:
    n_steps = 24 * TIME_STEPS_PER_HOUR
    index = pd.date_range(start=operating_day, periods=n_steps, freq=FREQUENCY)

    da_energy = np.full(n_steps, base_value)
    da_stage_rt_energy = np.full(n_steps, base_value + 1)
    rt_energy = np.full(n_steps, base_value + 2)

    da_prices = pd.Series(np.full(n_steps, 10.0 + base_value), index=index)
    da_stage_rt_prices = pd.Series(np.full(n_steps, 20.0 + base_value), index=index)
    rt_prices = pd.Series(np.full(n_steps, 30.0 + base_value), index=index)

    return DaySimulationResult(
        date=operating_day,
        da_energy_bids=da_energy,
        da_stage_rt_energy_bids=da_stage_rt_energy,
        rt_energy_bids=rt_energy,
        da_forecast_used=da_prices,
        da_stage_rt_forecast_used=da_stage_rt_prices,
        rt_forecast_used=rt_prices,
        da_stage_soc_schedule=np.full(n_steps + 1, 0.5),
        soc_schedule=np.full(n_steps + 1, 0.5),
        expected_revenue=0.0,
    )


def test_multi_day_simulation_aggregates_outputs(monkeypatch: pytest.MonkeyPatch):
    day1 = pd.Timestamp("2025-06-01")
    day2 = pd.Timestamp("2025-06-02")

    selected_day1 = _build_day_result(day1, 1.0)
    perfect_day1 = _build_day_result(day1, 101.0)
    selected_day2 = _build_day_result(day2, 2.0)
    perfect_day2 = _build_day_result(day2, 102.0)

    calls = [selected_day1, perfect_day1, selected_day2, perfect_day2]

    def fake_one_day_simulation(*args, **kwargs):
        return calls.pop(0)

    monkeypatch.setattr(
        "src.multi_day_simulation.one_day_simulation",
        fake_one_day_simulation,
    )

    result = multi_day_simulation(
        data=pd.DataFrame(),
        start_day=day1,
        end_day=day2,
        battery=BatteryParams(),
        da_stage_kwargs={"initial_soc": 0.5, "end_of_day_soc": 0.5},
        use_rt_mpc=True,
        rt_stage_kwargs={"end_of_day_soc": 0.5},
        da_stage_forecast_method="persistence",
        rt_stage_forecast_method="persistence",
    )

    n_steps = 24 * TIME_STEPS_PER_HOUR
    assert len(result.index) == 2 * n_steps
    assert np.all(result.da_energy_bids[:n_steps] == 1.0)
    assert np.all(result.da_energy_bids[n_steps:] == 2.0)
    assert np.all(result.da_energy_bids_perfect[:n_steps] == 101.0)
    assert np.all(result.da_energy_bids_perfect[n_steps:] == 102.0)
    assert result.da_forecast_used.iloc[0] == 11.0
    assert result.da_forecast_perfect.iloc[0] == 111.0


def test_multi_day_simulation_fails_fast_on_day_error(monkeypatch: pytest.MonkeyPatch):
    day1 = pd.Timestamp("2025-06-01")
    day2 = pd.Timestamp("2025-06-02")

    selected_day1 = _build_day_result(day1, 1.0)
    perfect_day1 = _build_day_result(day1, 101.0)

    calls = [selected_day1, perfect_day1, RuntimeError("boom")]

    def fake_one_day_simulation(*args, **kwargs):
        value = calls.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        "src.multi_day_simulation.one_day_simulation",
        fake_one_day_simulation,
    )

    with pytest.raises(RuntimeError, match="2025-06-02"):
        multi_day_simulation(
            data=pd.DataFrame(),
            start_day=day1,
            end_day=day2,
            battery=BatteryParams(),
        )


def test_multi_day_simulation_rejects_invalid_date_range():
    with pytest.raises(ValueError, match="end_day"):
        multi_day_simulation(
            data=pd.DataFrame(),
            start_day=pd.Timestamp("2025-06-02"),
            end_day=pd.Timestamp("2025-06-01"),
            battery=BatteryParams(),
        )
