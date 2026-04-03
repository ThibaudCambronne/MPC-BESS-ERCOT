import numpy as np
import pandas as pd
import pytest

from src.globals import TIME_STEPS_PER_HOUR
from src.multi_day_simulation import multi_day_simulation
from src.utils.battery_model import BatteryParams
from tests.helpers import build_day_result


def test_multi_day_simulation_aggregates_outputs(monkeypatch: pytest.MonkeyPatch):
    day1 = pd.Timestamp("2025-06-01")
    day2 = pd.Timestamp("2025-06-02")

    selected_day1 = build_day_result(day1, 1.0)
    perfect_day1 = build_day_result(day1, 101.0)
    selected_day2 = build_day_result(day2, 2.0)
    perfect_day2 = build_day_result(day2, 102.0)

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

    selected_day1 = build_day_result(day1, 1.0)
    perfect_day1 = build_day_result(day1, 101.0)

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
