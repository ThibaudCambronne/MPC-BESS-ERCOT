import numpy as np
import pandas as pd

from src.globals import DELTA_T
from src.stage1_da_scheduler import solve_da_schedule, solve_da_schedule_cvxpy
from src.utils.battery_model import BatteryParams
from src.utils.data_classes import DAScheduleResult

from .selectors import ALGO_CVXPY, ALGO_PYOMO


def _to_hourly_average_prices(price_series: pd.Series) -> pd.Series:
    if not isinstance(price_series.index, pd.DatetimeIndex):
        raise ValueError("Price series must use a DatetimeIndex.")
    return price_series.resample("h").mean()


def _compute_tb_spread_from_hourly(hourly_prices: pd.Series, n_pairs: int) -> float:
    if n_pairs * 2 >= len(hourly_prices):
        raise ValueError(
            f"n_pairs ({n_pairs}) is too large for the number of hourly prices. "
            f"It must be lower than {len(hourly_prices) // 2}."
        )

    values = np.sort(hourly_prices.dropna().to_numpy(dtype=float))

    return float(values[-n_pairs:].sum() - values[:n_pairs].sum())


def _tb_hourly_volume_mwh(battery: BatteryParams, n_pairs: int) -> float:
    usable_energy_mwh = max(
        (battery.soc_max - battery.soc_min) * battery.capacity_mwh,
        0.0,
    )
    energy_limited_mw = usable_energy_mwh / n_pairs
    return float(max(min(battery.power_max_mw, energy_limited_mw), 0.0))


def compute_tb_strategy_revenue(
    price_series: pd.Series,
    battery: BatteryParams,
    n_pairs: int,
) -> float:
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")

    hourly_prices = _to_hourly_average_prices(price_series)
    spread = _compute_tb_spread_from_hourly(
        hourly_prices=hourly_prices, n_pairs=n_pairs
    )

    volume_mwh_per_hour = _tb_hourly_volume_mwh(battery=battery, n_pairs=n_pairs)
    return float(spread * volume_mwh_per_hour)


def compute_daily_tb_strategy_revenues(
    da_prices: pd.Series,
    rt_prices: pd.Series,
    battery: BatteryParams,
) -> dict[str, float]:
    return {
        "tb2_da_revenue": compute_tb_strategy_revenue(
            price_series=da_prices,
            battery=battery,
            n_pairs=2,
        ),
        "tb2_rt_revenue": compute_tb_strategy_revenue(
            price_series=rt_prices,
            battery=battery,
            n_pairs=2,
        ),
        "tb4_da_revenue": compute_tb_strategy_revenue(
            price_series=da_prices,
            battery=battery,
            n_pairs=4,
        ),
        "tb4_rt_revenue": compute_tb_strategy_revenue(
            price_series=rt_prices,
            battery=battery,
            n_pairs=4,
        ),
    }


def solve_schedule_for_algorithm(
    algorithm: str,
    da_price_forecast: pd.Series,
    rt_price_forecast: pd.Series,
    battery: BatteryParams,
    initial_soc: float = 0.5,
    end_of_day_soc: float = 0.5,
) -> DAScheduleResult:
    if algorithm == ALGO_CVXPY:
        return solve_da_schedule_cvxpy(
            da_price_forecast=da_price_forecast,
            rt_price_forecast=rt_price_forecast,
            battery=battery,
            initial_soc=initial_soc,
            end_of_day_soc=end_of_day_soc,
        )

    if algorithm == ALGO_PYOMO:
        return solve_da_schedule(
            da_price_forecast=da_price_forecast,
            rt_price_forecast=rt_price_forecast,
            battery=battery,
            initial_soc=initial_soc,
            end_of_day_soc=end_of_day_soc,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def compute_revenue_series(
    da_forecast: pd.Series,
    rt_forecast: pd.Series,
    da_bids: np.ndarray,
    rt_bids: np.ndarray,
    da_forecast_perfect: pd.Series,
    rt_forecast_perfect: pd.Series,
    da_bids_perfect: np.ndarray,
    rt_bids_perfect: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planned_step_revenue = (
        -(da_forecast.values * da_bids + rt_forecast.values * rt_bids) * DELTA_T
    )
    realized_step_revenue = (
        -(da_forecast_perfect.values * da_bids + rt_forecast_perfect.values * rt_bids)
        * DELTA_T
    )
    perfect_step_revenue = (
        -(
            da_forecast_perfect.values * da_bids_perfect
            + rt_forecast_perfect.values * rt_bids_perfect
        )
        * DELTA_T
    )
    return planned_step_revenue, realized_step_revenue, perfect_step_revenue
