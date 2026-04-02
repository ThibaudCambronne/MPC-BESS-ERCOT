from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.globals import DELTA_T, TIME_STEPS_PER_HOUR

from .utils.battery_model import BatteryParams
from .utils.data_classes import DAScheduleResult

DEFAULT_INITIAL_SOC = 0.5
DEFAULT_END_OF_DAY_SOC = 0.5


def solve_da_schedule(
    da_price_forecast: pd.Series,
    rt_price_forecast: pd.Series,
    battery: BatteryParams,
    rt_price_uncertainty: Optional[pd.Series] = None,
    initial_soc: float = DEFAULT_INITIAL_SOC,
    end_of_day_soc: float = DEFAULT_END_OF_DAY_SOC,
    cvar_alpha: float = 0.90,
    cvar_weight: float = 0,
    rt_dispatch_penalty: float = 0,
    rt_uncertainty_default: float = 0,  # 20
    n_scenarios: int = 20,
    scenario_seed: Optional[int] = None,
    verbose: bool = False,
) -> DAScheduleResult:
    """
    Solve Stage 1 DA optimization problem with CVaR risk measure using Pyomo.

    CVaR is applied to total profit uncertainty across scenarios.
    We maximize: (1-λ) * E[profit] - λ * CVaR[cost]
    where CVaR[cost] protects against worst-case costs.

    Parameters
    ----------
    da_price_forecast : pd.Series
        Day-ahead energy prices for 24 hours [$/MWh]
    rt_price_forecast : pd.Series
        Real-time energy prices for 24 hours [$/MWh]
    battery : BatteryParams
        Battery parameters (capacity, power limits, efficiency, etc.)
    rt_price_uncertainty : Optional[pd.Series]
        Real-time price uncertainty/volatility (std dev) for each hour [$/MWh]. \
            If None, rt_uncertainty_default is used for each hour.
    initial_soc : float
        Initial state of charge [fraction, 0-1]
    end_of_day_soc : float
        Target state of charge at end of day [fraction, 0-1]
    cvar_alpha : float
        Confidence level for CVaR (0.95 = protect against worst 5%)
    cvar_weight : float
        λ, weight on CVaR term (0=risk-neutral, 1=full CVaR focus)
    rt_dispatch_penalty : float
        Penalty per MW of RT dispatch to discourage RT reliance [$/MW]
    n_scenarios : int
        Number of RT price scenarios to generate for CVaR calculation
    scenario_seed : Optional[int]
        Random seed for scenario generation

    Returns
    -------
    DAScheduleResult
        Optimization results
    """
    # Number of time periods
    T = len(rt_price_forecast)

    if da_price_forecast.isna().any():
        print("Warning: DA price forecast contains NaN values. Filling forward.")
        da_price_forecast = da_price_forecast.ffill()

    if rt_price_forecast.isna().any():
        print("Warning: RT price forecast contains NaN values. Filling forward.")
        rt_price_forecast = rt_price_forecast.ffill()

    if rt_price_uncertainty is not None and rt_price_uncertainty.isna().any():
        print("Warning: RT price uncertainty contains NaN values. Filling forward.")
        rt_price_uncertainty = rt_price_uncertainty.ffill()

    # Convert prices to numpy arrays
    da_prices = da_price_forecast.values
    rt_prices = rt_price_forecast.values

    # Default uncertainty if not provided
    if rt_price_uncertainty is not None:
        rt_uncertainty = rt_price_uncertainty.values
    else:
        rt_uncertainty = np.ones(len(rt_prices)) * rt_uncertainty_default

    # Generate RT price scenarios
    if scenario_seed is not None:
        np.random.seed(scenario_seed)

    rt_price_scenarios = np.random.normal(
        loc=rt_prices[:, np.newaxis],
        scale=rt_uncertainty[:, np.newaxis],
        size=(T, n_scenarios),
    )
    # Optional: clip extreme scenarios
    # rt_price_scenarios = np.clip(rt_price_scenarios, 0, 80)  # Adjust bounds as needed

    # ==================== Build Pyomo Model ====================

    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.RangeSet(0, T - 1)  # Time periods
    model.T_soc = pyo.RangeSet(0, T)  # Time periods for SoC (includes initial)
    model.S = pyo.RangeSet(0, n_scenarios - 1)  # Scenarios

    # ==================== Decision Variables ====================

    # Energy bids
    model.p_da = pyo.Var(model.T, bounds=(-battery.power_max_mw, battery.power_max_mw))
    model.p_rt = pyo.Var(model.T, bounds=(-battery.power_max_mw, battery.power_max_mw))

    # Actual dispatch schedule
    model.p_real = pyo.Var(
        model.T, bounds=(-battery.power_max_mw, battery.power_max_mw)
    )

    # Charge/discharge
    model.p_discharge = pyo.Var(model.T, bounds=(0, battery.power_max_mw))
    model.p_charge = pyo.Var(model.T, bounds=(0, battery.power_max_mw))

    # State of charge
    model.soc = pyo.Var(model.T_soc, bounds=(battery.soc_min, battery.soc_max))

    # CVaR variables - NOTE: We work with COSTS (negative profit)
    model.eta = pyo.Var()  # Value-at-Risk threshold
    model.z = pyo.Var(model.S, domain=pyo.NonNegativeReals)  # Excess cost beyond VaR

    # Auxiliary variable for RT dispatch absolute value (for penalty)
    model.p_rt_abs = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # ==================== Constraints ====================

    # Initial SoC
    model.initial_soc_con = pyo.Constraint(expr=model.soc[0] == initial_soc)

    # End of day SoC
    model.end_soc_con = pyo.Constraint(expr=model.soc[T] == end_of_day_soc)

    # DA bid must be constant within each hour
    def da_hourly_constant_rule(model, t):
        hour = int(t / TIME_STEPS_PER_HOUR)
        start_idx = int(hour * TIME_STEPS_PER_HOUR)
        if t == start_idx:
            return pyo.Constraint.Skip
        return model.p_da[t] == model.p_da[start_idx]

    model.da_hourly_constant = pyo.Constraint(model.T, rule=da_hourly_constant_rule)

    # Power decomposition
    def power_decomposition_rule(model, t):
        return model.p_real[t] == model.p_charge[t] - model.p_discharge[t]

    model.power_decomposition = pyo.Constraint(model.T, rule=power_decomposition_rule)

    # Power flow relationship
    def power_flow_rule(model, t):
        return model.p_real[t] == model.p_da[t] + model.p_rt[t]

    model.power_flow = pyo.Constraint(model.T, rule=power_flow_rule)

    # SoC dynamics
    def soc_dynamics_rule(model, t):
        return model.soc[t + 1] == model.soc[t] + (
            model.p_charge[t] * battery.efficiency_charge
            - model.p_discharge[t] / battery.efficiency_discharge
        ) / (TIME_STEPS_PER_HOUR * battery.capacity_mwh)

    model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)

    # Battery throughput constraint
    # model.p_real_abs = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # def abs_pos_rule(model, t):
    #     return model.p_real_abs[t] >= model.p_real[t]

    # model.abs_pos = pyo.Constraint(model.T, rule=abs_pos_rule)

    # def abs_neg_rule(model, t):
    #     return model.p_real_abs[t] >= -model.p_real[t]

    # model.abs_neg = pyo.Constraint(model.T, rule=abs_neg_rule)

    # model.throughput_con = pyo.Constraint(
    #     expr=sum(model.p_real_abs[t] for t in model.T) / TIME_STEPS_PER_HOUR
    #     <= battery.throughput_limit
    # )

    max_cycles = battery.throughput_limit / battery.capacity_mwh
    model.throughput_con = pyo.Constraint(
        expr=(
            sum(model.p_charge[t] * DELTA_T for t in model.T)
            <= max_cycles * (battery.soc_max - battery.soc_min) * battery.capacity_mwh
        )
    )

    # RT dispatch absolute value (for penalty term)
    def rt_abs_pos_rule(model, t):
        return model.p_rt_abs[t] >= model.p_rt[t]

    model.rt_abs_pos = pyo.Constraint(model.T, rule=rt_abs_pos_rule)

    def rt_abs_neg_rule(model, t):
        return model.p_rt_abs[t] >= -model.p_rt[t]

    model.rt_abs_neg = pyo.Constraint(model.T, rule=rt_abs_neg_rule)

    # ==================== Calculate Revenue/Cost Components ====================

    def calc_rt_revenue(p_rt, rt_prices_arr):
        """Calculate RT market revenue given an array of RT prices."""
        return -sum(rt_prices_arr[t] * p_rt[t] for t in model.T)

    # DA revenue (deterministic, same for all scenarios)
    # (sell at positive prices, buy at negative)
    # DA cost = price * power (positive when buying, negative when selling)
    da_revenue = -sum(da_prices[t] * model.p_da[t] for t in model.T)

    # RT dispatch penalty (deterministic, same for all scenarios)
    rt_penalty_cost = (
        rt_dispatch_penalty
        * sum(model.p_rt_abs[t] for t in model.T)
        / TIME_STEPS_PER_HOUR
    )

    # Helper function for scenario profit
    def scenario_profit_expr(model, s):
        rt_rev = calc_rt_revenue(model.p_rt, rt_price_scenarios[:, s])
        return da_revenue + rt_rev - rt_penalty_cost

    # CVaR constraints - one per scenario
    # We define cost = - revenue, so CVaR protects against high costs (low revenues)
    def cvar_rule(model, s):
        # Total profit in scenario s (negative cost)
        scenario_profit = scenario_profit_expr(model, s)

        # Cost = -profit
        scenario_cost = -scenario_profit

        # CVaR constraint: z[s] >= (scenario_cost - eta)
        return model.z[s] >= scenario_cost - model.eta

    model.cvar_constraint = pyo.Constraint(model.S, rule=cvar_rule)

    # ==================== Objective Function ====================

    # Expected profit across all scenarios
    expected_profit = sum(scenario_profit_expr(model, s) for s in model.S) / n_scenarios

    # CVaR term: eta + (1/(1-alpha)) * E[z]
    # This represents the conditional expected cost in the worst (1-alpha) scenarios
    cvar_cost = (
        model.eta
        + (1.0 / (1.0 - cvar_alpha)) * sum(model.z[s] for s in model.S) / n_scenarios
    )

    # CVaR term
    cvar_cost = (
        model.eta
        + (1.0 / (1.0 - cvar_alpha)) * sum(model.z[s] for s in model.S) / n_scenarios
    )

    # Objective
    model.obj = pyo.Objective(
        expr=-(1.0 - cvar_weight) * expected_profit + cvar_weight * cvar_cost,
        sense=pyo.minimize,
    )

    # ==================== Solve ====================

    solver = pyo.SolverFactory("cbc")
    solver.options["print_level"] = 5
    solver.options["max_iter"] = 3000
    solver.options["acceptable_tol"] = 1e-6
    solver.options["constr_viol_tol"] = 1e-6
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["max_iter"] = 9000
    results = solver.solve(model, tee=verbose)
    if verbose:
        print(results)

    # Check if solution was found
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise ValueError(
            f"Optimization failed with status: {results.solver.termination_condition}"
        )

    # ==================== Extract Results ====================

    da_energy_bids = np.array([pyo.value(model.p_da[t]) for t in model.T])
    rt_energy_bids = np.array([pyo.value(model.p_rt[t]) for t in model.T])
    power_dispatch_schedule = np.array([pyo.value(model.p_real[t]) for t in model.T])
    soc_schedule = np.array([pyo.value(model.soc[t]) for t in model.T_soc])
    discharge = np.array([pyo.value(model.p_discharge[t]) for t in model.T])
    charge = np.array([pyo.value(model.p_charge[t]) for t in model.T])

    # Calculate actual revenues
    da_revenue_val = -np.sum(da_prices * da_energy_bids)
    rt_revenue_val = -np.sum(rt_prices * rt_energy_bids)
    rt_penalty_val = (
        rt_dispatch_penalty * np.sum(np.abs(rt_energy_bids)) / TIME_STEPS_PER_HOUR
    )
    expected_profit_val = da_revenue_val + rt_revenue_val - rt_penalty_val

    # CVaR metrics
    eta_value = pyo.value(model.eta)
    cvar_value = (
        eta_value
        + (1.0 / (1.0 - cvar_alpha))
        * sum(pyo.value(model.z[s]) for s in model.S)
        / n_scenarios
    )

    # Calculate scenario profits for diagnostics
    scenario_profits = []
    scenario_costs = []
    for s in range(n_scenarios):
        scenario_rt_revenue = -np.sum(rt_price_scenarios[:, s] * rt_energy_bids)
        if np.any(np.isnan(rt_price_scenarios)):
            raise ValueError("RT price scenarios contain NaN values")
        if np.any(np.isinf(rt_price_scenarios)):
            raise ValueError("RT price scenarios contain infinite values")

        scenario_profit = da_revenue_val + scenario_rt_revenue - rt_penalty_val
        scenario_profits.append(scenario_profit)
        scenario_costs.append(-scenario_profit)

    # Calculate RT dispatch magnitude
    rt_dispatch_magnitude = np.sum(np.abs(rt_energy_bids)) / TIME_STEPS_PER_HOUR

    return DAScheduleResult(
        da_energy_bids=da_energy_bids,
        rt_energy_bids=rt_energy_bids,
        power_dispatch_schedule=power_dispatch_schedule,
        soc_schedule=soc_schedule,
        reg_up_capacity=np.zeros_like(
            da_energy_bids
        ),  # Placeholder, as we don't model regulation in this version
        reg_down_capacity=np.zeros_like(da_energy_bids),
        expected_revenue=expected_profit_val,
        diagnostic_information={
            "da_revenue": da_revenue_val,
            "rt_revenue": rt_revenue_val,
            "rt_penalty_cost": rt_penalty_val,
            "discharge": discharge,
            "charge": charge,
            "var_threshold": eta_value,
            "cvar_cost": cvar_value,
            "scenario_profits": scenario_profits,
            "scenario_costs": scenario_costs,
            "worst_case_profit": np.min(scenario_profits),
            "best_case_profit": np.max(scenario_profits),
            "profit_std": np.std(scenario_profits),
            "rt_dispatch_magnitude": rt_dispatch_magnitude,
            "rt_price_scenarios": rt_price_scenarios,
            "cvar_weight_used": cvar_weight,
            "cvar_alpha_used": cvar_alpha,
        },
    )


def get_optimization_problem(
    operating_day: pd.Timestamp,
    rt_price_forecast: pd.Series,
    da_price_forecast: pd.Series,
    battery: BatteryParams,
    initial_soc: float,
    end_of_day_soc: float,
    cvar_alpha: float,
    cvar_weight: float,
    rt_dispatch_penalty: float,
    rt_price_uncertainty: Optional[pd.Series],
    rt_uncertainty_default: float,
    n_scenarios: int,
    scenario_seed: Optional[int],
    da_commitments: Optional[np.ndarray] = None,
) -> tuple[cp.Problem, dict[str, cp.Variable]]:

    # Number of time periods
    T = len(rt_price_forecast)

    assert rt_price_forecast.notna().all(), "RT price forecast contains NaN values"
    assert da_price_forecast.notna().all(), "DA price forecast contains NaN values"

    # Get the end of day index of the operating day
    operating_day = operating_day.normalize()
    end_of_day_timestamp = (
        operating_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=int(DELTA_T * 60))
    )
    end_of_day_index = rt_price_forecast.index.get_loc(end_of_day_timestamp)
    assert isinstance(end_of_day_index, int), (
        "End of day timestamp not found in RT price forecast index"
    )

    # ==================== Variables ====================
    if da_commitments is None:
        p_da = cp.Variable(T, name="p_da")
    else:
        assert len(da_commitments) == T, (
            "Length of da_commitments must match number of time periods"
        )
        p_da = cp.Parameter(T, name="p_da", value=da_commitments)

    p_rt = cp.Variable(T, name="p_rt")
    p_real = cp.Variable(T, name="p_real")
    p_discharge = cp.Variable(T, name="p_discharge")
    p_charge = cp.Variable(T, name="p_charge")
    E = cp.Variable(T + 1, name="energy")

    # ==================== Constraints ====================
    # Power flow and decomposition
    constraints = [
        (p_real == p_da + p_rt).set_label("power_flow"),
        (p_real == p_charge - p_discharge).set_label("power_decomposition"),
    ]

    # Power bounds
    constraints += [
        (0 <= p_charge).set_label("charge_nonnegativity"),
        (0 <= p_discharge).set_label("discharge_nonnegativity"),
        (p_charge <= battery.power_max_mw).set_label("charge_power_limit"),
        (p_discharge <= battery.power_max_mw).set_label("discharge_power_limit"),
        (-battery.power_max_mw <= p_rt).set_label("rt_bid_lower_bound"),
        (p_rt <= battery.power_max_mw).set_label("rt_bid_upper_bound"),
    ]
    if da_commitments is None:
        constraints += [
            (-battery.power_max_mw <= p_da).set_label("da_bid_lower_bound"),
            (p_da <= battery.power_max_mw).set_label("da_bid_upper_bound"),
        ]

    # Energy dynamics
    constraints += [
        (E[0] == initial_soc * battery.capacity_mwh).set_label("initial_energy"),
        (E[end_of_day_index + 1] == end_of_day_soc * battery.capacity_mwh).set_label(
            "end_energy"
        ),
        (
            E[1:]
            == E[:-1]
            + (
                p_charge * battery.efficiency_charge
                - p_discharge / battery.efficiency_discharge
            )
            * DELTA_T
        ).set_label("soc_dynamics"),
        (battery.soc_min * battery.capacity_mwh <= E).set_label("energy_min"),
        (E <= battery.soc_max * battery.capacity_mwh).set_label("energy_max"),
    ]

    # Battery cycling limits
    max_cycles = battery.throughput_limit / battery.capacity_mwh
    constraints.append(
        (
            cp.sum(p_charge * DELTA_T)
            <= max_cycles * (battery.soc_max - battery.soc_min) * battery.capacity_mwh
        ).set_label("throughput_limit")
    )

    # DA bid must be constant within each hour
    if da_commitments is None:
        for t in range(T):
            hour = int(t / TIME_STEPS_PER_HOUR)
            start_idx = int(hour * TIME_STEPS_PER_HOUR)
            if t != start_idx:
                constraints.append(
                    (p_da[t] == p_da[start_idx]).set_label(f"da_hourly_{t}")
                )

    # ==================== Objective function ====================
    objective = cp.Minimize(
        (
            (cp.sum(da_price_forecast.values @ p_da) if da_commitments is None else 0)
            + cp.sum(rt_price_forecast.values @ p_rt)
        )
        * DELTA_T
    )

    variables = {
        "p_rt": p_rt,
        "p_real": p_real,
        "p_discharge": p_discharge,
        "p_charge": p_charge,
        "E": E,
    }
    if isinstance(p_da, cp.Variable):
        variables["p_da"] = p_da

    return cp.Problem(objective, constraints), variables  # type: ignore


def solve_da_schedule_cvxpy(
    da_price_forecast: pd.Series,
    rt_price_forecast: pd.Series,
    battery: BatteryParams,
    initial_soc: float = DEFAULT_INITIAL_SOC,
    end_of_day_soc: float = DEFAULT_END_OF_DAY_SOC,
    cvar_alpha: float = 0.90,
    cvar_weight: float = 0,
    rt_dispatch_penalty: float = 0,
    rt_price_uncertainty: Optional[pd.Series] = None,
    rt_uncertainty_default: float = 0,  # 20
    n_scenarios: int = 20,
    scenario_seed: Optional[int] = None,
    verbose: bool = False,
) -> DAScheduleResult:

    problem, variables = get_optimization_problem(
        operating_day=da_price_forecast.index[0].normalize(),
        rt_price_forecast=rt_price_forecast,
        da_price_forecast=da_price_forecast,
        battery=battery,
        initial_soc=initial_soc,
        end_of_day_soc=end_of_day_soc,
        cvar_alpha=cvar_alpha,
        cvar_weight=cvar_weight,
        rt_dispatch_penalty=rt_dispatch_penalty,
        rt_price_uncertainty=rt_price_uncertainty,
        rt_uncertainty_default=rt_uncertainty_default,
        n_scenarios=n_scenarios,
        scenario_seed=scenario_seed,
    )
    problem.solve(verbose=verbose)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    if problem.status == cp.OPTIMAL_INACCURATE:
        print("Warning: Optimization solved to optimality but is inaccurate.")

    # ==================== Extract results ====================
    da_revenue = -(da_price_forecast.to_numpy() @ variables["p_da"].value * DELTA_T)
    rt_revenue = -(rt_price_forecast.to_numpy() @ variables["p_rt"].value * DELTA_T)
    expected_revenue: float = da_revenue + rt_revenue  # type: ignore

    return DAScheduleResult(
        da_energy_bids=variables["p_da"].value,
        rt_energy_bids=variables["p_rt"].value,
        power_dispatch_schedule=variables["p_real"].value,
        soc_schedule=variables["E"].value / battery.capacity_mwh,
        reg_up_capacity=np.zeros_like(
            variables["p_da"].value
        ),  # Placeholder, as we don't model regulation in this version
        reg_down_capacity=np.zeros_like(variables["p_da"].value),
        expected_revenue=expected_revenue,
        diagnostic_information={
            "da_revenue": da_revenue,
            "rt_revenue": rt_revenue,
            "rt_penalty_cost": None,
            "discharge": variables["p_discharge"].value,
            "charge": variables["p_charge"].value,
            "var_threshold": None,
            "cvar_cost": None,
            "scenario_profits": None,
            "scenario_costs": None,
            "worst_case_profit": None,
            "best_case_profit": None,
            "profit_std": None,
            "rt_dispatch_magnitude": None,
            "rt_price_scenarios": None,
            "cvar_weight_used": cvar_weight,
            "cvar_alpha_used": cvar_alpha,
        },
    )
