import pandas as pd
import numpy as np
from pyomo.environ import *
from typing import Optional
from .battery_model import BatteryParams
from .utils import DAScheduleResult

def solve_da_schedule(
    da_price_forecast: pd.Series,
    rt_price_forecast: pd.Series,
    battery: BatteryParams,
    rt_price_uncertainty: Optional[pd.Series] = None,
    reg_up_price: Optional[pd.Series] = None,
    reg_down_price: Optional[pd.Series] = None,
    initial_soc: float = 0.5,
    rt_dispatches_per_hour: float = 4,
    end_of_day_soc: float = 0.5,
    cvar_alpha: float = 0.95,
    cvar_weight: float = 1.0,
    n_scenarios: int = 50,
    scenario_seed: Optional[int] = None,
) -> DAScheduleResult:
    """
    Solve Stage 1 DA optimization problem with CVaR risk measure using Pyomo.
    
    CVaR is applied only to RT cost uncertainty. DA cost is deterministic (known prices).
    
    Parameters
    ----------
    da_price_forecast : pd.Series
        Day-ahead energy prices for 24 hours [$/MWh]
    rt_price_forecast : pd.Series
        Real-time energy prices for 24 hours [$/MWh]
    battery : BatteryParams
        Battery parameters (capacity, power limits, efficiency, etc.)
    rt_price_uncertainty : Optional[pd.Series]
        Real-time price uncertainty/volatility for each hour
    reg_up_price : Optional[pd.Series]
        Regulation up capacity prices for 24 hours [$/MW]
    reg_down_price : Optional[pd.Series]
        Regulation down capacity prices for 24 hours [$/MW]
    initial_soc : float
        Initial state of charge [fraction, 0-1]
    rt_dispatches_per_hour : float
        Amount of power dispatches per hour [#/hour]
    end_of_day_soc : float
        Target state of charge at end of day [fraction, 0-1]
    cvar_alpha : float
        Confidence level for CVaR
    cvar_weight : float
        Weight on CVaR term
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
    
    # Convert prices to numpy arrays
    da_prices = da_price_forecast.values
    rt_prices = rt_price_forecast.values
    
    if rt_price_uncertainty is not None:
        rt_uncertainty = rt_price_uncertainty.values
    else:
        rt_uncertainty = np.ones(T) * 5
    
    # Generate RT price scenarios
    if scenario_seed is not None:
        np.random.seed(scenario_seed)
    
    rt_price_scenarios = np.random.normal(
        loc=rt_prices[:, np.newaxis],
        scale=rt_uncertainty[:, np.newaxis],
        size=(T, n_scenarios)
    )
    
    # Handle regulation prices
    if reg_up_price is not None:
        reg_up_prices = reg_up_price.values
    else:
        reg_up_prices = np.zeros(T)
    
    if reg_down_price is not None:
        reg_down_prices = reg_down_price.values
    else:
        reg_down_prices = np.zeros(T)
    
    # ==================== Build Pyomo Model ====================
    
    model = ConcreteModel()
    
    # Sets
    model.T = RangeSet(0, T-1)  # Time periods
    model.T_soc = RangeSet(0, T)  # Time periods for SoC (includes initial)
    model.S = RangeSet(0, n_scenarios-1)  # Scenarios
    
    # ==================== Decision Variables ====================
    
    # Energy bids
    model.p_da = Var(model.T, bounds=(-battery.power_max_mw, battery.power_max_mw))
    model.p_rt = Var(model.T, bounds=(-battery.power_max_mw, battery.power_max_mw))
    
    # Actual dispatch schedule
    model.p_real = Var(model.T, bounds=(-battery.power_max_mw, battery.power_max_mw))
    
    # Charge/discharge
    model.p_discharge = Var(model.T, bounds=(0, battery.power_max_mw))
    model.p_charge = Var(model.T, bounds=(0, battery.power_max_mw))
    
    # State of charge
    model.soc = Var(model.T_soc, bounds=(battery.soc_min, battery.soc_max))
    
    # CVaR variables (applied only to RT cost)
    model.tau = Var()  # Value-at-Risk for RT cost
    model.z = Var(model.S, domain=NonNegativeReals)  # Excess RT cost beyond VaR
    
    # ==================== Constraints ====================
    
    # Initial SoC
    model.initial_soc_con = Constraint(expr=model.soc[0] == initial_soc)
    
    # End of day SoC
    model.end_soc_con = Constraint(expr=model.soc[T] == end_of_day_soc)
    
    # DA bid must be constant within each hour
    def da_hourly_constant_rule(model, t):
        hour = int(t / rt_dispatches_per_hour)
        start_idx = int(hour * rt_dispatches_per_hour)
        if t == start_idx:
            return Constraint.Skip
        return model.p_da[t] == model.p_da[start_idx]
    
    model.da_hourly_constant = Constraint(model.T, rule=da_hourly_constant_rule)
    
    # Power decomposition
    def power_decomposition_rule(model, t):
        return model.p_real[t] == model.p_charge[t] - model.p_discharge[t]
    
    model.power_decomposition = Constraint(model.T, rule=power_decomposition_rule)
    
    # Power flow relationship
    def power_flow_rule(model, t):
        return model.p_real[t] == model.p_da[t] + model.p_rt[t]
    
    model.power_flow = Constraint(model.T, rule=power_flow_rule)
    
    # SoC dynamics
    def soc_dynamics_rule(model, t):
        return model.soc[t + 1] == model.soc[t] + \
            (model.p_charge[t] * battery.efficiency_charge - 
             model.p_discharge[t] / battery.efficiency_discharge) / \
            (rt_dispatches_per_hour * battery.capacity_mwh)
    
    model.soc_dynamics = Constraint(model.T, rule=soc_dynamics_rule)
    
    # Battery throughput constraint
    model.p_real_abs = Var(model.T, domain=NonNegativeReals)
    
    def abs_pos_rule(model, t):
        return model.p_real_abs[t] >= model.p_real[t]
    
    model.abs_pos = Constraint(model.T, rule=abs_pos_rule)
    
    def abs_neg_rule(model, t):
        return model.p_real_abs[t] >= -model.p_real[t]
    
    model.abs_neg = Constraint(model.T, rule=abs_neg_rule)
    
    model.throughput_con = Constraint(
        expr=sum(model.p_real_abs[t] for t in model.T) / rt_dispatches_per_hour <= battery.throughput_limit
    )
    
    # CVaR constraints (only for RT cost uncertainty)
    def cvar_rule(model, s):
        # RT cost in this scenario (uncertain)
        scenario_rt_cost = sum(rt_price_scenarios[t, s] * model.p_rt[t] for t in model.T)
        # CVaR is applied only to the uncertain RT cost
        return model.z[s] >= scenario_rt_cost - model.tau
    
    model.cvar_constraint = Constraint(model.S, rule=cvar_rule)
    
    # ==================== Objective Function ====================
    
    # DA energy cost (deterministic - known prices, no uncertainty)
    da_energy_cost_expr = sum(da_prices[t] * model.p_da[t] for t in model.T)
    
    # Expected RT energy cost (using mean forecast)
    rt_energy_cost_expected = sum(rt_prices[t] * model.p_rt[t] for t in model.T)
    
    # CVaR term for RT cost only
    # This represents: "VaR threshold + average of worst (1-alpha)% of RT cost outcomes"
    rt_cvar_term = model.tau + (1.0 / (1.0 - cvar_alpha)) * sum(model.z[s] for s in model.S) / n_scenarios
    
    # Combined objective:
    # - DA cost is deterministic (no risk adjustment)
    # - RT cost has both expected value and CVaR risk measure
    model.obj = Objective(
        expr=da_energy_cost_expr + 
             (1 - cvar_weight) * rt_energy_cost_expected + 
             cvar_weight * rt_cvar_term,
        sense=minimize
    )
    
    # ==================== Solve ====================
    
    # You can use different solvers: 'ipopt', 'gurobi', 'cplex', 'glpk', etc.
    solver = SolverFactory('ipopt')  # Change to your preferred solver
    results = solver.solve(model, tee=True)
    
    # Check if solution was found
    if results.solver.termination_condition != TerminationCondition.optimal:
        raise ValueError(f"Optimization failed with status: {results.solver.termination_condition}")
    
    # ==================== Extract Results ====================
    
    da_energy_bids = np.array([value(model.p_da[t]) for t in model.T])
    rt_energy_bids = np.array([value(model.p_rt[t]) for t in model.T])
    power_dispatch_schedule = np.array([value(model.p_real[t]) for t in model.T])
    soc_schedule = np.array([value(model.soc[t]) for t in model.T_soc])
    discharge = np.array([value(model.p_discharge[t]) for t in model.T])
    charge = np.array([value(model.p_charge[t]) for t in model.T])
    
    da_energy_cost = sum(da_prices[t] * value(model.p_da[t]) for t in model.T)
    
    # Calculate metrics
    expected_revenue = -value(model.obj)
    var_value = value(model.tau)
    cvar_value = var_value + (1.0 / (1.0 - cvar_alpha)) * sum(value(model.z[s]) for s in model.S) / n_scenarios
    
    # Calculate scenario costs for diagnostics
    scenario_costs = []
    scenario_rt_costs = []
    for s in range(n_scenarios):
        scenario_rt_cost = np.sum(rt_price_scenarios[:, s] * rt_energy_bids)
        scenario_total_cost = np.sum(da_prices * da_energy_bids) + scenario_rt_cost
        scenario_costs.append(scenario_total_cost)
        scenario_rt_costs.append(scenario_rt_cost)
    
    return DAScheduleResult(
        da_energy_bids=da_energy_bids,
        rt_energy_bids=rt_energy_bids,
        power_dispatch_schedule=power_dispatch_schedule,
        soc_schedule=soc_schedule,
        reg_up_capacity=0,
        reg_down_capacity=0,
        expected_revenue=expected_revenue,
        diagnostic_information={
            "da_energy_cost": da_energy_cost,
            "discharge": discharge,
            "charge": charge,
            "var_95": var_value,
            "cvar_95": cvar_value,
            "scenario_costs": scenario_costs,
            "scenario_rt_costs": scenario_rt_costs,
            "worst_case_cost": np.max(scenario_costs),
            "best_case_cost": np.min(scenario_costs),
            "worst_case_rt_cost": np.max(scenario_rt_costs),
            "best_case_rt_cost": np.min(scenario_rt_costs),
            "rt_price_scenarios": rt_price_scenarios,
        }
    )