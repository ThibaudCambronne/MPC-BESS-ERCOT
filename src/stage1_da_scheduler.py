import pandas as pd
import numpy as np
import cvxpy as cp
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
    cvar_alpha: float = 0.95,  # confidence level (95% = protect against worst 5%)
    cvar_weight: float = 1.0,  # weight on CVaR term vs expected cost
    n_scenarios: int = 50,     # number of price scenarios
    scenario_seed: Optional[int] = None,  # for reproducibility
) -> DAScheduleResult:
    """
    Solve Stage 1 DA optimization problem with CVaR risk measure.
    
    Additional Parameters
    ---------------------
    cvar_alpha : float
        Confidence level for CVaR (e.g., 0.95 means protect against worst 5% of scenarios)
    cvar_weight : float
        Weight on CVaR term. Higher = more risk-averse.
        0 = risk-neutral (ignore CVaR), 1 = balanced, >1 = very conservative
    n_scenarios : int
        Number of RT price scenarios to generate for CVaR calculation
    scenario_seed : Optional[int]
        Random seed for scenario generation (for reproducibility)
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
    # Assume normal distribution around forecast
    if scenario_seed is not None:
        np.random.seed(scenario_seed)
    
    rt_price_scenarios = np.random.normal(
        loc=rt_prices[:, np.newaxis],
        scale=rt_uncertainty[:, np.newaxis],
        size=(T, n_scenarios)
    )
    
    # by default not doing reg up down
    if reg_up_price is not None:
        reg_up_prices = reg_up_price.values
    else:
        reg_up_prices = np.zeros(T)
    
    if reg_down_price is not None:
        reg_down_prices = reg_down_price.values
    else:
        reg_down_prices = np.zeros(T)
    
    # ==================== Decision Variables ====================
    
    # Energy bids
    p_da = cp.Variable(T)
    p_rt = cp.Variable(T)
    p_real = cp.Variable(T)
    
    p_discharge = cp.Variable(T, nonneg=True)
    p_charge = cp.Variable(T, nonneg=True)
    
    # State of charge
    soc = cp.Variable(T + 1, nonneg=True)
    
    # CVaR variables
    tau = cp.Variable()  # Value-at-Risk (VaR) at alpha level
    z = cp.Variable(n_scenarios, nonneg=True)  # Excess cost beyond VaR for each scenario
    
    # ==================== Constraints ====================
    
    constraints = []
    
    # Initial SoC constraint
    constraints.append(soc[0] == initial_soc)
    constraints.append(soc <= battery.soc_max)
    constraints.append(soc >= battery.soc_min)
    constraints.append(soc[T] == end_of_day_soc)
    
    # DA bid must be constant within each hour
    for t in range(int(T / rt_dispatches_per_hour)):
        start_idx = int(t * rt_dispatches_per_hour)
        end_idx = int((t + 1) * rt_dispatches_per_hour)
        for t2 in range(start_idx + 1, end_idx):
            constraints.append(p_da[t2] == p_da[start_idx])
    
    # Power flow constraints for each time step
    for t in range(T):
        # Power decomposition
        constraints.append(p_real[t] == p_charge[t] - p_discharge[t])
        constraints.append(p_real[t] == p_da[t] + p_rt[t])
        
        # Power limits
        constraints.append(p_real[t] <= battery.power_max_mw)
        constraints.append(p_real[t] >= -battery.power_max_mw)
        constraints.append(p_rt[t] <= battery.power_max_mw)
        constraints.append(p_rt[t] >= -battery.power_max_mw)
        constraints.append(p_da[t] <= battery.power_max_mw)
        constraints.append(p_da[t] >= -battery.power_max_mw)
        constraints.append(p_discharge[t] <= battery.power_max_mw)
        constraints.append(p_charge[t] <= battery.power_max_mw)
        
        # SoC dynamics
        constraints.append(
            soc[t + 1] == soc[t] + 
            (p_charge[t] * battery.efficiency_charge - p_discharge[t] / battery.efficiency_discharge) / 
            (rt_dispatches_per_hour * battery.capacity_mwh)
        )
        constraints.append(soc[t + 1] >= battery.soc_min)
        constraints.append(soc[t + 1] <= battery.soc_max)
    
    # Battery throughput constraint
    constraints.append(cp.sum(cp.abs(p_real)) / rt_dispatches_per_hour <= battery.throughput_limit)
    
    # Expected DA cost (deterministic)
    da_energy_cost_expr = cp.sum(cp.multiply(da_prices, p_da))

    # ==================== CVaR Constraints ====================
    
    # For each scenario, calculate the total cost and constraint CVaR
    for s in range(n_scenarios):
        # Cost in this scenario
        scenario_rt_cost = cp.sum(cp.multiply(rt_price_scenarios[:, s], p_rt))
        scenario_total_cost = da_energy_cost_expr + scenario_rt_cost
        
        # CVaR constraint: z[s] >= (scenario_cost - tau)
        # z captures how much worse this scenario is than VaR threshold
        constraints.append(z[s] >= scenario_total_cost - tau)
    
    # ==================== Objective Function ====================
    
    
    # Expected RT cost (using mean forecast)
    rt_energy_cost_expected = cp.sum(cp.multiply(rt_prices, p_rt))
    
    # CVaR term: VaR + expected excess beyond VaR
    # This represents: "VaR threshold + average of worst (1-alpha)% of outcomes"
    cvar_term = tau + (1.0 / (1.0 - cvar_alpha)) * cp.sum(z) / n_scenarios
    
    # Combined objective
    # Option 1: Pure CVaR (ignore expected value)
    # objective = cp.Minimize(cvar_term)
    
    # Option 2: Weighted combination of expected cost and CVaR
    objective = cp.Minimize(
        (1 - cvar_weight) * (da_energy_cost_expr + rt_energy_cost_expected) + 
        cvar_weight * cvar_term
    )
    
    # ==================== Solve ====================
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=True)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    # ==================== Extract Results ====================
    
    da_energy_bids = p_da.value
    rt_energy_bids = p_rt.value
    power_dispatch_schedule = p_real.value
    soc_schedule = soc.value
    discharge = p_discharge.value
    charge = p_charge.value
    
    # Calculate metrics
    expected_revenue = -problem.value  # Negative because we're minimizing cost
    var_value = tau.value
    cvar_value = var_value + (1.0 / (1.0 - cvar_alpha)) * np.sum(z.value) / n_scenarios
    
    # Calculate scenario costs for diagnostics
    scenario_costs = []
    for s in range(n_scenarios):
        scenario_rt_cost = np.sum(rt_price_scenarios[:, s] * rt_energy_bids)
        scenario_total_cost = np.sum(da_prices * da_energy_bids) + scenario_rt_cost
        scenario_costs.append(scenario_total_cost)
    
    return DAScheduleResult(
        da_energy_bids=da_energy_bids,
        rt_energy_bids=rt_energy_bids,
        power_dispatch_schedule=power_dispatch_schedule,
        soc_schedule=soc_schedule,
        reg_up_capacity=0,
        reg_down_capacity=0,
        expected_revenue=expected_revenue,
        diagnostic_information={
            "discharge": discharge,
            "charge": charge,
            "var_95": var_value,
            "cvar_95": cvar_value,
            "scenario_costs": scenario_costs,
            "worst_case_cost": np.max(scenario_costs),
            "best_case_cost": np.min(scenario_costs),
            "rt_price_scenarios": rt_price_scenarios,
        }
    )