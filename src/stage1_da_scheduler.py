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
    cvar_alpha: float = 0.90,
    cvar_weight: float = 0.2,
    rt_dispatch_penalty: float = 0, 
    rt_uncertainty_default: float = 20,
    n_scenarios: int = 50,
    scenario_seed: Optional[int] = None,
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
        Real-time price uncertainty/volatility (std dev) for each hour
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
        Confidence level for CVaR (0.95 = protect against worst 5%)
    cvar_weight : float
        Weight on CVaR term (0=risk-neutral, 1=full CVaR focus)
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
    da_price_forecast.ffill()
    rt_price_forecast.ffill()
    if rt_price_uncertainty is not None:
        rt_price_uncertainty.ffill()
    
    # Convert prices to numpy arrays
    da_prices = da_price_forecast.values
    rt_prices = rt_price_forecast.values
    
    # Default uncertainty if not provided
    if rt_price_uncertainty is not None:
        rt_uncertainty = rt_price_uncertainty.values
    else:
        # Default: 10% of price or minimum of $5/MWh
        rt_uncertainty = np.ones(len(rt_prices)) *  rt_uncertainty_default
    
    # Generate RT price scenarios
    if scenario_seed is not None:
        np.random.seed(scenario_seed)
    
    rt_price_scenarios = np.random.normal(
        loc=rt_prices[:, np.newaxis],
        scale=rt_uncertainty[:, np.newaxis],
        size=(T, n_scenarios)
    )
        # Optional: clip extreme scenarios
    rt_price_scenarios = np.clip(rt_price_scenarios, 0, 80)  # Adjust bounds as needed
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
    
    # CVaR variables - NOTE: We work with COSTS (negative profit)
    model.eta = Var()  # Value-at-Risk threshold
    model.z = Var(model.S, domain=NonNegativeReals)  # Excess cost beyond VaR
    
    # Auxiliary variable for RT dispatch absolute value (for penalty)
    model.p_rt_abs = Var(model.T, domain=NonNegativeReals)
    
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
    
    # RT dispatch absolute value (for penalty term)
    def rt_abs_pos_rule(model, t):
        return model.p_rt_abs[t] >= model.p_rt[t]
    
    model.rt_abs_pos = Constraint(model.T, rule=rt_abs_pos_rule)
    
    def rt_abs_neg_rule(model, t):
        return model.p_rt_abs[t] >= -model.p_rt[t]
    
    model.rt_abs_neg = Constraint(model.T, rule=rt_abs_neg_rule)
    
    # CVaR constraints - one per scenario
    # We define cost = - revenue, so CVaR protects against high costs (low revenues)
    def cvar_rule(model, s):
        # Revenue from DA market (sell at positive prices, buy at negative)
        # DA cost = price * power (positive when buying, negative when selling)
        da_revenue = -sum(da_prices[t] * model.p_da[t] for t in model.T)
        
        # Revenue from RT market in this scenario
        rt_revenue = -sum(rt_price_scenarios[t, s] * model.p_rt[t] for t in model.T)
        
        # RT dispatch penalty cost
        rt_penalty_cost = rt_dispatch_penalty * sum(model.p_rt_abs[t] for t in model.T) / rt_dispatches_per_hour
        
        # Total profit in scenario s (negative cost)
        scenario_profit = da_revenue + rt_revenue - rt_penalty_cost
        
        # Cost = -profit
        scenario_cost = -scenario_profit
        
        # CVaR constraint: z[s] >= (scenario_cost - eta)
        return model.z[s] >= scenario_cost - model.eta
    
    model.cvar_constraint = Constraint(model.S, rule=cvar_rule)
    
    # ==================== Objective Function ====================
    
    # Expected revenue from DA market (deterministic)
    da_revenue = -sum(da_prices[t] * model.p_da[t] for t in model.T)
    
    # Expected revenue from RT market (using mean forecast)
    rt_revenue = -sum(rt_prices[t] * model.p_rt[t] for t in model.T)
    
    # RT dispatch penalty
    rt_penalty_cost = rt_dispatch_penalty * sum(model.p_rt_abs[t] for t in model.T) / rt_dispatches_per_hour
    
    # Expected profit
    expected_profit = da_revenue + rt_revenue - rt_penalty_cost
    
    # CVaR term: eta + (1/(1-alpha)) * E[z]
    # This represents the conditional expected cost in the worst (1-alpha) scenarios
    cvar_cost = model.eta + (1.0 / (1.0 - cvar_alpha)) * sum(model.z[s] for s in model.S) / n_scenarios
    
    # Combined objective: maximize expected profit while minimizing CVaR of cost
    # = minimize: -(1-λ) * E[profit] + λ * CVaR[cost]
    # model.obj = Objective(
    #     expr=-(1.0 - cvar_weight) * expected_profit + cvar_weight * cvar_cost,
    #     sense=minimize
    # )
    
    # Define scenario profit expression
    def scenario_profit_expr(model, s):
        da_rev = -sum(da_prices[t] * model.p_da[t] for t in model.T)
        rt_rev = -sum(rt_price_scenarios[t, s] * model.p_rt[t] for t in model.T)
        penalty = rt_dispatch_penalty * sum(model.p_rt_abs[t] for t in model.T) / rt_dispatches_per_hour
        return da_rev + rt_rev - penalty

    # Expected profit
    expected_profit = sum(scenario_profit_expr(model, s) for s in model.S) / n_scenarios

    # CVaR term
    cvar_cost = model.eta + (1.0 / (1.0 - cvar_alpha)) * sum(model.z[s] for s in model.S) / n_scenarios

    # Objective
    model.obj = Objective(
        expr=-(1.0 - cvar_weight) * expected_profit + cvar_weight * cvar_cost,
        sense=minimize
    )

    # ==================== Solve ====================
    
    solver = SolverFactory('ipopt')
    solver.options['print_level'] = 5
    solver.options['max_iter'] = 3000
    solver.options['acceptable_tol'] = 1e-6
    solver.options['constr_viol_tol'] = 1e-6
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['max_iter'] = 9000
    results = solver.solve(model)
    print(results)
    
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
    
    # Calculate actual revenues
    da_revenue_val = -np.sum(da_prices * da_energy_bids)
    rt_revenue_val = -np.sum(rt_prices * rt_energy_bids)
    rt_penalty_val = rt_dispatch_penalty * np.sum(np.abs(rt_energy_bids)) / rt_dispatches_per_hour
    expected_profit_val = da_revenue_val + rt_revenue_val - rt_penalty_val
    
    # CVaR metrics
    eta_value = value(model.eta)
    cvar_value = eta_value + (1.0 / (1.0 - cvar_alpha)) * sum(value(model.z[s]) for s in model.S) / n_scenarios
    
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
    rt_dispatch_magnitude = np.sum(np.abs(rt_energy_bids)) / rt_dispatches_per_hour
    
    return DAScheduleResult(
        da_energy_bids=da_energy_bids,
        rt_energy_bids=rt_energy_bids,
        power_dispatch_schedule=power_dispatch_schedule,
        soc_schedule=soc_schedule,
        reg_up_capacity=0,
        reg_down_capacity=0,
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
        }
    )


from .forecaster import get_forecasts_for_da
from .utils import load_ercot_data


AMT_DAYS = 2

def main():
    data = load_ercot_data()
    current_time = pd.Timestamp("2025-06-02 10:00:00")
    print(data.head())
    
    # 1. Prices for the scheduler (Persistence Forecast)
    da_prices_forecast, rt_prices_forecast = get_forecasts_for_da(
        data,
        current_time=current_time,
        horizon_hours=24 * AMT_DAYS,
        method="regression",
        verbose=False,
    )
    
    # 2. Prices for Real Revenue Calculation (Perfect Forecast / Real Prices)
    da_prices_real, rt_prices_real = get_forecasts_for_da(
        data,
        current_time=current_time,
        horizon_hours=24 * AMT_DAYS,
        method="perfect",
        verbose=False,
    )
    
    # --- CALCULATE PERFECT UNCERTAINTY FORECAST ---
    # The magnitude of the error between the persistence forecast and the real price
    # is used as a perfect proxy for the expected uncertainty (std dev).
    perfect_uncertainty_forecast = (rt_prices_real - rt_prices_forecast).abs()
    
    battery = BatteryParams()
    
    # --- Define Scenarios for Comparison ---
    scenarios = {
        "Baseline (w=0, p=0, Unc=0)": {
            "cvar_weight": 0,
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 0,
            "rt_price_uncertainty": None, # Use default/float
            "forecast_input": (da_prices_forecast, rt_prices_forecast),
            "color": "tab:blue",
            "linestyle": "-"
        },
        "Risk-Averse Regression (w=0.5, Unc=20)": {
            "cvar_weight": 0.1,
            "rt_uncertainty_default": 10,
            "rt_dispatch_penalty": 0,
            "rt_price_uncertainty": None, # Use default/float
            "forecast_input": (da_prices_forecast, rt_prices_forecast),
            "color": "tab:orange",
            "linestyle": "--"
        },
        "Conservative Regression (w=0.1, Unc=20)": {
            "cvar_weight": 0.1,
            "rt_uncertainty_default": 5,
            "rt_dispatch_penalty": 0,
            "rt_price_uncertainty": None, # Use default/float
            "forecast_input": (da_prices_forecast, rt_prices_forecast),
            "color": "tab:green",
            "linestyle": ":"
        },
        "Perfect Uncertainty Regression (w=0.5)": { # NEW SCENARIO
            "cvar_weight": 0.1, 
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 0,
            "rt_price_uncertainty": perfect_uncertainty_forecast, # <-- Use the Series
            "forecast_input": (da_prices_forecast, rt_prices_forecast),
            "color": "tab:purple",
            "linestyle": "-."
        },
    }
    
    results_comparison = {}
    
    print("Solving DA Schedule for different scenarios...")
    
    for name, params in scenarios.items():
        print(f"  -> Running scenario: {name}")
        
        # Determine the price input for the solver
        da_input, rt_input = params["forecast_input"]

        # Solve DA schedule with specific parameters
        result = solve_da_schedule(
            da_price_forecast=da_input,
            rt_price_forecast=rt_input,
            battery=battery,
            cvar_weight=params["cvar_weight"],
            rt_uncertainty_default=params["rt_uncertainty_default"],
            rt_dispatch_penalty=params["rt_dispatch_penalty"],
            rt_price_uncertainty=params["rt_price_uncertainty"],
        )
        print(result)

if __name__ == "__main__":
    main()