import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Optional
from .battery_model import BatteryParams
from .utils import DAScheduleResult

def solve_da_schedule(
    da_price_forecast: pd.Series,
    initial_soc: float,
    battery: BatteryParams,
    reg_up_price: Optional[pd.Series] = None,
    reg_down_price: Optional[pd.Series] = None
) -> DAScheduleResult:
    """
    Solve Stage 1 DA optimization problem.
    
    This function formulates and solves a convex optimization problem to determine
    optimal day-ahead bids for energy and ancillary services (regulation up/down).
    
    Parameters
    ----------
    da_price_forecast : pd.Series
        Day-ahead energy prices for 24 hours [$/MWh]
    initial_soc : float
        Initial state of charge [fraction, 0-1]. Determined from previous optimization (either DA or RA)
        Alternatively, could just be set to 0.5, and EOD SOC could be constrained to 0.5
    battery : BatteryParams
        Battery parameters (capacity, power limits, efficiency, etc.)
    reg_up_price : Optional[pd.Series]
        Regulation up capacity prices for 24 hours [$/MW]. Currently not doing this.
    reg_down_price : Optional[pd.Series]
        Regulation down capacity prices for 24 hours [$/MW]. Currently not doing this. 
    
    Returns
    -------
    DAScheduleResult
        Optimization results including DA energy bids, regulation capacity,
        planned SoC trajectory, and expected revenue
    """
    # Number of time periods (just 24 hours generally)
    T = len(da_price_forecast)
    
    # Convert prices to numpy arrays
    da_prices = da_price_forecast.values
    
    # by default not doing reg up down
    if reg_up_price is not None:
        reg_up_prices = reg_up_price.values
    else:
        reg_up_prices = np.zeros(T)
    
    if reg_down_price is not None:
        reg_down_prices = reg_down_price.values
    else:
        reg_down_prices = np.zeros(T)
    
    # Decision var
    # power
    # positive = discharge, negative = charge
    p_da = cp.Variable(T)  
    
    # Regulation capacity [MW] - always non-negative
    # Maybe constrain to zero using bounds(0,0) if just having 0 prices doesn't work? 
    r_up = cp.Variable(T, nonneg=True)    # Reg up capacity
    r_down = cp.Variable(T, nonneg=True)  # Reg down capacity
    
    # State of charge % of total capacity
    soc = cp.Variable(T + 1)  # all soc variables, taking initial and producing next initial
    
    # variables for charge/discharge, makes constraints easier
    p_charge = cp.Variable(T, nonneg=True)    # Charging power [MW]
    p_discharge = cp.Variable(T, nonneg=True)  # Discharging power [MW]
    
    # Constraints list
    constraints = []
    
    # Initial SoC constraint
    constraints.append(soc[0] == initial_soc)
    
    # SoC bounds
    constraints.append(soc >= battery.soc_min)
    constraints.append(soc <= battery.soc_max)
    
    # Power flow constraints for each time step
    for t in range(T):
        # DA power decomposition: p_da = p_discharge - p_charge
        # Must always be positive, so just going to get p_da = p_discharge if positive, p_da = p_charge if negative
        constraints.append(p_da[t] == p_discharge[t] - p_charge[t])
        
        # Power limits considering regulation reserves
        # When providing reg up, we must be able to increase discharge by r_up
        # When providing reg down, we must be able to increase charge by r_down
        constraints.append(p_discharge[t] + r_up[t] <= battery.power_max_mw)
        constraints.append(p_charge[t] + r_down[t] <= battery.power_max_mw)
        
        # dynamucs - 1 hour time step
        # SoC decreases when discharging, increases when charging
        constraints.append(
            soc[t + 1] == soc[t] + \
                (- p_discharge[t] / battery.efficiency_discharge
                + p_charge[t] * battery.efficiency_charge) \
                / battery.capacity_mwh
        )
    
    # Objective: Maximize total expected revenue
    # Revenue from DA energy market (discharge positive, charge negative)
    da_energy_revenue = cp.sum(cp.multiply(da_prices, p_da))
    
    # Revenue from regulation capacity - Should always be zero for now
    reg_up_revenue = cp.sum(cp.multiply(reg_up_prices, r_up))
    reg_down_revenue = cp.sum(cp.multiply(reg_down_prices, r_down))
    
    # Optional: battery lifecycle degradation cost
    if battery.degradation_cost > 0:
        throughput_cost = battery.degradation_cost * cp.sum(p_discharge + p_charge)
    else:
        throughput_cost = 0
    
    # Total objective
    objective = cp.Maximize(
        da_energy_revenue + 
        reg_up_revenue + reg_down_revenue - 
        throughput_cost
    )
    
    # Formulate and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)
    
    # Check if solution was found
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    # Extract results
    da_energy_bids = p_da.value
    reg_up_capacity = r_up.value
    reg_down_capacity = r_down.value
    planned_soc = soc.value
    
    # Convert SoC to mWh
    planned_soc_mwh = planned_soc * battery.capacity_mwh
    
    # Calculate expected revenue
    expected_revenue = problem.value
    
    return DAScheduleResult(
        da_energy_bids=da_energy_bids,
        reg_up_capacity=reg_up_capacity,
        reg_down_capacity=reg_down_capacity,
        planned_soc=planned_soc,
        expected_revenue=expected_revenue
    )
