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
    risk_aversion: float = 2
) -> DAScheduleResult:
    """
    Solve Stage 1 DA optimization problem.
    
    This function formulates and solves a convex optimization problem to determine
    optimal day-ahead bids for energy and ancillary services (regulation up/down).
    
    Parameters
    ----------
    da_price_forecast : pd.Series
        Day-ahead energy prices for 24 hours [$/MWh]
    rt_price_forecast : pd.Series
        Real-time energy prices for 24 hours [$/MWh]
    initial_soc : float
        Initial state of charge [fraction, 0-1]. Determined from previous optimization (either DA or RA)
        Alternatively, could just be set to 0.5, and EOD SOC could be constrained to 0.5
    battery : BatteryParams
        Battery parameters (capacity, power limits, efficiency, etc.)
    reg_up_price : Optional[pd.Series]
        Regulation up capacity prices for 24 hours [$/MW]. Currently not doing this.
    reg_down_price : Optional[pd.Series]
        Regulation down capacity prices for 24 hours [$/MW]. Currently not doing this. 
    rt_risk_factor : float
        Risk factor for real-time dispatch (0-1), since forecast is more uncertain 
        than day-ahead
    rt_dispatches_per_hour : float
        amount of power dispatches per hour [#/hour]. Currently dispatches are done at 5 minute increments
    
    Returns
    -------
    DAScheduleResult
        Optimization results including DA energy bids, regulation capacity,
        planned SoC trajectory, and expected revenue
    """
    # Number of time periods (just 24 hours * 5 min price per hour, generally)
    T = len(rt_price_forecast)
    
    # Convert prices to numpy arrays
    da_prices = da_price_forecast.values
    rt_prices = rt_price_forecast.values

    if rt_price_uncertainty is not None:
        rt_uncertainty = rt_price_uncertainty.values
    else:
        rt_uncertainty = np.ones(T) * 5
    
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

    # energy bid for day ahead market 
    p_da = cp.Variable(T) 

    # energy bid for real time market
    p_rt = cp.Variable(T)

    # actual dispatch schedule 
    p_real = cp.Variable(T)

    p_discharge = cp.Variable(T, nonneg = True)
    p_charge = cp.Variable(T, nonneg = True)
    
    # State of charge % of total capacity
    soc = cp.Variable(T + 1, nonneg = True)  # all soc variables, taking initial and producing next initial
    
    # Constraints list
    constraints = []
    
    # Initial SoC constraint
    constraints.append(soc[0] == initial_soc)
    
    constraints.append(soc <= battery.soc_max)
    constraints.append(soc >= battery.soc_min)
    
    # doing start and end at 50% constrained for now 
    constraints.append(soc[T] == end_of_day_soc)

    # constraint for da bid amount 
    for t in range(int(T / rt_dispatches_per_hour)):
        start_idx = int(t * rt_dispatches_per_hour)
        end_idx = int((t + 1) * rt_dispatches_per_hour)
        for t2 in range(start_idx+1, end_idx):
            constraints.append(p_da[t2] == p_da[start_idx])
    
    # Power flow constraints for each time step
    for t in range(T):
        # DA power decomposition: p_da = p_discharge - p_charge
        # Must always be positive, so just going to get p_da = p_discharge if positive, p_da = p_charge if negative
        
        # charge and discharge
                
        constraints.append(p_real[t] == p_discharge[t] - p_charge[t]) 

        # getting bids
        constraints.append(p_real[t] == p_da[t] + p_rt[t])

        constraints.append(p_real[t] <= battery.power_max_mw)
        constraints.append(p_real[t] >= - battery.power_max_mw)
        
        # power limits
        constraints.append(p_rt[t] <= battery.power_max_mw)
        constraints.append(p_rt[t] >= - battery.power_max_mw)
        constraints.append(p_da[t] <= battery.power_max_mw)
        constraints.append(p_da[t] >= - battery.power_max_mw)
        
        constraints.append(p_discharge[t] <= battery.power_max_mw)
        constraints.append(p_charge[t] <= battery.power_max_mw)

        # KEY CHANGE: SoC dynamics with different efficiencies
        # When charging (p_charge > 0): energy stored = p_charge * eta_charge
        # When discharging (p_discharge > 0): energy removed = p_discharge / eta_discharge
        constraints.append(
            soc[t + 1] == soc[t] + 
            (p_charge[t] * battery.efficiency_charge - p_discharge[t] / battery.efficiency_discharge) / 
            (rt_dispatches_per_hour * battery.capacity_mwh)
        )
        # soc constraints, can't go below min or above max 
        constraints.append(soc[t+1] >= battery.soc_min)
        constraints.append(soc[t+1] <= battery.soc_max)
    

    # battery throughpot constraint, respecting warranty scenario
    constraints.append(cp.sum(cp.abs(p_real)) / rt_dispatches_per_hour <= battery.throughput_limit)
    
    # Objective: Maximize total expected revenue
    # Revenue from DA energy market (discharge positive, charge negative
    
    # Total objective
    # objective = cp.Maximize(
    #     da_energy_revenue + rt_enegy_revenue +
    #     reg_up_revenue + reg_down_revenue 
    # )

    # 1. Penalize RT positions proportional to uncertainty
    rt_downside_cost = risk_aversion * cp.sum(
        cp.multiply(rt_uncertainty, cp.abs(p_rt))
    )
    
    # 2. Hard cap on RT exposure during high uncertainty
    high_uncertainty_threshold = np.percentile(rt_uncertainty, 75)
    
    for t in range(T):
        if rt_uncertainty[t] > high_uncertainty_threshold:
            # Severely limit RT positions during very uncertain periods
            constraints.append(cp.abs(p_rt[t]) <= 0.3 * battery.power_max_mw)
    
        # Revenue from selling energy (positive when discharging at high prices)
    da_revenue = cp.sum(cp.multiply(da_prices, p_da))
    rt_revenue = cp.sum(cp.multiply(rt_prices, p_rt))

    # Risk penalty for RT uncertainty
    rt_risk_penalty = risk_aversion * cp.sum(
        cp.multiply(rt_uncertainty, cp.abs(p_rt))
    )

    # Maximize profit = revenue - risk penalty
    objective = cp.Maximize(da_revenue + rt_revenue - rt_risk_penalty)
        
    # Formulate and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=True)
    
    # Check if solution was found
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    # Extract results
    da_energy_bids = p_da.value
    # reg_up_capacity = r_up.value
    # reg_down_capacity = r_down.value
    reg_up_capacity = 0
    reg_down_capacity = 0
    soc_schedule = soc.value
    power_dispatch_schedule = p_real.value
    rt_energy_bids = p_rt.value
    discharge = p_discharge.value
    charge = p_charge.value
    # Calculate expected revenue
    expected_revenue = problem.value
    
    return DAScheduleResult(
        da_energy_bids=da_energy_bids,
        rt_energy_bids=rt_energy_bids,
        power_dispatch_schedule=power_dispatch_schedule,
        soc_schedule=soc_schedule,
        reg_up_capacity=reg_up_capacity,
        reg_down_capacity=reg_down_capacity,
        expected_revenue=expected_revenue,
        diagnostic_information={
            "discharge": discharge,
            "charge": charge,}
    )
