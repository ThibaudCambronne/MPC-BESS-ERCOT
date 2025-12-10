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
    reg_up_price: Optional[pd.Series] = None,
    reg_down_price: Optional[pd.Series] = None,
    initial_soc: float = 0.5,
    rt_risk_factor: float = 0,
    rt_dispatches_per_hour: float = 12,
    end_of_day_soc: float = 0.5
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

    # variables for charge/discharge and frequency, makes constraints easier
    p_charge = cp.Variable(T, nonneg=True)    # Charging power [MW]
    p_discharge = cp.Variable(T, nonneg=True)  # Discharging power [MW]
    p_da_freq_rt = cp.Variable(T)  # 5 minute frequency power that will become day ahead bid [MW]
    
    # Regulation capacity [MW] - always non-negative
    # Maybe constrain to zero using bounds(0,0) if just having 0 prices doesn't work? 
    r_up = cp.Variable(T, nonneg=True)    # Reg up capacity
    r_down = cp.Variable(T, nonneg=True)  # Reg down capacity
    
    # State of charge % of total capacity
    soc = cp.Variable(T + 1, nonneg = True)  # all soc variables, taking initial and producing next initial
    
    # Constraints list
    constraints = []
    
    # Initial SoC constraint
    constraints.append(soc[0] == initial_soc)
    
    # SoC bounds
    constraints.append(soc[0] >= battery.soc_min)
    constraints.append(soc[0] <= battery.soc_max)
    
    # doing start and end at 50% constrained for now 
    constraints.append(soc[T] == end_of_day_soc)

    # constraint for da bid amount 
    for t in range(int(T / rt_dispatches_per_hour)):
        constraints.append(p_da[t] == p_da_freq_rt[t:(t+1)*rt_dispatches_per_hour].mean())
    
    # Power flow constraints for each time step
    for t in range(T):
        # DA power decomposition: p_da = p_discharge - p_charge
        # Must always be positive, so just going to get p_da = p_discharge if positive, p_da = p_charge if negative

        # power decomposition
        constraints.append(p_real[t] == p_discharge[t] - p_charge[t])
        
        # Discharge power should be the positive part of p_real
        constraints.append(p_discharge[t] == cp.maximum(0, p_real[t]))  # Only positive part contributes to discharge
        
        # Charge power should be the negative part of p_real
        constraints.append(p_charge[t] == cp.maximum(0, -p_real[t]))  # Only negative part contributes to charge


        # getting bids
        constraints.append(p_real[t] == p_da_freq_rt[t] + p_rt[t])
        
        # power limits
        constraints.append(p_rt[t] <= battery.power_max_mw)
        constraints.append(p_rt[t] >= - battery.power_max_mw)
        constraints.append(p_da_freq_rt[t] <= battery.power_max_mw)
        constraints.append(p_da_freq_rt[t] >= - battery.power_max_mw)
        
        # Power limits
        constraints.append(p_discharge[t] <= battery.power_max_mw)
        constraints.append(p_charge[t] <= battery.power_max_mw)
        
        # Power limits considering regulation reserves
        # When providing reg up, we must be able to increase discharge by r_up
        # When providing reg down, we must be able to increase charge by r_down
        # constraints.append(p_discharge[t] + r_up[t] <= battery.power_max_mw)
        # constraints.append(p_charge[t] + r_down[t] <= battery.power_max_mw)
        
        # dynamucs - 5 min timestep
        # SoC decreases when discharging, increases when charging
        # dividing by rt_dispatchers_per_hour to get MWh
        constraints.append(
            soc[t + 1] == soc[t] + \
                (- p_discharge[t] / battery.efficiency_discharge / rt_dispatches_per_hour 
                + p_charge[t] * battery.efficiency_charge / rt_dispatches_per_hour) \
                / battery.capacity_mwh
        )
        # soc constraints, can't go below min or above max 
        constraints.append(soc[t+1] >= battery.soc_min)
        constraints.append(soc[t+1] <= battery.soc_max)
        
    # battery throughpot constraint, respecting warranty scenario
    constraints.append(cp.sum(p_discharge) / rt_dispatches_per_hour <= battery.throughput_limit)
    
    # Objective: Maximize total expected revenue
    # Revenue from DA energy market (discharge positive, charge negative)
    da_energy_revenue = cp.sum(cp.multiply(da_prices, p_da))
    
    # Revenue from regulation capacity - Should always be zero for now
    # reg_up_revenue = cp.sum(cp.multiply(reg_up_prices, r_up))
    # reg_down_revenue = cp.sum(cp.multiply(reg_down_prices, r_down))
    
    # Revenue from RT energy market
    rt_enegy_revenue = cp.sum(cp.multiply(rt_prices, p_rt)) * (1 - rt_risk_factor)
    
    # Total objective
    # objective = cp.Maximize(
    #     da_energy_revenue + rt_enegy_revenue +
    #     reg_up_revenue + reg_down_revenue 
    # )
    objective = cp.Maximize(
        da_energy_revenue + rt_enegy_revenue
    )
    
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
    )
