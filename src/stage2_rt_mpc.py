import pyomo.environ as pyo
import numpy as np
import pandas as pd
from typing import Literal
from .battery_model import BatteryParams
from .utils import DAScheduleResult, RTMPCResult

def solve_rt_mpc(
    current_time: pd.Timestamp,
    current_soc: float,
    rt_price_forecast: pd.Series,
    da_commitments: DAScheduleResult,
    battery: BatteryParams,
    horizon_type: Literal["shrinking", "receding"] = "receding",
    horizon_hours: int = 24
) -> RTMPCResult:
    """
    Solve Stage 2 Real-Time (RT) MPC problem using Pyomo/Ipopt.
    """
    
    # --- 1. Setup Horizon & Data ---
    delta_t = 0.25 # 15 min
    freq = "15min"
    
    if horizon_type == "shrinking":
        end_time = (current_time + pd.Timedelta(days=1)).normalize()
    else:
        end_time = current_time + pd.Timedelta(hours=horizon_hours)
        
    horizon_index = pd.date_range(start=current_time, end=end_time, freq=freq, inclusive='left')
    T = len(horizon_index)
    
    if T == 0:
        return RTMPCResult(0.0, np.array([current_soc]), "empty_horizon")

    # Align Data
    c_rt_data = rt_price_forecast.reindex(horizon_index).ffill().fillna(0.0).values
    
    # Day-Ahead Commitments mapping
    da_start = current_time.normalize()
    n_da = len(da_commitments.da_energy_bids)
    da_index = pd.date_range(start=da_start, periods=n_da, freq=freq)
    da_series = pd.Series(da_commitments.da_energy_bids, index=da_index)
    p_da_data = da_series.reindex(horizon_index).fillna(0.0).values

    # --- 2. Pyomo Model Construction ---
    model = pyo.ConcreteModel()

    model.T = pyo.RangeSet(0, T-1) 
    model.T_E = pyo.RangeSet(0, T) 

    # Parameters
    model.P_DA = pyo.Param(model.T, initialize=lambda m, t: p_da_data[t], mutable=True)
    model.c_RT = pyo.Param(model.T, initialize=lambda m, t: c_rt_data[t], mutable=True)
    
    E_min = battery.soc_min * battery.capacity_mwh
    E_max = battery.soc_max * battery.capacity_mwh
    Z = 2.0 
    
    # Variables
    model.P_RT = pyo.Var(model.T, domain=pyo.Reals)
    model.P_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, battery.power_max_mw))
    model.P_dis = pyo.Var(model.T, domain=pyo.NonPositiveReals, bounds=(-battery.power_max_mw, 0))
    model.E = pyo.Var(model.T_E, bounds=(E_min, E_max))

    # --- 3. Constraints ---
    def init_condition_rule(m):
        return m.E[0] == current_soc * battery.capacity_mwh
    model.init_condition = pyo.Constraint(rule=init_condition_rule)

    def power_balance_rule(m, t):
        return m.P_DA[t] + m.P_RT[t] == m.P_ch[t] + m.P_dis[t]
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    def dynamics_rule(m, t):
        if t < T:
            power_flow = (m.P_ch[t] * battery.efficiency_charge + 
                          m.P_dis[t] / battery.efficiency_discharge)
            return m.E[t+1] == m.E[t] + power_flow * delta_t
        return pyo.Constraint.Skip
    model.dynamics = pyo.Constraint(model.T, rule=dynamics_rule)

    def market_constraints_rule(m, t):
        return (-battery.power_max_mw, m.P_RT[t], battery.power_max_mw)
    model.market_constraints = pyo.Constraint(model.T, rule=market_constraints_rule)

    def throughput_rule(m):
        total_charge_energy = sum(m.P_ch[t] for t in m.T) * delta_t
        limit = Z * (E_max - E_min)
        return total_charge_energy <= limit
    model.throughput_check = pyo.Constraint(rule=throughput_rule)

    # --- 4. Objective ---
    def obj_rule(m):
        return sum(m.c_RT[t] * m.P_RT[t] for t in m.T) * delta_t
    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- 5. Solve with Robust Options ---
    solver = pyo.SolverFactory('ipopt')
    
    # Increase iterations and relax tolerance slightly for robustness
    solver.options['max_iter'] = 5000
    solver.options['tol'] = 1e-6
    solver.options['print_level'] = 0
    
    try:
        result = solver.solve(model, tee=False)
    except Exception as e:
        return RTMPCResult(0.0, np.array([current_soc]), f"Solver Error: {str(e)}")

    # --- 6. Extract Results ---
    # Accept 'optimal' OR 'maxIterations' (feasible but hit limit)
    term_cond = result.solver.termination_condition
    if (result.solver.status == pyo.SolverStatus.ok and 
        (term_cond == pyo.TerminationCondition.optimal or 
         term_cond == pyo.TerminationCondition.maxIterations)):
        
        p_rt_opt = pyo.value(model.P_RT[0])
        p_da_fixed = pyo.value(model.P_DA[0])
        power_setpoint = p_da_fixed + p_rt_opt
        
        soc_traj = np.array([pyo.value(model.E[t]) for t in model.T_E]) / battery.capacity_mwh
        
        status_msg = "optimal" if term_cond == pyo.TerminationCondition.optimal else "max_iter"
        
        return RTMPCResult(
            power_setpoint=power_setpoint,
            predicted_soc=soc_traj,
            solve_status=status_msg
        )
    else:
        return RTMPCResult(
            power_setpoint=0.0,
            predicted_soc=np.array([current_soc]),
            solve_status=str(term_cond)
        )