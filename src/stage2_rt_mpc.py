from typing import Literal

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.globals import DELTA_T, FREQUENCY

from .battery_model import BatteryParams
from .utils import DAScheduleResult, RTMPCResult


def solve_rt_mpc(
    current_time: pd.Timestamp,
    current_soc: float,
    rt_price_forecast: pd.Series,
    da_commitments: DAScheduleResult,
    battery: BatteryParams,
    horizon_type: Literal["shrinking", "receding"] = "receding",
    horizon_hours: int = 24,
) -> RTMPCResult:
    """
    Solve Stage 2 Real-Time (RT) MPC problem using Pyomo/Ipopt.
    """

    # --- 1. Setup Horizon & Data ---
    if horizon_type == "shrinking":
        end_time = (current_time + pd.Timedelta(days=1)).normalize()
    else:
        end_time = current_time + pd.Timedelta(hours=horizon_hours)

    horizon_index = pd.date_range(
        start=current_time, end=end_time, freq=FREQUENCY, inclusive="left"
    )
    T = len(horizon_index)

    if T == 0:
        return RTMPCResult(0.0, np.array([current_soc]), "empty_horizon")

    # Align RT Price Data
    c_rt_data = rt_price_forecast.reindex(horizon_index).ffill().fillna(0.0).values

    # --- Day-Ahead Commitments mapping ---
    da_start = current_time.normalize()
    n_da = len(da_commitments.da_energy_bids)
    
    # Map 96-step DA bids to time index
    da_index = pd.date_range(start=da_start, periods=n_da, freq=FREQUENCY)
    da_power_series = pd.Series(da_commitments.da_energy_bids, index=da_index)
    p_da_data = da_power_series.reindex(horizon_index).fillna(0.0).values
    
    # Map DA SoC Target (for tracking cost)
    # The soc_schedule has T+1 points. We align the first T points to the horizon.
    da_soc_series = pd.Series(da_commitments.soc_schedule[:-1], index=da_index)
    soc_target_data = da_soc_series.reindex(horizon_index).fillna(0.5).values

    # --- 2. Pyomo Model Construction ---
    model = pyo.ConcreteModel()

    model.T = pyo.RangeSet(0, T - 1)
    model.T_E = pyo.RangeSet(0, T)

    # Parameters
    model.P_DA = pyo.Param(model.T, initialize=lambda m, t: p_da_data[t], mutable=True)
    model.c_RT = pyo.Param(model.T, initialize=lambda m, t: c_rt_data[t], mutable=True)
    model.SoC_Target = pyo.Param(model.T, initialize=lambda m, t: soc_target_data[t], mutable=True)

    E_min = battery.soc_min * battery.capacity_mwh
    E_max = battery.soc_max * battery.capacity_mwh
    Z = 2.0
    
    # Tuned Weights
    w_tracking = 0.1  # Reduced from 10.0 to allow more freedom for RT arbitrage
    w_slack = 1e5     # Penalty for violating soft constraints

    # Variables
    model.P_RT = pyo.Var(model.T, domain=pyo.Reals)
    model.P_ch = pyo.Var(
        model.T, domain=pyo.NonNegativeReals, bounds=(0, battery.power_max_mw)
    )
    model.P_dis = pyo.Var(
        model.T, domain=pyo.NonPositiveReals, bounds=(-battery.power_max_mw, 0)
    )
    model.E = pyo.Var(model.T_E, bounds=(E_min, E_max))
    
    # Slack for Robustness
    model.s_soc = pyo.Var(model.T_E, domain=pyo.NonNegativeReals)

    # --- 3. Constraints ---
    model.init_condition = pyo.Constraint(expr=model.E[0] == current_soc * battery.capacity_mwh)

    def power_balance_rule(m, t):
        return m.P_DA[t] + m.P_RT[t] == m.P_ch[t] + m.P_dis[t]
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    def dynamics_rule(m, t):
        if t < T:
            power_flow = (
                m.P_ch[t] * battery.efficiency_charge
                + m.P_dis[t] / battery.efficiency_discharge
            )
            return m.E[t + 1] == m.E[t] + power_flow * DELTA_T
        return pyo.Constraint.Skip
    model.dynamics = pyo.Constraint(model.T, rule=dynamics_rule)

    def market_constraints_rule(m, t):
        return (-battery.power_max_mw, m.P_RT[t], battery.power_max_mw)
    model.market_constraints = pyo.Constraint(model.T, rule=market_constraints_rule)
    
    # Soft SoC Constraints
    def soc_min_soft(m, t):
        return m.E[t] >= E_min - m.s_soc[t]
    model.soc_min_con = pyo.Constraint(model.T_E, rule=soc_min_soft)
    
    def soc_max_soft(m, t):
        return m.E[t] <= E_max + m.s_soc[t]
    model.soc_max_con = pyo.Constraint(model.T_E, rule=soc_max_soft)

    def throughput_rule(m):
        total_charge_energy = sum(m.P_ch[t] for t in m.T) * DELTA_T
        limit = Z * (E_max - E_min)
        return total_charge_energy <= limit
    model.throughput_check = pyo.Constraint(rule=throughput_rule)

    # --- 4. Objective ---
    def obj_rule(m):
        # Economic Profit (Minimize Cost)
        cost_market = sum(m.c_RT[t] * m.P_RT[t] for t in m.T) * DELTA_T
        
        # Tracking Cost (Stay close to DA plan unless profitable deviation exists)
        # Using MWh differences. 
        cost_tracking = sum(w_tracking * (m.E[t] - m.SoC_Target[t] * battery.capacity_mwh)**2 for t in m.T)
        
        # Slack Cost
        cost_slack = sum(w_slack * m.s_soc[t] for t in m.T_E)
        
        return cost_market + cost_tracking + cost_slack

    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- 5. Solve ---
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 3000
    solver.options["tol"] = 1e-6
    solver.options["print_level"] = 0

    try:
        result = solver.solve(model, tee=False)
    except Exception as e:
        return RTMPCResult(0.0, np.array([current_soc]), f"Solver Error: {str(e)}")

    # --- 6. Extract Results ---
    term_cond = result.solver.termination_condition
    if result.solver.status == pyo.SolverStatus.ok and (
        term_cond == pyo.TerminationCondition.optimal
        or term_cond == pyo.TerminationCondition.maxIterations
    ):
        p_rt_opt = pyo.value(model.P_RT[0])
        p_da_fixed = pyo.value(model.P_DA[0])
        power_setpoint = p_da_fixed + p_rt_opt

        soc_traj = (
            np.array([pyo.value(model.E[t]) for t in model.T_E]) / battery.capacity_mwh
        )
        
        slack_val = sum(pyo.value(model.s_soc[t]) for t in model.T_E)
        status_msg = "optimal" if slack_val < 1e-3 else "optimal_with_slack"

        return RTMPCResult(
            power_setpoint=power_setpoint,
            predicted_soc=soc_traj,
            solve_status=status_msg,
        )
    else:
        return RTMPCResult(
            power_setpoint=0.0,
            predicted_soc=np.array([current_soc]),
            solve_status=str(term_cond),
        )