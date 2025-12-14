# MPC-BESS-ERCOT Simulation Guide

## Table of Contents
1. [Overview](#overview)
2. [Sign Conventions](#sign-conventions)
3. [How the Simulation Works](#how-the-simulation-works)
4. [Revenue Calculation](#revenue-calculation)
5. [Debugging Guide](#debugging-guide)
6. [Key Parameters](#key-parameters)

## Overview

This simulator implements a two-stage Model Predictive Control (MPC) strategy for a Battery Energy Storage System (BESS) participating in ERCOT electricity markets.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SIMULATION LOOP                         │
│  (runs for n_days, starting at start_date)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │         For Each Day (0:00 AM)         │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   STAGE 1: Day-Ahead Scheduling        │
         │   (solve_da_schedule)                  │
         │                                        │
         │   • Runs ONCE per day at midnight      │
         │   • Uses 24h DA/RT price forecasts     │
         │   • Optimizes DA and RT bids           │
         │   • Plans SoC trajectory for the day   │
         │   • Solver: Pyomo/IPOPT                │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   STAGE 2: Real-Time MPC Loop          │
         │   (solve_rt_mpc)                       │
         │                                        │
         │   • Runs every 15 minutes (96x/day)    │
         │   • Uses current SOC + RT forecast     │
         │   • Considers DA commitments           │
         │   • Computes optimal power setpoint    │
         │   • Solver: Pyomo/IPOPT                │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │     Update Battery State (SOC)         │
         │                                        │
         │   • Apply power setpoint               │
         │   • Account for charge/discharge eff.  │
         │   • Clamp to SOC limits [0.1, 0.9]     │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │      Calculate Revenue                 │
         │                                        │
         │   revenue = -sum(price × power × dt)   │
         └────────────────────────────────────────┘
```

## Sign Conventions

**CRITICAL: All modules use consistent sign conventions**

### Power Sign Convention
```
Positive power (+)  = CHARGING   (battery buying from grid)
Negative power (-)  = DISCHARGING (battery selling to grid)
```

**Examples:**
- `power = +20 MW` → Battery is charging at 20 MW
- `power = -15 MW` → Battery is discharging at 15 MW
- `power = 0 MW`   → Battery is idle

### Revenue Calculation
```
When DISCHARGING (p < 0): We SELL power → Revenue POSITIVE
When CHARGING    (p > 0): We BUY power  → Revenue NEGATIVE

Formula: revenue = -sum(price × power × dt)
```

**Example:**
```python
# At time t:
price = $50/MWh
power = -20 MW  # discharging
dt = 0.25 hours # 15 minutes

# Revenue for this interval:
revenue = -(50 × (-20) × 0.25) = $250  # Positive!

# If charging instead:
power = +20 MW  # charging
revenue = -(50 × 20 × 0.25) = -$250  # Negative (we pay)
```

### SOC Dynamics
```python
if power > 0:  # Charging
    energy_added = power × efficiency_charge × dt
    soc_next = soc + energy_added / capacity

else:  # Discharging (power < 0)
    energy_removed = power / efficiency_discharge × dt  # Note: power is negative
    soc_next = soc + energy_removed / capacity
```

## How the Simulation Works

### 1. Initialization
```python
from src.simulator import run_simulation
from src.utils import load_ercot_data
import pandas as pd

data = load_ercot_data()

results = run_simulation(
    data=data,
    start_date=pd.Timestamp("2020-01-02"),
    n_days=3,                    # Simulate 3 days
    forecast_method="perfect",   # Use actual future prices
    initial_soc=0.5,            # Start at 50% SOC
    end_of_day_soc=0.5          # Target 50% at end of each day
)
```

### 2. Day-by-Day Execution

For each day:

#### **Step 1: Day-Ahead Optimization (00:00 AM)**

The DA scheduler ([stage1_da_scheduler.py](src/stage1_da_scheduler.py)) runs once at midnight:

**Inputs:**
- DA price forecast (24 hours, hourly)
- RT price forecast (24 hours, 15-min)
- Initial SOC (from previous day or 0.5 for first day)
- Battery parameters

**Decision Variables:**
- `p_da[t]`: Day-ahead energy bids (MW) - constant per hour
- `p_rt[t]`: Real-time energy bids (MW) - 15-min resolution
- `p_real[t]`: Actual power dispatch (MW)
- `soc[t]`: State of charge trajectory

**Constraints:**
- Power limits: `-25 MW ≤ power ≤ 25 MW`
- SOC limits: `0.1 ≤ SOC ≤ 0.9`
- SOC dynamics with efficiencies
- Throughput limit (battery warranty)
- End-of-day SOC = 0.5

**Objective:**
Minimize expected cost (= maximize revenue) considering both DA and RT markets with CVaR risk measure.

**Output:**
- `DAScheduleResult` containing planned bids and SOC trajectory

#### **Step 2: Real-Time MPC (Every 15 Minutes)**

The RT MPC ([stage2_rt_mpc.py](src/stage2_rt_mpc.py)) runs 96 times per day:

**Inputs:**
- Current time and SOC
- RT price forecast (next 24 hours)
- DA commitments from Step 1

**Decision Variables:**
- `P_RT[t]`: RT market adjustments
- `P_ch[t]`: Charging power (≥ 0)
- `P_dis[t]`: Discharging power (≤ 0)
- `E[t]`: Energy stored in battery

**Constraints:**
- Initial condition: `E[0] = current_soc × capacity`
- Power balance: `P_DA[t] + P_RT[t] = P_ch[t] + P_dis[t]`
- Dynamics: `E[t+1] = E[t] + (P_ch × η_ch + P_dis / η_dis) × dt`
- Power/SOC limits

**Objective:**
Minimize RT market cost over the horizon.

**Output:**
- `RTMPCResult` with power setpoint for current interval

#### **Step 3: State Update**

Apply the power setpoint from RT MPC:

```python
power_setpoint = rt_result.power_setpoint

if power_setpoint > 0:  # Charging
    energy_change = power_setpoint × 0.95 × 0.25  # efficiency × dt
else:  # Discharging
    energy_change = power_setpoint / 0.95 × 0.25  # 1/efficiency × dt

soc_next = soc_current + energy_change / 100  # capacity = 100 MWh
```

#### **Step 4: Revenue Calculation**

After all 96 intervals:

```python
revenue = -sum(rt_prices × power_trajectory × 0.25)
```

### 3. Multi-Day Continuation

The final SOC from each day becomes the initial SOC for the next day:

```python
for day in simulation:
    day_result = simulate_day(..., initial_soc=current_soc, end_of_day_soc=0.5)
    current_soc = day_result.final_soc  # Carry over to next day
```

## Revenue Calculation

### Why Revenue Was Negative (Bug Fixed)

**OLD CODE (Wrong):**
```python
# This was backwards!
da_revenue = -sum(da_bids × prices × dt)
rt_revenue = -sum(rt_bids × prices × dt)
```

**Problem:** The optimizers minimize COST, so their outputs are already in cost convention. Adding another negative sign made profitable discharge operations appear as losses.

**NEW CODE (Correct):**
```python
# Revenue based on ACTUAL dispatched power
revenue = -sum(rt_prices × power_trajectory × dt)

# Where power_trajectory comes from RT MPC:
#   negative power = discharge = sell power = positive revenue
#   positive power = charge = buy power = negative revenue
```

### Revenue Breakdown

The simulator now reports:
- **Total Revenue**: Sum of all intervals' revenue
- **DA Revenue**: Set to 0 (actual DA market not implemented in current version)
- **RT Revenue**: All revenue attributed to RT market settlement

### Expected Revenue Range

For a 100 MWh / 25 MW battery:
- **Perfect forecast**: $500-2000/day depending on price volatility
- **Persistence forecast**: $200-800/day (lower due to forecast error)

## Debugging Guide

### 1. Check Sign Conventions

```python
from src.simulator import simulate_day

result = simulate_day(data, date, 0.5, battery, "perfect")

print("Power trajectory (first 10 intervals):")
print(result.power_trajectory[:10])
print(f"Min: {result.power_trajectory.min():.2f} MW")
print(f"Max: {result.power_trajectory.max():.2f} MW")

# Expectations:
# - Both positive and negative values
# - Range: [-25, 25] MW
# - Negative values when prices are high (discharge to sell)
# - Positive values when prices are low (charge to buy)
```

### 2. Check SOC Trajectory

```python
print("\nSOC trajectory:")
print(f"Start: {result.soc_trajectory[0]:.2%}")
print(f"Min: {result.soc_trajectory.min():.2%}")
print(f"Max: {result.soc_trajectory.max():.2%}")
print(f"End: {result.soc_trajectory[-1]:.2%}")

# Expectations:
# - Should vary between 10% and 90%
# - Should decrease when discharging (negative power)
# - Should increase when charging (positive power)
# - End SOC may not exactly match end_of_day_soc due to RT MPC deviations
```

### 3. Check Revenue Calculation

```python
import numpy as np

# Manual revenue check for one interval:
t = 10  # Interval index
price = data.loc[result.date]['HB_SOUTH_RTM'].values[t]
power = result.power_trajectory[t]
dt = 0.25

interval_revenue = -price * power * dt

print(f"\nInterval {t}:")
print(f"  Price: ${price:.2f}/MWh")
print(f"  Power: {power:.2f} MW")
print(f"  Revenue: ${interval_revenue:.2f}")

if power < 0:
    print(f"  → Discharging {abs(power):.2f} MW, earning ${interval_revenue:.2f}")
else:
    print(f"  → Charging {power:.2f} MW, paying ${abs(interval_revenue):.2f}")
```

### 4. Check Price-Power Correlation

```python
import matplotlib.pyplot as plt

prices = data.loc[result.date]['HB_SOUTH_RTM'].values[:96]
powers = result.power_trajectory

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(prices, label='RT Price')
plt.ylabel('Price [$/MWh]')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(powers, label='Power', color='purple')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.ylabel('Power [MW]')
plt.xlabel('15-min Interval')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Expectations:
# - When prices are high: power should be NEGATIVE (discharging to sell)
# - When prices are low: power should be POSITIVE (charging to buy cheap)
```

### 5. Solver Status

Check if optimizations are solving successfully:

```python
# In simulator.py, the RT MPC returns solve status:
print(f"RT MPC Status: {rt_result.solve_status}")

# Should be "optimal" most of the time
# If "max_iter" or other status, may need to adjust solver parameters
```

## Key Parameters

### Battery Parameters

```python
from src.battery_model import BatteryParams

battery = BatteryParams(
    capacity_mwh=100.0,         # Energy capacity
    power_max_mw=25.0,          # Max charge/discharge rate (C-rate = 0.25)
    soc_min=0.1,                # Min 10% to protect battery
    soc_max=0.9,                # Max 90% to protect battery
    efficiency_charge=0.95,     # 95% charging efficiency
    efficiency_discharge=0.95,  # 95% discharging efficiency
    throughput_limit=200.0      # Max daily energy cycled [MWh]
)
```

### Forecast Methods

- **`"perfect"`**: Uses actual future prices
  - Upper bound on performance
  - Useful for benchmarking and debugging

- **`"persistence"`**: Uses previous day's prices
  - More realistic but lower performance
  - Simulates naive forecasting

### Horizon Types

- **`"receding"`** (recommended): Fixed 24-hour horizon
  - Standard MPC approach
  - Consistent decision-making throughout day

- **`"shrinking"`**: Horizon shrinks to end of day
  - May cause aggressive behavior near midnight
  - Less common in practice

### SOC Constraints

- **`initial_soc`**: Starting SOC for first day (default: 0.5)
- **`end_of_day_soc`**: Target SOC at end of each day (default: 0.5)
  - Currently enforced in DA scheduler
  - RT MPC may deviate slightly due to forecast updates

## Common Issues

### Issue: Negative Revenue

**Symptom:** Total revenue is large and negative

**Cause:** Sign convention mismatch (now fixed!)

**Check:**
```python
# Revenue should be positive for profitable operations
assert results.total_revenue > 0, "Revenue should be positive!"
```

### Issue: SOC Not Reaching Target

**Symptom:** `final_soc != end_of_day_soc`

**Cause:** RT MPC doesn't enforce terminal constraint

**Explanation:** The DA scheduler plans for `end_of_day_soc = 0.5`, but RT MPC re-optimizes based on updated forecasts and may deviate. This is normal MPC behavior (plan updates).

**Solution:** Accept small deviations (~5-10%) or add terminal cost in RT MPC.

### Issue: Solver Failures

**Symptom:** "Solver Error" or "infeasible" status

**Possible Causes:**
1. Insufficient forecast data (check data coverage)
2. Infeasible SOC constraints (e.g., can't reach end_soc with power limits)
3. Solver numerical issues

**Debug:**
```python
# Enable solver output
# In stage2_rt_mpc.py, line 120:
solver.options['print_level'] = 5  # Increase verbosity
```

## Next Steps

1. **Validate Results:** Compare against known benchmarks
2. **Sensitivity Analysis:** Vary battery parameters and observe revenue
3. **Forecast Comparison:** Test "perfect" vs "persistence" methods
4. **Risk Analysis:** Examine CVaR metrics in DA scheduler
5. **Longer Simulations:** Run for weeks/months to get statistical significance
