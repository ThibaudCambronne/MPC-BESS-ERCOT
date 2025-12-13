# MPC-BESS-ERCOT

A 2-stage MPC controller producing bidding strategy and battery control for Day-Ahead and Real-Time Energy and Ancillary Services ERCOT markets.

## Overview

This project implements a two-stage optimization framework for battery energy storage systems (BESS) participating in ERCOT electricity markets:

1. **Stage 1 (Day-Ahead Scheduling)**: Solves a convex optimization problem once per day to determine optimal day-ahead and real-time energy bids
2. **Stage 2 (Real-Time MPC)**: Runs every 15 minutes using model predictive control to adjust power setpoints based on updated forecasts

## Installation

This project uses `uv` for fast, reliable Python package management. If you don't have it installed:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

Install dependencies:

```bash
uv sync
```

## Data Setup

The simulator requires three CSV files in the `data/` directory:
- `All_2020_2024_with_AS.csv` - Day-ahead market training data
- `All_2025_with_AS.csv` - Day-ahead market testing data
- `RTM_all_2020_2025_enriched.csv` - Real-time market data

## Running Simulations

### Basic Usage

Run the default 3-day simulation:

```bash
uv run main.py
```

### Custom Simulation

The API has been simplified to use `n_days` instead of `end_date`:

```python
import pandas as pd
from src.utils import load_ercot_data
from src.battery_model import BatteryParams
from src.simulator import run_simulation

# Load data
data = load_ercot_data()

# Configure battery parameters (optional - uses defaults if not specified)
battery = BatteryParams(
    capacity_mwh=100.0,         # Energy capacity [MWh]
    power_max_mw=25.0,          # Max charge/discharge [MW]
    soc_min=0.1,                # Min state of charge [0-1]
    soc_max=0.9,                # Max state of charge [0-1]
    efficiency_charge=0.95,     # Charging efficiency
    efficiency_discharge=0.95,  # Discharging efficiency
    throughput_limit=200.0      # Daily throughput limit [MWh]
)

# Run simulation (simplified API)
results = run_simulation(
    data=data,
    start_date=pd.Timestamp("2020-01-02"),
    n_days=30,                  # Simulate 30 days
    battery=battery,            # Optional
    forecast_method="perfect",  # or "persistence"
    horizon_type="receding",    # or "shrinking"
    initial_soc=0.5,           # Start at 50% SOC
    end_of_day_soc=0.5         # Target 50% at end of each day
)

# Access results
print(f"Total Revenue: ${results.total_revenue:,.2f}")
for day in results.daily_results:
    print(f"{day.date.date()}: ${day.total_revenue:,.2f}")
```

## Architecture

### Modules

- [src/simulator.py](src/simulator.py) - Main simulation loop
- [src/stage1_da_scheduler.py](src/stage1_da_scheduler.py) - Day-ahead optimization (CVXPY)
- [src/stage2_rt_mpc.py](src/stage2_rt_mpc.py) - Real-time MPC (Pyomo/Ipopt)
- [src/forecaster.py](src/forecaster.py) - Price forecasting (persistence/perfect)
- [src/battery_model.py](src/battery_model.py) - Battery parameter configuration
- [src/utils.py](src/utils.py) - Data loading and result structures

### Key Features

- **Separate charge/discharge efficiencies**: Models asymmetric battery losses
- **Throughput constraints**: Respects battery warranty limits
- **Receding horizon MPC**: Re-optimizes every 15 minutes with updated forecasts
- **Multiple forecast methods**:
  - `perfect`: Uses actual future prices (upper bound)
  - `persistence`: Uses previous day's prices at same time
- **Revenue tracking**: Separate accounting for DA and RT revenues

## Example Output

```
Loading ERCOT data...
Data loaded: 204768 intervals from 2020-01-01 00:00:00 to 2025-11-07 23:45:00

Running simulation from 2020-01-02 to 2020-01-04...
Simulating day 1/3: 2020-01-02
Simulating day 2/3: 2020-01-03
Simulating day 3/3: 2020-01-04

============================================================
SIMULATION RESULTS
============================================================
Total Revenue: $8,448.61

Daily Breakdown:
  Day 1 (2020-01-02): $2,576.51
    DA Revenue: $518.90
    RT Revenue: $2,057.61
    Final SOC: 10.00%
  Day 2 (2020-01-03): $3,280.14
    DA Revenue: $3,472.69
    RT Revenue: $-192.56
    Final SOC: 10.00%
  Day 3 (2020-01-04): $2,591.97
    DA Revenue: $2,614.04
    RT Revenue: $-22.08
    Final SOC: 10.00%
```

## Development

Run tests:

```bash
uv run pytest
```

## License

See LICENSE file for details.
