# MPC-BESS-ERCOT

A 2-stage MPC controller producing bidding strategy and battery control for Day-Ahead and Real-Time Energy and Ancillary Services ERCOT markets.

## Overview

This project implements a two-stage optimization framework for battery energy storage systems (BESS) participating in ERCOT electricity markets:

1. **Stage 1 (Day-Ahead Scheduling)**: Solves a convex optimization problem once per day to determine optimal day-ahead bids and provisional real-time energy bids
2. **Stage 2 (Real-Time MPC)**: Runs every 15 minutes using model predictive control to adjust RT bids based on updated forecasts

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

## Running The Streamlit App

Start the multipage dashboard:

```bash
uv run streamlit run app/main.py
```

### Custom Simulation

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

## Optimization Formulation

### Day Ahead

#### Decision Variables

| *Notation* | *Description* | *Units* |
| ---------- | ------------- | ------- |
| $p_{da}(t)$ | Day-ahead power bid, $t \in \{0, \ldots, T-1\}$ | [MW] |
| $p_{rt}(t)$ | Real-time power bid, $t \in \{0, \ldots, T-1\}$ | [MW] |
| $p_{real}(t)$ | Actual net power delivered, $t \in \{0, \ldots, T-1\}$ | [MW] |
| $p_{dis}(t)$ | Net discharge power, $t \in \{0, \ldots, T-1\}$ | [MW] |
| $p_{ch}(t)$ | Net charge power, $t \in \{0, \ldots, T-1\}$ | [MW] |
| $E(t)$ | Energy stored in the battery,  $t \in \{0, \ldots, T\}$ | [MWh] |

*Note: $p_{da}(t)$ and $p_{rt}(t)$ can be positive (buying/charging) or negative (selling/discharging)*

#### Parameters

| *Notation* | *Description* | *Units* |
| ---------- | ------------- | ------- |
| $T$ | Final time step of the optimization horizon | [.] |
| $\Delta t$ | Time step duration (e.g., 0.25 for 15 minutes) | [hours] |
| $c_{da}(t)$ | Day-ahead price, $t \in \{0, \ldots, T-1\}$ | [$/MWh] |
| $c_{rt}(t)$ | Real-time price, $t \in \{0, \ldots, T-1\}$ | [$/MWh] |
| $\eta_{ch}$ | Charging efficiency (between 0 and 1) | [.] |
| $\eta_{dis}$ | Discharging efficiency (between 0 and 1) | [.] |
| $E_{\min}$ | Minimum energy level | [MWh] |
| $E_{\max}$ | Maximum energy level | [MWh] |
| $E_0$ | Initial and terminal energy level at $t=0$ and $t=T$ | [MWh] |
| $P_{\max}$ | Maximum charge/discharge power | [MW] |
| $\delta_{cycle}$ | Max number of battery cycles per day | [.] |

#### Objective

Minimize costs (maximize profits), which is costs from day-ahead and real-time markets:
$$ \begin{align}
\min_{\text{decision variables}} \quad & \sum_{t=0}^{T-1} \left( c_{da}(t) p_{da}(t) + c_{rt}(t) p_{rt}(t) \right) \Delta t

\end{align} $$

#### Constraints
$$
\begin{align}
\text{Net Power flow:}
&& p_{real}(t) &= p_{da}(t) + p_{rt}(t)
&& \forall t \in \{0, \ldots, T-1\} \\

\text{Net power decomposition:}
&& p_{real}(t) &= p_{ch}(t) - p_{dis}(t)
&& \forall t \in \{0, \ldots, T-1\} \\

\text{Charge/discharge limits:}
&& 0 &\leq p_{ch}(t) \leq P_{\max}
&& \forall t \in \{0, \ldots, T-1\} \\
&& 0 &\leq p_{dis}(t) \leq P_{\max}
&& \forall t \in \{0, \ldots, T-1\} \\

\text{Market bids power limits:}
&& -P_{\max} &\leq p_{da}(t) \leq P_{\max}
&& \forall t \in \{0, \ldots, T-1\} \\
&& -P_{\max} &\leq p_{rt}(t) \leq P_{\max}
&& \forall t \in \{0, \ldots, T-1\} \\

\text{Energy dynamics:}
&& E(t+1) &= E(t) + \eta_{ch} p_{ch}(t) \Delta t - \frac{p_{dis}(t)}{\eta_{dis}} \Delta t
&& \forall t \in \{0, \ldots, T-1\} \\

\text{Energy limits:}
&& E_{\min} &\leq E(t) \leq E_{\max}
&& \forall t \in \{0, \ldots, T\} \\

\text{Energy bounds at start and end:}
&& E(0) &= E_0  \\
&& E(T) &= E_0 \\

\text{Battery cycling limit (prevents excessive cycling):}
&& \sum_{t=0}^{T-1} p_{ch}(t) \Delta t &\leq \delta_{cycle} (E_{\max} - E_{\min}) \\


\end{align}
$$

## Architecture

### Modules

- [src/simulator.py](src/simulator.py) - Main simulation loop
- [src/stage1_da_scheduler.py](src/stage1_da_scheduler.py) - Day-ahead optimization (CVXPY)
- [src/stage2_rt_mpc.py](src/stage2_rt_mpc.py) - Real-time MPC (Pyomo/Ipopt)
- [src/forecaster.py](src/forecaster.py) - Price forecasting (persistence/perfect)
- [src/battery_model.py](src/battery_model.py) - Battery parameter configuration
- [src/utils.py](src/utils.py) - Data loading and result structures

## License

See LICENSE file for details.
