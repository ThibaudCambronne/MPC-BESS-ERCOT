from dataclasses import dataclass

@dataclass
class BatteryParams:
    capacity_mwh: float = 100.0      # Energy capacity [MWh]
    power_max_mw: float = 25.0       # Max charge/discharge [MW]
    soc_min: float = 0.1             # Min SoC [fraction]
    soc_max: float = 0.9             # Max SoC [fraction]
    efficiency_charge: float = 0.95  # Charging efficiency
    efficiency_discharge: float = 0.95  # Discharging efficiency
    degradation_cost: float = 0.0    # $/MWh throughput (optional)
