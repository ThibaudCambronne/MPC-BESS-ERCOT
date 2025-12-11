from dataclasses import dataclass

@dataclass
class BatteryParams:
    capacity_mwh: float = 100.0      # Energy capacity [MWh]
    power_max_mw: float = 25.0       # Max charge/discharge [MW]
    soc_min: float = 0.1             # Min SoC [fraction]
    soc_max: float = 0.9             # Max SoC [fraction]
    rte: float = 0.9  # Round trip efficiency
    throughput_limit: float = 2 * 100  # Limit on discharge for warranty [MWh]
