# Energy — Decision Chain

> **Status:** Live — 1 implementation; decision points mapped across all 5 phases.

## About this sector

The energy sector involves infrastructure siting, generation planning, storage optimization, distribution, and market trading. OR methods optimize unit commitment, economic dispatch, grid operations under uncertainty, and energy trading strategies. The sector features large-scale network flow problems, stochastic programming for renewable intermittency, and robust optimization for market participation.

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Infrastructure | Power Plant Siting | Facility Location / CFLP | — | Planned |
| 01 Infrastructure | Transmission Network Design | Network Design / MST | — | Planned |
| 02 Generation | Unit Commitment | MIP (binary on/off decisions) | — | Planned |
| 02 Generation | Capacity Expansion | Network Design / LP | — | Planned |
| 03 Storage | Battery Dispatch Scheduling | LP / Inventory Management | — | Planned |
| 03 Storage | Storage Sizing | Knapsack / Capacity Planning | — | Planned |
| 04 Distribution | Load Balancing | Network Flow | `energy_grid.py` | Live |
| 05 Trading | Energy Portfolio Optimization | Robust Portfolio | — | Planned |
| 05 Trading | Bidding Strategy | Two-Stage Stochastic Programming | — | Planned |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| Facility Location | Location | `problems/5_location_covering/facility_location/` |
| Network Design | Network | `problems/6_network_flow_design/network_design/` |
| MST | Network | `problems/6_network_flow_design/min_spanning_tree/` |
| Network Flow | Network | `problems/6_network_flow_design/max_flow/` |
| Knapsack | Packing | `problems/3_packing_cutting/knapsack/` |
| Robust Portfolio | Uncertainty | `problems/9_uncertainty_modeling/robust_portfolio/` |
| Two-Stage SP | Uncertainty | `problems/9_uncertainty_modeling/two_stage_sp/` |
| LP | Continuous | `problems/continuous/linear_programming/` |

## Existing implementations
- `energy_grid.py` — Energy grid load balancing as network flow

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations
