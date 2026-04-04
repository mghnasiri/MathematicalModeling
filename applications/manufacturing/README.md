# Manufacturing — Decision Chain

> **Status:** Live — 3 implementations; decision points mapped across all 6 phases.

## About this sector

Manufacturing operations span plant design, production planning, shop floor scheduling, assembly line balancing, supply chain management, and maintenance optimization. OR methods optimize production sequences (flow shop, job shop), balance assembly lines, size production lots, and plan preventive maintenance. The sector features some of the most classical OR problems — flow shop scheduling dates to Johnson (1954).

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Strategic | Plant Location | Facility Location / CFLP | — | Planned |
| 02 Production | Production Lot Sizing | Capacitated Lot Sizing (CLSP) | — | Planned |
| 02 Production | Material Requirements | Wagner-Whitin / Lot Sizing | — | Planned |
| 03 Shop Floor | Job Scheduling | Job Shop / Flow Shop | `manufacturing_scheduling.py` | Live |
| 03 Shop Floor | Resource Packing | Bin Packing / Knapsack | `cloud_resource_packing.py` | Live |
| 04 Assembly | Assembly Line Balancing | SALBP | — | Planned |
| 05 Supply Chain | Supply Chain Network Design | Network Design / Facility Location | `supply_chain_network.py` | Live |
| 06 Maintenance | Preventive Maintenance Scheduling | Single Machine / Batch Scheduling | — | Planned |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| Flow Shop | Scheduling | `problems/1_scheduling/flow_shop/` |
| Job Shop | Scheduling | `problems/1_scheduling/job_shop/` |
| Single Machine | Scheduling | `problems/1_scheduling/single_machine/` |
| Assembly Line Balancing | Scheduling | `problems/1_scheduling/assembly_line_balancing/` |
| Bin Packing | Packing | `problems/3_packing_cutting/bin_packing/` |
| Facility Location | Location | `problems/5_location_covering/facility_location/` |
| Network Design | Network | `problems/6_network_flow_design/network_design/` |
| Capacitated Lot Sizing | Inventory | `problems/7_inventory_lotsizing/capacitated_lot_sizing/` |
| Wagner-Whitin | Inventory | `problems/7_inventory_lotsizing/wagner_whitin/` |

## Existing implementations
- `manufacturing_scheduling.py` — Job shop scheduling for manufacturing
- `cloud_resource_packing.py` — Resource packing as flexible manufacturing analogy
- `supply_chain_network.py` — Supply chain network design

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations
