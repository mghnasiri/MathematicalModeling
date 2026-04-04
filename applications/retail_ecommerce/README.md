# Retail & E-Commerce — Decision Chain

> **Status:** Live — 3 implementations; decision points mapped across all 4 phases.

## About this sector

Retail and e-commerce operations span store/warehouse location, inventory management, order fulfillment, and last-mile delivery. OR methods optimize the end-to-end supply chain — from strategic network design (where to locate warehouses) through inventory policies (how much to stock) to operational routing (how to deliver). The newsvendor model and facility location problem are foundational to retail OR.

## Decision chain

| Phase | Decision point | OR problem | Implementation | Status |
|---|---|---|---|---|
| 01 Network | Warehouse / DC Location | UFLP / CFLP | `warehouse_location.py` | Live |
| 01 Network | Assortment Optimization | Portfolio / Knapsack | `finance_portfolio.py` | Live |
| 02 Inventory | Retail Inventory Management | Newsvendor / (s,S) Policy | `retail_inventory.py` | Live |
| 02 Inventory | Safety Stock Optimization | Safety Stock / EOQ | — | Planned |
| 03 Fulfillment | Order Picking & Packing | Bin Packing / Assignment | — | Planned |
| 03 Fulfillment | Batch Order Scheduling | Batch Scheduling | — | Planned |
| 04 Last Mile | Last-Mile Delivery Routing | CVRP / VRPTW | — | Planned |
| 04 Last Mile | Crowd-Sourced Delivery | Assignment / Matching | — | Planned |

## Canonical problems used

| OR Problem | Problem Family | Repository Path |
|---|---|---|
| Facility Location (UFLP) | Location | `problems/5_location_covering/facility_location/` |
| Newsvendor | Uncertainty | `problems/9_uncertainty_modeling/newsvendor/` |
| Safety Stock | Inventory | `problems/7_inventory_lotsizing/safety_stock/` |
| EOQ | Inventory | `problems/7_inventory_lotsizing/eoq/` |
| CVRP | Routing | `problems/2_routing/cvrp/` |
| VRPTW | Routing | `problems/2_routing/vrptw/` |
| Bin Packing | Packing | `problems/3_packing_cutting/bin_packing/` |
| Assignment | Assignment | `problems/4_assignment_matching/assignment/` |
| Knapsack | Packing | `problems/3_packing_cutting/knapsack/` |

## Existing implementations
- `warehouse_location.py` — Warehouse location as facility location
- `retail_inventory.py` — Retail inventory management
- `finance_portfolio.py` — Portfolio optimization (assortment analogy)

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations
