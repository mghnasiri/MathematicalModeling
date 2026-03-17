# Retail & E-Commerce — Decision Chain

> **Status:** Structure in place — decision points being populated progressively.

## About this sector
Retail and e-commerce operations span store/warehouse location, inventory management, order fulfillment, and last-mile delivery. OR methods optimize the end-to-end supply chain from network design to customer delivery.

## Decision chain

| Phase | Decision point | OR problem | Status |
|---|---|---|---|
| 01 Network | Warehouse Location | UFLP/LRP | Live |
| 02 Inventory | Retail Inventory | Newsvendor/(s,S) | Live |

## Existing implementations
- `warehouse_location.py` — Warehouse location as facility location
- `retail_inventory.py` — Retail inventory management
- `finance_portfolio.py` — Portfolio optimization (assortment analogy)

## See also
- `TAXONOMY.md` — full taxonomy specification
- `problems/` — canonical problem implementations
