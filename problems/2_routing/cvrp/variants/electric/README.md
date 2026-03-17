# Electric Vehicle Routing Problem (EVRP)

## Problem Definition

Extends CVRP with battery-constrained electric vehicles. Vehicles have limited battery range and must visit charging stations to recharge. Each charging station visit fully replenishes the battery.

**Objective:** Minimize total travel distance across all routes.

**Constraints:**
- Each customer visited exactly once
- Vehicle load capacity respected
- Battery level never drops below zero
- Vehicles may visit charging stations to recharge

## Complexity

NP-hard (generalizes CVRP).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor (energy-aware) | Heuristic | Erdogan & Miller-Hooks (2012) |
| Simulated Annealing | Metaheuristic | Schneider, Stenger & Goeke (2014) |
