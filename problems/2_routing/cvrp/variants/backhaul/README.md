# VRP with Backhauls (VRPB)

## Problem Definition

Extends CVRP with two customer types: linehaul (delivery) and backhaul (pickup). All linehaul customers on a route must be served before any backhaul customers to avoid cargo interference.

**Objective:** Minimize total travel distance.

**Constraints:**
- Each customer visited exactly once
- Linehaul customers served before backhaul customers on each route
- Separate capacity constraints for deliveries and pickups

## Complexity

NP-hard (generalizes CVRP).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor (phased) | Heuristic | Toth & Vigo (1999) |
| Simulated Annealing | Metaheuristic | Toth & Vigo (1999) |
