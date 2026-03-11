# Multi-Compartment VRP (MCVRP)

## Problem Definition

Vehicles have multiple compartments, each with its own capacity for a specific product type. Customers demand specific product types that must be delivered from their designated compartment. Minimize total travel distance while respecting per-compartment capacity constraints.

## Complexity

NP-hard (generalizes CVRP).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor | Heuristic | Derigs, Gottlieb & Kalkoff (2011) |
| Clarke-Wright Savings | Heuristic | Derigs, Gottlieb & Kalkoff (2011) |
| Simulated Annealing | Metaheuristic | Derigs, Gottlieb & Kalkoff (2011) |
