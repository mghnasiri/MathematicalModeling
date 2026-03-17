# VRP with Backhauls (VRPB)

## Problem Definition

Customers split into linehaul (delivery) and backhaul (pickup). Each route must serve all linehaul before any backhaul customers. Separate capacity constraints for deliveries and pickups.

## Complexity

NP-hard (generalizes CVRP).

## Applications

- **Grocery distribution**: deliver goods, collect returns
- **Beverage delivery**: deliver full bottles, collect empties
- **Postal service**: deliver parcels, collect outgoing mail

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Nearest Neighbor VRPB | Heuristic | Goetschalckx & Jacobs-Blecha (1989) |
| Simulated Annealing | Metaheuristic | Toth & Vigo (1999) |
