# VRP with Backhauls — Alternate Formulation (VRPB) (CVRP Variant)

## What Changes

This is an alternate implementation of the VRP with Backhauls, focusing on the application-driven perspective. In grocery distribution, beverage delivery, and postal logistics, vehicles deliver goods on the outbound leg and collect returns, empties, or outgoing mail on the return leg. The linehaul-before-backhaul precedence constraint reflects practical loading constraints: picked-up items placed in the cargo hold would obstruct access to remaining deliveries.

Compared to the base CVRP:
- Customers are partitioned into linehaul (delivery) set L and backhaul (pickup) set B.
- A strict precedence constraint requires all linehaul customers to be visited before any backhaul customer on each route.
- Delivery and pickup loads are subject to separate capacity limits.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Customer partition:** N = L u B (disjoint linehaul and backhaul sets).
- **Precedence:** On each route r, position(i) < position(j) for all i in L_r, j in B_r.
- **Capacity:** sum_{i in L_r} d_i <= Q_delivery and sum_{j in B_r} d_j <= Q_pickup for each route r.

The variant here separates delivery capacity Q_delivery and pickup capacity Q_pickup, which may differ depending on vehicle configuration.

## Complexity

NP-hard (generalizes CVRP). The precedence constraint reduces the feasible neighborhood for local search, which can make metaheuristic convergence slower but limits the search to structurally feasible solutions.

Note that the VRPB is distinct from the VRP with Simultaneous Pickup and Delivery (VRPSPD), where linehaul and backhaul customers can be freely interleaved. The strict ordering in VRPB simplifies feasibility checking but generally produces longer routes than VRPSPD for the same customer set. It also differs from the VRP with Mixed Backhauls, which relaxes the precedence constraint.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest Neighbor VRPB | Heuristic | Yes | Greedy nearest with precedence enforcement |
| Cluster-first, route-second | Heuristic | Yes | Group customers into routes, then sequence |
| Simulated Annealing | Metaheuristic | Yes | Relocate/swap with linehaul-before-backhaul preservation |

## Applications

- **Grocery distribution**: deliver goods to stores, collect returns and packaging
- **Beverage delivery**: deliver full bottles/kegs, collect empties
- **Postal service**: deliver parcels, collect outgoing mail
- **Waste management**: deliver empty containers, collect full ones

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `VRPBInstance` dataclass with linehaul/backhaul sets, capacity parameters, and validation |
| `heuristics.py` | Cluster-first route-second; nearest neighbor with precedence constraint |
| `metaheuristics.py` | Simulated Annealing with precedence-preserving relocate/swap moves |

## Key References

- Goetschalckx, M. & Jacobs-Blecha, C. (1989). The vehicle routing problem with backhauls. *European Journal of Operational Research*, 42(1), 39-51. https://doi.org/10.1016/0377-2217(89)90057-X
- Toth, P. & Vigo, D. (1999). A heuristic algorithm for the symmetric and asymmetric vehicle routing problems with backhauls. *European Journal of Operational Research*, 113(3), 528-543. https://doi.org/10.1016/S0377-2217(98)00022-8
- Toth, P. & Vigo, D. (2002). *The Vehicle Routing Problem*. SIAM Monographs on Discrete Mathematics and Applications, Chapter 8. [TODO: verify DOI]
