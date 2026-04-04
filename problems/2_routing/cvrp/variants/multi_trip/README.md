# Multi-Trip VRP (MTVRP) (CVRP Variant)

## What Changes

Standard CVRP assumes each vehicle makes exactly one trip (depot to customers to depot). The MTVRP relaxes this by allowing vehicles to return to the depot after completing a route and depart on a new trip. This is relevant when the vehicle fleet is small relative to the number of customers, or when vehicles are expensive assets that should be utilized fully within a planning horizon. Typical applications include urban delivery with small vans making multiple rounds per day and retail replenishment from a central warehouse.

Compared to the base CVRP:
- Each vehicle may perform multiple sequential trips within the planning period.
- The number of vehicles is explicitly limited (K_max), making fleet size a binding constraint.
- Each individual trip respects the vehicle capacity Q.
- An optional maximum route duration or working-day time limit may apply.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Trip assignment:** Each vehicle k in {1, ..., K_max} performs a sequence of trips T_{k,1}, T_{k,2}, ... Each trip is a depot-to-depot route.
- **Per-trip capacity:** sum_{i in T_{k,t}} d_i <= Q for each trip t of vehicle k.
- **Fleet limit:** At most K_max vehicles are used.
- **Coverage:** Each customer appears in exactly one trip of one vehicle.
- **Sequencing (optional):** If a time horizon [0, T_max] applies, the cumulative duration of all trips for vehicle k (including travel and service times) must not exceed T_max.

The objective is to minimize total travel distance across all trips of all vehicles.

## Complexity

NP-hard (generalizes CVRP, since K_max = n recovers standard CVRP where each vehicle makes at most one trip). The multi-trip dimension introduces a bin-packing subproblem: assigning trips to vehicles to satisfy the fleet limit, which itself is NP-hard.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Greedy NN + bin-packing | Heuristic | Yes | Build single-trip routes via NN, then pack trips into vehicles |
| Simulated Annealing | Metaheuristic | Yes | Trip reassignment and intra-trip relocate/swap/2-opt moves |

The two-phase heuristic first builds routes ignoring fleet limits (treating it as standard CVRP), then assigns routes to vehicles using a bin-packing approach. The SA improves both the routing and the trip-to-vehicle assignment simultaneously.

## Applications

- **Urban delivery with small vans**: limited fleet making multiple delivery rounds per day
- **Retail replenishment**: a single truck restocking multiple store groups from a central warehouse
- **Courier services**: drivers completing multiple collection/delivery circuits within a shift
- **Waste collection**: when trucks must return to the depot to empty before continuing

The MTVRP becomes particularly important when fleet acquisition costs are high, forcing operators to maximize utilization of each vehicle rather than adding vehicles to the fleet.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `MTVRPInstance` dataclass with fleet size limit, trip capacity, and multi-trip validation |
| `heuristics.py` | Greedy bin-packing of nearest-neighbor routes to a limited vehicle fleet |
| `metaheuristics.py` | Simulated Annealing with trip reassignment and intra-trip improvement moves |

## Key References

- Taillard, E.D., Laporte, G. & Gendreau, M. (1996). Vehicle routeing with multiple use of vehicles. *Journal of the Operational Research Society*, 47(8), 1065-1070. https://doi.org/10.1057/jors.1996.133
- Olivera, A. & Viera, O. (2007). Adaptive memory programming for the vehicle routing problem with multiple trips. *Computers & Operations Research*, 34(1), 28-47. https://doi.org/10.1016/j.cor.2005.02.044
- Cattaruzza, D., Absi, N., Feillet, D. & Gonzalez-Feliu, J. (2017). Vehicle routing problems for city logistics. *EURO Journal on Transportation and Logistics*, 6(1), 51-79. [TODO: verify DOI]
- Petch, R.J. & Salhi, S. (2004). A multi-phase constructive heuristic for the vehicle routing problem with multiple trips. *Discrete Applied Mathematics*, 133(1-3), 69-92. [TODO: verify DOI]
