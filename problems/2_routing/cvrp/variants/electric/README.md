# Electric Vehicle Routing Problem (EVRP) (CVRP Variant)

## What Changes

The EVRP extends CVRP to model fleets of battery electric vehicles (BEVs) with limited driving range. Vehicles consume energy proportional to distance traveled, and their battery may not last an entire route. Charging stations are placed at known locations in the network, and vehicles can detour to recharge during a route. Each charging station visit fully replenishes the battery. This reflects the operational reality of last-mile electric delivery fleets, where range anxiety and charging infrastructure planning are critical.

Compared to the base CVRP:
- A battery capacity B and energy consumption rate (energy per unit distance) are added.
- A set of charging station nodes S is introduced (distinct from customers and depot).
- Routes may include detours to charging stations; the battery level must remain non-negative at all times.
- Charging stations may be visited multiple times across different routes.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Energy constraint:** For each arc (i, j) traversed, battery decreases by e * d(i,j), where e is energy consumption per unit distance. Battery level must satisfy b_j = b_i - e * d(i,j) >= 0 at every node j.
- **Recharging:** When visiting a charging station s in S, the battery is reset to B (full capacity).
- **Station visits:** Charging stations are optional nodes that may appear zero or more times across all routes. They do not have demands.
- **Customer constraints:** Each customer still visited exactly once; capacity Q still respected.

The objective remains minimizing total travel distance (including detours to stations).

## Complexity

NP-hard (generalizes CVRP, since removing all energy constraints and stations recovers standard CVRP). The added energy dimension and station insertion decisions increase the combinatorial complexity significantly.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest Neighbor (energy-aware) | Heuristic | Yes | Inserts station visits when battery insufficient for next leg |
| Greedy savings (EVRP) | Heuristic | Yes | Clarke-Wright adapted with energy feasibility checks |
| Simulated Annealing | Metaheuristic | Yes | Relocate, swap, station insertion/removal neighborhoods |

A key challenge in EVRP heuristics is deciding when and where to insert charging station detours. The energy-aware nearest neighbor inserts the nearest reachable station whenever the battery would be insufficient to reach the next customer or return to depot.

## Applications

- **Last-mile electric delivery**: parcel delivery fleets (e.g., urban logistics) with BEV vans
- **Electric utility vehicles**: municipal service fleets transitioning from diesel to electric
- **Autonomous electric shuttles**: on-demand transportation with range constraints
- **Green logistics**: carriers seeking to minimize carbon footprint with zero-emission vehicles

Extensions include partial recharging (battery not fully replenished), heterogeneous charging rates at different stations, and battery degradation models.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `EVRPInstance` dataclass with battery capacity, energy rate, charging station locations, and energy validation |
| `heuristics.py` | Nearest neighbor with energy-aware station insertion; greedy savings adapted for EVRP |
| `metaheuristics.py` | Simulated Annealing with relocate, swap, and station insertion/removal moves |

## Key References

- Erdogan, S. & Miller-Hooks, E. (2012). A green vehicle routing problem. *Transportation Research Part E*, 48(1), 100-114. https://doi.org/10.1016/j.tre.2011.08.001
- Schneider, M., Stenger, A. & Goeke, D. (2014). The electric vehicle routing problem with time windows and recharging stations. *Transportation Science*, 48(4), 500-520. https://doi.org/10.1287/trsc.2013.0490
- Pelletier, S., Jabali, O. & Laporte, G. (2016). Goods distribution with electric vehicles: Review and research perspectives. *Transportation Science*, 50(1), 3-22. [TODO: verify DOI]
- Hiermann, G., Puchinger, J., Ropke, S. & Hartl, R.F. (2016). The electric fleet size and mix vehicle routing problem with time windows and recharging stations. *European Journal of Operational Research*, 252(3), 995-1018. [TODO: verify DOI]
