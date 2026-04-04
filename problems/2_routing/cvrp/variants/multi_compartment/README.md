# Multi-Compartment VRP (MCVRP) (CVRP Variant)

## What Changes

In standard CVRP, all goods are treated as a single product sharing one cargo space. The MCVRP models vehicles equipped with multiple physically separated compartments, each dedicated to a specific product type with its own capacity. Customers demand specific product types that must be loaded into and delivered from the correct compartment. This arises in fuel distribution (gasoline, diesel, and premium in separate tanker compartments), food logistics (frozen, chilled, and ambient goods requiring different temperature zones), and waste collection (sorted recyclables).

Compared to the base CVRP:
- Each vehicle has K compartments with individual capacities Q_1, Q_2, ..., Q_K.
- Each customer demands specific quantities of one or more product types.
- The single aggregate capacity constraint is replaced by K per-compartment constraints.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Compartment capacity:** For each route r and compartment k in {1, ..., K}: sum_{i in route_r} d_{i,k} <= Q_k, where d_{i,k} is the demand of customer i for product type k.
- **Product assignment:** Each customer's demand for product type k must be served from compartment k. Cross-compartment substitution is not allowed.
- **Aggregate capacity:** The total load across all compartments must also fit in the vehicle: sum_k load_k <= Q_total (if an overall weight/volume limit applies).

The objective remains minimizing total travel distance.

## Complexity

NP-hard (generalizes CVRP, since K = 1 with a single compartment recovers standard CVRP). The per-compartment capacity constraints tighten feasibility, potentially requiring more vehicles than a single-compartment formulation with the same total capacity.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest Neighbor | Heuristic | Yes | Compartment capacity checks at each insertion |
| Clarke-Wright Savings | Heuristic | Yes | Route merging with per-compartment feasibility |
| Simulated Annealing | Metaheuristic | Yes | Relocate and swap moves with multi-compartment validation |

Standard CVRP heuristics adapt naturally by replacing the single capacity check with K per-compartment checks. The savings heuristic merges two routes only if the combined load is feasible in every compartment.

## Applications

- **Fuel distribution**: tanker trucks with separate compartments for gasoline, diesel, and premium
- **Food logistics**: refrigerated vehicles with frozen, chilled, and ambient temperature zones
- **Waste collection**: trucks with separated bins for paper, glass, plastic, and organic waste
- **Chemical distribution**: incompatible chemicals requiring physical separation during transport

The key modeling distinction from standard CVRP is that two customers with small total demand may still be infeasible on the same route if they overload a single compartment, even when the overall vehicle has spare capacity. This makes route feasibility checking more expensive than in standard CVRP, as all K compartment capacities must be verified.

Compartment sizes may be fixed or configurable. In the fixed case, compartment capacities are given; in the configurable case, the total vehicle volume can be partitioned among compartments, adding another decision layer.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `MCVRPInstance` dataclass with per-compartment capacities, product-type demands, and validation |
| `heuristics.py` | Nearest neighbor with compartment checks; Clarke-Wright savings with multi-compartment feasibility |
| `metaheuristics.py` | Simulated Annealing with relocate and swap moves under per-compartment capacity constraints |

## Key References

- Derigs, U., Gottlieb, J. & Kalkoff, J. (2011). Vehicle routing with compartments: Applications, modelling and heuristics. *OR Spectrum*, 33(4), 885-914. https://doi.org/10.1007/s00291-009-0175-6
- Chajakis, E.D. & Guignard, M. (2003). Scheduling deliveries in vehicles with multiple compartments. *Journal of Global Optimization*, 26(1), 43-78. https://doi.org/10.1023/A:1023067016014
- Coelho, L.C. & Laporte, G. (2015). Classification, models and exact algorithms for multi-compartment delivery problems. *European Journal of Operational Research*, 242(3), 854-864. [TODO: verify DOI]
- Lahyani, R., Coelho, L.C., Khemakhem, M., Laporte, G. & Semet, F. (2015). A multi-compartment vehicle routing problem arising in the collection of olive oil in Tunisia. *Omega*, 51, 1-10. [TODO: verify DOI]
