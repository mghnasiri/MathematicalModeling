# VRP with Backhauls (VRPB) (CVRP Variant)

## What Changes

In standard CVRP, all customers receive deliveries. VRPB introduces a second customer type -- backhaul customers -- who require pickups (goods collected and returned to the depot). This models industries where outbound delivery vehicles can utilize their return trip to collect goods, avoiding empty backhauls. The key structural constraint is that all linehaul (delivery) customers on a route must be served before any backhaul (pickup) customers, because loading picked-up goods on top of undelivered cargo causes interference and operational difficulty.

Compared to the base CVRP:
- Customers are partitioned into linehaul set L and backhaul set B.
- A precedence constraint enforces linehaul-before-backhaul ordering within each route.
- Separate capacity constraints apply to deliveries (sum of linehaul demands <= Q) and pickups (sum of backhaul demands <= Q) per route.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Customer partition:** N = L u B, where L = linehaul customers, B = backhaul customers.
- **Precedence constraint:** For each route, if customer i in L and customer j in B appear on the same route, then i must precede j in the visit sequence.
- **Dual capacity:** sum_{i in L on route k} d_i <= Q and sum_{j in B on route k} d_j <= Q for all routes k.

The objective remains minimizing total travel distance across all routes.
Note that Q may be the same for deliveries and pickups, or different depending on the vehicle design.

## Complexity

NP-hard (generalizes CVRP, since setting B = empty set recovers standard CVRP). The precedence constraint further restricts the feasible solution space, making the problem structurally different from mixed pickup-and-delivery variants.

## Applications

- **Grocery distribution**: deliver goods to retail stores, collect expired products and reusable crates on the return leg
- **Beverage logistics**: deliver full bottles and kegs, pick up empties for recycling or refill
- **Postal service**: deliver parcels and mail, collect outgoing mail from post offices
- **Manufacturing supply chains**: deliver raw materials, collect finished goods or empty pallets
- **Library services**: deliver requested books, collect returned books

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest Neighbor (phased) | Heuristic | Yes | Serves nearest linehaul first, then nearest backhaul |
| Cluster-first, route-second | Heuristic | Yes | Groups linehaul+backhaul into routes, then sequences |
| Simulated Annealing | Metaheuristic | Yes | Relocate/swap moves preserving linehaul-first ordering |

Standard CVRP heuristics (Clarke-Wright, sweep) require modification to enforce the precedence constraint. The phased nearest-neighbor approach naturally respects this by completing all deliveries before starting pickups on each route.

Note that the VRPB differs from the VRP with Simultaneous Pickup and Delivery (VRPSPD), where pickups and deliveries can be interleaved. The strict precedence constraint in VRPB simplifies feasibility checking but may increase total distance compared to VRPSPD solutions. It also differs from the VRP with Mixed Backhauls, which allows arbitrary interleaving of linehaul and backhaul customers.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `VRPBInstance` dataclass with linehaul/backhaul customer sets and capacity validation |
| `heuristics.py` | Nearest neighbor with linehaul-first constraint; cluster-first route-second |
| `metaheuristics.py` | Simulated Annealing with precedence-preserving relocate/swap neighborhoods |

## Key References

- Goetschalckx, M. & Jacobs-Blecha, C. (1989). The vehicle routing problem with backhauls. *European Journal of Operational Research*, 42(1), 39-51. https://doi.org/10.1016/0377-2217(89)90057-X
- Toth, P. & Vigo, D. (1999). A heuristic algorithm for the symmetric and asymmetric vehicle routing problem with backhauls. *European Journal of Operational Research*, 113(3), 528-543. https://doi.org/10.1016/S0377-2217(98)00012-6
- Mingozzi, A., Giorgi, S. & Baldacci, R. (1999). An exact method for the vehicle routing problem with backhauls. *Transportation Science*, 33(3), 315-329. [TODO: verify DOI]
- Toth, P. & Vigo, D. (2002). *The Vehicle Routing Problem*. SIAM Monographs on Discrete Mathematics and Applications. [TODO: verify DOI]
