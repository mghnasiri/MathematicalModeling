# Multi-Depot Vehicle Routing Problem (MDVRP) (CVRP Variant)

## What Changes

Standard CVRP assumes a single depot from which all vehicles depart and return. The MDVRP introduces multiple depot locations, each with its own vehicle fleet. Each customer must be assigned to exactly one depot and served by a vehicle from that depot. This models distribution networks with multiple warehouses, regional distribution centers serving overlapping territories, or municipal services operating from several garages.

The problem involves two interdependent decisions: (1) assigning customers to depots, and (2) building capacity-feasible routes within each depot's assigned customer set. These two decisions interact -- a customer assignment that minimizes depot-customer distances may produce poor routing, and vice versa.

Compared to the base CVRP:
- Multiple depot nodes D = {d_1, ..., d_p} replace the single depot.
- Each depot has its own fleet of vehicles (possibly with different fleet sizes).
- Each customer is assigned to exactly one depot.
- Routes start and end at the same depot.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Depot set:** D = {d_1, ..., d_p}, each with fleet size m_j.
- **Customer assignment:** Each customer i is assigned to exactly one depot: sum_{j=1}^{p} y_{ij} = 1 for all customers i, where y_{ij} = 1 if customer i is assigned to depot j.
- **Route origin:** Each route from depot d_j starts at d_j and returns to d_j.
- **Fleet constraint:** The number of routes originating at depot d_j does not exceed m_j.

The objective is to minimize total travel distance across all depots and routes.

## Complexity

NP-hard (generalizes CVRP, since p = 1 recovers single-depot CVRP). The customer-to-depot assignment adds a combinatorial layer on top of the routing problem. Even with fixed assignments, each depot faces an independent CVRP subproblem.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest-Depot NN | Heuristic | Yes | Assigns each customer to nearest depot, then NN per depot |
| Simulated Annealing | Metaheuristic | Yes | Relocate between routes/depots, swap, intra-route 2-opt |

A common decomposition strategy is to first assign customers to depots (e.g., nearest depot, or Voronoi partition), then solve independent CVRP subproblems per depot. The SA implementation allows inter-depot customer transfers to escape poor initial assignments.

## Applications

- **Multi-warehouse distribution**: regional distribution centers serving overlapping customer territories
- **Municipal services**: trash collection, snow plowing, or road maintenance operating from multiple garages
- **Parcel delivery**: hub-and-spoke networks where multiple hubs each serve local delivery routes
- **Emergency response**: fire stations or ambulance bases dispatching vehicles to incidents

The quality of the initial customer-to-depot assignment strongly affects overall solution quality. Nearest-depot assignment is fast but may produce imbalanced workloads across depots.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `MDVRPInstance` dataclass with multiple depot coordinates, per-depot fleets, and customer assignment |
| `heuristics.py` | Nearest-depot assignment followed by nearest-neighbor routing per depot; O(n^2 * D) |
| `metaheuristics.py` | Simulated Annealing with inter-depot relocate, swap, and intra-route 2-opt neighborhoods |

## Key References

- Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search heuristic for periodic and multi-depot vehicle routing problems. *Networks*, 30(2), 105-119. https://doi.org/10.1002/(SICI)1097-0037(199709)30:2<105::AID-NET5>3.0.CO;2-G
- Renaud, J., Laporte, G. & Boctor, F.F. (1996). A tabu search heuristic for the multi-depot vehicle routing problem. *Computers & Operations Research*, 23(3), 229-235. [TODO: verify DOI]
- Montoya-Torres, J.R., Franco, J.L., Isaza, S.N., Jimenez, H.F. & Herazo-Padilla, N. (2015). A literature review on the vehicle routing problem with multiple depots. *Computers & Industrial Engineering*, 79, 115-129. [TODO: verify DOI]
