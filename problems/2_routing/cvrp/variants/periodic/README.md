# Periodic Vehicle Routing Problem (PVRP) (CVRP Variant)

## What Changes

Standard CVRP plans routes for a single day. The PVRP extends this to a multi-day planning horizon of T periods (e.g., a work week of 5 days). Each customer requires a specified number of visits over the horizon, and the planner must decide both which days to visit each customer and how to build capacity-feasible routes for each day. This arises in recurring service operations such as waste collection (certain areas need pickup every 2 days), retail store replenishment (stores needing 2-3 deliveries per week), vending machine servicing, and industrial gas delivery.

The PVRP couples a combinatorial day-selection problem with a routing problem for each day, making it substantially harder than single-day CVRP.

Compared to the base CVRP:
- A planning horizon of T periods replaces the single-period model.
- Each customer i has a visit frequency f_i (number of required visits over T periods).
- A set of allowable visit-day combinations C_i is defined for each customer.
- Routes must be built for each day independently, all respecting vehicle capacity.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Day selection:** For each customer i, select one visit-day combination c in C_i, where C_i contains all valid subsets of {1, ..., T} with |c| = f_i.
- **Daily routing:** For each day t in {1, ..., T}, the set of customers scheduled for day t forms a CVRP subproblem with capacity Q.
- **Frequency satisfaction:** Each customer i is visited exactly f_i times over the horizon.
- **Workload balancing (optional):** The total route distance or number of customers per day may be balanced across the horizon.

The objective is to minimize total travel distance summed over all days.

## Complexity

NP-hard (generalizes CVRP, since T = 1 with all f_i = 1 recovers single-day CVRP). The day-combination selection introduces an additional combinatorial dimension that grows exponentially with T. Even with fixed day assignments, each day is an independent CVRP subproblem.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Spread-then-Route | Heuristic | Yes | Spread visits evenly across days, then NN per day |
| Simulated Annealing | Metaheuristic | Yes | Day-swap, intra-day relocate/2-opt neighborhoods |

The spread-then-route heuristic first assigns each customer's visits to days (spreading them as evenly as possible), then solves each day's routing problem independently using nearest neighbor. The SA can modify both day assignments and within-day routing.

## Applications

- **Waste collection**: residential areas requiring pickup 2-3 times per week on specific day patterns
- **Retail store replenishment**: stores needing regular deliveries (e.g., convenience stores restocked every 2 days)
- **Vending machine servicing**: machines needing refill based on consumption frequency
- **Industrial gas delivery**: factories receiving gas deliveries on a fixed weekly schedule

The interaction between day selection and routing quality is the defining challenge of PVRP. Assigning geographically clustered customers to the same days produces better daily routes, but may violate frequency or workload balancing requirements.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `PVRPInstance` dataclass with planning horizon, visit frequencies, day-combination sets, and validation |
| `heuristics.py` | Spread-then-route: even day assignment followed by nearest-neighbor routing per day |
| `metaheuristics.py` | Simulated Annealing with day-swap moves and intra-day relocate/2-opt improvement |

## Key References

- Christofides, N. & Beasley, J.E. (1984). The period routing problem. *Networks*, 14(2), 237-256. https://doi.org/10.1002/net.3230140205
- Cordeau, J.-F., Gendreau, M. & Laporte, G. (1997). A tabu search heuristic for periodic and multi-depot vehicle routing problems. *Networks*, 30(2), 105-119. https://doi.org/10.1002/(SICI)1097-0037(199709)30:2<105::AID-NET5>3.0.CO;2-G
- Vidal, T., Crainic, T.G., Gendreau, M., Lahrichi, N. & Rei, W. (2012). A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. *Operations Research*, 60(3), 611-624. [TODO: verify DOI]
