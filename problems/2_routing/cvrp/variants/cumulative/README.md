# Cumulative VRP (CumVRP) (CVRP Variant)

## What Changes

Standard CVRP minimizes total travel distance (a carrier-centric objective). The Cumulative VRP instead minimizes the total arrival time at all customers -- the sum of arrival times across all routes. This is also known as the Minimum Latency VRP or the Multi-Vehicle Traveling Repairman Problem. The objective shift fundamentally changes solution structure: CumVRP favors serving nearby customers first on each route, while CVRP may prefer longer initial legs if they enable efficient subsequent visits.

This matters in time-critical applications such as disaster relief (minimizing total waiting time of affected populations), humanitarian logistics, and repair service routing where customer waiting time is the primary concern rather than fleet operating cost.

Compared to the base CVRP:
- The objective changes from min sum of route distances to min sum of customer arrival times.
- Capacity constraints remain the same.
- Customer visit uniqueness constraints remain the same.

## Mathematical Formulation

The base CVRP formulation (see parent README) is modified as follows:

- **Objective:** min sum_{k} sum_{i in route_k} a_i, where a_i is the arrival time at customer i (cumulative travel time from the depot to i along the route).
- **Arrival time:** a_i = sum of edge travel times from depot to customer i along route k, following the visit sequence.

All other constraints (capacity, visit-once, depot start/end) remain unchanged from base CVRP.

## Complexity

NP-hard (generalizes the Traveling Repairman Problem / Minimum Latency Problem, which is NP-hard even for a single vehicle). The cumulative objective makes the problem harder to approximate than standard CVRP because the contribution of each edge to the total cost depends on how many customers follow it.

## Applications

- **Disaster relief**: minimize total waiting time of affected populations for aid delivery
- **Humanitarian logistics**: food and medical supply distribution where delay costs lives
- **Repair/maintenance services**: technician routing where customer downtime is the primary cost
- **Home healthcare**: visiting patients where accumulated waiting time impacts care quality

The cumulative objective captures a fundamentally different notion of service quality than total distance. Two solutions with the same total distance can have vastly different total latencies depending on visit ordering.

## Solution Approaches

| Method | Type | Works? | Notes |
|--------|------|--------|-------|
| Nearest Neighbor (latency-aware) | Heuristic | Yes | Greedy nearest customer, good for cumulative objectives |
| Nearest-first insertion | Heuristic | Yes | Inserts customers minimizing arrival time increase |
| Simulated Annealing | Metaheuristic | Yes | Relocate/swap/2-opt optimizing total latency |

For CumVRP, nearest-neighbor performs relatively well as a constructive heuristic because the greedy strategy of visiting nearby customers first naturally aligns with minimizing cumulative arrival times.

## Implementations

| File | Description |
|------|-------------|
| `instance.py` | `CumVRPInstance` dataclass with latency-based objective computation |
| `heuristics.py` | Nearest neighbor (latency-aware); nearest-first insertion |
| `metaheuristics.py` | Simulated Annealing with relocate/swap/2-opt moves evaluated on total latency |

## Key References

- Ngueveu, S.U., Prins, C. & Wolfler Calvo, R. (2010). An effective memetic algorithm for the cumulative capacitated vehicle routing problem. *Computers & Operations Research*, 37(11), 1877-1885. https://doi.org/10.1016/j.cor.2009.06.014
- Kara, I., Kara, B.Y. & Yetis, M.K. (2008). Energy minimizing vehicle routing problem. *Combinatorial Optimization and Applications*, LNCS 5165, 62-71. [TODO: verify DOI]
- Ribeiro, G.M. & Laporte, G. (2012). An adaptive large neighborhood search heuristic for the cumulative capacitated vehicle routing problem. *Computers & Operations Research*, 39(3), 627-633. [TODO: verify DOI]
