# Problem Taxonomy — Full Classification

> Complete classification of all OR problems implemented in this repository, with complexity classes, key algorithms, and links to interactive application pages.

---

## Family 1 · Scheduling

**Reference page:** [`families/scheduling.html`](families/scheduling.html)

### 1. Single Machine Scheduling

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total completion | 1 \|\| ΣCj | P | SPT |
| Weighted completion | 1 \|\| ΣwjCj | P | WSPT |
| Maximum lateness | 1 \|\| Lmax | P | EDD |
| Number of tardy jobs | 1 \|\| ΣUj | P | Moore's |
| Total tardiness | 1 \|\| ΣTj | NP-hard | DP |
| Weighted tardiness | 1 \|\| ΣwjTj | NP-hard (strongly) | B&B, SA |

### 2. Parallel Machine Scheduling

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Identical, makespan | Pm \|\| Cmax | NP-hard | LPT (4/3) |
| Uniform machines | Qm \|\| Cmax | NP-hard | |
| Unrelated machines | Rm \|\| Cmax | NP-hard (strongly) | LP rounding |

### 3. Flow Shop Scheduling

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 2-machine makespan | F2 \|\| Cmax | P | Johnson's rule |
| m-machine makespan | Fm \| prmu \| Cmax | NP-hard (m >= 3) | NEH, IG |
| Blocking flow shop | Fm \| block \| Cmax | NP-hard (m >= 3) | |
| No-wait flow shop | Fm \| no-wait \| Cmax | NP-hard (m >= 3) | TSP reduction |

### 4. Job Shop Scheduling

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Makespan | Jm \|\| Cmax | NP-hard (m >= 2) | Tabu search, CP |

**Applications:** [Job Shop Scheduling](applications/job-shop-scheduling.html), [Steel Production](applications/steel-production.html), [Print Shop](applications/print-shop.html)

### 5. Flexible Job Shop Scheduling

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Total flexibility | T-FJSP | NP-hard | Integrated TS, GA |
| Partial flexibility | P-FJSP | NP-hard | Hierarchical TS |

**Applications:** [Flexible Manufacturing](applications/flexible-manufacturing.html)

### 6. Resource-Constrained Project Scheduling (RCPSP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Basic | RCPSP | NP-hard (strongly) | GA + SGS |

**Applications:** [Construction Project Scheduling](applications/construction-project-scheduling.html), [JWST Scheduling](applications/jwst-scheduling.html), [Layered Defense Scheduling](applications/layered-defense-scheduling.html)

---

## Family 2 · Routing

**Reference page:** [`families/routing.html`](families/routing.html)

### 7. Traveling Salesman Problem (TSP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Symmetric TSP | TSP | NP-hard | Held-Karp DP, B&B |
| Asymmetric TSP | ATSP | NP-hard | B&B |

**Applications:** [Gravity-Assist Sequence](applications/gravity-assist-sequence.html), [Delivery Routing](applications/delivery-routing.html)

### 8. Capacitated Vehicle Routing (CVRP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| CVRP | CVRP | NP-hard | Clarke-Wright, Sweep |

**Applications:** [Crop Transport](applications/crop-transport.html), [Medical Supply](applications/medical-supply.html), [Garbage Collection](applications/garbage-collection.html)

### 9. VRP with Time Windows (VRPTW)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| VRPTW | VRPTW | NP-hard | Solomon I1, SA, GA |

**Applications:** [Home Visits](applications/home-visits.html), [Fertilizer Routing](applications/fertilizer-routing.html), [Technician Routing](applications/technician-routing.html)

### 10. Arc Routing (CARP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| CARP | CARP | NP-hard | Route-first cluster-second |

---

## Family 3 · Packing & Cutting

**Reference page:** [`families/packing.html`](families/packing.html)

### 11. 0-1 Knapsack

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 0-1 Knapsack | KP01 | NP-hard (weakly) | DP O(nW), B&B |

**Applications:** [Construction Portfolio Selection](applications/construction-portfolio-selection.html), [Budget Allocation](applications/budget-allocation.html)

### 12. 1D Bin Packing

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 1D Bin Packing | BPP1D | NP-hard (strongly) | FFD (11/9 approx) |

**Applications:** [Cargo Loading](applications/cargo-loading.html), [Silo Packing](applications/silo-packing.html), [Warehouse Packing](applications/warehouse-packing.html)

### 13. 1D Cutting Stock

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| 1D Cutting Stock | CSP1D | NP-hard | Column generation |

**Applications:** [Cutting Stock](applications/cutting-stock.html)

---

## Family 4 · Assignment & Matching

**Reference page:** [`families/assignment.html`](families/assignment.html)

### 14. Linear Assignment (LAP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Square LAP | LAP | P | Hungarian O(n^3) |

**Applications:** [Nurse Rostering](applications/nurse-rostering.html), [Field Assignment](applications/field-assignment.html), [Workforce Assignment](applications/workforce-assignment.html), [Construction Crew Assignment](applications/construction-crew-assignment.html)

### 15. Quadratic Assignment (QAP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| QAP | QAP | NP-hard | SA, TS |

**Applications:** [Warehouse Slotting](applications/warehouse-slotting.html)

### 16. Generalized Assignment (GAP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| GAP | GAP | NP-hard | Greedy, LP relaxation |

**Applications:** [Swarm Task Allocation](applications/swarm-task-allocation.html), [Gate Assignment](applications/gate-assignment.html)

---

## Family 5 · Location & Covering

**Reference page:** [`families/location.html`](families/location.html)

### 17. Facility Location

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Uncapacitated | UFLP | NP-hard | Greedy add/drop, SA |
| Capacitated | CFLP | NP-hard | — |

**Applications:** [Plant Location](applications/plant-location.html), [DC Location](applications/dc-location.html), [Energy Plant Siting](applications/energy-plant-siting.html), [Construction Site Selection](applications/construction-site-selection.html)

### 18. p-Median / p-Center

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| p-Median | PMP | NP-hard | Greedy, Teitz-Bart |

**Applications:** [School Location](applications/school-location.html), [Fire Station Siting](applications/fire-station-siting.html), [Ambulance Placement](applications/ambulance-placement.html)

### 19. Maximal Covering Location (MCLP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| MCLP | MCLP | NP-hard | Greedy, ILP |

**Applications:** [Radar Network Placement](applications/radar-network-placement.html), [Store Location](applications/store-location.html)

### 20. Set Covering / Set Partitioning

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Set Covering | SCP | NP-hard | Greedy (ln n approx) |

**Applications:** [Ecology — Reserve Selection](applications/ecology-conservation.html), [Police Patrol](applications/police-patrol.html)

### 21. Set Packing

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Set Packing | SPP | NP-hard | Greedy |

---

## Family 6 · Network Flow & Design

**Reference page:** [`families/network.html`](families/network.html)

### 22. Shortest Path

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Non-negative SSSP | SPP | P | Dijkstra O((V+E) log V) |
| General SSSP | SPP | P | Bellman-Ford O(VE) |

**Applications:** [Emergency Routing](applications/emergency-routing.html), [Evacuation Routing](applications/evacuation-routing.html)

### 23. Maximum Flow / Min Cut

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Max Flow / Min Cut | Max-Flow | P | Edmonds-Karp O(VE²) |

**Applications:** [Patient Flow](applications/patient-flow.html), [Energy Smart Grid Routing](applications/energy-smart-grid-routing.html)

### 24. Minimum Spanning Tree

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| MST | MST | P | Kruskal O(E log E), Prim O(E log V) |

**Applications:** [Irrigation Network](applications/irrigation-network.html), [Energy Transmission Network](applications/energy-transmission-network.html)

### 25. Network Interdiction

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Max-Flow Interdiction | NIP | NP-hard | Bilevel LP, Enumeration |

**Applications:** [Supply Line Interdiction](applications/supply-line-interdiction.html)

---

## Family 7 · Inventory & Lot Sizing

**Reference page:** [`families/inventory.html`](families/inventory.html)

### 26. EOQ Models

Classical economic order quantity with extensions.

### 27. Lot Sizing

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Uncapacitated | ULSP | P | Wagner-Whitin DP |

**Applications:** [Lot Sizing](applications/lot-sizing.html), [Seed Inventory](applications/seed-inventory.html)

### 28. Capacitated Lot Sizing

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| CLSP | CLSP | NP-hard | MIP, Lagrangian relaxation |

### 29. Multi-Echelon Inventory

Multi-level inventory optimization across supply chain tiers.

### 30. Newsvendor / Stochastic Inventory

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Newsvendor | — | P | Closed-form F⁻¹(cᵤ/(cᵤ+cₒ)) |

**Applications:** [Resource Stockpiling](applications/resource-stockpiling.html), [Grocery Ordering](applications/grocery-ordering.html)

---

## Family 8 · Integrated Structural

**Reference page:** [`families/integrated.html`](families/integrated.html)

### 31. Location-Routing Problem (LRP)

Combined facility location and vehicle routing.

**Applications:** [Distribution Center](applications/distribution-center.html), [Depot Location](applications/depot-location.html)

### 32. Inventory-Routing Problem (IRP)

Combined inventory replenishment and delivery routing.

### 33. Assembly Line Balancing (SALBP)

Combined task assignment and scheduling with cycle time constraints.

**Applications:** [Assembly Line Balancing](applications/assembly-line-balancing.html), [Manufacturing Balancing](applications/manufacturing-balancing.html)

---

## Family 9 · Uncertainty Modeling

**Reference page:** [`families/uncertainty.html`](families/uncertainty.html)

Paradigms for optimization under uncertainty, applied across all problem families.

| Paradigm | Key Idea | Algorithm |
|----------|----------|-----------|
| Stochastic Programming | Expected value over scenarios | Two-Stage SP, SAA |
| Robust Optimization | Worst-case over uncertainty set | Min-max, Min-max regret |
| Chance-Constrained | Probability of constraint satisfaction | Scenario-based, reformulation |
| DRO | Worst-case over distribution family | Wasserstein ball |

**Applications:** [Market Timing](applications/market-timing.html), [Portfolio Optimization](applications/portfolio.html), [Resource Stockpiling](applications/resource-stockpiling.html)

---

## Family 10 · Game Theory & Adversarial Optimization

### 34. Weapon-Target Assignment (WTA)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Static WTA | WTA | NP-complete | Greedy, MMR, B&B |

**Applications:** [Interceptor Assignment](applications/interceptor-assignment.html)

### 35. Stackelberg Security Games

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| SSE (Strong Stackelberg) | SSG | P (for single follower) | LP, ERASER |

**Applications:** [Patrol Route Optimization](applications/patrol-route-optimization.html), [Perimeter Defense Game](applications/perimeter-defense-game.html)

### 36. Tri-Level Defender-Attacker-Defender

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| DAD | DAD | NP-hard | Benders decomposition |

**Applications:** [Infrastructure Fortification](applications/infrastructure-fortification.html)

---

## Family 11 · Sequential Decision Making

### 37. Markov Decision Processes (MDP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Finite MDP | MDP | P | Value iteration, Policy iteration |

### 38. Approximate Dynamic Programming (ADP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| ADP | ADP | — | Rollout, value function approximation |

**Applications:** [Dynamic Defense Allocation](applications/dynamic-defense-allocation.html)

### 39. Partially Observable MDP (POMDP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| Finite POMDP | POMDP | PSPACE-hard | Belief update, PBVI |

**Applications:** [Reconnaissance Planning](applications/reconnaissance-planning.html), [Space Debris CAM](applications/space-debris-cam.html)

---

## Family 12 · Multi-Agent Planning

### 40. Multi-Agent Path Planning (MAPP)

| Variant | Notation | Complexity | Key Algorithm |
|---------|----------|------------|--------------|
| MAPP | MAPF | NP-hard | CBS, Reynolds flocking |

**Applications:** [Drone Swarm Coordination](applications/drone-swarm-coordination.html)

---

## Cross-Reference: OR Problem → Application Pages

| OR Problem | Application Pages |
|-----------|-------------------|
| RCPSP | Construction Scheduling, JWST, Layered Defense |
| TSP | Gravity-Assist, Delivery Routing |
| CVRP | Crop Transport, Medical Supply, Garbage Collection |
| VRPTW | Home Visits, Fertilizer Routing, Technician Routing |
| Facility Location | Plant Location, DC Location, Energy Plant Siting |
| p-Median | School Location, Fire Station, Ambulance Placement |
| MCLP | Radar Network, Store Location |
| Assignment | Nurse Rostering, Field Assignment, Crew Assignment |
| Bin Packing | Cargo Loading, Silo Packing, Warehouse Packing |
| Network Interdiction | Supply Line Interdiction |
| WTA | Interceptor Assignment |
| Security Games | Patrol Routes, Perimeter Defense |
| POMDP | Reconnaissance, Space Debris |
| Newsvendor | Resource Stockpiling, Grocery Ordering |
| Max Flow | Patient Flow, Smart Grid Routing |
| MST | Irrigation Network, Transmission Network |
| Job Shop | Steel Production, Print Shop |
| Lot Sizing | Lot Sizing, Seed Inventory |
