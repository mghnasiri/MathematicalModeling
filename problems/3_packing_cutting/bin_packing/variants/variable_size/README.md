# Variable-Size Bin Packing (VS-BPP)

## Problem Definition

Pack n items into bins of K different types (with capacities C_k and costs c_k) to minimize total bin cost.

## Complexity

NP-hard (generalizes standard BPP).

## Applications

- **Cloud computing**: selecting VM instance types
- **Container shipping**: choosing container sizes
- **Memory allocation**: page size selection
- **Fleet selection**: choosing vehicle types for deliveries

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| FFD Best-Type | Heuristic | Friesen & Langston (1986) |
| Cost-Ratio Greedy | Heuristic | Friesen & Langston (1986) |
| Simulated Annealing | Metaheuristic | — |
