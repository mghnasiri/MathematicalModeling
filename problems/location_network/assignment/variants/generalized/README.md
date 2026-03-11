# Generalized Assignment Problem (GAP)

## Problem Definition

Assign n jobs to m agents. Each agent i has capacity b_i. Assigning job j to agent i consumes a_ij resources and yields cost c_ij. Minimize total cost subject to capacity constraints.

**Objective:** min Σ_j c_{a(j),j} where a(j) is the agent assigned to job j.

**Constraints:**
- Each job assigned to exactly one agent
- Σ_j a_{ij} · x_{ij} ≤ b_i for each agent i (capacity)

## Complexity

Strongly NP-hard. Even feasibility checking is NP-complete.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Greedy ratio | Heuristic | Martello & Toth (1990) |
| First-fit decreasing | Heuristic | Martello & Toth (1990) |
| Simulated Annealing | Metaheuristic | Osman (1995) |
