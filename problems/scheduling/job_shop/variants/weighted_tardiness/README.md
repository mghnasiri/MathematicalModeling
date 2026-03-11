# Job Shop with Weighted Tardiness (Jm || ΣwjTj)

## Problem Definition

JSP with due dates d_j and weights w_j. Minimize total weighted tardiness ΣwjTj where Tj = max(0, Cj - dj).

## Complexity

Strongly NP-hard (even 1||ΣwjTj is strongly NP-hard).

## Applications

- **Make-to-order manufacturing**: customer orders with deadlines and priorities
- **Semiconductor fab**: wafer lots with varying urgency
- **Batch chemical processing**: product batches with delivery commitments

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| ATC Dispatching | Heuristic | Vepsalainen & Morton (1987) |
| WSPT Dispatching | Heuristic | — |
| Simulated Annealing | Metaheuristic | Singer & Pinedo (1998) |
