# Flexible Job Shop with Tardiness (FJm || ΣwjTj)

## Problem Definition

Combines the flexibility of FJSP (machine assignment) with tardiness minimization. Each operation has a set of eligible machines with machine-dependent processing times. The objective is to minimize total weighted tardiness.

**Objective:** min ΣwjTj where Tj = max(0, Cj - dj)

**Constraints:**
- Operations within a job follow precedence order
- Each machine processes at most one operation at a time
- Each operation assigned to exactly one eligible machine

## Complexity

Strongly NP-hard (generalizes both FJSP and 1||ΣwjTj).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| EDD + ECT | Heuristic | Brandimarte (1993) |
| WATC dispatching | Heuristic | Vepsalainen & Morton (1987) |
| Simulated Annealing | Metaheuristic | Mastrolilli & Gambardella (2000) |
