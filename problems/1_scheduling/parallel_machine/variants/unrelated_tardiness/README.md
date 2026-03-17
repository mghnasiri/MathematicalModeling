# Unrelated Parallel Machine with Total Tardiness (Rm || ΣTj)

## Problem Definition

n jobs assigned to m unrelated parallel machines where processing time depends on both job and machine. Each job has a due date. Minimize total tardiness.

**Objective:** min ΣTj where Tj = max(0, Cj - dj)

## Complexity

Strongly NP-hard.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| EDD-ECT | Heuristic | Pinedo (2016) |
| ATC dispatching | Heuristic | Pinedo (2016) |
| Simulated Annealing | Metaheuristic | Weng, Lu & Ren (2001) |
