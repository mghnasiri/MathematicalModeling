# Single Machine Batch Scheduling (1 | batch, sj | ΣwjCj)

## Problem Definition

Jobs are grouped into batches processed on a single machine. All jobs in a batch start together; batch processing time equals the maximum processing time of its jobs. A setup time is incurred between consecutive batches. Minimize total weighted completion time.

## Complexity

NP-hard even with equal setup times.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| WSPT Single | Heuristic | Smith (1956) |
| Greedy Batching | Heuristic | Webster & Baker (1995) |
| Simulated Annealing | Metaheuristic | Potts & Kovalyov (2000) |
