# Preemptive Single Machine Scheduling (1 | pmtn, rj | ΣCj)

## Problem Definition

Single machine with job release dates. Preemption allowed. Minimize total (weighted) completion time.

## Complexity

Polynomial for 1|pmtn,rj|ΣCj (SRPT is optimal). NP-hard for 1|pmtn,rj|ΣwjCj.

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| SRPT | Exact | Schrage (1968) |
| Weighted SRPT | Heuristic | Baker & Trietsch (2009) |
