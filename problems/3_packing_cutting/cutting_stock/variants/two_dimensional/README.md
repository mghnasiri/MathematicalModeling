# Two-Dimensional Cutting Stock Problem (2D-CSP)

## Problem Definition

Cut rectangular items from large stock sheets to satisfy demands, minimizing the number of stock sheets used. Items may optionally be rotated by 90 degrees.

**Objective:** Minimize number of stock sheets used.

**Constraints:**
- All demands must be satisfied
- Items must fit within sheet boundaries
- Items must not overlap
- Only orthogonal (guillotine-free) placements

## Complexity

Strongly NP-hard (generalizes 2D bin packing).

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Bottom-Left FFD | Heuristic | Baker, Coffman & Rivest (1980) |
| Shelf NFDH | Heuristic | Baker, Coffman & Rivest (1980) |
| Simulated Annealing | Metaheuristic | Lodi, Martello & Vigo (1999) |
