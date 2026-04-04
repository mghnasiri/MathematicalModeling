# Steiner Tree Problem in Graphs (MST Variant)

## What Changes

In the **Steiner tree** variant, only a subset S of vertices (called terminals)
must be connected, rather than all vertices as in the MST. The tree may include
non-terminal vertices (Steiner nodes) if they reduce the total edge weight.
The base MST connects all V vertices; the Steiner tree connects only |S|
terminals, potentially using a subset of the remaining vertices as intermediate
relay points. This models network design where only certain nodes need
connectivity -- connecting branch offices through a backbone network, VLSI
routing where only certain pins must be wired, and phylogenetic tree
reconstruction where only observed species are terminals.

The key structural difference is that vertex selection (which Steiner nodes
to include) becomes part of the optimization, making the problem NP-hard
unlike the polynomial MST.

## Mathematical Formulation

The base MST formulation gains terminal subset selection:

```
min  sum_{(u,v) in T}  w(u, v)
s.t. T is a tree (connected, acyclic)
     S is a subset of V(T)    (all terminals are in the tree)
     V(T) is a subset of V    (may include Steiner nodes from V \ S)
```

**Special cases:**
- |S| = |V|: reduces to MST (all vertices are terminals)
- |S| = 2: reduces to shortest path between two terminals

## Complexity

| Variant | Complexity | Reference |
|---------|-----------|-----------|
| General Steiner tree | NP-hard | Karp (1972) |
| \|S\| = 2 | Polynomial (shortest path) | -- |
| \|S\| = \|V\| | Polynomial (MST) | -- |
| Best known approx | 1 + ln(3)/2 ~ 1.55 | Byrka et al. (2013) |

NP-hard for general |S|. The KMB heuristic provides a 2(1 - 1/l)
approximation where l is the number of leaves in the optimal Steiner tree.
The best known polynomial-time approximation ratio is approximately 1.39
by Byrka et al. (2013). [TODO: verify exact ratio and year]

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| Kruskal/Prim (base MST exact) | No | Connects all vertices, not just terminals |
| KMB heuristic (variant) | Yes | MST on terminal-distance graph, 2-approx |
| Shortest path heuristic (variant) | Yes | Iteratively connect nearest terminal |

## Implementations

Python files in this directory:
- `instance.py` -- SteinerTreeInstance, terminal subset, graph structure
- `heuristics.py` -- KMB (Kou-Markowsky-Berman), shortest path heuristic
- `tests/test_steiner.py` -- 16 tests

## Applications

- Telecommunications network design (connecting branch offices)
- VLSI routing (connecting pins on a circuit board)
- Pipeline network design (connecting wells to processing plants)
- Phylogenetic tree reconstruction (connecting observed species)

## Key References

- Kou, L., Markowsky, G. & Berman, L. (1981). "A fast algorithm for Steiner trees." Acta Informatica 15(2), 141-145.
- Hwang, F.K., Richards, D.S. & Winter, P. (1992). The Steiner Tree Problem. Annals of Discrete Mathematics 53, North-Holland.
- Byrka, J., Grandoni, F., Rothvoss, T. & Sanita, L. (2013). "Steiner tree approximation via iterative randomized rounding." JACM 60(1), Article 6. [TODO: verify]
