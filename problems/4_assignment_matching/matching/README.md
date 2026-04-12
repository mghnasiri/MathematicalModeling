# Graph Matching Problem

## 1. Problem Definition

- **Input:** Graph $G = (V, E)$ with optional edge weights $w_e$
- **Decision:** Select a subset $M \subseteq E$ of pairwise non-adjacent edges (a **matching**)
- **Objective:** Maximize $|M|$ (maximum cardinality) or $\sum_{e \in M} w_e$ (maximum weight)
- **Constraints:** Each vertex incident to at most one edge in $M$
- **Classification:** **Polynomial** — $O(V \cdot E)$ for bipartite, $O(V^3)$ for general graphs

---

## 2. Variants

| Variant | Graph | Objective | Complexity |
|---------|-------|-----------|-----------|
| Maximum Cardinality (bipartite) | Bipartite | Max $\|M\|$ | $O(\sqrt{V} \cdot E)$ Hopcroft-Karp |
| Maximum Weight (bipartite) | Bipartite | Max $\sum w_e$ | $O(V^3)$ Hungarian |
| Maximum Cardinality (general) | General | Max $\|M\|$ | $O(V \cdot E)$ Edmonds |
| Maximum Weight (general) | General | Max $\sum w_e$ | $O(V^3)$ Edmonds |
| Stable Matching | Bipartite | Stability | $O(V^2)$ Gale-Shapley |

---

### Small Illustrative Instance

```
Bipartite graph: L = {a, b, c}, R = {1, 2, 3}
Edges: (a,1), (a,2), (b,1), (b,3), (c,2)
Maximum matching: M = {(a,2), (b,1), (c,2)} — wait, c can only go to 2.
Actually: M = {(a,1), (b,3), (c,2)} → |M| = 3 (perfect matching)
Augmenting path from initial M={}: a-1, then b-3, then c-2.
```

---

## 3. Key Concepts

- **Perfect matching:** Every vertex is matched ($|M| = |V|/2$)
- **Augmenting path:** Alternating path between two unmatched vertices — existence iff matching is not maximum (Berge's theorem)
- **Blossom algorithm:** Edmonds' algorithm for general graphs, contracts odd cycles
- **Hall's theorem:** A bipartite graph $G = (L \cup R, E)$ has a matching saturating all of $L$ iff $|N(S)| \geq |S|$ for every $S \subseteq L$

#### Augmenting Path Algorithm Pseudocode (Bipartite)

```
MAX_MATCHING_BIPARTITE(G = (L ∪ R, E)):
    M = {}  (empty matching)
    for each unmatched vertex u in L:
        visited = {}
        if AUGMENT(u, M, visited):
            |M| increases by 1
    return M

AUGMENT(u, M, visited):
    for each neighbor v of u in R:
        if v not in visited:
            visited.add(v)
            if v is unmatched in M, or AUGMENT(match(v), M, visited):
                M = M ∪ {(u, v)}   # (and remove old edge of v if matched)
                return TRUE
    return FALSE
```

**Complexity:** $O(V \cdot E)$ for bipartite maximum cardinality matching. Hopcroft-Karp improves this to $O(\sqrt{V} \cdot E)$ by finding multiple augmenting paths per phase.

### Edmonds' Blossom Algorithm (General Graphs)

For non-bipartite graphs, augmenting paths may encounter odd-length cycles, which cannot occur in bipartite graphs. Edmonds' blossom algorithm (1965) handles this by **contracting** odd cycles (blossoms) into single super-vertices, searching for augmenting paths in the contracted graph, and then expanding blossoms to recover the actual matching. The blossom algorithm runs in $O(V^3)$ for maximum weight matching and $O(V \cdot E)$ for maximum cardinality.

### Konig's Theorem

In bipartite graphs, the size of a maximum matching equals the size of a minimum vertex cover: $\nu(G) = \tau(G)$. This is a cornerstone of combinatorial optimization and does **not** hold for general graphs (where $\tau(G)$ can be up to $2\nu(G)$).

### Applications

- **Job assignment** (matching workers to tasks)
- **Organ donor matching** (kidney exchange programs)
- **Ride-sharing** (matching drivers to passengers)
- **Stable marriage** (college admissions via Gale-Shapley)
- **Course scheduling** (matching students to course sections)
- **Network flow** (matching as a special case of network flow on bipartite networks)

### Weighted vs. Cardinality Matching

For **cardinality** matching, the goal is simply to maximize the number of edges in $M$. The augmenting path method above is the standard approach. For **weighted** matching, the goal is to maximize $\sum_{e \in M} w_e$, which requires maintaining dual variables and adjusting them to preserve complementary slackness. The Hungarian method (for bipartite) and weighted blossom algorithm (for general) both achieve $O(V^3)$.

---

## 4. Implementation

> This folder is reserved for dedicated matching implementations. Related assignment implementations are in the parent folder.

See [`../assignment/`](../assignment/) — LAP as bipartite weighted matching.

---

## 5. Key References

- Edmonds, J. (1965). Paths, trees, and flowers. *Canadian J. Math.*, 17, 449-467.
- Hopcroft, J.E. & Karp, R.M. (1973). An $n^{5/2}$ algorithm for maximum matchings in bipartite graphs. *SIAM J. Comput.*, 2(4), 225-231.
- Gale, D. & Shapley, L.S. (1962). College admissions and the stability of marriage. *Amer. Math. Monthly*, 69(1), 9-15.
- Lovász, L. & Plummer, M.D. (1986). *Matching Theory*. North-Holland.
- Berge, C. (1957). Two theorems in graph theory. *Proc. Nat. Acad. Sci.*, 43(9), 842-844.
