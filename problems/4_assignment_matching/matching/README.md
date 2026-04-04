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

## 3. Key Concepts

- **Perfect matching:** Every vertex is matched ($|M| = |V|/2$)
- **Augmenting path:** Alternating path between two unmatched vertices — existence iff matching is not maximum (Berge's theorem)
- **Blossom algorithm:** Edmonds' algorithm for general graphs, contracts odd cycles

---

## 4. Implementation

> This folder is reserved for dedicated matching implementations. Related assignment implementations are in the parent folder.

See [`../assignment/`](../assignment/) — LAP as bipartite weighted matching.

---

## 5. Key References

- Edmonds, J. (1965). Paths, trees, and flowers. *Canadian J. Math.*, 17, 449-467.
- Hopcroft, J.E. & Karp, R.M. (1973). An $n^{5/2}$ algorithm for maximum matchings in bipartite graphs. *SIAM J. Comput.*, 2(4), 225-231.
- Gale, D. & Shapley, L.S. (1962). College admissions and the stability of marriage. *Amer. Math. Monthly*, 69(1), 9-15.
