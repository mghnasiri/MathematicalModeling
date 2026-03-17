# 1D Bin Packing Problem (BPP)

## Problem Definition

Given $n$ items with sizes $s_i$ and an unlimited supply of bins with capacity $C$, pack all items into the minimum number of bins such that the total size in each bin does not exceed $C$.

## Mathematical Formulation

$$\min \sum_{j=1}^{n} y_j$$

$$\text{s.t.} \quad \sum_{i=1}^{n} s_i x_{ij} \leq C \cdot y_j \quad \forall j, \quad \sum_{j=1}^{n} x_{ij} = 1 \quad \forall i$$

where $y_j = 1$ if bin $j$ is used, $x_{ij} = 1$ if item $i$ is in bin $j$.

## Complexity

NP-hard (Garey & Johnson, 1979). Strong NP-hardness (no FPTAS unless P=NP).

## Solution Approaches

| Method | Complexity | Approximation | Description |
|--------|-----------|---------------|-------------|
| First Fit (FF) | $O(n^2)$ | $\leq 1.7 \cdot OPT + 1$ | Place in first bin that fits |
| First Fit Decreasing (FFD) | $O(n \log n)$ | $\leq 11/9 \cdot OPT + 6/9$ | Sort descending, then FF |
| Best Fit Decreasing (BFD) | $O(n \log n)$ | $\leq 11/9 \cdot OPT + 6/9$ | Sort descending, place in tightest bin |
| Genetic Algorithm | $O(\text{pop} \cdot \text{gen} \cdot n)$ | — | Permutation encoding, FF decoder |

## Variants

| Variant | Directory | Description |
|---------|-----------|-------------|
| [Online Bin Packing](variants/online/) | `variants/online/` | Items arrive one at a time; irrevocable placement decisions |
| [Variable-Size Bin Packing (VS-BPP)](variants/variable_size/) | `variants/variable_size/` | Bins of different sizes with associated costs |
| [2D Bin Packing (2D-BPP)](variants/two_dimensional/) | `variants/two_dimensional/` | Pack rectangular items into 2D bins |

## Key References

- Garey, M.R. & Johnson, D.S. (1979). *Computers and Intractability*. W.H. Freeman.
- Johnson, D.S. et al. (1974). Worst-case performance bounds for simple 1D packing algorithms. *SIAM J. Comput.*, 3(4), 299-325. https://doi.org/10.1137/0203025
- Dósa, G. (2007). The tight bound of FFD is $11/9 \cdot OPT + 6/9$. *ESCAPE*, LNCS 4614. https://doi.org/10.1007/978-3-540-74450-4_1
- Martello, S. & Toth, P. (1990). Lower bounds and reduction procedures for BPP. *DAM*, 28(1), 59-70. https://doi.org/10.1016/0166-218X(90)90094-S
