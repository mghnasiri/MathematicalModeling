# Online Bin Packing (BPP Variant)

## What Changes

In the **online bin packing problem**, items arrive one at a time in an arbitrary
(adversarial) order and must be irrevocably assigned to a bin upon arrival. The
algorithm has no knowledge of future items and cannot repack previously placed
items. This models real-time resource allocation scenarios where decisions must
be made immediately -- incoming network packets assigned to buffers, real-time
task scheduling on servers, and warehouse receiving where pallets are shelved
on arrival.

Unlike the offline variant (where FFD achieves 11/9 OPT + 6/9), the lack of
sorting and lookahead fundamentally limits performance. The competitive ratio
framework replaces approximation ratios as the primary quality measure.

## Mathematical Formulation

The base BPP formulation is unchanged, but the decision process is constrained:

**Online constraint:** Item i must be assigned to a bin before item i+1 is revealed:
```
assign(i) is determined using only {s_1, ..., s_i}
```

**No repacking:** Once item i is placed in bin b, it cannot be moved:
```
bin(i, t) = bin(i, t')    for all t' > t  (assignment is permanent)
```

**Performance measure:** Competitive ratio r such that ALG(I) <= r * OPT(I) + c
for all instances I, where ALG is the online algorithm and OPT is the offline
optimum.

## Complexity

| Bound | Value | Reference |
|-------|-------|-----------|
| Lower bound (any online alg) | 1.54037... | Balogh, Bekesi & Galambos (2012) |
| First Fit / Best Fit (upper) | 1.7 asymptotic | Johnson (1973) |
| Next Fit (upper) | 2 | Johnson (1973) |
| Harmonic (upper) | 1.691... | Lee & Lee (1985) |

No online algorithm can achieve competitive ratio below ~1.54, regardless of
computational resources. This is a fundamental information-theoretic limitation.

## Solution Approaches

| Method | Works? | Notes |
|--------|--------|-------|
| FFD (base heuristic) | No | Requires sorting -- items unsorted in online setting |
| BFD (base heuristic) | No | Requires sorting -- items unsorted in online setting |
| Next Fit (variant) | Yes | O(n), simplest: pack in current bin or open new |
| First Fit (variant) | Yes | O(n^2), scan all open bins for first fit |
| Best Fit (variant) | Yes | O(n^2), scan all open bins for tightest fit |

## Implementations

Python files in this directory:
- `instance.py` -- OnlineBPInstance, item-stream simulation
- `heuristics.py` -- Next Fit, First Fit, Best Fit (all online, no sorting)
- `tests/test_online_bp.py` -- 17 tests

## Applications

- Network packet buffering (assign to queues on arrival)
- Cloud VM placement (allocate VMs to servers in real time)
- Warehouse receiving (assign pallets to shelves on delivery)
- Real-time memory allocation (allocate blocks without compaction)

## Key References

- Johnson, D.S. (1973). "Near-optimal bin packing algorithms." PhD thesis, MIT.
- Yao, A.C. (1980). "New algorithms for bin packing." JACM 27(2), 207-227.
- Lee, C.C. & Lee, D.T. (1985). "A simple on-line bin-packing algorithm." JACM 32(3), 562-572.
- Balogh, J., Bekesi, J. & Galambos, G. (2012). "New lower bounds for certain classes of bin packing algorithms." Theoretical Computer Science 440, 1-13. [TODO: verify exact bound value]
