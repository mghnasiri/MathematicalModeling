# Online Bin Packing

## Problem Definition

Items arrive one at a time and must be irrevocably placed into a bin upon arrival. No repacking or lookahead allowed. Minimize the number of bins used.

## Complexity

No online algorithm achieves competitive ratio better than 1.5 (Yao, 1980). First Fit has asymptotic ratio 17/10.

## Algorithms

| Algorithm | Type | Competitive Ratio | Reference |
|-----------|------|-------------------|-----------|
| Next Fit | Online | 2 | Johnson (1973) |
| First Fit | Online | 17/10 | Johnson (1973) |
| Best Fit | Online | 17/10 | Johnson (1973) |
