"""
No-Wait Flow Shop Scheduling Problem (NWFSP)

In the no-wait variant, once a job starts processing on the first machine,
it must proceed through all machines without any waiting time between
consecutive operations. This means each job's processing on consecutive
machines must be contiguous in time.

Notation: Fm | prmu, no-wait | Cmax
Complexity: NP-hard for m >= 3 (Roeck, 1984)
            Polynomial for m = 2 via reduction to TSP on a line (Gilmore & Gomory, 1964)

Applications:
    - Steel manufacturing (continuous casting)
    - Chemical processing (reactions that cannot be interrupted)
    - Food processing (temperature-sensitive products)
    - Pharmaceutical production
"""
