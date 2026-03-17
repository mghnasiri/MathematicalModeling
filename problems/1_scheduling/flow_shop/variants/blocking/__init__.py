"""
Blocking Flow Shop Scheduling Problem (BFSP)

In the blocking variant, there are no intermediate storage buffers between
machines. A job that has completed processing on machine i cannot leave
that machine until machine i+1 becomes available. This means the job
"blocks" machine i, preventing the next job from starting.

Notation: Fm | prmu, blocking | Cmax
Complexity: NP-hard for m >= 3 (Hall & Sriskandarajah, 1996)

Applications:
    - Manufacturing with limited buffer space
    - Robotic cells (robot transfers between machines)
    - Paint shops (parts must move immediately to avoid drying)
    - Concrete production (material must be poured immediately)

Reference: Hall, N.G. & Sriskandarajah, C. (1996). "A Survey of Machine
           Scheduling Problems with Blocking and No-Wait in Process"
           Operations Research, 44(3):510-525.
           DOI: 10.1287/opre.44.3.510
"""
