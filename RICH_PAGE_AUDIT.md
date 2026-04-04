# RICH PAGE AUDIT — Pre-Edit Safety Assessment

> Generated: 2026-04-04

## Risk Matrix

| Page | Lines | Cards | JS Refs | Risk | Edit Strategy |
|------|-------|-------|---------|------|---------------|
| continuous.html | 4,564 | 3 | 124 | SAFE | Update counts + verify cards |
| supply-chain.html | 4,887 | 6 | 121 | SAFE | Update counts + verify cards |
| combinatorial.html | 5,073 | 8 | 118 | SAFE | Update counts + verify cards |
| multi-objective.html | 5,224 | 3 | 119 | SAFE | Update counts + verify cards |
| stochastic.html | 6,555 | 27 | 152 | MEDIUM | Update counts, may add cards |
| packing.html | 6,043 | 0* | 151 | MEDIUM | Uses benchmark-card pattern |
| network.html | 6,611 | 9 | 201 | HIGH | Text-only updates |
| routing.html | 6,750 | 3 | 180 | HIGH | Text-only + possibly add cards |
| scheduling.html | 7,729 | 10 | 199 | HIGH | Text-only updates |

*packing.html uses `benchmark-card` divs instead of `problem-card`

## Shared JS Elements (DO NOT MODIFY)

These IDs appear across most pages and are tightly coupled to JS:
- `algoInfoCard`, `algoInfoOverlay` — algorithm info modals
- `cmdInput`, `cmdPalette`, `cmdResults` — command palette
- `backToTop`, `fsCanvas`, `fsCloseBtn` — UI controls

## Update Types (Safest → Riskiest)

1. **Text in meta-description tags** — zero JS coupling
2. **Numbers in hero stats** — display only, no JS binding
3. **Algorithm counts in meta badges** — display only
4. **New rows in static HTML tables** — safe if table has no JS sorting
5. **New problem cards** — safe if following exact existing pattern
6. **New entries in JS data arrays** — safe but requires understanding data structure
7. **Changing element IDs** — NEVER do this
