# SYNC MAP — GitHub Pages ↔ Content Layer

> Generated: 2026-04-04 | View Sync Pass Phase 0

## Architecture Summary

- **CSS:** Inline in each HTML file (~500-900 lines). No shared stylesheet.
- **JS:** Inline + Bootstrap 5.3.0 CDN. Interactive: scroll nav, theme toggle, sortable tables, expandable sections, solver panels.
- **Fonts:** Cormorant Garamond (headings), Inter (body), JetBrains Mono (code). Google Fonts CDN.
- **Math:** NO KaTeX/MathJax. Math shown as HTML entities (sub/sup tags, &ge;, &le;, etc.).
- **Analytics:** Google Analytics G-X3F4KE0WSF on every page.
- **Build system:** None. Hand-crafted standalone HTML files.
- **Template pattern:** Each page self-contained with identical CSS variables + Bootstrap grid.

## Page Categories

### Family Pages (14 in docs/families/)

| HTML Page | Lines | README Source | README Lines | Status |
|-----------|-------|--------------|--------------|--------|
| `scheduling.html` | 7,729 | `1_scheduling/` (11 subfamilies) | 655+534+454+402+241+184+144 | Rich — may need algorithm count updates |
| `routing.html` | 6,750 | `2_routing/` (8 subfamilies) | 563+498+434 | Rich — may need variant/algorithm updates |
| `packing.html` | 6,043 | `3_packing_cutting/` (7 subfamilies) | 400+356+356 | Rich — may need algorithm updates |
| `network.html` | 6,611 | `6_network_flow_design/` (5 subfamilies) | 423+105+111 | Rich |
| `stochastic.html` | 6,555 | `9_uncertainty_modeling/` (9 subfamilies) | 400+358+... | Rich |
| `combinatorial.html` | 5,073 | `combinatorial/` (7 subfamilies) | 107+105+... | Rich |
| `multi-objective.html` | 5,224 | `multi_objective/` (3 subfamilies) | 109+106+105 | Rich |
| `continuous.html` | 4,564 | `continuous/` (4 subfamilies) | 109+119+100+100 | Rich |
| `supply-chain.html` | 4,887 | `7_inventory_lotsizing/` (7 subfamilies) | 114+106+... | Rich |
| **`assignment.html`** | **296** | `4_assignment_matching/` (4 subfamilies) | 104+101+110+... | **STUB — needs full build** |
| **`integrated.html`** | **296** | `8_integrated_structural/` (3 subfamilies) | IRP+LRP+ALB | **STUB — needs full build** |
| **`inventory.html`** | **296** | `7_inventory_lotsizing/` | 114+106+103+... | **STUB — needs full build** |
| **`location.html`** | **296** | `5_location_covering/` (6 subfamilies) | 388+106+... | **STUB — needs full build** |
| **`uncertainty.html`** | **296** | `9_uncertainty_modeling/` | 400+358+... | **STUB — needs full build** |

### Application Pages (~115 in docs/applications/)

| Category | Count | Lines Range | Status |
|----------|-------|-------------|--------|
| Domain overviews (agriculture, construction, etc.) | 8 | 4,000-5,600 | Rich — content matches pre-enrichment |
| Individual applications | ~107 | 4,500-5,600 | Rich — hand-crafted with interactive elements |

### Index Page

| Page | Lines | Status |
|------|-------|--------|
| `index.html` | 4,915 | Rich — stats may need updating (57→74 problems, 2339→2556 tests) |

## Priority Actions

### P0: Update index page stats
- Problem count: 57 → 74
- Test count: 2,339 → 2,556 (verified)
- Family count: 10 → 13 (with legacy families)

### P1: Fill 5 stub family pages (296 lines each → ~3,000-5,000)
These are the biggest gap. Each needs:
1. Hero section (already has title + subtitle)
2. Quick navigation
3. Problem cards for each subfamily (notation, description, algorithm table)
4. References section
5. Related families section

Order: assignment, location, inventory, uncertainty, integrated

### P2: Update rich family pages
Algorithm counts and variant counts may be stale. Quick audits needed for:
- scheduling.html: Now 11 subfamilies (was showing 6). Algorithm counts need updating.
- routing.html: Now 8 subfamilies. Variant counts updated.
- packing.html: Now 7 subfamilies.

### P3: Application pages — likely need minimal updates
The domain overview pages (agriculture.html, healthcare.html etc.) may need decision chain table updates.

## Reference Template

**Best page for content pattern:** `docs/families/scheduling.html`
- Most complete, most problem cards, most interactive elements
- CSS is the gold standard for the design system
- Problem card structure: h3 title + notation + meta badges + description + formulation + tags + algo table

**Best stub page for structure:** `docs/families/assignment.html`
- Shows the minimal Chrome (nav, breadcrumb, hero, footer, JS) that every page shares
- The "Content in progress" section needs to be replaced with actual problem cards
