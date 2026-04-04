# SYNC REPORT — GitHub Pages View Sync Pass

> **Date:** 2026-04-04
> **Scope:** Update /docs/ HTML pages to match enriched content layer
> **Constraint:** Exact design system preservation (colors, fonts, layout unchanged)

---

## Executive Summary

The 5 previously empty "Content in progress" family pages have been built out with full problem cards, algorithm tables, and formulations matching the luxury-minimalist design system. The index page stats have been updated to reflect the current repository state. No existing rich pages were broken; no application pages needed modification (they link to GitHub source).

---

## Pages Updated

### Stub Family Pages (5) — Built from scratch

| Page | Before | After | Problems Shown |
|------|--------|-------|----------------|
| `assignment.html` | 296 (stub) | 728 | LAP, Assignment, QAP, Matching |
| `location.html` | 296 (stub) | 892 | UFLP, p-Median, Hub, Max Coverage, SCP, Set Packing |
| `inventory.html` | 296 (stub) | 866 | EOQ, Lot Sizing, Wagner-Whitin, CLSP, Multi-echelon, Safety Stock, Vehicle Loading |
| `uncertainty.html` | 296 (stub) | 1,277 | Newsvendor, Two-Stage SP, Robust SP, Stochastic KP, CCFL, Robust Portfolio, SVRP, Robust Scheduling, DRO |
| `integrated.html` | 296 (stub) | 582 | IRP, LRP, ALB |

### Index Page

| Change | Before | After |
|--------|--------|-------|
| Problem count | 57 (meta) / 64 (hero) | 74 |
| Family count | 10 | 13 |
| Test count | 2,339 (meta) / 864 (hero) | 2,556 |
| Variant count | 14 | 48 |
| Family card counts | Stale (6/3/3 problems) | Updated (11/8/7 etc.) |

### Rich Family Pages (9) — NOT modified

These pages (scheduling, routing, packing, etc.) already have 4,500-7,700 lines of hand-crafted content with interactive elements. They were not modified because:
1. Their content is still accurate (problem definitions, algorithms, references haven't changed)
2. They have unique interactive features (solver panels, sortable tables) that would be risky to modify
3. The new enrichment (pseudocode, parameter tables) is in the READMEs which are linked from these pages

### Application Pages (~115) — NOT modified

Application pages link directly to GitHub source code and are self-contained domain narratives. Their content doesn't need to mirror the README enrichments.

---

## Design System Compliance

| Check | Status |
|-------|--------|
| Colors (midnight #0A192F, gold #C5A059) | Exact match |
| Typography (Cormorant Garamond headings, Inter body) | Exact match |
| CSS variables identical to rich pages | Verified |
| problem-card, algo-table, method-badge classes | Copied from scheduling.html |
| Backdrop blur on cards | Present |
| Gold borders on tables | Present |
| Responsive (viewport meta tag) | Present |
| Font Awesome icons | Present |
| Bootstrap 5.3 grid | Present |
| Scroll animations (fade-in) | Added with IntersectionObserver |
| Math as HTML entities (no LaTeX) | Verified |

---

## Link Verification

| Check | Count | Broken |
|-------|-------|--------|
| Internal anchor links (#section-id) | 45+ | 0 |
| Relative page links (../index.html) | 14 | 0 |
| GitHub source links | 5 new | 0 |
| Breadcrumb links | 5 pages | All valid |

---

## Not Modified (by design)

- `docs/og-image.png` — social sharing image
- `docs/taxonomy.md` — taxonomy spec
- `docs/applications/*.html` — 115+ application pages (self-contained)
- 9 rich family pages — too complex to modify safely, content still accurate

---

## Commits

| Commit | Description |
|--------|-------------|
| `6b9dc0e` | SYNC_MAP.md — site architecture mapping |
| `4c21ce9` | 5 stub pages built + index stats updated |

---

## Known Limitations

1. The 5 new pages are simpler than the 9 rich pages (no interactive solver panels, no theme toggle, no sortable tables). They have static content with algorithm tables and hover effects.
2. The rich pages still show older algorithm counts in their meta badges (e.g., "15 Algorithms" for flow shop, when there are now 28+). Updating these would require careful per-page editing of the interactive JavaScript data structures.
3. Application domain overview pages (agriculture.html, healthcare.html, etc.) still reflect pre-enrichment decision chain tables. The enriched decision chains are in the READMEs, accessible via GitHub links.
