# Contributing to Mathematical Modeling & Operations Research

Thank you for your interest in contributing! This project has two main components:

1. **Interactive web pages** (`docs/`) — self-contained HTML pages with browser-based solvers
2. **Algorithm implementations** (`problems/`) — Python/C++ implementations with benchmarks

---

## Table of Contents

- [Adding a New Application Page](#adding-a-new-application-page)
- [Adding a New Problem Family Page](#adding-a-new-problem-family-page)
- [Adding an Algorithm Implementation](#adding-an-algorithm-implementation)
- [Design System Reference](#design-system-reference)
- [Code Style](#code-style)
- [Literature Guidelines](#literature-guidelines)
- [Checklist Before Submitting](#checklist-before-submitting)

---

## Adding a New Application Page

Application pages live in `docs/applications/` and are self-contained HTML files.

### Required Sections

Every application page must include:

| Section | Purpose |
|---------|---------|
| **Hero** | Domain-themed header with problem title and one-line description |
| **Breadcrumb** | 3-level: `Home > [Sector/Domain] > [Page Title]` |
| **OR Problem Card** | Canonical problem name, complexity class, solver realism badge |
| **Mathematical Formulation** | Objective function, decision variables, constraints |
| **Real-World → OR Mapping Table** | Maps domain terms to mathematical notation |
| **Interactive Solver** | JavaScript solver with parameter inputs and Run button |
| **Canvas Visualization** | Visual output (Gantt chart, route map, network graph, etc.) |
| **Evidence Panel** | Academic references, industry adoption data |
| **Educational Disclaimer** | States this is a teaching tool, not production software |

### Page Template Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>[Problem Name] — Mathematical Modeling</title>
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-X3F4KE0WSF"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-X3F4KE0WSF');
    </script>
    <!-- Bootstrap 5.3 + FontAwesome 6 + Google Fonts -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* All CSS inline — follow the design system tokens below */
    </style>
</head>
<body>
    <!-- Breadcrumb -->
    <!-- Hero section -->
    <!-- OR Problem Card + Formulation -->
    <!-- Mapping Table -->
    <!-- Interactive Solver -->
    <!-- Canvas Visualization -->
    <!-- Evidence Panel -->
    <!-- Footer with disclaimer -->
    <script>
        /* All JS inline — solver logic, canvas rendering */
    </script>
</body>
</html>
```

### Technical Rules for Application Pages

1. **Self-contained** — all CSS and JS must be inline. No external solver libraries.
2. **Safe DOM manipulation** — use `createElement` / `textContent`, never `innerHTML` for user data.
3. **Variable declarations** — use `var` (not `const`/`let`) for maximum browser compatibility.
4. **GA tag** — every page must include the `G-X3F4KE0WSF` analytics tag.
5. **Breadcrumb** — always 3-level: `Home > [Sector] > [Page Title]`.
6. **Solver realism badge** — classify honestly:
   - ★★★ **Exact** — optimal solution guaranteed (small instances)
   - ★★☆ **Heuristic** — good solutions, no optimality guarantee
   - ★☆☆ **Educational Demo** — simplified for teaching

### Linking Your Page

After creating the page, add it to:

1. **`docs/index.html`** — add an `app-card` in the appropriate group (Industry or Science & Research)
2. **Domain landing page** — add to the application grid in the relevant `*-domain.html`
3. **Command palette** — add a search entry in the `commandItems` array in `index.html`

---

## Adding a New Problem Family Page

Problem family reference pages live in `docs/families/` and provide an overview of a problem class with:

- Problem definition and variants
- Complexity classification
- Key algorithms and their guarantees
- Links to all application pages that use this problem type
- Interactive mini-solver demonstrating the core algorithm

Existing families: Scheduling, Routing, Packing, Assignment, Location, Network, Inventory, Integrated, Stochastic, Combinatorial, Continuous, Multi-Objective, Supply Chain, Uncertainty.

---

## Adding an Algorithm Implementation

Algorithm implementations live in `problems/{family}/{problem_name}/`.

### Directory Structure

```
problems/{family}/{problem_name}/
├── README.md             # Definition, formulation, complexity
├── BENCHMARKS.md         # Standard test instances & sources
├── LITERATURE.md         # Key papers & recent articles (links only)
├── exact/                # Exact algorithms (B&B, DP, ILP)
├── heuristics/           # Constructive & improvement heuristics
├── metaheuristics/       # GA, SA, Tabu, ACO, etc.
└── tests/                # Validation against known benchmarks
```

### Implementation Template

```python
"""
Algorithm Name for Problem Name

Notation: alpha | beta | gamma  (for scheduling problems)
Complexity: O(...)
Reference: Author (Year) - "Paper Title"
"""

from dataclasses import dataclass

@dataclass
class Instance:
    """Problem instance data."""
    ...

@dataclass
class Solution:
    """Solution representation."""
    objective: float
    ...

def solve(instance: Instance) -> Solution:
    """Main solving function."""
    ...

if __name__ == "__main__":
    # Example usage with a small instance
    ...
```

### Requirements

- Type hints on all functions
- Docstrings with algorithm description and complexity
- Reference to the original paper
- A `solve()` function as the main entry point
- An `if __name__ == "__main__"` block with a small example
- Tests that validate against known benchmark results

### Adding Tests

Tests go in `problems/{family}/{problem_name}/tests/` and should:

1. Test with small hand-crafted instances (verify correctness)
2. Test with at least one benchmark instance (verify against known optimal / best-known)
3. Use `pytest` with clear test names

```python
def test_spt_minimizes_total_completion_time():
    """SPT should give optimal total completion time on single machine."""
    ...

def test_against_benchmark_instance():
    """Validate against known optimal for OR-Library instance."""
    ...
```

---

## Design System Reference

### Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| Midnight | `#0A192F` | Page backgrounds, hero sections |
| Gold | `#C5A059` | Accents, buttons, highlights |
| Ivory | `#F8F9FA` | Cards, body backgrounds |
| White | `#FFFFFF` | Panel backgrounds |
| Text Primary | `#333333` | Body text |
| Text Secondary | `#6C757D` | Captions, meta |

### Typography

| Element | Font | Weight | Size |
|---------|------|--------|------|
| h1, h2 (display) | Cormorant Garamond | 700 | 2.2rem / 1.8rem |
| h3, h4 | Cormorant Garamond | 600 | 1.4rem / 1.2rem |
| Body | Inter | 400 | 1rem |
| Math / Code | JetBrains Mono | 400 | 0.9rem |

### Domain-Specific Hero Themes

Each sector/domain has a unique CSS `hero::before` pattern. When adding a page, match the existing pattern for its domain:

| Domain | Pattern | Animation |
|--------|---------|-----------|
| Healthcare | Heartbeat echoes | Pulse ripple |
| Manufacturing | Blueprint grid | Grid scan |
| Logistics | Route dots | Flowing dots |
| Agriculture | Growing leaves | Leaf growth |
| Retail | Shopping grid | Price flicker |
| Finance | Candlestick bars | Market pulse |
| Construction | Structural beams | Crane swing |
| Energy | Power waves | Wave flow |
| Public Services | Civic stars | Badge spin |
| Astronomy | Star field | Twinkle |
| Genomics | DNA helix | Strand rotate |
| Ecology | Ecosystem web | Web pulse |
| Climate | Weather fronts | Front drift |
| Neuroscience | Neural sparks | Synapse fire |
| Physics | Particle tracks | Orbit spin |
| Chemistry | Molecular bonds | Bond vibrate |
| Mathematics | Proof steps | Step reveal |

---

## Code Style

### HTML/CSS/JS (Application Pages)

- All styles and scripts inline — no external files
- Bootstrap 5.3 for grid layout only
- FontAwesome 6 for icons
- `createElement` + `textContent` for DOM manipulation
- `var` declarations for browser compatibility
- Canvas API for visualizations — no charting libraries

### Python (Algorithm Implementations)

- Python 3.10+
- Type hints everywhere
- `dataclass` for data structures
- `numpy` for numerical operations
- `matplotlib` for visualization
- Follow PEP 8

---

## Literature Guidelines (Copyright Compliance)

- Include only **bibliographic data**: author, year, title, journal, DOI link
- Write 1-sentence descriptions of what each paper contributes (in your own words)
- Link to the official DOI or publisher page — never host PDFs
- For recent articles, focus on 2020+ publications from top OR journals

---

## Checklist Before Submitting

### For Application Pages

- [ ] Page is self-contained (all CSS/JS inline)
- [ ] GA tag `G-X3F4KE0WSF` is present
- [ ] Breadcrumb follows `Home > [Sector] > [Title]` pattern
- [ ] Mathematical formulation includes objective, variables, and constraints
- [ ] Mapping table connects real-world terms to OR notation
- [ ] Solver runs correctly and produces visible output
- [ ] Canvas visualization renders without errors
- [ ] Solver realism badge is honest (★★★ / ★★☆ / ★☆☆)
- [ ] Evidence panel has at least 2 academic references
- [ ] Educational disclaimer is present
- [ ] Domain-specific hero pattern matches existing pages in same domain
- [ ] Page linked from `index.html` and relevant domain landing page
- [ ] Page added to command palette search entries
- [ ] Mobile responsive (tested at 640px width)

### For Algorithm Implementations

- [ ] `solve()` function as main entry point
- [ ] Type hints and docstrings
- [ ] Reference to original paper
- [ ] `if __name__ == "__main__"` with small example
- [ ] Tests against known benchmarks
- [ ] `README.md` with formulation and complexity

---

## Questions?

Open an issue on [GitHub](https://github.com/mghnasiri/MathematicalModeling/issues) or check the [live site](https://mghnasiri.github.io/MathematicalModeling/) for examples of existing pages.
