# Mathematical Modeling & Operations Research

[![GitHub Pages](https://img.shields.io/badge/Live_Site-mghnasiri.github.io/MathematicalModeling-0A192F?style=for-the-badge&logo=github)](https://mghnasiri.github.io/MathematicalModeling/)
[![Pages](https://img.shields.io/badge/Pages-135+-C5A059?style=for-the-badge)](#site-map)
[![OR Problems](https://img.shields.io/badge/OR_Problems-30+-4A90D9?style=for-the-badge)](#problem-taxonomy)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An interactive encyclopedia of **Operations Research** problems — featuring mathematical formulations, real-world applications across 18 domains, and browser-based solvers you can run instantly.

> **Live site:** [mghnasiri.github.io/MathematicalModeling](https://mghnasiri.github.io/MathematicalModeling/)

---

## What This Project Is

This is a **GitHub Pages site** with **135 self-contained HTML pages**, each presenting an OR problem in a real-world context with:

- Canonical **mathematical formulation** (objective, constraints, decision variables)
- **Interactive solver** — runs in your browser, no installation required
- **HTML5 Canvas visualizations** — Gantt charts, route maps, network graphs, assignment diagrams
- **Mapping table** connecting the real-world scenario to the abstract OR model
- **Evidence panel** with academic references and industry adoption data
- **Solver realism badge** — ★★★ Exact, ★★☆ Heuristic, ★☆☆ Educational Demo

---

## Domains & Applications

### Industry Applications (9 sectors)

| Sector | Apps | Example Problems |
|--------|------|-----------------|
| **Healthcare** | 8 | Nurse Rostering, OR Scheduling, Patient Flow, Ambulance Placement |
| **Manufacturing** | 14 | Job Shop, Flow Shop, Assembly Line Balancing, Lot Sizing |
| **Logistics & Transport** | 12 | CVRP, VRPTW, Hub-Spoke Design, Fleet Sizing, Gate Assignment |
| **Agriculture** | 13 | Crop Selection, Harvest Scheduling, Irrigation Network, Water Allocation |
| **Retail & E-Commerce** | 8 | Assortment Optimization, Store Location, Order Picking, Grocery Ordering |
| **Finance & Insurance** | 15 | Markowitz, CVaR, Robust Portfolio, Black-Scholes, Binomial, Monte Carlo, VaR/ES, Vasicek/CIR, Merton Credit, ALM, Optimal Execution, Real Options, Insurance Ratemaking, Systemic Risk |
| **Construction** | 8 | Project Scheduling, Site Selection, Crew Assignment, Resource Leveling |
| **Energy** | 8 | Unit Commitment, Economic Dispatch, Plant Siting, Storage Dispatch |
| **Public Services** | 8 | School Location, Fire Station Siting, Transit Network, Evacuation Routing |

### Science & Research Applications (9 domains)

| Domain | Apps | OR Problems |
|--------|------|-------------|
| **Astronomy & Space** | 4 | JWST Scheduling, Gravity-Assist TSP, Radio Array Design, Debris Tracking |
| **Genomics & Bioinformatics** | 3 | Genome Assembly (Hamiltonian), MSA, Protein Design |
| **Ecology & Conservation** | 3 | Reserve Selection (Set Covering), Wildlife Corridors (Steiner Tree) |
| **Climate & Earth Sciences** | 3 | Sensor Placement (Facility Location), Submodular Optimization |
| **Neuroscience** | 3 | Brain Parcellation (Graph Clustering), L1 Optimization |
| **Physics & Materials** | 3 | Crystal Prediction (QUBO), Accelerator Scheduling (RCPSP) |
| **Computational Chemistry** | 3 | Retrosynthesis (MCTS), Molecular Docking (Branch & Bound) |
| **Mathematics & Theory** | 3 | Four Color Theorem, Kepler Conjecture, Ramsey Numbers |
| **Planetary Defense** | 12 | WTA, MCLP, Security Games, POMDP, Network Interdiction |

### Problem Family Reference Pages (14 families)

Interactive taxonomy pages covering: Scheduling, Routing, Packing, Assignment, Location, Network Flow, Inventory, Integrated, Stochastic, Combinatorial, Continuous, Multi-Objective, Supply Chain, Uncertainty.

---

## Problem Taxonomy

```
Operations Research Problems
│
├── 1. Scheduling
│   ├── Single Machine (SPT, EDD, Moore's)
│   ├── Parallel Machine (LPT)
│   ├── Flow Shop (Johnson's, NEH, IG)
│   ├── Job Shop (Tabu Search, CP)
│   ├── Flexible Job Shop (Integrated TS, GA)
│   └── RCPSP (GA + SGS)
│
├── 2. Routing
│   ├── TSP (Held-Karp DP, Branch & Bound)
│   ├── CVRP (Clarke-Wright, Sweep)
│   ├── VRPTW (Solomon I1, SA, GA)
│   └── Arc Routing (CARP)
│
├── 3. Packing & Cutting
│   ├── Knapsack (0-1, Bounded, Multidimensional)
│   ├── Bin Packing (1D, 2D, 3D — FFD)
│   └── Cutting Stock (Column Generation)
│
├── 4. Assignment & Matching
│   ├── Linear Assignment (Hungarian O(n^3))
│   ├── Quadratic Assignment (SA, TS)
│   └── Graph Matching
│
├── 5. Location & Covering
│   ├── Facility Location (UFL, CFL)
│   ├── p-Median / p-Center
│   ├── MCLP (Maximal Covering)
│   ├── Set Covering / Set Partitioning
│   └── Set Packing
│
├── 6. Network Flow & Design
│   ├── Shortest Path (Dijkstra, Bellman-Ford)
│   ├── Max Flow / Min Cut (Edmonds-Karp)
│   ├── Minimum Spanning Tree (Kruskal, Prim)
│   └── Network Interdiction
│
├── 7. Inventory & Lot Sizing
│   ├── EOQ Models
│   ├── Lot Sizing (Wagner-Whitin)
│   ├── Capacitated Lot Sizing
│   └── Newsvendor (Stochastic)
│
├── 8. Integrated Structural
│   ├── Location-Routing (LRP)
│   ├── Inventory-Routing (IRP)
│   └── Assembly Line Balancing (SALBP)
│
├── 9. Uncertainty & Stochastic
│   ├── Stochastic Programming (Two-Stage SP)
│   ├── Robust Optimization
│   ├── Chance-Constrained Programming
│   └── Distributionally Robust Optimization (DRO)
│
├── 10. Game Theory & Adversarial
│   ├── Stackelberg Security Games
│   ├── Weapon-Target Assignment (WTA)
│   └── Tri-Level Defender-Attacker-Defender
│
└── 11. Sequential Decision Making
    ├── Markov Decision Processes (MDP)
    ├── Approximate Dynamic Programming (ADP)
    └── POMDP (Partially Observable)
```

---

## Repository Structure

```
MathematicalModeling/
├── docs/                         # GitHub Pages site root
│   ├── index.html                # Main landing page — 135 app links
│   ├── applications/             # 120 interactive application pages
│   │   ├── healthcare-domain.html    # Domain landing page
│   │   ├── nurse-rostering.html      # Individual app with solver
│   │   ├── operating-room.html
│   │   └── ...
│   └── families/                 # 14 problem family reference pages
│       ├── scheduling.html
│       ├── routing.html
│       └── ...
├── problems/                     # Algorithm implementations (Python/C++)
│   └── {family}/{problem}/
│       ├── README.md
│       ├── exact/
│       ├── heuristics/
│       ├── metaheuristics/
│       └── tests/
├── shared/                       # Reusable solver components
│   ├── metaheuristics/
│   ├── parsers/
│   └── visualization/
├── CONTRIBUTING.md               # Contribution guidelines
└── README.md                     # This file
```

---

## Technology

| Layer | Stack |
|-------|-------|
| **Hosting** | GitHub Pages (static site) |
| **Pages** | Self-contained HTML — all CSS & JS inline, zero build step |
| **Styling** | Bootstrap 5.3 grid + custom midnight/gold design system |
| **Fonts** | Cormorant Garamond (headings) · Inter (body) · JetBrains Mono (math) |
| **Icons** | FontAwesome 6 |
| **Solvers** | Pure JavaScript — no external solver libraries |
| **Visualization** | HTML5 Canvas (Gantt charts, route maps, network graphs) |
| **Analytics** | Google Analytics (G-X3F4KE0WSF) |

---

## Design System

The site uses a consistent academic visual identity:

| Token | Value | Usage |
|-------|-------|-------|
| Midnight | `#0A192F` | Primary background, headers |
| Gold | `#C5A059` | Accent, highlights, buttons |
| Ivory | `#F8F9FA` | Card backgrounds, body |

Each domain has a unique CSS hero pattern (star fields for astronomy, heartbeat echoes for healthcare, blueprint grids for manufacturing, etc.) with matching icon animations.

---

## Getting Started

### Browse the site

Visit **[mghnasiri.github.io/MathematicalModeling](https://mghnasiri.github.io/MathematicalModeling/)** — every solver runs in your browser.

### Run locally

```bash
# Clone the repository
git clone https://github.com/mghnasiri/MathematicalModeling.git
cd MathematicalModeling

# Serve locally (any static file server works)
python -m http.server 8000 --directory docs
# Open http://localhost:8000
```

### Run algorithm implementations

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run a specific algorithm
python problems/1_scheduling/job_shop/heuristics/dispatching_rules.py

# Run tests against benchmarks
pytest problems/1_scheduling/job_shop/tests/
```

---

## Scheduling Notation

This project uses the standard **three-field notation** $\alpha \mid \beta \mid \gamma$ introduced by Graham et al. (1979):

| Field | Meaning | Examples |
|-------|---------|---------|
| $\alpha$ | Machine environment | $1$ (single), $P_m$ (parallel identical), $F_m$ (flow shop), $J_m$ (job shop) |
| $\beta$ | Job characteristics & constraints | $r_j$ (release dates), $p_{ij}$ (machine-dependent), $prec$ (precedence) |
| $\gamma$ | Objective function | $C_{\max}$ (makespan), $\sum C_j$ (total completion), $\sum w_j T_j$ (weighted tardiness) |

---

## License

MIT — see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding problems, applications, and interactive solvers.
