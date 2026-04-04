# TEMPLATE STANDARD

> Defines the exact structure every problem and application README must follow.
> All folders must match this standard for `[EXCELLENT]` status in AUDIT_MANIFEST.md.
>
> Last updated: 2026-04-04

---

## A. Problem Family README Template

Every `/problems/{family}/{problem}/README.md` must contain **all 8 sections** below.
If a section genuinely does not apply, include the heading with a one-line explanation of why.

```markdown
# {Problem Name} ({Abbreviation})

## 1. Problem Definition

Formal definition in plain language, then:

- **Input:** sets, parameters, data structures
- **Decision variables:** what the solver chooses
- **Objective:** what is minimized/maximized
- **Constraints:** list all constraints in natural language
- **Classification:** combinatorial / continuous / mixed-integer / stochastic / multi-objective
- **Complexity:** P / NP-hard / NP-complete / strongly NP-hard — with citation
  - Include proof sketch or reduction reference (e.g., "reduces from 3-PARTITION")
  - Special cases that become polynomial (e.g., "P for m=2 machines")

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition | Domain |
|--------|-----------|--------|
| $n$ | Number of ... | $\mathbb{Z}^+$ |
| ... | ... | ... |

### Formulation (name, e.g., "DFJ Formulation" or "Position-Based MILP")

$$\min \quad \text{objective}$$

$$\text{s.t.} \quad \text{constraint 1} \quad \forall i \in N \quad (1)$$
$$\quad\quad\quad \text{constraint 2} \quad \forall j \in M \quad (2)$$
$$\quad\quad\quad x_{ij} \in \{0, 1\} \quad (3)$$

Each constraint numbered and explained.

### Alternative Formulations (if they exist)

Same format for each alternative (e.g., MTZ vs DFJ for TSP).
Note: strengths/weaknesses, LP relaxation quality.

## 3. Variants

For EACH known variant:

### 3.k {Variant Name} ({notation})

- **Added/modified constraints:** what changes from the base problem
- **Complexity:** if different from base
- **Key references:** 1-2 seminal papers
- **Solution methods that apply:** cross-ref to Section 5
- **Implementation:** link to `variants/{name}/` if implemented

## 4. Benchmark Instances

### Standard Libraries

| Library | # Instances | Size Range | URL |
|---------|------------|-----------|-----|
| ... | ... | ... | ... |

### Instance Format

Describe the file format used by standard benchmarks.

### Best Known Solutions (BKS)

| Instance | BKS | Method | Author | Year |
|----------|-----|--------|--------|------|
| ... | ... | ... | ... | ... |

(Include at least the canonical small/medium instances.)

### Small Illustrative Instance

Include 1-2 tiny instances directly in the README for quick testing.

## 5. Solution Methods

### 5.1 Exact Methods

For EACH method:

**{Method Name}** (Author, Year)

- **Idea:** 2-3 sentence description
- **Complexity:** time and space
- **Practical limit:** approximate instance size
- **Key implementation detail:** (e.g., branching strategy, valid inequalities)

Pseudocode:
```
ALGORITHM MethodName(instance)
  ...
  RETURN solution
```

### 5.2 Constructive Heuristics

Same format as 5.1 but also include:
- **Approximation ratio:** if known (with citation)
- **Quality in practice:** typical gap to optimum

### 5.3 Improvement Heuristics / Local Search

For each neighborhood structure:
- **Move definition:** what changes in a single move
- **Neighborhood size:** $O(?)$
- **Move evaluation:** naive vs. incremental cost

### 5.4 Metaheuristics

For EACH metaheuristic:
- **Encoding:** how solutions are represented
- **Operators:** crossover/mutation (GA), neighborhood (SA/TS), etc.
- **Key parameters:** and recommended values with citation
- **Pseudocode**
- **Performance:** typical results on benchmarks

### 5.5 Hybrid and Advanced Methods (if applicable)

- Matheuristics, decomposition, ML-assisted, parallel approaches

## 6. Implementation Guide

- How to model this problem in Pyomo / OR-Tools / Gurobi Python API
- Key implementation pitfalls
- Solver parameter tuning advice
- Data preprocessing pipeline for real-world instances

## 7. Computational Results Summary

| Method | Category | Time (small) | Gap (small) | Time (large) | Gap (large) |
|--------|----------|-------------|------------|-------------|------------|
| ... | Exact | ... | 0% | ... | ... |
| ... | Heuristic | ... | ...% | ... | ...% |
| ... | Metaheuristic | ... | ...% | ... | ...% |

- What works best at each scale
- State-of-the-art method and reference

## 8. Key References

- **Seminal papers:** [Author (Year)] Title. *Journal*. DOI: ...
- **Best surveys:** ...
- **Textbook chapters:** ...
- **Recent state-of-the-art:** ...
```

---

## B. Variant README Template (Lighter)

Every `/problems/{family}/{problem}/variants/{variant}/README.md`:

```markdown
# {Variant Name} ({scheduling notation if applicable})

## Problem Definition

How this differs from the base problem. 1-2 paragraphs.

## Additional Constraints / Modified Objective

LaTeX formulation of what changes.

## Complexity

If different from base. Otherwise: "Same as base — NP-hard."

## Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| ... | Exact/Heuristic/Metaheuristic | Author (Year) |

For each algorithm, 2-3 sentences describing the approach.

## Implementation

```
variants/{name}/
├── instance.py    # Modified instance/solution dataclasses
├── heuristics.py  # Variant-specific algorithms
└── tests/
    └── test_{name}.py
```

## Key References

1-3 papers specific to this variant.
```

---

## C. Application Domain README Template

Every `/applications/{domain}/README.md`:

```markdown
# {Application Domain}

## 1. Domain Overview

- What this application area is about
- Why OR/optimization matters here
- Scale of impact: industry size, documented cost savings
- Key decision-makers who use these models

## 2. Decision Chain

| Phase | Decision Point | OR Problem Family | Link |
|-------|---------------|-------------------|------|
| ... | ... | ... | `/problems/...` |

## 3. Key Optimization Problems

For EACH decision point:

### 3.k {Decision Point Name}

- **Real-world question:** what managers actually ask
- **OR formulation:** which problem family models this
- **Typical data:** what input data is needed
- **Typical scale:** number of variables/constraints in practice
- **Link:** to the canonical problem in `/problems/`

## 4. Real-World Constraints

Constraints not captured in textbook formulations:
- Regulatory, union, safety, fairness constraints
- Data availability and quality challenges
- Organizational / political constraints

## 5. Case Studies

For each case study (2-3 minimum):

### 5.k {Company/Organization}

- **Problem:** what they needed to solve
- **Scale:** problem dimensions
- **Method:** what approach was used
- **Result:** quantified improvement
- **Reference:** citation

## 6. Modeling Walkthrough

Step-by-step: from messy real-world problem to clean OR formulation:
1. Problem scoping
2. Data collection and preprocessing
3. Model formulation
4. Solver selection
5. Validation and sensitivity analysis
6. Deployment considerations

## 7. Implementations

List and briefly describe each Python file:

| File | Problem Type | OR Family |
|------|-------------|-----------|
| `{domain}_*.py` | ... | ... |

## 8. Current Research Frontiers

- Open problems in this domain
- Emerging trends (robustness, fairness, sustainability, real-time)
- Recent high-impact papers (last 3 years)

## 9. Key References

- **Industry reports:** ...
- **Academic surveys:** ...
- **Textbook chapters:** ...
```

---

## D. Application Phase README Template

Every `/applications/{domain}/{NN_phase}/README.md`:

```markdown
# Phase: {Phase Name}

## Decision Points

| Decision | OR Problem | Scale | Frequency |
|----------|-----------|-------|-----------|
| ... | ... | ... | Daily/Weekly/Annual |

## Detailed Formulations

For each decision point:

### {Decision Point Name}

**Sets and Parameters:**
...

**Decision Variables:**
...

**Objective:**
...

**Constraints:**
...

## Real-World Considerations

- Data sources and quality issues
- Stakeholder requirements
- Computational requirements

## See Also

- `/problems/{family}/{problem}/` — canonical formulation
- Other phases that feed into / depend on this phase
```

---

## E. Code File Standards

Every `.py` file must have:

1. **Module docstring:** algorithm name, problem notation, complexity, 1-2 key references with DOI
2. **Type hints** on all function signatures
3. **Google-style docstrings** on public functions (Args, Returns, Raises)
4. **`if __name__ == "__main__"` block** with a small runnable example
5. **`seed` parameter** for all randomized algorithms
6. **No global state** — all state passed via parameters
7. **Numpy arrays** for numerical data

---

## F. Notation Conventions (Cross-Repository)

To ensure consistency across all READMEs:

| Concept | Symbol | Notes |
|---------|--------|-------|
| Number of jobs / cities / items | $n$ | Primary problem dimension |
| Number of machines / vehicles / bins | $m$ | Secondary dimension |
| Processing time | $p_j$ or $p_{ij}$ | Job $j$ (on machine $i$) |
| Due date | $d_j$ | Job $j$ |
| Weight / priority | $w_j$ | Job $j$ |
| Distance / cost matrix | $d_{ij}$ or $c_{ij}$ | From $i$ to $j$ |
| Capacity | $Q$ or $C$ | Vehicle or bin |
| Binary decision variable | $x_{ij}$ | Assignment of $i$ to $j$ |
| Makespan | $C_{\max}$ | Completion time of last job |
| Tardiness | $T_j = \max(0, C_j - d_j)$ | |
| Optimal solution value | $z^*$ or OPT | |
| Approximation ratio | $\rho$ | Worst-case guarantee |
