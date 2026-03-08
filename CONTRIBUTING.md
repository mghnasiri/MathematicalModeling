# Contributing

## Adding a New Problem

1. Create directory: `problems/{family}/{problem_name}/`
2. Add subdirectories: `exact/`, `heuristics/`, `metaheuristics/`, `tests/`
3. Create three documentation files:

| File | Content |
|------|---------|
| `README.md` | Problem definition, mathematical formulation, complexity, solution approaches |
| `BENCHMARKS.md` | Standard test instances with URLs and format descriptions |
| `LITERATURE.md` | Foundational papers, key algorithms, surveys, and recent articles (links only) |

## Adding an Algorithm

### File Structure

Each algorithm implementation should follow this pattern:

```python
"""
Algorithm Name for Problem Name

Notation: α | β | γ
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

## Adding Tests

Tests go in `problems/{family}/{problem_name}/tests/` and should:

1. Test with small hand-crafted instances (verify correctness)
2. Test with at least one benchmark instance (verify against known optimal/best-known)
3. Use `pytest` with clear test names

```python
def test_spt_minimizes_total_completion_time():
    """SPT should give optimal total completion time on single machine."""
    ...

def test_against_benchmark_instance():
    """Validate against known optimal for OR-Library instance."""
    ...
```

## Literature Guidelines (Copyright Compliance)

- Include only **bibliographic data**: author, year, title, journal, DOI link
- Write 1-sentence descriptions of what each paper contributes (in your own words)
- Link to the official DOI or publisher page — never host PDFs
- For recent/hot articles, focus on 2020+ publications from top OR journals

## Code Style

- Python 3.10+
- Type hints everywhere
- `dataclass` for data structures
- `numpy` for numerical operations
- `matplotlib` for visualization
- Follow PEP 8
