# Flow Shop Scheduling — Benchmarks

## Standard Test Instance Libraries

### Taillard Instances (1993) — The Gold Standard
- **URL**: http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/flowshop.dir/best_lb_up.txt
- **Instances**: 120 instances across 12 classes
- **Sizes**: $n \times m$ — 20×5, 20×10, 20×20, 50×5, 50×10, 50×20, 100×5, 100×10, 100×20, 200×10, 200×20, 500×20
- **10 instances per class**
- **Processing times**: $p_{ij} \sim U[1, 99]$
- **Best known solutions**: Maintained and updated
- **Usage**: Most widely used benchmark for PFSP

### VFR Instances (Vallada, Ruiz & Framinan, 2015)
- **URL**: http://soa.iti.es/instancias-702
- **Instances**: 480 instances, designed as harder alternatives to Taillard
- **Sizes**: 100×20 to 800×60
- **Note**: Designed to be challenging for modern metaheuristics

### Carlier Instances (1978)
- **Instances**: 8 small instances
- **Sizes**: $n \times m$ from 5×5 to 12×7
- **Usage**: Good for validating exact algorithms

### Reeves Instances (1995)
- **Instances**: 21 instances
- **Sizes**: 20×5, 20×10, 20×15, 20×20, 30×10, 30×15, 50×10, 75×20
- **Usage**: Common for GA-based approaches

### Heller Instances (1960)
- **Instances**: Small, classic instances
- **Usage**: Early benchmarking

---

## Blocking Flow Shop Instances

### Taillard-based Blocking
- Same instance data as Taillard, evaluated with blocking constraint
- Best known solutions maintained separately

---

## No-Wait Flow Shop Instances

### Taillard-based No-Wait
- Same instance data, evaluated with no-wait constraint
- Reeves & Yamada (1998) provide results

---

## Instance Format

Taillard format:
```
n m
p_11 p_12 ... p_1n     (processing times on machine 1)
p_21 p_22 ... p_2n     (processing times on machine 2)
...
p_m1 p_m2 ... p_mn     (processing times on machine m)
```

---

## Best Known Solutions

- Taillard instances: continuously updated at Taillard's website
- VFR instances: maintained at http://soa.iti.es/
- See also: Ruiz & Stützle (2007) Iterated Greedy results
