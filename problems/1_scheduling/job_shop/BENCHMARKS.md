# Job Shop Scheduling — Benchmarks

## Standard Test Instance Libraries

### Fisher & Thompson (1963) — The Classics
- **Instances**: `ft06` (6×6), `ft10` (10×10), `ft20` (20×5)
- **Historical significance**: `ft10` was unsolved for 26 years (optimal = 930)
- **Usage**: Validation of exact methods

### Lawrence Instances (1984)
- **Instances**: `la01`–`la40` (40 instances)
- **Sizes**: 10×5, 15×5, 20×5, 10×10, 15×10, 20×10, 30×10, 15×15
- **Usage**: Widely used medium-size benchmark

### Adams, Balas & Zawack (1988)
- **Instances**: `abz5`–`abz9` (5 instances)
- **Sizes**: 10×10, 20×15
- **Usage**: Hard instances for testing

### Taillard Instances (1993)
- **URL**: http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/jobshop.dir/best_lb_up.txt
- **Instances**: 80 instances across 8 classes
- **Sizes**: 15×15, 20×15, 20×20, 30×15, 30×20, 50×15, 50×20, 100×20
- **10 instances per class**
- **Usage**: Standard large-scale benchmark

### Demirkol, Mehta & Uzsoy (DMU, 1998)
- **Instances**: `dmu01`–`dmu80` (80 instances)
- **Sizes**: 20×15 to 50×20
- **Usage**: Harder modern instances, many still open

### Yamada & Nakano (1992)
- **Instances**: `yn1`–`yn4`
- **Sizes**: 20×20
- **Usage**: Hard instances

### Storer, Wu & Vaccari (SWV, 1992)
- **Instances**: `swv01`–`swv20`
- **Sizes**: 20×10, 20×15, 50×10
- **Usage**: Well-studied instances

### ORB Instances (Applegate & Cook, 1991)
- **Instances**: `orb01`–`orb10` (10 instances)
- **Sizes**: 10×10
- **Usage**: Challenging despite small size

---

## Online Resources

- **Comprehensive list of JSP instances and best known**: http://optimizizer.com/jobshop.php
- **Taillard's page**: http://mistic.heig-vd.ch/taillard/
- **OR-Library**: http://people.brunel.ac.uk/~mastjjb/jeb/info.html

---

## Instance Format

Standard format:
```
n  m
machine_1  time_1  machine_2  time_2  ...  machine_m  time_m    (job 1)
machine_1  time_1  machine_2  time_2  ...  machine_m  time_m    (job 2)
...
```

Each row defines a job's route: pairs of (machine_index, processing_time) in operation order.

---

## Best Known Solutions

- Many Taillard and DMU instances remain **open** (gap between best known upper and lower bounds)
- Best known solutions are maintained at:
  - Taillard's website
  - http://optimizizer.com/jobshop.php
