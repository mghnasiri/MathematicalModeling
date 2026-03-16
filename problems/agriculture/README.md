# Agriculture Problem Family

Operations Research applications in the agriculture sector. Each sub-problem
applies established OR models to real-world agricultural planning decisions.

## Sub-Problems

| Problem | OR Model | Algorithms | Tests |
|---------|----------|------------|-------|
| **Crop Harvest Planning** | Newsvendor (stochastic inventory) | Critical fractile, marginal allocation, independent+scale | 34 |
| **Feed Procurement** | EOQ + Silver-Meal + Wagner-Whitin (lot sizing) | Classic EOQ, dynamic heuristic, optimal DP | 26 |
| **Farm Delivery Routing** | CVRP + Stochastic VRP | Clarke-Wright, sweep, CC-CW, SA with recourse | 28 |
| **Irrigation Network** | MST + Max Flow + Shortest Path | Kruskal, Edmonds-Karp, Dijkstra | 24 |
| **Crop Rotation** | LP + Multi-Objective (epsilon-constraint) | HiGHS LP, Pareto front | 28 |
| **Cold Storage & Packaging** | Bin Packing + Cutting Stock | FF, FFD, BFD, GA, greedy CSP | 27 |

**Total: 167 tests**

## Structure

```
agriculture/
├── README.md
├── crop_harvest/
│   ├── instance.py              # CropHarvestInstance, CropProfile
│   ├── heuristics/
│   │   └── harvest_planning.py  # Critical fractile, marginal allocation
│   └── tests/
│       └── test_crop_harvest.py # 34 tests
├── feed_procurement/
│   ├── instance.py              # FeedProcurementInstance, FarmInputProfile
│   ├── heuristics/
│   │   └── procurement_planning.py  # EOQ, Silver-Meal, Wagner-Whitin
│   └── tests/
│       └── test_feed_procurement.py # 26 tests
├── farm_delivery/
│   ├── instance.py              # FarmDeliveryInstance, DeliveryPoint
│   ├── heuristics/
│   │   └── deterministic_routing.py # Clarke-Wright, Sweep
│   ├── metaheuristics/
│   │   └── stochastic_routing.py    # CC-CW, SA with recourse
│   └── tests/
│       └── test_farm_delivery.py    # 28 tests
├── irrigation_network/
│   ├── instance.py              # IrrigationNetworkInstance, NetworkNode
│   ├── exact/
│   │   └── network_analysis.py  # Kruskal + Edmonds-Karp + Dijkstra
│   └── tests/
│       └── test_irrigation_network.py # 24 tests
├── crop_rotation/
│   ├── instance.py              # CropRotationInstance, FieldProfile
│   ├── exact/
│   │   └── lp_allocation.py     # LP + epsilon-constraint Pareto
│   └── tests/
│       └── test_crop_rotation.py    # 28 tests
└── cold_storage/
    ├── instance.py              # ColdStorageInstance, PackagingInstance
    ├── heuristics/
    │   └── packing_solver.py    # FF, FFD, BFD, GA, greedy CSP
    └── tests/
        └── test_cold_storage.py     # 27 tests
```

## Benchmark Instances

Each sub-problem includes benchmark instances based on a Quebec agricultural
setting:

- **Crop Harvest**: 8-crop Quebec vegetable farm with $2,500/day labor budget
- **Feed Procurement**: 500-head Quebec dairy farm, 12-month planning horizon
- **Farm Delivery**: 15 delivery points across Quebec City (markets, restaurants, stores, food banks)
- **Irrigation Network**: 10-node farm with 5 field zones and 16 pipe segments
- **Crop Rotation**: 6 fields (145 ha), 5 crops with water/labor/diversity constraints
- **Cold Storage**: 20 produce lots, 4 packaging sheet types

## Running Tests

```bash
# Run all agriculture tests (167 tests)
python -m pytest problems/agriculture/ -v

# Run individual sub-problem tests
python -m pytest problems/agriculture/crop_harvest/tests/ -v
python -m pytest problems/agriculture/feed_procurement/tests/ -v
python -m pytest problems/agriculture/farm_delivery/tests/ -v
python -m pytest problems/agriculture/irrigation_network/tests/ -v
python -m pytest problems/agriculture/crop_rotation/tests/ -v
python -m pytest problems/agriculture/cold_storage/tests/ -v
```
