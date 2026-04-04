# Economic Order Quantity (EOQ)

## 1. Problem Definition

- **Input:** Demand rate $D$ (units/year), ordering cost $K$ ($/order), holding cost $h$ ($/unit/year); optionally backorder cost $b$ ($/unit/year) or quantity discount schedule
- **Decision:** Order quantity $Q$
- **Objective:** Minimize total annual cost = ordering cost + holding cost (+ backorder cost if applicable)
- **Classification:** $O(1)$ closed-form for basic and backorder models; $O(B \log B)$ for $B$ discount breakpoints

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $D$ | Annual demand rate (units/year) |
| $K$ | Fixed ordering cost per order |
| $h$ | Holding cost per unit per year |
| $b$ | Backorder cost per unit per year |
| $Q$ | Order quantity (decision variable) |

### Classic EOQ (Harris, 1913)

$$\text{TC}(Q) = \frac{D}{Q} \cdot K + \frac{Q}{2} \cdot h \tag{1}$$

$$Q^* = \sqrt{\frac{2DK}{h}} \tag{2}$$

$$\text{TC}^* = \sqrt{2DKh} \tag{3}$$

### EOQ with Planned Backorders

$$Q^* = \sqrt{\frac{2DK}{h} \cdot \frac{h + b}{b}} \tag{4}$$

When $b \to \infty$, this reduces to the classic EOQ.

### EOQ with Quantity Discounts

For $B$ price tiers $[0, q_1), [q_1, q_2), \ldots$, evaluate the EOQ at each tier's unit cost, select the feasible quantity with minimum total cost (purchase + ordering + holding).

### Small Illustrative Instance

```
D = 1000 units/year, K = $50/order, h = $2/unit/year
Q* = sqrt(2 * 1000 * 50 / 2) = sqrt(50000) ≈ 224 units
TC* = sqrt(2 * 1000 * 50 * 2) = sqrt(200000) ≈ $447/year
Orders per year = 1000/224 ≈ 4.5
```

---

## 3. Solution Methods

| Method | Variant | Complexity | Description |
|--------|---------|-----------|-------------|
| Classic formula | Basic EOQ | $O(1)$ | Closed-form $\sqrt{2DK/h}$ |
| Backorder formula | EOQ + backorders | $O(1)$ | Adjusted formula with $(h+b)/b$ factor |
| Discount evaluation | Quantity discounts | $O(B \log B)$ | Evaluate per tier, pick minimum |

---

## 4. Implementations in This Repository

```
eoq/
├── instance.py                    # EOQInstance, EOQSolution
│                                  #   - Fields: demand_rate, ordering_cost, holding_cost,
│                                  #     backorder_cost, discount_breaks, discount_prices
├── exact/
│   └── eoq_formula.py             # classic_eoq(), backorder_eoq(), discount_eoq()
└── tests/
    └── test_eoq.py                # EOQ test suite
```

---

## 5. Key References

- Harris, F.W. (1913). How many parts to make at once. *Factory, The Magazine of Management*, 10(2), 135-136, 152.
- Hadley, G. & Whitin, T.M. (1963). *Analysis of Inventory Systems*. Prentice-Hall.
- Zipkin, P.H. (2000). *Foundations of Inventory Management*. McGraw-Hill.
