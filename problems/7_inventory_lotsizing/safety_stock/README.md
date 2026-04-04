# Safety Stock Optimization

## 1. Problem Definition

- **Input:** $n$ SKUs, each with mean demand $\mu_D$, demand std $\sigma_D$, mean lead time $\mu_L$, lead time std $\sigma_L$, holding cost $h$, and target service level $\text{SL}$
- **Decision:** Safety stock level and reorder point for each SKU
- **Objective:** Minimize total holding cost while meeting the service level target per SKU
- **Classification:** $O(n)$ — closed-form under normal demand/lead time assumptions

---

## 2. Mathematical Formulation

### Notation Table

| Symbol | Definition |
|--------|-----------|
| $n$ | Number of SKUs |
| $\mu_D$ | Mean demand per period |
| $\sigma_D$ | Standard deviation of demand per period |
| $\mu_L$ | Mean lead time (periods) |
| $\sigma_L$ | Standard deviation of lead time |
| $\text{SL}$ | Target cycle service level |
| $z$ | Safety factor $= \Phi^{-1}(\text{SL})$ |

### Demand During Lead Time (DDLT)

$$\sigma_{\text{DDLT}} = \sqrt{\mu_L \cdot \sigma_D^2 + \mu_D^2 \cdot \sigma_L^2} \tag{1}$$

### Safety Stock and Reorder Point

$$\text{SS} = z \cdot \sigma_{\text{DDLT}} = \Phi^{-1}(\text{SL}) \cdot \sigma_{\text{DDLT}} \tag{2}$$

$$\text{ROP} = \mu_D \cdot \mu_L + \text{SS} \tag{3}$$

### Small Illustrative Instance

```
μ_D = 100 units/week, σ_D = 20, μ_L = 2 weeks, σ_L = 0.5
SL = 0.95 → z = 1.645

σ_DDLT = sqrt(2·400 + 10000·0.25) = sqrt(3300) ≈ 57.4
SS = 1.645 × 57.4 ≈ 94.5 units
ROP = 100×2 + 94.5 = 294.5 units
```

---

## 3. Solution Methods

| Method | Type | Complexity | Description |
|--------|------|-----------|-------------|
| Analytical formulas | Exact | $O(n)$ | Closed-form $\sigma_{\text{DDLT}}$, SS, ROP per SKU |

### Analytical Safety Stock

For each SKU, compute $\sigma_{\text{DDLT}}$ from demand and lead-time variability, then apply the safety factor $z = \Phi^{-1}(\text{SL})$. The formula accounts for both demand uncertainty and lead-time uncertainty through the convolution of independent random variables.

```
REORDER-POINT-CALCULATION(SKUs, SL):
  z ← Φ⁻¹(SL)                          // inverse normal CDF
  for each SKU i = 1, ..., n:
    σ_DDLT ← sqrt(μ_L[i] · σ_D[i]² + μ_D[i]² · σ_L[i]²)
    SS[i]  ← z · σ_DDLT
    ROP[i] ← μ_D[i] · μ_L[i] + SS[i]
  return SS, ROP
```

---

## 4. Implementations in This Repository

```
safety_stock/
├── instance.py                    # SafetyStockInstance, SafetyStockSolution
│                                  #   - Fields: n, mean_demands, std_demands,
│                                  #     mean_lead_times, std_lead_times, holding_costs,
│                                  #     service_level
├── exact/
│   └── analytical_ss.py           # compute_sigma_ddlt(), safety stock formulas
└── tests/
    └── test_safety_stock.py       # Safety stock test suite
```

---

## 5. Key References

- Silver, E.A., Pyke, D.F. & Thomas, D.J. (2016). *Inventory and Production Management in Supply Chains*. 4th ed. CRC Press.
- Chopra, S. & Meindl, P. (2019). *Supply Chain Management: Strategy, Planning, and Operation*. 7th ed. Pearson.

---

## 6. Notes

- When lead-time variability is negligible ($\sigma_L \approx 0$), the DDLT formula simplifies to $\sigma_{\text{DDLT}} = \sigma_D \sqrt{\mu_L}$.
- For fill-rate (Type II) service, the safety stock calculation requires the unit normal loss function $G(z)$ rather than $\Phi^{-1}(\text{SL})$.
- The formulas above assume normally distributed demand. For non-normal distributions, use the empirical quantile $Q_{\text{SL}}$ of the DDLT distribution.
- Axsater, S. (2006). *Inventory Control*. 2nd ed. Springer. https://doi.org/10.1007/0-387-33331-2
