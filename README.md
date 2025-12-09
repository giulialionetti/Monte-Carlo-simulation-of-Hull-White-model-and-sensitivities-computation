# Monte Carlo simulation of Hull-White model and sensitivities computation

The goal of this project is to implement Monte Carlo simulation for pricing and sensitivity analysis of fixed income instruments under the Hull-White one-factor short-rate model using CUDA.

## Results

### Q1: Zero-Coupon Bond Pricing
```
Monte Carlo Paths:     1,048,576 × 2 (antithetic variates)
Time Steps:            1,000
Simulation Time:       5.36 ms
Throughput:            391 M paths/sec

Validation:
  P(0,0)  = 1.000000 ✓
  P(0,10) = 0.876844 ✓
  f(0,0)  = 1.21% ✓
```

### Q2a: Theta Calibration
```
Method:    Finite differences on forward rates
Formula:   θ(T) = ∂f/∂T + a·f(0,T) + σ²/(2a)·(1-e^(-2aT))
Max Error: 1.56e-03 (at T=0)
Mean Error: 2.49e-04
Status:    SUCCESS (< 0.01 threshold)
```

### Q2b: Option Pricing with Optimal Control Variate
```
Option:              European call on P(5,10)
Strike:              K = e^(-0.1) = 0.904837
Method:              Optimal beta control variate

Control Variate Analysis:
  Beta (optimal):      0.166447 ± 0.000163
  Beta (naive):        1.0 (theoretical)
  Correlation:         0.673
  Variance Reduction:  +20.4%

ZBC Price:           0.03549203 ± 0.00000825 (95% CI)
Precision:           ±0.0496% (CV)

Performance:         1.97 ms, 1064 M paths/sec
```
### Q3: Sensitivity Analysis (Vega)
```
Method                      Vega        Diff from Pathwise
Pathwise Derivative:        0.229895    ---
Finite Difference:          0.230316    0.18%
FD (recalibrated P,f):      0.523552    127.74%

Statistical Validation (20 independent runs):
  Mean Vega:           0.230189 ± 0.000260 (95% CI)
  Precision:           ±0.24% (CV)
  Z-score:             4.73 (statistically significant)

Conclusion: Pathwise and FD agree within 0.18%. 
            Recalibration degrades accuracy due to accumulated 
            Monte Carlo noise from re-simulating P(0,T) curves.
```

## Mathematical Model

### Hull-White SDE

$$dr(t) = [\theta(t) - ar(t)]dt + \sigma dW_t$$

**Parameters:** $r_0 = 0.012$, $a = 1.0$, $\sigma = 0.1$

### Exact Discretization

$$r(t+\Delta t) = r(t)e^{-a\Delta t} + \text{drift} + \sigma\sqrt{\frac{1-e^{-2a\Delta t}}{2a}}\,G$$

where $G \sim \mathcal{N}(0,1)$

### Analytical Bond Formula

$$P(t,T) = A(t,T)\exp(-B(t,T)r(t))$$

where:

$$B(t,T) = \frac{1 - e^{-a(T-t)}}{a}$$

$$A(t,T) = \frac{P(0,T)}{P(0,t)} \exp\left[B(t,T)f(0,t) - \frac{\sigma^2(1-e^{-2at})}{4a}B(t,T)^2\right]$$

### Optimal Control Variate

For option payoff $X$ with control $Y = \text{discount} \cdot P(S_1, S_2)$:

$$\beta^* = \frac{\text{Cov}(X,Y)}{\text{Var}(Y)} \quad \text{(empirically computed)}$$

$$X_{\text{CV}} = X - \beta^*(Y - \mathbb{E}[Y])$$

**Theoretical β = 1**, but **optimal β* ≈ 0.166** due to non-linear payoff structure.

## Implementation

### Variance Reduction
- **Antithetic Variates**: Simulate with $\pm G$ per path
- **Optimal Control Variates**: Empirically compute $\beta^* = \text{Cov}(X,Y)/\text{Var}(Y)$ achieving 20% variance reduction
- **Common Random Numbers**: Shared RNG states for finite differences
- **Statistical Validation**: 20 independent runs to quantify precision

### GPU Optimizations
- **Warp Shuffle Reductions**: $O(\log N)$ complexity
- **Constant Memory**: Precomputed drift tables
- **Fast Math**: `__expf()`, `__fmaf_rn()` intrinsics
- **100% Occupancy**: 1024 threads/block, no register spilling (32 regs/thread)
- **Atomic Operations**: Lock-free accumulation across blocks

### Code Structure
```
include/
  ├── common.cuh          # Core kernels & utilities
  ├── market_data.cuh     # Bond pricing & calibration
  ├── output.cuh          # JSON/CSV output
  └── perf_benchmark.cuh  # Reduction benchmarks

src/
  ├── 1_bond_pricing.cu             # Q1: P(0,T) and f(0,T)
  ├── 2_option_pricing.cu           # Q2: Calibration & ZBC with optimal CV
  ├── 3_sensitivity_analysis.cu     # Q3: Pathwise + FD + validation
  └── benchmark_reductions.cu       # Performance comparison
```

## Building & Running
```bash
# Compile
make all

# Run workflow
make run-all

# Generate plots
make analyze

# Clean
make clean
```

**Requirements:**
- CUDA Toolkit 11.0+
- GPU with compute capability 7.0+ (V100, A100, RTX 3000+)
- Python 3.8+ (matplotlib, pandas, numpy)

## Performance Details

### GPU Occupancy (Tesla V100)
```
Theoretical Occupancy:  100.0%
Registers/thread:       32
Shared mem/block:       128 bytes
Local mem/thread:       0 (no spilling)
Blocks per SM:          2 (limited by registers)
Active threads/SM:      2048/2048
```

### Throughput Comparison
```
Task                    Time (ms)    Throughput (M paths/s)
Q1: Bond Pricing        5.36         391
Q2b: Option Pricing     1.97         1064
Q3: Sensitivity         2.06         509
```

## Findings

1. **control variate**: Empirically determined β* = 0.166 achieved 20% variance reduction with strong correlation (ρ = 0.67), compared to -42% with naive β = 1. This 62 percentage point improvement highlights the critical importance of proper calibration for non-linear payoffs.

2. **Statistical validation**: 20 independent Monte Carlo runs achieved sub-0.3% precision (CV) for all estimates, with 95% confidence intervals confirming robustness.

3. **Method agreement**: Pathwise derivative and finite difference agree within 0.18%, despite statistical significance (z = 4.73) due to inherent methodological differences (tangent vs secant approximation).

4. **Antithetic variates**: Effective for variance reduction with minimal computational overhead.

5. **Recalibration counterproductive**: Re-simulating P(0,T) curves introduces Monte Carlo noise, increasing error from 0.18% to 127%.

6. **GPU efficiency**: Achieved 100% occupancy with optimized warp shuffle reductions and zero register spilling.

---

**Authors**: Giulia Lionetti, Francesco Zattoni  
**Institution**: Sorbonne Université  
**Course**: Advanced High Performance Computing Algorithms and Programming Many-Core Architectures  
**Academic Year**: 2024-2025