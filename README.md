# Monte Carlo simulation of Hull-White model and sensitivities computation

The goal of this project is to implement Monte Carlo simulation for pricing and sensitivity analysis
of fixed income instruments under the Hull-White one-factor short-rate model using CUDA.

## Results

### Q1: Zero-Coupon Bond Pricing
```
Monte Carlo Paths:     1,048,576 × 2 (antithetic variates)
Time Steps:            1,000
Simulation Time:       6.18 ms
Throughput:            339 M paths/sec

Validation:
  P(0,0)  = 1.000000 ✓
  P(0,10) = 0.876813
  f(0,0)  = 1.21%
```

### Q2a: Theta Calibration
```
Method:    Finite differences on forward rates
Formula:   θ(T) = ∂f/∂T + a·f(0,T) + σ²/(2a)·(1-e^(-2aT))
Max Error: 1.56e-03 (at T=0)
Mean Error: 2.27e-04
Status:    SUCCESS (< 0.01 threshold)
```

### Q2b: Option Pricing
```
Option:              European call on P(5,10)
Strike:              K = e^(-0.1) = 0.904837
Method:              Control variate variance reduction

ZBC (raw):           0.03546107
Control mean:        0.87679142
Expected (P(0,10)):  0.87681299
Control deviation:   -0.00002158
ZBC (adjusted):      0.03548265

Performance:         1.67 ms, 1260 M paths/sec
```

### Q3: Sensitivity Analysis (Vega)
```
Method                      Vega        Diff from Pathwise
Pathwise Derivative:        0.230562    ---
Finite Difference:          0.230173    0.17%
FD (recalibrated P,f):      0.523699    127.14%

Conclusion: Recalibration degrades accuracy due to accumulated 
            Monte Carlo noise from re-simulating P(0,T) curves.
```

## Mathematical Model

### Hull-White SDE
```
dr(t) = [θ(t) - a·r(t)]dt + σ·dW(t)
```

Parameters: r₀=0.012, a=1.0, σ=0.1

### Exact Discretization
```
r(t+Δt) = r(t)·e^(-a·Δt) + drift + σ·√[(1-e^(-2a·Δt))/(2a)]·G
```

### Analytical Bond Formula
```
P(t,T) = A(t,T)·exp(-B(t,T)·r(t))
```

## Implementation

### Variance Reduction
- **Antithetic Variates**: Simulate with ±G per path
- **Control Variates**: Use E[discount·P] = P(0,S₂)
- **Common Random Numbers**: Shared RNG states for finite differences

### GPU Optimizations
- **Warp Shuffle Reductions**: O(log N) complexity
- **Constant Memory**: Precomputed drift tables
- **Fast Math**: `__expf()`, `__fmaf_rn()` intrinsics
- **100% Occupancy**: 1024 threads/block, no register spilling (32 regs/thread)

### Code Structure
```
include/
  ├── common.cuh          # Core kernels & utilities
  ├── market_data.cuh     # Bond pricing & calibration
  ├── output.cuh          # JSON/CSV output
  └── perf_benchmark.cuh  # Reduction benchmarks

src/
  ├── 1_bond_pricing.cu             # Q1: P(0,T) and f(0,T)
  ├── 2_option_pricing.cu           # Q2: Calibration & ZBC
  ├── 3_sensitivity_analysis.cu     # Q3: Pathwise + FD
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
Q1: Bond Pricing        6.18         339
Q2b: Option Pricing     1.67         1260
Q3: Sensitivity         1.74         602
```

## Key Findings

1. **Antithetic variates**: Effective for variance reduction with minimal overhead
2. **Control variates**: Deviation of 0.00002 validates simulation quality
3. **Pathwise vs FD agreement**: 0.17% difference demonstrates numerical consistency
4. **Recalibration counterproductive**: Adding MC noise to P(0,T) curves increases error from 0.17% to 127%
5. **Warp shuffles**: Efficient reduction pattern for GPU kernels

## Experimental: Market Data Recalibration

Testing whether re-simulating P(0,T) and f(0,T) at perturbed volatilities improves finite difference accuracy:

**Hypothesis**: Recalibrating market data ensures model consistency

**Result**: Vega error increased from 0.17% → 127.14%

**Explanation**: Re-running Q1 three times (at σ-ε, σ, σ+ε) introduces Monte Carlo noise that outweighs any theoretical benefit. Original Q1 data quality was sufficient.

---

**Authors**: Giulia Lionetti, Francesco Zattoni  
**Institution**: Sorbonne Université  
**Course**: Advanced High Performance Computing Algorithms and Programming Many-Core Architectures  
**Academic Year**: 2025-2026
