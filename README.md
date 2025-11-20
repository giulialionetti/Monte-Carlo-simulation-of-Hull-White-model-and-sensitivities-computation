# Monte-Carlo-simulation-of-Hull-White-model-and-sensitivities-computation

## project overview

The objective of this project is to implement Monte Carlo simulation for pricing and sensitivity analysis of fixed income instruments under the Hull-White one-factor short-rate model using CUDA.

This project implements Monte Carlo simulation for pricing and sensitivity analysis of fixed income instruments under the Hull-White one-factor short-rate model using CUDA.

## Mathematical Framework

### Hull-White One-Factor Model

The short-rate model is defined by the stochastic differential equation:

```
dr(t) = [θ(t) - ar(t)]dt + σdW_t
```

**Parameters:**
- `r(0) = 0.012` (initial short rate)
- `a = 1` (mean reversion speed)
- `σ = 0.1` (volatility)
- `θ(t)` (piecewise linear function defined on [0, 10]):
  ```
  θ(t) = [0.012 + 0.0014·t]  for 0 ≤ t < 5
         [0.019 + 0.001·(t-5)] for 5 ≤ t ≤ 10
  ```

### Simulation Formula

The short rate r(t) at time t, given its value r_s at time s < t, follows a Gaussian distribution:

```
r(t) = m_{s,t} + Σ_{s,t}·G
```

where G ~ N(0,1) and:

```
m_{s,t} = r(s)·e^{-a(t-s)} + ∫_s^t e^{-a(t-u)}θ(u)du

Σ_{s,t} = √[σ²(1 - e^{-2a(t-s)})/(2a)]
```

### Zero Coupon Bond Pricing

The zero coupon bond price represents the amount to pay at time t to receive 1 at maturity T:

```
P(t,T) = E_t[exp(-∫_t^T r_s ds)]
```

**Forward Rate:**
```
f(0,T) = ∂ln(P(0,T))/∂T
```

**Analytical Expression (Hull-White):**
```
P(t,T) = A(t,T)·exp(-B(t,T)·r(t))

where:
B(t,T) = (1 - e^{-a(T-t)})/a

A(t,T) = [P(0,T)/P(0,t)]·exp[B(t,T)·f(0,t) - σ²(1-e^{-2aT})·B(t,T)²/(4a)]
```

### Calibration Formula

In practice, P(0,T) values are market-quoted, and θ is recovered using:

```
θ(T) = ∂f(0,T)/∂T + a·f(0,T) + σ²(1-e^{-2aT})/(2a)
```

### Zero Coupon Bond Call Option

European call option on a zero coupon bond:

```
ZBC(S₁, S₂, K) = E[e^{-∫_0^{S₁} r_s ds}·(P(S₁,S₂) - K)⁺]
```

where (x)⁺ = max(x, 0) and 0 < S₁ < S₂ ≤ 10.

### Sensitivity (Greeks)

Derivative with respect to volatility σ:

```
∂_σ ZBC(S₁,S₂,K) = E[∂_σ P(S₁,S₂)·e^{-∫_0^{S₁} r_s ds}·1_{P(S₁,S₂)>K}
                      - [∫_0^{S₁} ∂_σ r_s ds]·e^{-∫_0^{S₁} r_s ds}·(P(S₁,S₂)-K)⁺]
```

where ∂_σ r(t) follows the induction:

```
∂_σ r(t) = M_{s,t} + (Σ_{s,t}/σ)·G

M_{s,t} = ∂_σ r(s)·e^{-a(t-s)} + 2σe^{-at}[cosh(at) - cosh(as)]/a²

∂_σ r(0) = 0
```

## Project Tasks

### Question 1: Monte Carlo for Zero Coupon Bonds (8 points)

**Objective:** Compute Monte Carlo estimates of P(0,T) and f(0,T) for T ∈ [0,10]

**Requirements:**
- Implement CUDA kernel for short-rate simulation using equation (8)
- Use trapezoidal rule on uniform time grid to approximate ∫_0^T r_s ds
- Compute both bond prices P(0,T) and forward rates f(0,T)
- Generate results for a range of maturities

**Key Implementation Details:**
- Discretize time interval [0,10] uniformly
- For each Monte Carlo path:
  - Simulate r(t) trajectory using Gaussian increments
  - Accumulate integral using trapezoidal rule
  - Compute exp(-∫_0^T r_s ds)
- Average over all paths to get P(0,T)
- Compute forward rate using finite differences: f(0,T) ≈ ∂ln(P(0,T))/∂T

### Question 2: Calibration and Option Pricing

#### 2a) Calibration 

**Objective:** Recover the piecewise linear θ(t) from equation (7) using market data

**Requirements:**
- Use P(0,T) and f(0,T) values from Question 1
- Implement CUDA kernel to apply calibration formula (10)
- Verify recovered θ(t) matches the specified piecewise linear form

**Implementation:**
- Compute ∂f(0,T)/∂T numerically from grid of f values
- Apply formula: θ(T) = ∂f(0,T)/∂T + a·f(0,T) + σ²(1-e^{-2aT})/(2a)

#### 2b) Option Pricing 

**Objective:** Price the zero coupon bond call option ZBC(5, 10, e^{-0.1})

**Requirements:**
- Use analytical formula for P(t,T)
- Simulate r(t) trajectories from t=0 to t=5
- Approximate integral ∫_0^5 r_s ds using trapezoidal rule
- Compute option payoff: e^{-∫_0^5 r_s ds}·(P(5,10) - e^{-0.1})⁺
- Average over Monte Carlo paths

### Question 3: Sensitivity Analysis

**Objective:** Compute ∂_σ ZBC(5, 10, e^{-0.1}) and validate with finite differences

**Requirements:**
- Implement pathwise derivative method
- Simulate both r(t) and ∂_σ r(t) along the same paths using same random numbers
- Compute sensitivity using the analytical derivative formula
- **Validation:** Compare with finite difference approximation:
  ```
  [ZBC(σ + ε) - ZBC(σ - ε)]/(2ε)  for small ε
  ```

**Implementation Details:**
- Initialize ∂_σ r(0) = 0
- At each time step, update both r(t) and ∂_σ r(t) using the same G
- Compute ∂_σ P(S₁,S₂) using chain rule on analytical formula
- Accumulate ∫_0^{S₁} ∂_σ r_s ds using trapezoidal rule

## Technical Requirements

### CUDA Implementation
- Efficient use of GPU global and shared memory
- Proper random number generation (cuRAND)
- Thread-safe Monte Carlo path generation
- Numerical stability for exponential and integration calculations

### Numerical Methods
- Trapezoidal rule for integral approximation
- Finite difference for derivative approximation
- Appropriate time discretization (balance accuracy vs. computation)

### Testing & Validation
- Compare Question 2a results with known θ(t) expression
- Verify finite difference approximation matches pathwise derivative
- Check numerical stability and convergence

## Deliverables

1. **Source Code:**
   - Well-commented CUDA kernels
   - Host code for data management
   - Random number generation setup

2. **Presentation (6-8 minutes + 4-6 minutes Q&A):**
   - Numerical results for P(0,T), f(0,T), θ(T)
   - Option price ZBC(5, 10, e^{-0.1})
   - Sensitivity ∂_σ ZBC vs. finite difference validation
   - Code explanation (1-2 minutes)
   - Performance analysis (execution time, convergence)

3. **Slides (3-6 slides for results):**
   - Plots of P(0,T) and f(0,T) curves
   - Calibrated θ(t) vs. theoretical
   - Option price with confidence interval
   - Sensitivity comparison (pathwise vs. finite difference)
   - Execution time analysis

## Submission

**Deadline:** December 17th, 2025 (email to instructor)

**Presentation:** December 18th, 2025 (12 minutes total)
