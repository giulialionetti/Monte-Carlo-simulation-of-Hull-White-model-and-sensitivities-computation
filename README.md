# Monte Carlo Simulation of Hull-White Model and Sensitivities Computation

## Project Overview

This project implements Monte Carlo simulation for pricing and sensitivity analysis of fixed income instruments under the Hull-White one-factor short-rate model using CUDA.

## Mathematical Framework

### Hull-White One-Factor Model

The short-rate model is defined by the stochastic differential equation:

$$dr(t) = [\theta(t) - ar(t)]dt + \sigma dW_t$$

**Parameters:**
- $r(0) = 0.012$ (initial short rate)
- $a = 1$ (mean reversion speed)
- $\sigma = 0.1$ (volatility)
- $\theta(t)$ (piecewise linear function defined on $[0, 10]$):

$$\theta(t) = \begin{cases}
0.012 + 0.0014t & \text{for } 0 \leq t < 5 \\
0.019 + 0.001(t-5) & \text{for } 5 \leq t \leq 10
\end{cases}$$

### Simulation Formula

The short rate $r(t)$ at time $t$, given its value $r_s$ at time $s < t$, follows a Gaussian distribution:

$$r(t) = m_{s,t} + \Sigma_{s,t} G$$

where $G \sim \mathcal{N}(0,1)$ and:

$$m_{s,t} = r(s)e^{-a(t-s)} + \int_s^t e^{-a(t-u)}\theta(u)du$$

$$\Sigma_{s,t} = \sqrt{\frac{\sigma^2(1 - e^{-2a(t-s)})}{2a}}$$

### Zero Coupon Bond Pricing

The zero coupon bond price represents the amount to pay at time $t$ to receive 1 at maturity $T$:

$$P(t,T) = \mathbb{E}_t\left[\exp\left(-\int_t^T r_s ds\right)\right]$$

**Forward Rate:**

$$f(0,T) = \frac{\partial \ln(P(0,T))}{\partial T}$$

**Analytical Expression (Hull-White):**

$$P(t,T) = A(t,T) \exp(-B(t,T)r(t))$$

where:

$$B(t,T) = \frac{1 - e^{-a(T-t)}}{a}$$

$$A(t,T) = \frac{P(0,T)}{P(0,t)} \exp\left[B(t,T)f(0,t) - \frac{\sigma^2(1-e^{-2aT})}{4a}B(t,T)^2\right]$$

### Calibration Formula

In practice, $P(0,T)$ values are market-quoted, and $\theta$ is recovered using:

$$\theta(T) = \frac{\partial f(0,T)}{\partial T} + af(0,T) + \frac{\sigma^2(1-e^{-2aT})}{2a}$$

### Zero Coupon Bond Call Option

European call option on a zero coupon bond:

$$\text{ZBC}(S_1, S_2, K) = \mathbb{E}\left[e^{-\int_0^{S_1} r_s ds}(P(S_1,S_2) - K)^+\right]$$

where $(x)^+ = \max(x, 0)$ and $0 < S_1 < S_2 \leq 10$.

### Sensitivity (Greeks)

Derivative with respect to volatility $\sigma$:

$$\frac{\partial}{\partial\sigma}\text{ZBC}(S_1,S_2,K) = \mathbb{E}\left[\frac{\partial P(S_1,S_2)}{\partial\sigma} e^{-\int_0^{S_1} r_s ds}\mathbf{1}_{P(S_1,S_2)>K} - \left[\int_0^{S_1} \frac{\partial r_s}{\partial\sigma} ds\right]e^{-\int_0^{S_1} r_s ds}(P(S_1,S_2)-K)^+\right]$$

where $\frac{\partial r(t)}{\partial\sigma}$ follows the induction:

$$\frac{\partial r(t)}{\partial\sigma} = M_{s,t} + \frac{\Sigma_{s,t}}{\sigma}G$$

$$M_{s,t} = \frac{\partial r(s)}{\partial\sigma}e^{-a(t-s)} + \frac{2\sigma e^{-at}[\cosh(at) - \cosh(as)]}{a^2}$$

$$\frac{\partial r(0)}{\partial\sigma} = 0$$

## Project Tasks

### Question 1: Monte Carlo for Zero Coupon Bonds (8 points)

**Objective:** Compute Monte Carlo estimates of $P(0,T)$ and $f(0,T)$ for $T \in [0,10]$

**Requirements:**
- Implement CUDA kernel for short-rate simulation using equation (8)
- Use trapezoidal rule on uniform time grid to approximate $\int_0^T r_s ds$
- Compute both bond prices $P(0,T)$ and forward rates $f(0,T)$
- Generate results for a range of maturities

**Key Implementation Details:**
- Discretize time interval $[0,10]$ uniformly
- For each Monte Carlo path:
  - Simulate $r(t)$ trajectory using Gaussian increments
  - Accumulate integral using trapezoidal rule
  - Compute $\exp\left(-\int_0^T r_s ds\right)$
- Average over all paths to get $P(0,T)$
- Compute forward rate using finite differences: $f(0,T) \approx \frac{\partial \ln(P(0,T))}{\partial T}$

### Question 2: Calibration and Option Pricing

#### 2a) Calibration 

**Objective:** Recover the piecewise linear $\theta(t)$ from equation (7) using market data

**Requirements:**
- Use $P(0,T)$ and $f(0,T)$ values from Question 1
- Implement CUDA kernel to apply calibration formula (10)
- Verify recovered $\theta(t)$ matches the specified piecewise linear form

**Implementation:**
- Compute $\frac{\partial f(0,T)}{\partial T}$ numerically from grid of $f$ values
- Apply formula: $\theta(T) = \frac{\partial f(0,T)}{\partial T} + af(0,T) + \frac{\sigma^2(1-e^{-2aT})}{2a}$

#### 2b) Option Pricing 

**Objective:** Price the zero coupon bond call option $\text{ZBC}(5, 10, e^{-0.1})$

**Requirements:**
- Use analytical formula for $P(t,T)$
- Simulate $r(t)$ trajectories from $t=0$ to $t=5$
- Approximate integral $\int_0^5 r_s ds$ using trapezoidal rule
- Compute option payoff: $e^{-\int_0^5 r_s ds} \cdot (P(5,10) - e^{-0.1})^+$
- Average over Monte Carlo paths

### Question 3: Sensitivity Analysis 
**Objective:** Compute $\frac{\partial}{\partial\sigma}\text{ZBC}(5, 10, e^{-0.1})$ and validate with finite differences

**Requirements:**
- Implement pathwise derivative method
- Simulate both $r(t)$ and $\frac{\partial r(t)}{\partial\sigma}$ along the same paths using same random numbers
- Compute sensitivity using the analytical derivative formula
- **Validation:** Compare with finite difference approximation:
  
  $$\frac{\text{ZBC}(\sigma + \varepsilon) - \text{ZBC}(\sigma - \varepsilon)}{2\varepsilon} \quad \text{for small } \varepsilon$$

**Implementation Details:**
- Initialize $\frac{\partial r(0)}{\partial\sigma} = 0$
- At each time step, update both $r(t)$ and $\frac{\partial r(t)}{\partial\sigma}$ using the same $G$
- Compute $\frac{\partial P(S_1,S_2)}{\partial\sigma}$ using chain rule on analytical formula
- Accumulate $\int_0^{S_1} \frac{\partial r_s}{\partial\sigma} ds$ using trapezoidal rule

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

2. **Presentation:**
   - Numerical results for $P(0,T)$, $f(0,T)$, $\theta(T)$
   - Option price $\text{ZBC}(5, 10, e^{-0.1})$
   - Sensitivity $\frac{\partial}{\partial\sigma}\text{ZBC}$ vs. finite difference validation
   - Code explanation (1-2 minutes)
   - Performance analysis (execution time, convergence)

3. **Slides:**
   - Plots of $P(0,T)$ and $f(0,T)$ curves
   - Calibrated $\theta(t)$ vs. theoretical
   - Option price with confidence interval
   - Sensitivity comparison (pathwise vs. finite difference)
   - Execution time analysis

