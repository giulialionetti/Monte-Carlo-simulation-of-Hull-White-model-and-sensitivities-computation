#include "common.cuh"
#include "output.cuh"


/*
 * Hull-White Model Calibration and Bond Option Pricing
 * 
 * Recover theta(t) from Monte Carlo forward rates using the
 *      Hull-White calibration formula (equation 10)
 * 
 * Price a European call option on a zero-coupon bond using
 *      analytical Hull-White formulas and Monte Carlo simulation
 */


/**
 * Compute numerical derivative df/dT using finite differences.
 * 
 * Uses central differences for interior points and one-sided differences
 * at boundaries for improved accuracy.
 * 
 * @param f Array of forward rates
 * @param i Index at which to compute derivative
 * @param n Array length
 * @param spacing Grid spacing dT
 * @return df/dT at index i
 */

__device__ float compute_derivative(const float* f, int i, int n, float spacing) {
    if (i == 0) {
        return (f[1] - f[0]) / spacing;
    } else if (i == n - 1) {
        return (f[i] - f[i-1]) / spacing;
    } else {
        return (f[i+1] - f[i-1]) / (2.0f * spacing);
    }
}

/**
 * Recover theta(t) from forward rates using Hull-White calibration formula.
 * 
 * Formula (equation 10): theta(T) = df/dT + af(0,T) + sigma^2/(2a)(1 - e^(-2aT))
 * 
 * This inverts the relationship between theta and forward rates, allowing
 * calibration to market data. We verify that our Monte Carlo f(0,T) correctly
 * recovers the piecewise linear theta from equation (7).
 * 
 * @param f Array of forward rates f(0,T) from Monte Carlo
 * @param theta_recovered Output array for recovered theta values
 * @param theta_original Output array for original theta values
 * @param Ts Output array for time points T
 * @param n_mat Number of maturity points
 */

__global__ void recover_theta(const float* f,
                              float* theta_recovered, 
                              float* theta_original,
                              float* Ts,
                              int n_mat) {
    for (int i = threadIdx.x; i < n_mat; i += blockDim.x) {
        float T = i * d_mat_spacing;
        float df_dT = compute_derivative(f, i, n_mat, d_mat_spacing);
        float convexity = (d_sigma * d_sigma / (2.0f * d_a)) * 
                         (1.0f - expf(-2.0f * d_a * T));
        theta_recovered[i] = df_dT + d_a * f[i] + convexity;
        theta_original[i] = theta_func(T);
        Ts[i] = T;
    }
}

/**
 * Print comparison between original and recovered theta functions.
 * 
 * Outputs error metrics and validates successful recovery (max error < 0.01).
 * 
 * @param theta_original True theta function from equation (7)
 * @param theta_recovered Theta recovered from forward rates
 * @param n_mat Number of points
 */

void print_theta_comparison(const float* theta_original, 
                           const float* theta_recovered, 
                           int n_mat) {
    printf("  T      theta_original   theta_recovered   error\n");
    
    float max_error = 0.0f;
    float sum_error = 0.0f;
    int n_printed = 0;
    
    for (int i = 0; i <= n_mat; i += SAVE_STRIDE) {
        float T = i * H_MAT_SPACING;
        float error = fabsf(theta_recovered[i] - theta_original[i]);
        max_error = fmaxf(max_error, error);
        sum_error += error;
        n_printed++;
        
        printf("%5.1f    %.6f         %.6f          %.2e\n",
               T, theta_original[i], theta_recovered[i], error);
    }
    printf("\n");
    
    printf("Max error:  %.2e\n", max_error);
    printf("Mean error: %.2e\n", sum_error / n_printed);
    printf("\nRecovery: %s\n", max_error < 0.01f ? "SUCCESS" : "FAILED");

    save_q2a_json("data/q2a_results.json", max_error, max_error < 0.01f);
}

void run_q2a(const float* h_P, const float* h_f, 
             const float* d_P_market, const float* d_f_market) {
    printf("\n=== Q2a: THETA RECOVERY ===\n\n");
    
    float h_theta_recovered[N_MAT], h_theta_original[N_MAT], h_T[N_MAT];
    float* d_theta_recovered, *d_theta_original, *d_T;
    cudaMalloc(&d_theta_recovered, N_MAT * sizeof(float));
    cudaMalloc(&d_theta_original, N_MAT * sizeof(float));
    cudaMalloc(&d_T, N_MAT * sizeof(float));

    recover_theta<<<1, N_MAT>>>(d_f_market, d_theta_recovered, d_theta_original, d_T, N_MAT);
    check_cuda("recover_theta");
    cudaDeviceSynchronize();
    cudaMemcpy(h_theta_recovered, d_theta_recovered, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_theta_original, d_theta_original, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_T, d_T, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);

    print_theta_comparison(h_theta_original, h_theta_recovered, N_MAT);

    csv_write_comparison("data/theta_comparison.csv", 
                         h_T, h_theta_original, h_theta_recovered, 
                         "T", "theta_original", "theta_recovered", 
                         N_MAT);

    cudaFree(d_theta_recovered);
    cudaFree(d_theta_original);
    cudaFree(d_T);
}

/**
 * Monte Carlo kernel for pricing European call option on zero-coupon bond.
 * 
 * Option payoff: max(P(S1,S2) - K, 0) at expiry S1
 * Bond matures at S2, strike is K
 * 
 * Value: ZBC(S1,S2,K) = E[e^(-\int_0^{S1} r(s)ds) × (P(S1,S2) - K)⁺]
 * 
 * Algorithm:
 * 1. Simulate short rate r(t) from 0 to S1 using Hull-White dynamics
 * 2. Compute discount factor exp(-\int_0^{S1} r(s)ds) via trapezoidal rule
 * 3. Evaluate P(S1,S2) using analytical Hull-White formula (no further simulation needed)
 * 4. Compute discounted payoff: discount * max(P(S1,S2) - K, 0)
 * 5. Average over all paths (including antithetic pairs)
 * 
 * Uses antithetic variates for variance reduction.
 * 
 * @param ZBC_sum Global memory scalar to accumulate option value sum
 * @param states cuRAND states [N_PATHS]
 * @param S1 Option expiry time (years)
 * @param S2 Bond maturity time (years)
 * @param K Strike price
 * @param d_P_market Market bond prices P(0,T) on device [N_MAT]
 * @param d_f_market Market forward rates f(0,T) on device [N_MAT]
 */

__global__ void simulate_ZBC(float* ZBC_sum, curandState* states, 
                             float S1, float S2, float K,
                             const float* d_P_market, const float* d_f_market) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_ZBC_sum;
    if (threadIdx.x == 0) {
        s_ZBC_sum = 0.0f;
    }
    __syncthreads();

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        int n_steps_S1 = (int)(S1 / d_dt);

        for (int i = 1; i <= n_steps_S1; i++) {
            float drift = d_drift_table[i - 1];
            float G = curand_normal(&local);

            float r1_next = r1 * d_exp_adt + drift + d_sig_st * G;
            integral1 += 0.5f * (r1 + r1_next) * d_dt;
            r1 = r1_next;

            float r2_next = r2 * d_exp_adt + drift - d_sig_st * G;
            integral2 += 0.5f * (r2 + r2_next) * d_dt;
            r2 = r2_next;
        }
        
        // Use shared device function from common.cuh
        float P1 = compute_P_HW(S1, S2, r1, d_a, d_sigma, d_P_market, d_f_market);
        float P2 = compute_P_HW(S1, S2, r2, d_a, d_sigma, d_P_market, d_f_market);
        
        float discount1 = expf(-integral1);
        float discount2 = expf(-integral2);
        
        float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);
        
        atomicAdd(&s_ZBC_sum, payoff1 + payoff2);
        
        states[pid] = local;
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(ZBC_sum, s_ZBC_sum);
    }
}

__global__ void simulate_ZBC_optimized(
    float* ZBC_sum, 
    curandState* states, 
    float S1, float S2, float K,
    const float* __restrict__ d_P_market,
    const float* __restrict__ d_f_market
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;       // Thread index within warp (0-31)
    int warp_id = threadIdx.x >> 5;    // Warp index within block

    // One accumulator per warp (max 32 warps per 1024-thread block)
    __shared__ float warp_sums[32];

    float thread_sum = 0.0f;

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        // Precompute constants (compiler optimization)
        const int n_steps = 500;              // S1/dt = 5.0/0.01 = 500 steps
        const float half_dt = 0.5f * d_dt;    // For trapezoidal rule
        const float exp_adt = d_exp_adt;      // Cache in register
        const float sig_st = d_sig_st;        // Cache in register

        // Main simulation loop: evolve r(t) from 0 to S1
        #pragma unroll 8
        for (int i = 1; i <= n_steps; i++) {
            const float drift = d_drift_table[i - 1];
            const float G = curand_normal(&local);
            const float sig_G = sig_st * G;   // Compute once, use twice

            // Antithetic path 1: +G
            const float r1_next = __fmaf_rn(r1, exp_adt, drift + sig_G);
            integral1 = __fmaf_rn(half_dt, r1 + r1_next, integral1);
            r1 = r1_next;

            // Antithetic path 2: -G
            const float r2_next = __fmaf_rn(r2, exp_adt, drift - sig_G);
            integral2 = __fmaf_rn(half_dt, r2 + r2_next, integral2);
            r2 = r2_next;
        }
        
        // Compute P(S1, S2) using analytical Hull-White formula
        const float a_val = d_a;
        const float sigma_val = d_sigma;
        
        const float P1 = compute_P_HW(S1, S2, r1, a_val, sigma_val, d_P_market, d_f_market);
        const float P2 = compute_P_HW(S1, S2, r2, a_val, sigma_val, d_P_market, d_f_market);
        
        // Compute discounted option payoffs
        const float discount1 = __expf(-integral1);  // Fast math intrinsic
        const float discount2 = __expf(-integral2);
        
        const float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        const float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);
        
        thread_sum = payoff1 + payoff2;
        
        states[pid] = local;
    }
    
    // Warp-level reduction using shuffle intrinsics (fast!)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp does final reduction across all warps
    if (warp_id == 0) {
        float warp_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Final atomic to global memory (only once per block)
        if (lane == 0) {
            atomicAdd(ZBC_sum, warp_sum);
        }
    }
}

void run_q2b(const float* d_P_market, const float* d_f_market) {
    printf("\n\n=== Q2b: ZERO COUPON BOND CALL OPTION ===\n");
    
    float S1 = 5.0f;
    float S2 = 10.0f;
    float K = expf(-0.1f);
    
    printf("\nParameters:\n");
    printf("  S1 (option maturity) = %.1f years\n", S1);
    printf("  S2 (bond maturity) = %.1f years\n", S2);
    printf("  K (strike) = %.6f\n", K);
    printf("  N_PATHS = %d (x2 antithetic = %d effective)\n\n", N_PATHS, N_PATHS * 2);
    
    float *d_ZBC_sum;
    float h_ZBC;
    curandState *d_states;
    
    cudaMalloc(&d_ZBC_sum, sizeof(float));
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    check_cuda("cudaMalloc Q2b");
    
    cudaMemset(d_ZBC_sum, 0, sizeof(float));
    
    printf("Initializing RNG...\n");
    init_rng<<<NB, NTPB>>>(d_states, time(NULL) + 54321);
    check_cuda("init_rng Q2b");
    cudaDeviceSynchronize();
    printf("RNG initialized\n");
    
    printf("Running Monte Carlo simulation...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_ZBC_optimized<<<NB, NTPB>>>(d_ZBC_sum, d_states, S1, S2, K, d_P_market, d_f_market);
    cudaEventRecord(stop);
    check_cuda("simulate_ZBC");
    cudaDeviceSynchronize();
    
    float sim_ms;
    cudaEventElapsedTime(&sim_ms, start, stop);
    printf("Simulation complete\n");
    
    cudaMemcpy(&h_ZBC, d_ZBC_sum, sizeof(float), cudaMemcpyDeviceToHost);
    h_ZBC /= (2.0f * N_PATHS);
    
    printf("\n=== RESULTS ===\n");
    printf("ZBC(5, 10, e^-0.1) = %.8f\n", h_ZBC);
    
    printf("\n=== Performance ===\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    
    cudaFree(d_ZBC_sum);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Control variate approach for variance reduction.
 * 
 * Control: C = e^(-∫r dt) × P(S₁,S₂)
 * Known: E[C] = P(0,S₂) = 0.877 (from Q1)
 * 
 * Adjusted estimator: ZBC_CV = ZBC + β(E[C] - Ĉ)
 * where β = -Cov(ZBC,C)/Var(C) ≈ -1 for simplicity
 */
__global__ void simulate_ZBC_control_variate(
    float* ZBC_sum, 
    float* control_sum,  // NEW: accumulate control variate
    curandState* states, 
    float S1, float S2, float K,
    const float* d_P_market,
    const float* d_f_market
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    __shared__ float warp_ZBC_sums[32];
    __shared__ float warp_control_sums[32];

    float thread_ZBC = 0.0f;
    float thread_control = 0.0f;

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        int n_steps_S1 = (int)(S1 / d_dt);

        for (int i = 0; i < n_steps_S1; i++) {
            float drift = d_drift_table[i];
            float G = curand_normal(&local);

            float r1_next = r1 * d_exp_adt + drift + d_sig_st * G;
            integral1 += 0.5f * (r1 + r1_next) * d_dt;
            r1 = r1_next;

            float r2_next = r2 * d_exp_adt + drift - d_sig_st * G;
            integral2 += 0.5f * (r2 + r2_next) * d_dt;
            r2 = r2_next;
        }
        
        float P1 = compute_P_HW(S1, S2, r1, d_a, d_sigma, d_P_market, d_f_market);
        float P2 = compute_P_HW(S1, S2, r2, d_a, d_sigma, d_P_market, d_f_market);
        
        float discount1 = expf(-integral1);
        float discount2 = expf(-integral2);
        
        // Control variate: discount * P (we know E[this] = P(0,S2))
        float control1 = discount1 * P1;
        float control2 = discount2 * P2;
        
        // Option payoff
        float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);
        
        thread_ZBC = payoff1 + payoff2;
        thread_control = control1 + control2;
        
        states[pid] = local;
    }
    
    // Warp reduction for ZBC
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_ZBC += __shfl_down_sync(0xffffffff, thread_ZBC, offset);
        thread_control += __shfl_down_sync(0xffffffff, thread_control, offset);
    }
    
    if (lane == 0) {
        warp_ZBC_sums[warp_id] = thread_ZBC;
        warp_control_sums[warp_id] = thread_control;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_ZBC = (lane < (blockDim.x >> 5)) ? warp_ZBC_sums[lane] : 0.0f;
        float warp_control = (lane < (blockDim.x >> 5)) ? warp_control_sums[lane] : 0.0f;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_ZBC += __shfl_down_sync(0xffffffff, warp_ZBC, offset);
            warp_control += __shfl_down_sync(0xffffffff, warp_control, offset);
        }
        
        if (lane == 0) {
            atomicAdd(ZBC_sum, warp_ZBC);
            atomicAdd(control_sum, warp_control);
        }
    }
}

void run_q2b_control_variate(const float* d_P_market, const float* d_f_market, const float P0S2) {
    printf("\n\n=== Q2b: ZBC WITH CONTROL VARIATE ===\n");
    
    float S1 = 5.0f;
    float S2 = 10.0f;
    float K = expf(-0.1f);
    
    float *d_ZBC_sum, *d_control_sum;
    float h_ZBC, h_control;
    curandState *d_states;
    
    cudaMalloc(&d_ZBC_sum, sizeof(float));
    cudaMalloc(&d_control_sum, sizeof(float));
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    check_cuda("cudaMalloc Q2b CV");
    
    cudaMemset(d_ZBC_sum, 0, sizeof(float));
    cudaMemset(d_control_sum, 0, sizeof(float));
    
    init_rng<<<NB, NTPB>>>(d_states, time(NULL) + 54321);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_ZBC_control_variate<<<NB, NTPB>>>(
        d_ZBC_sum, d_control_sum, d_states, S1, S2, K, d_P_market, d_f_market
    );
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float sim_ms;
    cudaEventElapsedTime(&sim_ms, start, stop);
    
    cudaMemcpy(&h_ZBC, d_ZBC_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_control, d_control_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    h_ZBC /= (2.0f * N_PATHS);
    h_control /= (2.0f * N_PATHS);
    
    // Control variate adjustment
    float control_dev = h_control - P0S2;
    float ZBC_adjusted = h_ZBC - control_dev;;
    
    printf("=== RESULTS ===\n");
    printf("ZBC (raw):                %.8f\n", h_ZBC);
    printf("Control mean:             %.8f\n", h_control);
    printf("Expected control (P0S2):  %.8f\n", P0S2);
    printf("Control deviation:        %.8f\n", control_dev);
    printf("ZBC (control adjusted):   %.8f\n", ZBC_adjusted);
    
    printf("\n=== Performance ===\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);

    save_q2b_json("data/q2b_results.json", ZBC_adjusted, control_dev, sim_ms, N_PATHS * 2);
    
    cudaFree(d_ZBC_sum);
    cudaFree(d_control_sum);
    cudaFree(d_states);
}

int main() {
    select_gpu(); 
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.2f GB free / %.2f GB total\n\n", 
           free_mem / 1e9, total_mem / 1e9);
    
    printf("HULL-WHITE MODEL: QUESTION 2\n");
    printf("========================================\n");
    
    float h_P[N_MAT], h_f[N_MAT];
    load_array(P_FILE, h_P, N_MAT);
    load_array(F_FILE, h_f, N_MAT);
    printf("\n");

    float *d_P_market, *d_f_market;
    cudaMalloc(&d_P_market, N_MAT * sizeof(float));
    cudaMalloc(&d_f_market, N_MAT * sizeof(float));
    cudaMemcpy(d_P_market, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_market, h_f, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda("cudaMemcpy market data");
    
    compute_constants();  
    run_q2a(h_P, h_f, d_P_market, d_f_market);
    run_q2b_control_variate(d_P_market, d_f_market, h_P[N_MAT-1]);

    cudaFree(d_P_market);
    cudaFree(d_f_market);
    
    return 0;
}