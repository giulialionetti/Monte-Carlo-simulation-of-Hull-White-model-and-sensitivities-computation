#include "common.cuh"


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
 * @param spacing Grid spacing ΔT
 * @return df/dT at index i
 */

float compute_derivative(const float* f, int i, int n, float spacing) {
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
 * Formula (equation 10): θ(T) = df/dT + af(0,T) + σ²/(2a)(1 - e^(-2aT))
 * 
 * This inverts the relationship between theta and forward rates, allowing
 * calibration to market data. We verify that our Monte Carlo f(0,T) correctly
 * recovers the piecewise linear theta from equation (7).
 * 
 * @param f Array of forward rates f(0,T) from Monte Carlo
 * @param theta_recovered Output array for recovered theta values
 * @param n_mat Number of maturity points
 */

void recover_theta(const float* f, float* theta_recovered, int n_mat) {
    for (int i = 0; i < n_mat; i++) {
        float T = i * H_MAT_SPACING;
        float df_dT = compute_derivative(f, i, n_mat, H_MAT_SPACING);
        float convexity = (H_SIGMA * H_SIGMA / (2.0f * H_A)) * 
                         (1.0f - expf(-2.0f * H_A * T));
        theta_recovered[i] = df_dT + H_A * f[i] + convexity;
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
    
    for (int i = 0; i <= 100; i += 10) {
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
    printf("\nRecovery: %s\n", max_error < 0.01f ? "SUCCESS ✓" : "FAILED ✗");
}

void run_q2a(const float* h_P, const float* h_f) {
    printf("\n=== Q2a: THETA RECOVERY ===\n\n");
    
    float h_theta_recovered[N_MAT];
    float h_theta_original[N_MAT];
    
    recover_theta(h_f, h_theta_recovered, N_MAT);
    
    for (int i = 0; i < N_MAT; i++) {
        float T = i * H_MAT_SPACING;
        h_theta_original[i] = theta_func(T);
    }
    
    print_theta_comparison(h_theta_original, h_theta_recovered, N_MAT);
}

/**
 * Monte Carlo kernel for pricing European call option on zero-coupon bond.
 * 
 * Option payoff: max(P(S₁,S₂) - K, 0) at expiry S₁
 * Bond matures at S₂, strike is K
 * 
 * Value: ZBC(S₁,S₂,K) = E[e^(-∫₀^S₁ r(s)ds) × (P(S₁,S₂) - K)⁺]
 * 
 * Algorithm:
 * 1. Simulate short rate r(t) from 0 to S₁ using Hull-White dynamics
 * 2. Compute discount factor exp(-∫₀^S₁ r(s)ds) via trapezoidal rule
 * 3. Evaluate P(S₁,S₂) using analytical Hull-White formula (no further simulation needed)
 * 4. Compute discounted payoff: discount × max(P(S₁,S₂) - K, 0)
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

void run_q2b(const float* h_P, const float* h_f) {
    printf("\n\n=== Q2b: ZERO COUPON BOND CALL OPTION ===\n");
    
    float S1 = 5.0f;
    float S2 = 10.0f;
    float K = expf(-0.1f);
    
    printf("\nParameters:\n");
    printf("  S1 (option maturity) = %.1f years\n", S1);
    printf("  S2 (bond maturity) = %.1f years\n", S2);
    printf("  K (strike) = %.6f\n", K);
    printf("  N_PATHS = %d (x2 antithetic = %d effective)\n\n", N_PATHS, N_PATHS * 2);
    
    float *d_ZBC_sum, *d_P_market, *d_f_market;
    float h_ZBC;
    curandState *d_states;
    
    cudaMalloc(&d_ZBC_sum, sizeof(float));
    cudaMalloc(&d_P_market, N_MAT * sizeof(float));
    cudaMalloc(&d_f_market, N_MAT * sizeof(float));
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    check_cuda("cudaMalloc Q2b");
    
    cudaMemcpy(d_P_market, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_market, h_f, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda("cudaMemcpy market data");
    
    cudaMemset(d_ZBC_sum, 0, sizeof(float));
    
    printf("Initializing RNG...\n");
    init_rng<<<NB, NTPB>>>(d_states, time(NULL) + 54321);
    check_cuda("init_rng Q2b");
    cudaDeviceSynchronize();
    printf("RNG initialized ✓\n");
    
    printf("Running Monte Carlo simulation...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_ZBC<<<NB, NTPB>>>(d_ZBC_sum, d_states, S1, S2, K, d_P_market, d_f_market);
    cudaEventRecord(stop);
    check_cuda("simulate_ZBC");
    cudaDeviceSynchronize();
    
    float sim_ms;
    cudaEventElapsedTime(&sim_ms, start, stop);
    printf("Simulation complete ✓\n");
    
    cudaMemcpy(&h_ZBC, d_ZBC_sum, sizeof(float), cudaMemcpyDeviceToHost);
    h_ZBC /= (2.0f * N_PATHS);
    
    printf("\n=== RESULTS ===\n");
    printf("ZBC(5, 10, e^-0.1) = %.8f\n", h_ZBC);
    
    printf("\n=== Performance ===\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    
    cudaFree(d_ZBC_sum);
    cudaFree(d_P_market);
    cudaFree(d_f_market);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
    
    compute_constants();  
    run_q2a(h_P, h_f);
    run_q2b(h_P, h_f);
    
    return 0;
}