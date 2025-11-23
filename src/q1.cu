/*
 *  this part of the code computes P(0,T) and f(0,T) for T ∈ [0, 10] using Monte Carlo simulation.
 * 
 *   the following optimizations have been implemented:
 *   - Antithetic variates (variance reduction)
 *   - Precomputed drift table (constant memory)
 *   - Shared memory reduction
 *   - Fast math compiler flag


 * Output:
 * - P(0,T) for T ∈ [0, 10] years (saved to data/P.bin)
 * - f(0,T) for T ∈ [0, 10] years (saved to data/f.bin)
 */


#include "common.cuh"

/**
 * Simulate zero-coupon bond prices using Hull-White model with antithetic variates.
 * 
 * For each path, we simulate two trajectories using G and -G (antithetic pair)
 * to reduce variance. The short rate evolution follows:
 * 
 *   r(t+Δt) = r(t)e^(-aΔt) + drift_integral + σ√[(1-e^(-2aΔt))/(2a)] × G
 * 
 * Bond prices are computed via: P(0,T) = E[exp(-∫₀ᵀ r(s)ds)]
 * 
 * The integral ∫r(s)ds is approximated using the trapezoidal rule.
 * Results are accumulated in shared memory before writing to global memory
 * to reduce atomic operation overhead.
 * 
 * @param P_sum Global memory array to accumulate bond price sums [N_MAT]
 * @param states cuRAND states for random number generation [N_PATHS]
 */

__global__ void simulate_zcb(float* P_sum, curandState* states) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_P_sum[N_MAT];
    if (threadIdx.x < N_MAT) {
        s_P_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (pid < N_PATHS) {
        curandState local = states[pid];

        atomicAdd(&s_P_sum[0], 2.0f);

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        for (int i = 1; i <= N_STEPS; i++) {
            float drift = d_drift_table[i - 1];
            float G = curand_normal(&local);

            float r1_next = r1 * d_exp_adt + drift + d_sig_st * G;
            integral1 += 0.5f * (r1 + r1_next) * d_dt;
            r1 = r1_next;

            float r2_next = r2 * d_exp_adt + drift - d_sig_st * G;
            integral2 += 0.5f * (r2 + r2_next) * d_dt;
            r2 = r2_next;

            if (i % SAVE_STRIDE == 0) {
                int m = i / SAVE_STRIDE;
                if (m < N_MAT) {
                    atomicAdd(&s_P_sum[m], expf(-integral1) + expf(-integral2));
                }
            }
        }
        
        states[pid] = local;
    }
    
    __syncthreads();
    if (threadIdx.x < N_MAT) {
        atomicAdd(&P_sum[threadIdx.x], s_P_sum[threadIdx.x]);
    }
}

/**
 * Compute zero-coupon bond prices and forward rates from simulation results.
 * 
 * Bond prices: P(0,T) = P_sum[T] / (2 × N_PATHS)
 * Forward rates: f(0,T) = -∂ln(P(0,T))/∂T
 * 
 * Forward rates are computed using finite differences:
 * - Central difference for interior points: [ln P(T+ΔT) - ln P(T-ΔT)] / (2ΔT)
 * - Forward difference at T=0: [ln P(ΔT) - ln P(0)] / ΔT
 * - Backward difference at T=T_max: [ln P(T_max) - ln P(T_max-ΔT)] / ΔT
 * 
 * @param d_P Output array of averaged bond prices [N_MAT]
 * @param d_f Output array of forward rates [N_MAT]
 * @param d_P_sum Input array of accumulated sums from simulation [N_MAT]
 * @param n_mat Number of maturity points
 * @param n_paths Total number of paths (including antithetic)
 * @param dT Maturity grid spacing
 */


__global__ void compute_average_and_forward(
    float* d_P,  // Output: averaged bond prices
    float* d_f,  // Output: forward rates
     float* d_P_sum, // Input: raw sums from simulation
    int n_mat, // Number of maturities
     int n_paths, // Number of paths (2×N_PATHS for antithetic)
      float dT // Maturity spacing
) {
    __shared__ float s_P[N_MAT];
    int m = threadIdx.x;
    
     // Compute averages
    if (m < n_mat) {
        float avg = d_P_sum[m] / (float)n_paths;
        s_P[m] = avg;
        d_P[m] = avg;
    }
    __syncthreads();
    
    //Compute forward rates via finite differences
    if (m < n_mat) {
        if (m == 0) {
             // Forward difference at left boundary
            d_f[m] = -(logf(s_P[1]) - logf(s_P[0])) / dT;
        } else if (m == n_mat - 1) {
             // Backward difference at right boundary
            d_f[m] = -(logf(s_P[m]) - logf(s_P[m-1])) / dT;
        } else {
             // Central difference for interior points
            d_f[m] = -(logf(s_P[m+1]) - logf(s_P[m-1])) / (2.0f * dT);
        }
    }
}


int main() {
   
    printf("ZERO COUPON BOND PRICING\n");
   
    
    printf("Parameters:\n");
    printf("  N_PATHS = %d (x2 antithetic = %d effective)\n", N_PATHS, N_PATHS * 2);
    printf("  N_STEPS = %d, N_MAT = %d, T = %.1f years\n", N_STEPS, N_MAT, T_FINAL);
    printf("  a = %.2f, sigma = %.2f, r0 = %.4f\n\n", H_A, H_SIGMA, H_R0);

    // Allocate memory
    float *d_P_sum, *d_P, *d_f;
    float *h_P, *h_f;
    curandState *d_states;
    
    cudaMalloc(&d_P_sum, N_MAT * sizeof(float));
    cudaMalloc(&d_P, N_MAT * sizeof(float));
    cudaMalloc(&d_f, N_MAT * sizeof(float));
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    check_cuda("cudaMalloc");
    
    h_P = (float*)malloc(N_MAT * sizeof(float));
    h_f = (float*)malloc(N_MAT * sizeof(float));
    
    cudaMemset(d_P_sum, 0, N_MAT * sizeof(float));
    
    compute_constants();
    
    // Initialize RNG
    printf("Initializing RNG...\n");
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    check_cuda("init_rng");
    cudaDeviceSynchronize();
    printf("RNG initialized ✓\n");
    
    // Run simulation
    printf("Running Monte Carlo simulation...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_zcb<<<NB, NTPB>>>(d_P_sum, d_states);
    cudaEventRecord(stop);
    check_cuda("simulate_zcb");
    cudaDeviceSynchronize();
    
    float sim_ms;
    cudaEventElapsedTime(&sim_ms, start, stop);
    printf("Simulation complete ✓\n");
    
    // Compute averages and forward rates
    compute_average_and_forward<<<1, 128>>>(
        d_P, d_f, d_P_sum, N_MAT, 2 * N_PATHS, H_MAT_SPACING
    );
    check_cuda("compute_average_and_forward");
    cudaDeviceSynchronize();
    
    // Copy results to host
    cudaMemcpy(h_P, d_P, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f, d_f, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("\n"); 
   
    printf("RESULTS\n");
    printf("T (years)    P(0,T)         f(0,T)\n");
    
    for (int i = 0; i <= 100; i += 10) {
        printf("%5.1f        %.6f       %7.4f%%\n", 
               i * H_MAT_SPACING, h_P[i], h_f[i] * 100.0f);
    }
   

    // Sanity checks
    printf("\n=== Sanity Checks ===\n");
    printf("P(0,0) = 1.0:      %.6f %s\n", h_P[0], 
           (h_P[0] > 0.99f && h_P[0] < 1.01f) ? "✓" : "✗");
    printf("P(0,10) ~ 0.87:    %.6f %s\n", h_P[100], 
           (h_P[100] > 0.3f && h_P[100] < 0.9f) ? "✓" : "✗");
    printf("f(0,0) ~ 1.2%%:     %.4f%% %s\n", h_f[0] * 100.0f, 
           (h_f[0] > 0.01f && h_f[0] < 0.02f) ? "✓" : "✗");
    
    // Performance
    printf("\n=== Performance ===\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Effective paths: %d\n", N_PATHS * 2);
    printf("Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    
    // Save results
    printf("\n=== Saving Results ===\n");
    save_array(P_FILE, h_P, N_MAT);
    save_array(F_FILE, h_f, N_MAT);
    printf("Results saved for Q2/Q3 ✓\n");
    
    // Cleanup
    free(h_P);
    free(h_f);
    cudaFree(d_P_sum);
    cudaFree(d_P);
    cudaFree(d_f);
    cudaFree(d_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}