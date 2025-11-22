
/*
 *  this part of the code computes P(0,T) and f(0,T) for T ∈ [0, 10] using Monte Carlo simulation.
 * 
 *   the following optimizations have been implemented:
 *   - Antithetic variates (variance reduction)
 *   - Precomputed drift table (constant memory)
 *   - Shared memory reduction
 *   - Fast math compiler flag
 */


#include "common.cuh"

/* 
 * Constant memory for device access
 * 
 * These values are computed once on host and copied to GPU constant memory
 * for fast, cached access by all threads.
 *  
 */
__constant__ float d_a;
__constant__ float d_sigma;
__constant__ float d_r0;
__constant__ float d_dt;
__constant__ float d_exp_adt;
__constant__ float d_sig_st;
__constant__ float d_one_minus_exp_adt_over_a;
__constant__ float d_one_minus_exp_adt_over_a_sq;
__constant__ float d_drift_table[N_STEPS];


/*
 * Initialize constant memory with precomputed values
 * 
 * The Hull-White transition density is:
 *   r(t+Δt) = m_{s,t} + Σ_{s,t}·G,  where G ~ N(0,1)
 * 
 * m_{s,t} = r(s)·e^{-aΔt} + ∫ₛᵗ e^{-a(t-u)}θ(u)du
 * Σ_{s,t} = σ·√[(1-e^{-2aΔt})/(2a)]
 * 
 * We precompute the drift integral for each time step to avoid
 * redundant computation across all paths.
  */
void compute_constants() {
    float h_exp_adt = expf(-H_A * H_DT);
    float h_sig_st = H_SIGMA * sqrtf((1.0f - expf(-2.0f * H_A * H_DT)) / (2.0f * H_A));
    float h_one_minus_exp_adt_over_a = (1.0f - h_exp_adt) / H_A;
    float h_one_minus_exp_adt_over_a_sq = h_one_minus_exp_adt_over_a / H_A;

    cudaMemcpyToSymbol(d_a, &H_A, sizeof(float));
    cudaMemcpyToSymbol(d_sigma, &H_SIGMA, sizeof(float));
    cudaMemcpyToSymbol(d_r0, &H_R0, sizeof(float));
    cudaMemcpyToSymbol(d_dt, &H_DT, sizeof(float));
    cudaMemcpyToSymbol(d_exp_adt, &h_exp_adt, sizeof(float));
    cudaMemcpyToSymbol(d_sig_st, &h_sig_st, sizeof(float));
    cudaMemcpyToSymbol(d_one_minus_exp_adt_over_a, &h_one_minus_exp_adt_over_a, sizeof(float));
    cudaMemcpyToSymbol(d_one_minus_exp_adt_over_a_sq, &h_one_minus_exp_adt_over_a_sq, sizeof(float));

    // Precompute drift integral table
    float h_drift[N_STEPS];
    for (int i = 0; i < N_STEPS; i++) {
        float t = i * H_DT;
        float first_term = ((t + H_DT) - h_exp_adt * t) / H_A - h_one_minus_exp_adt_over_a_sq;
        h_drift[i] = (t < 5.0f) ? 
            (0.0014f * first_term + 0.012f * h_one_minus_exp_adt_over_a) :
            (0.001f * first_term + 0.014f * h_one_minus_exp_adt_over_a);
    }
    cudaMemcpyToSymbol(d_drift_table, h_drift, N_STEPS * sizeof(float));
}

/* 
 * Monte Carlo simulation kernel with antithetic variates
 * 
 * For each random number G, we simulate two paths:
 *   Path 1: uses +G
 *   Path 2: uses -G (antithetic)
 * 
 * These paths are negatively correlated, reducing variance when averaged.
 * The integral ∫r(s)ds is computed using the trapezoidal rule.
 * 
 * Shared memory is used to accumulate partial sums within each block
 * before writing to global memory, reducing atomic contention.
 *  */
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

/* 
 * Compute averages and forward rates in a single kernel
 * 
 * P(0,T) = P_sum[T] / n_paths
 * f(0,T) = -∂ln(P)/∂T ≈ -(ln P[T+1] - ln P[T-1]) / (2·ΔT)
 * 
 * Central differences are used for interior points.
 * Forward/backward differences are used at boundaries.
 * */
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