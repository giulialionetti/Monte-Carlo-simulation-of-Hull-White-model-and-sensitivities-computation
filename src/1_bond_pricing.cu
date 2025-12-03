/*
 *  This part of the code computes P(0,T) and f(0,T) for T in [0, 10] using Monte Carlo simulation.
 * 
 *   the following optimizations have been implemented:
 *   - Antithetic variates (variance reduction)
 *   - Precomputed drift table (constant memory)
 *   - Shared memory reduction
 *   - Fast math compiler flag


 * Output:
 * - P(0,T) for T in [0, 10] years (saved to data/P.bin)
 * - f(0,T) for T in [0, 10] years (saved to data/f.bin)
 */


#include "common.cuh"
#include "output.cuh"

/**
 * Simulate zero-coupon bond prices using Hull-White model with antithetic variates.
 * 
 * For each path, we simulate two trajectories using G and -G (antithetic pair)
 * to reduce variance. The short rate evolution follows:
 * 
 *   r(t+dt) = r(t)e^(-a*dt) + drift_integral + sigma * √[(1-e^(-2a*dt))/(2a)] * G
 * 
 * Bond prices are computed via: P(0,T) = E[exp(-\int_0^T r(s)ds)]
 * 
 * The integral (\int_0^T r(s)ds) is approximated using the trapezoidal rule.
 * Results are accumulated in shared memory before writing to global memory
 * to reduce atomic operation overhead.
 * 
 * @param P_sum Global memory array to accumulate bond price sums [N_MAT]
 * @param states cuRAND states for random number generation [N_PATHS]
 */

__global__ void simulate_zcb(float* P_sum, curandState* states) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    __shared__ float s_P_sum[N_MAT];
    if (threadIdx.x < N_MAT) {
        s_P_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float p0_warp = 2.0f * 32;
        
        if (lane == 0) {
            atomicAdd(&s_P_sum[0], p0_warp);
        }

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;
        const float exp_adt = d_exp_adt;      // Cache in register
        const float sig_st = d_sig_st;        // Cache in register

        for (int i = 1; i <= N_STEPS; i++) {
            float drift = d_drift_table[i - 1];
            float G = curand_normal(&local);
            const float sig_G = sig_st * G;

            evolve_hull_white_step(
                &r1, &integral1, drift, 
                sig_G, exp_adt, d_dt
            );
            evolve_hull_white_step(
                &r2, &integral2, drift, 
                -sig_G, exp_adt, d_dt
            );

            if (i % SAVE_STRIDE == 0) {
                int m = i / SAVE_STRIDE;
                if (m < N_MAT) {
                    float p0_m = expf(-integral1) + expf(-integral2);
                    warp_reduce(p0_m);
                    if (lane == 0) {
                        atomicAdd(&s_P_sum[m], p0_m);
                    }
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
 * Bond prices: P(0,T) = P_sum[T] / (2 * N_PATHS)
 * Forward rates: f(0,T) = -\partial ln(P(0,T)) / \partial T
 * 
 * Forward rates are computed using finite differences:
 * - Central difference for interior points: [ln P(T+dT) - ln P(T-dT)] / (2dT)
 * - Forward difference at T=0: [ln P(dT) - ln P(0)] / dT
 * - Backward difference at T=T_max: [ln P(T_max) - ln P(T_max-dT)] / dT
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
    int n_paths, // Number of paths (2*N_PATHS for antithetic)
    float inv_dT // (1 / Maturity spacing)
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
        int first_idx = (m == 0) ? 0 : m - 1;
        int last_idx = (m == n_mat - 1) ? n_mat - 1 : m + 1;
        float scale = ((m == 0) || (m == n_mat - 1)) ? 1.0f : 0.5f;
        d_f[m] = -scale * inv_dT * (logf(s_P[last_idx]) - logf(s_P[first_idx]));
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
    printf("RNG initialized\n");
    
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
    printf("Simulation complete\n");
    
    // Compute averages and forward rates
    compute_average_and_forward<<<1, 128>>>(
        d_P, d_f, d_P_sum, N_MAT, 2 * N_PATHS, 1 / H_MAT_SPACING
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
    
    for (int i = 0; i < N_MAT; i += SAVE_STRIDE) {
        printf("%5.1f        %.6f       %7.4f%%\n", 
               i * H_MAT_SPACING, h_P[i], h_f[i] * 100.0f);
    }
   

    // Sanity checks
    printf("\n=== Sanity Checks ===\n");
    printf("P(0,0) = 1.0:      %.6f %s\n", h_P[0], 
           (h_P[0] > 0.99f && h_P[0] < 1.01f) ? "OK" : "ERROR");
    printf("P(0,10) ~ 0.87:    %.6f %s\n", h_P[100], 
           (h_P[100] > 0.3f && h_P[100] < 0.9f) ? "OK" : "ERROR");
    printf("f(0,0) ~ 1.2%%:     %.4f%% %s\n", h_f[0] * 100.0f, 
           (h_f[0] > 0.01f && h_f[0] < 0.02f) ? "OK" : "ERROR");
    
    // Performance
    printf("\n=== Performance ===\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Effective paths: %d\n", N_PATHS * 2);
    printf("Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    
    summary_init("data/summary.txt");

    // Save results
    printf("\n=== Saving Results ===\n");
    save_array(P_FILE, h_P, N_MAT);
    save_array(F_FILE, h_f, N_MAT);
    printf("Results saved for Q2/Q3\n");

    // Write JSON output
    FILE* json = json_open("data/q1_results.json", "Q1: Zero-Coupon Bond Pricing");
    if (json) {
        json_write_array(json, "P", h_P, N_MAT);
        fprintf(json, ",\n");
        json_write_array(json, "f", h_f, N_MAT);
        fprintf(json, ",\n");
        json_write_performance(json, sim_ms, N_PATHS * 2);
        fprintf(json, ",\n");
        
        fprintf(json, "  \"validation\": {\n");
        fprintf(json, "    \"P_0_0\": %.8f,\n", h_P[0]);
        fprintf(json, "    \"P_0_10\": %.8f,\n", h_P[100]);
        fprintf(json, "    \"f_0_0\": %.8f\n", h_f[0]);
        fprintf(json, "  }\n");
        
        json_close(json);
        printf("Saved data/q1_results.json\n");
    }
    
    // Write CSV for plotting
    csv_write_timeseries("data/P_curve.csv", "P(0 T)", h_P, N_MAT, H_MAT_SPACING);
    csv_write_timeseries("data/f_curve.csv", "f(0 T)", h_f, N_MAT, H_MAT_SPACING);
    
    // Append to summary
    summary_append("data/summary.txt", "Q1: ZERO-COUPON BOND PRICING");
    FILE* sum = fopen("data/summary.txt", "a");
    fprintf(sum, "\nKey Results:\n");
    fprintf(sum, "  P(0,0) = %.8f (expected: 1.0)\n", h_P[0]);
    fprintf(sum, "  P(0,10) = %.8f\n", h_P[100]);
    fprintf(sum, "  f(0,0) = %.4f%% (expected: ~1.2%%)\n", h_f[0] * 100.0f);
    fprintf(sum, "\nPerformance:\n");
    fprintf(sum, "  Simulation time: %.2f ms\n", sim_ms);
    fprintf(sum, "  Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    fclose(sum);
    
    
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