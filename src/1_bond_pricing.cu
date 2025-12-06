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

// note: the kernels were moved to market_data.cuh to be available globally.

#include "common.cuh"
#include "output.cuh"
#include "market_data.cuh"


int main() {
   
    printf("ZERO COUPON BOND PRICING\n");
   
    
    printf("Parameters:\n");
    printf("  N_PATHS = %d (x2 antithetic = %d effective)\n", N_PATHS, N_PATHS * 2);
    printf("  N_STEPS = %d, N_MAT = %d, T = %.1f years\n", N_STEPS, N_MAT, T_FINAL);
    printf("  a = %.2f, sigma = %.2f, r0 = %.4f\n\n", H_A, H_SIGMA, H_R0);

    
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
    
   
    printf("Initializing RNG...\n");
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    check_cuda("init_rng");
    cudaDeviceSynchronize();
    printf("RNG initialized\n");
    
   
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
    
   
    compute_average_and_forward<<<1, 128>>>(
        d_P, d_f, d_P_sum, N_MAT, 2 * N_PATHS, 1 / H_MAT_SPACING
    );
    check_cuda("compute_average_and_forward");
    cudaDeviceSynchronize();
    
   
    cudaMemcpy(h_P, d_P, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f, d_f, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    
  
    printf("\n"); 
   
    printf("RESULTS\n");
    printf("T (years)    P(0,T)         f(0,T)\n");
    
    for (int i = 0; i < N_MAT; i += SAVE_STRIDE) {
        printf("%5.1f        %.6f       %7.4f%%\n", 
               i * H_MAT_SPACING, h_P[i], h_f[i] * 100.0f);
    }
   

   
    printf("\nChecks\n");
    printf("P(0,0) = 1.0:      %.6f %s\n", h_P[0], 
           (h_P[0] > 0.99f && h_P[0] < 1.01f) ? "OK" : "ERROR");
    printf("P(0,10) ~ 0.87:    %.6f %s\n", h_P[100], 
           (h_P[100] > 0.3f && h_P[100] < 0.9f) ? "OK" : "ERROR");
    printf("f(0,0) ~ 1.2%%:     %.4f%% %s\n", h_f[0] * 100.0f, 
           (h_f[0] > 0.01f && h_f[0] < 0.02f) ? "OK" : "ERROR");
    
  
    printf("\nPerformance\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Effective paths: %d\n", N_PATHS * 2);
    printf("Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    
    summary_init("data/summary.txt");

    
    printf("\nSaving Results\n");
    save_array(P_FILE, h_P, N_MAT);
    save_array(F_FILE, h_f, N_MAT);
    printf("Results saved.\n");

    
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
    
   
    csv_write_timeseries("data/P_curve.csv", "P(0 T)", h_P, N_MAT, H_MAT_SPACING);
    csv_write_timeseries("data/f_curve.csv", "f(0 T)", h_f, N_MAT, H_MAT_SPACING);
    
  
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
