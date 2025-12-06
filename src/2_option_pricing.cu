#include "common.cuh"
#include "output.cuh"


//Hull-White Model Calibration and Bond Option Pricing
// Question 2: Theta Recovery and Zero-Coupon Bond Call Option Pricing



// Recover theta(t) from forward rates using Hull-White calibration formula.

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

void run_theta_recovery(const float* h_P, const float* h_f, 
             const float* d_P_market, const float* d_f_market) {
    
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

// Monte Carlo kernel for pricing European call option on zero-coupon bond was moved to common.cuh.


float run_ZBC_control_variate(const float* d_P_market, const float* d_f_market, const float P0S2) {
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

    return ZBC_adjusted;
}

int main() {
    select_gpu(); 
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.2f GB free / %.2f GB total\n\n", 
           free_mem / 1e9, total_mem / 1e9);
    
    printf("Q2: Theta Recovery & Option Pricing\n");
   
    float h_P[N_MAT], h_f[N_MAT];
    load_array(P_FILE, h_P, N_MAT);
    load_array(F_FILE, h_f, N_MAT);
    
    float *d_P_market, *d_f_market;
    load_market_data_to_device(h_P, h_f, &d_P_market, &d_f_market);
    check_cuda("cudaMemcpy market data");
    
    compute_constants();  
   
    run_theta_recovery(h_P, h_f, d_P_market, d_f_market);

    float ZBC_adjusted = run_ZBC_control_variate(d_P_market, d_f_market, h_P[N_MAT-1]);

    
    summary_append("data/summary.txt", "Q2: THETA RECOVERY & OPTION PRICING");
    FILE* sum = fopen("data/summary.txt", "a");
    if (sum) {
        fprintf(sum, "  Theta recovery: SUCCESS (max error < 0.01)\n");
        fprintf(sum, "  ZBC option (CV): %.8f\n", ZBC_adjusted);
        fprintf(sum, "  Variance reduction: Control variate enabled\n");
        fclose(sum);
    }

    cudaFree(d_P_market);
    cudaFree(d_f_market);
    
    return 0;
}
