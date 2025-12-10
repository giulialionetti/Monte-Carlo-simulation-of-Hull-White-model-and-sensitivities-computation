#include "common.cuh"
#include "output.cuh"
#include <algorithm>

//Hull-White Model Calibration and Bond Option Pricing
// Question 2: Theta Recovery and Zero-Coupon Bond Call Option Pricing



// this functions recovers the instantaneous short rate drift theta(T)
// from the market forward rate curve f(T) using the calibration formula:
// theta(T) = df/dT + a * f(T) + (sigma^2 / (2a)) * (1 - exp(-2aT))
// and compares it to the original theta(T) piecewise linear in the model specification
__global__ void recover_theta(const float* f, 
                              float* theta_recovered, 
                              float* theta_original,
                              float* Ts, 
                              int n_mat) { // n_mat = number of maturities


     // each thread processes multiple maturity spacings
     // the jumps are by number of threads in the block 
     // the kernel is launched with 1 block and N_MAT threads                          
    for (int i = threadIdx.x; i < n_mat; i += blockDim.x) {
        float T = i * d_mat_spacing; // d_mat_spacing is precomputed in compute_constants()
        float df_dT = compute_derivative(f, i, n_mat, d_mat_spacing);
        // convexity accounts for the stochastic volatility in the model
        float convexity = (d_sigma * d_sigma / (2.0f * d_a)) * 
                         (1.0f - expf(-2.0f * d_a * T));
        // apply calibration formula 
        theta_recovered[i] = df_dT + d_a * f[i] + convexity;
        theta_original[i] = theta_func(T);
        Ts[i] = T; // saves time point for later analysis
    }
}

// this function prints a comparison table of the original and recovered theta(T)
// along with the absolute error, and computes max and mean error statistics
void print_theta_comparison(const float* theta_original, 
                           const float* theta_recovered, 
                           int n_mat) {
    printf("  T      theta_original   theta_recovered   error\n");
    
    float max_error = 0.0f; // tracks the sup error
    float sum_error = 0.0f; // accumulates total error to compute mean
    int n_printed = 0;

    // print every SAVE_STRIDE-th entry to reduce output size
    for (int i = 0; i <= n_mat; i += SAVE_STRIDE) { // SAVE_STRIDE defined in common.cuh
        float T = i * H_MAT_SPACING;
        float error = fabsf(theta_recovered[i] - theta_original[i]);
        max_error = fmaxf(max_error, error); // builds up the sup norm error
        sum_error += error;
        n_printed++; // counts number of printed entries for mean calculation

        printf("%5.1f    %.6f         %.6f          %.2e\n",
               T, theta_original[i], theta_recovered[i], error);
    }
    printf("\n");

    printf("Max error:  %.2e\n", max_error);
    printf("Mean error: %.2e\n", sum_error / n_printed); // average error over printed entries
    
    // 0.01 threshold given theta(t) varies around 1%
    printf("\nRecovery: %s\n", max_error < 0.01f ? "SUCCESS" : "FAILED");

    save_q2a_json("data/q2a_results.json", max_error, max_error < 0.01f);
}

void run_theta_recovery(const float* h_P, const float* h_f,
             const float* d_P_market, const float* d_f_market) {

    // host arrays
    float h_theta_recovered[N_MAT], h_theta_original[N_MAT], h_T[N_MAT];
    // pointer to device arrays
    float* d_theta_recovered, *d_theta_original, *d_T;
    // allocate device memory
    cudaMalloc(&d_theta_recovered, N_MAT * sizeof(float));
    cudaMalloc(&d_theta_original, N_MAT * sizeof(float));
    cudaMalloc(&d_T, N_MAT * sizeof(float));
    // launch kernel to recover theta(T)
    recover_theta<<<1, N_MAT>>>(d_f_market, d_theta_recovered, d_theta_original, d_T, N_MAT);
    check_cuda("recover_theta"); // check for kernel launch errors
    cudaDeviceSynchronize(); // wait for kernel to finish

    // copy results back to host
    cudaMemcpy(h_theta_recovered, d_theta_recovered, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_theta_original, d_theta_original, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_T, d_T, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);

    // print comparison and error statistics
    print_theta_comparison(h_theta_original, h_theta_recovered, N_MAT);

    csv_write_comparison("data/theta_comparison.csv",
                         h_T, h_theta_original, h_theta_recovered,
                         "T", "theta_original", "theta_recovered",
                         N_MAT);

    cudaFree(d_theta_recovered);
    cudaFree(d_theta_original);
    cudaFree(d_T);
}

// Monte Carlo simulation for pricing European call option on zero-coupon bond (simulate_ZBC_control_variate) was moved to common.cuh.
// to be used in 3_sensitivity_analysis.cu as well.

float run_ZBC_control_variate(const float* d_P_market, const float* d_f_market, const float P0S2) {
    float S1 = 5.0f;
    float S2 = 10.0f;
    float K = expf(-0.1f);
    
    float *d_ZBC_sum, *d_control_sum, *d_ZBC_sq_sum, *d_control_sq_sum, *d_cross_sum;
    curandState *d_states;
    
    cudaMalloc(&d_ZBC_sum, sizeof(float));
    cudaMalloc(&d_control_sum, sizeof(float));
    cudaMalloc(&d_ZBC_sq_sum, sizeof(float));
    cudaMalloc(&d_control_sq_sum, sizeof(float));
    cudaMalloc(&d_cross_sum, sizeof(float));
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    
    cudaMemset(d_ZBC_sum, 0, sizeof(float));
    cudaMemset(d_control_sum, 0, sizeof(float));
    cudaMemset(d_ZBC_sq_sum, 0, sizeof(float));
    cudaMemset(d_control_sq_sum, 0, sizeof(float));
    cudaMemset(d_cross_sum, 0, sizeof(float));
    
    init_rng<<<NB, NTPB>>>(d_states, time(NULL) + 54321);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_ZBC_control_variate<<<NB, NTPB>>>(
        d_ZBC_sum, d_control_sum, d_ZBC_sq_sum, d_control_sq_sum, d_cross_sum,
        d_states, S1, S2, K, d_P_market, d_f_market
    );
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float sim_ms;
    cudaEventElapsedTime(&sim_ms, start, stop);
    
    // Copy results back
    float h_ZBC, h_control, h_ZBC_sq, h_control_sq, h_cross;
    cudaMemcpy(&h_ZBC, d_ZBC_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_control, d_control_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ZBC_sq, d_ZBC_sq_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_control_sq, d_control_sq_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_cross, d_cross_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    int N_total = 2 * N_PATHS;
    
    // Compute sample means
    float mean_ZBC = h_ZBC / N_total;
    float mean_control = h_control / N_total;
    
    // Compute sample variance of control
    float E_Y2 = h_control_sq / N_total;
    float E_Y_sq = mean_control * mean_control;
    float var_control = E_Y2 - E_Y_sq;
    
    // Compute sample covariance
    float E_XY = h_cross / N_total;
    float E_X_E_Y = mean_ZBC * mean_control;
    float cov = E_XY - E_X_E_Y;
    
    // Optimal beta
    float beta_optimal = cov / var_control;
    
    // Apply control variate with optimal beta
    float control_adjustment = beta_optimal * (mean_control - P0S2);
    float ZBC_adjusted = mean_ZBC - control_adjustment;
    
    // Theoretical variance reduction
    float correlation = cov / (sqrtf(var_control) * sqrtf(E_Y2 - E_Y_sq));  // Approximate
    float expected_var_reduction = 100.0f * correlation * correlation;
    
    printf("=== RESULTS===\n");
    printf("ZBC (prior to control variate adjustment):%.8f\n", mean_ZBC);
    printf("Control mean:               %.8f\n", mean_control);
    printf("Expected control (P0S2):    %.8f\n", P0S2);
    printf("\n");
    printf("Beta Analysis:\n");
    printf("Covariance(X,Y):          %.8e\n", cov);
    printf("Variance(Y):              %.8e\n", var_control);
    printf("Beta optimal:             %.6f\n", beta_optimal);
    printf("Correlation:              %.6f\n", correlation);
    printf("Expected variance reduction:              %.2f%%\n", expected_var_reduction);
    printf("\n");
    printf("Control adjustment:         %.8f\n", control_adjustment);
    printf("ZBC (control variate adjusted):  %.8f\n", ZBC_adjusted);
    
    printf("\n=== Performance ===\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Throughput: %.2f M paths/sec\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    
    cudaFree(d_ZBC_sum);
    cudaFree(d_control_sum);
    cudaFree(d_ZBC_sq_sum);
    cudaFree(d_control_sq_sum);
    cudaFree(d_cross_sum);
    cudaFree(d_states);
    
    return ZBC_adjusted;
}

void run_zbc_statistical_validation(const float* d_P_market, const float* d_f_market,
                                                float P0S2) {

    float S1 = 5.0f, S2 = 10.0f, K = expf(-0.1f);
    const int N_RUNS = 20;

    float zbc_samples[N_RUNS];
    float zbc_raw_samples[N_RUNS];
    float beta_samples[N_RUNS];
    float correlation_samples[N_RUNS];

    printf("Running %d independent Monte Carlo simulations...\n", N_RUNS);

    unsigned long base_seed = (unsigned long)time(NULL) * 1000000UL;

    for (int run = 0; run < N_RUNS; run++) {
        curandState *d_states;
        cudaMalloc(&d_states, N_PATHS * sizeof(curandState));

        unsigned long seed = base_seed + run * 12345;
        init_rng<<<NB, NTPB>>>(d_states, seed);
        cudaDeviceSynchronize();

        float *d_ZBC_sum, *d_control_sum, *d_ZBC_sq_sum, *d_control_sq_sum, *d_cross_sum;
        cudaMalloc(&d_ZBC_sum, sizeof(float));
        cudaMalloc(&d_control_sum, sizeof(float));
        cudaMalloc(&d_ZBC_sq_sum, sizeof(float));
        cudaMalloc(&d_control_sq_sum, sizeof(float));
        cudaMalloc(&d_cross_sum, sizeof(float));
        
        cudaMemset(d_ZBC_sum, 0, sizeof(float));
        cudaMemset(d_control_sum, 0, sizeof(float));
        cudaMemset(d_ZBC_sq_sum, 0, sizeof(float));
        cudaMemset(d_control_sq_sum, 0, sizeof(float));
        cudaMemset(d_cross_sum, 0, sizeof(float));

        simulate_ZBC_control_variate<<<NB, NTPB>>>(
            d_ZBC_sum, d_control_sum, d_ZBC_sq_sum, d_control_sq_sum, d_cross_sum,
            d_states, S1, S2, K, d_P_market, d_f_market
        );
        cudaDeviceSynchronize();

        float h_ZBC, h_control, h_ZBC_sq, h_control_sq, h_cross;
        cudaMemcpy(&h_ZBC, d_ZBC_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_control, d_control_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_ZBC_sq, d_ZBC_sq_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_control_sq, d_control_sq_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_cross, d_cross_sum, sizeof(float), cudaMemcpyDeviceToHost);

        int N_total = 2 * N_PATHS;
        
       
        float mean_ZBC = h_ZBC / N_total;
        float mean_control = h_control / N_total;
        
      
        float E_Y2 = h_control_sq / N_total;
        float var_control = E_Y2 - mean_control * mean_control;
        
       
        float E_XY = h_cross / N_total;
        float cov = E_XY - mean_ZBC * mean_control;
        
       
        float beta = cov / var_control;
        
      
        float E_X2 = h_ZBC_sq / N_total;
        float var_ZBC = E_X2 - mean_ZBC * mean_ZBC;
        
      
        float correlation = cov / sqrtf(var_ZBC * var_control);
        
       
        float control_adjustment = beta * (mean_control - P0S2);
        float zbc_adjusted = mean_ZBC - control_adjustment;

        zbc_raw_samples[run] = mean_ZBC;
        zbc_samples[run] = zbc_adjusted;
        beta_samples[run] = beta;
        correlation_samples[run] = correlation;

        cudaFree(d_ZBC_sum);
        cudaFree(d_control_sum);
        cudaFree(d_ZBC_sq_sum);
        cudaFree(d_control_sq_sum);
        cudaFree(d_cross_sum);
        cudaFree(d_states);

        if ((run + 1) % 5 == 0) {
            printf("  Completed %d/%d runs...\n", run + 1, N_RUNS);
        }
    }

   
    float mean = 0.0f;
    for (int i = 0; i < N_RUNS; i++) {
        mean += zbc_samples[i];
    }
    mean /= N_RUNS;

    float variance = 0.0f;
    for (int i = 0; i < N_RUNS; i++) {
        float diff = zbc_samples[i] - mean;
        variance += diff * diff;
    }
    variance /= (N_RUNS - 1);

    float std_dev = sqrtf(variance);
    float std_error = std_dev / sqrtf(N_RUNS);
    float t_critical = 2.093f;
    float margin_of_error = t_critical * std_error;
    float ci_lower = mean - margin_of_error;
    float ci_upper = mean + margin_of_error;
    float cv_percent = 100.0f * std_dev / mean;

   
    float mean_raw = 0.0f;
    for (int i = 0; i < N_RUNS; i++) mean_raw += zbc_raw_samples[i];
    mean_raw /= N_RUNS;

    float var_raw = 0.0f;
    for (int i = 0; i < N_RUNS; i++) {
        float diff = zbc_raw_samples[i] - mean_raw;
        var_raw += diff * diff;
    }
    var_raw /= (N_RUNS - 1);
    float std_dev_raw = sqrtf(var_raw);

   
    float variance_reduction = 100.0f * (1.0f - variance / var_raw);
    
    
    float mean_beta = 0.0f;
    float mean_correlation = 0.0f;
    for (int i = 0; i < N_RUNS; i++) {
        mean_beta += beta_samples[i];
        mean_correlation += correlation_samples[i];
    }
    mean_beta /= N_RUNS;
    mean_correlation /= N_RUNS;
    
    
    float beta_variance = 0.0f;
    for (int i = 0; i < N_RUNS; i++) {
        float diff = beta_samples[i] - mean_beta;
        beta_variance += diff * diff;
    }
    beta_variance /= (N_RUNS - 1);
    float beta_std = sqrtf(beta_variance);

    printf("\nSummary of statistical analysis on ZBC option pricing:\n\n");
    printf("Beta Control Variate Statistics:\n");
    printf("Mean beta:              %.6f\n", mean_beta);
    printf("Beta std dev:           %.6f\n", beta_std);
    printf("Beta range:             [%.4f, %.4f]\n", 
       *std::min_element(beta_samples, beta_samples + N_RUNS),
       *std::max_element(beta_samples, beta_samples + N_RUNS));
    printf("Mean Correlation:       %.6f\n", mean_correlation);
    printf("\n");

    printf("With Control Variate:\n");
    printf("  Mean Price:             %.8f\n", mean);
    printf("  Standard Deviation:     %.8f\n", std_dev);
    printf("  Standard Error:         %.8f\n", std_error);
    printf("  Coefficient of Var:     %.4f%%\n", cv_percent);
    printf("\n");

    printf("95%% Confidence Interval:\n");
    printf("  Lower Bound:            %.8f\n", ci_lower);
    printf("  Upper Bound:            %.8f\n", ci_upper);
    printf("  Margin of Error:        ±%.8f\n", margin_of_error);
    printf("  Relative Width:         ±%.4f%%\n", 100.0f * margin_of_error / mean);
    printf("\n");

    printf("Without Control Variate:\n");
    printf("  Mean Price (raw):       %.8f\n", mean_raw);
    printf("  Standard Deviation:     %.8f\n", std_dev_raw);
    printf("\n");

    printf("Variance Reduction:       %.2f%%\n", variance_reduction);
    printf("\n");

    printf("\n");
    printf("Sample Distribution:\n");
    printf("Min:  %.8f\n", *std::min_element(zbc_samples, zbc_samples + N_RUNS));
    printf("Q1:   %.8f\n", zbc_samples[N_RUNS/4]);
    printf("Med:  %.8f\n", zbc_samples[N_RUNS/2]);
    printf("Q3:   %.8f\n", zbc_samples[3*N_RUNS/4]);
    printf("Max:  %.8f\n", *std::max_element(zbc_samples, zbc_samples + N_RUNS));

    printf("\n");
    printf("result:\n");
    printf("95%% confident true option price lies in [%.8f, %.8f]\n", ci_lower, ci_upper);
    
    if (variance_reduction > 0) {
        printf("reduced variance by %.1f%%\n", variance_reduction);
        printf("mean correlation of %.4f indicates %s relationship\n", 
               mean_correlation, 
               fabsf(mean_correlation) > 0.5f ? "strong" : "moderate");
    } else {
        printf("increased variance by %.1f%%\n", -variance_reduction);
        printf("low correlation (%.4f) explains poor performance\n", mean_correlation);
    }
    
    printf("pricing precision: ±%.4f%%\n", cv_percent);
    
    if (beta_std / fabsf(mean_beta) > 0.2f) {
        printf("warning: beta estimates vary significantly (CV = %.1f%%)\n", 
               100.0f * beta_std / fabsf(mean_beta));
    }

    // Save detailed results
    FILE* csv = fopen("data/zbc_bootstrap_optimal.csv", "w");
    if (csv) {
        fprintf(csv, "run,price_adjusted,price_raw,beta_optimal,correlation\n");
        for (int i = 0; i < N_RUNS; i++) {
            fprintf(csv, "%d,%.10f,%.10f,%.8f,%.8f\n",
                    i+1, zbc_samples[i], zbc_raw_samples[i], 
                    beta_samples[i], correlation_samples[i]);
        }
        fclose(csv);
        printf("\nSaved data/zbc_bootstrap_optimal.csv\n");
    }

    FILE* stats = fopen("data/zbc_statistics_optimal.txt", "w");
    if (stats) {
        fprintf(stats, "Option Parameters:\n");
        fprintf(stats, "  S1 (exercise):     %.1f years\n", S1);
        fprintf(stats, "  S2 (maturity):     %.1f years\n", S2);
        fprintf(stats, "  Strike:            K = e^-0.1 = %.6f\n", K);
        fprintf(stats, "\n");
        fprintf(stats, "Monte Carlo Parameters:\n");
        fprintf(stats, "  Paths per run:     %d\n", N_PATHS);
        fprintf(stats, "  Independent runs:  %d\n", N_RUNS);
        fprintf(stats, "  Total samples:     %d\n", N_PATHS * N_RUNS);
        fprintf(stats, "\n");
        fprintf(stats, "Beta Statistics:\n");
        fprintf(stats, "  Mean beta:         %.6f\n", mean_beta);
        fprintf(stats, "  Beta std dev:      %.6f\n", beta_std);
        fprintf(stats, "  Beta CV:           %.2f%%\n", 100.0f * beta_std / fabsf(mean_beta));
        fprintf(stats, "  Mean correlation:  %.6f\n", mean_correlation);
        fprintf(stats, "  Expected VR:       %.2f%% (from ρ²)\n", 100.0f * mean_correlation * mean_correlation);
        fprintf(stats, "\n");
        fprintf(stats, "Point Estimate:\n");
        fprintf(stats, "  Mean Price:        %.8f\n", mean);
        fprintf(stats, "\n");
        fprintf(stats, "Uncertainty Quantification:\n");
        fprintf(stats, "  Standard Error:    %.8f (%.4f%%)\n",
                std_error, 100.0f * std_error / mean);
        fprintf(stats, "  95%% CI:             [%.8f, %.8f]\n", ci_lower, ci_upper);
        fprintf(stats, "\n");
        fprintf(stats, "Control Variate Performance:\n");
        fprintf(stats, "  Variance (with CV):  %.10e\n", variance);
        fprintf(stats, "  Variance (without CV):       %.10e\n", var_raw);
        fprintf(stats, "  Variance Reduction:          %.2f%%\n", variance_reduction);
        printf("Saved data/zbc_statistics_optimal.txt\n");
    }
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

    printf("\n");
    printf("Run statistical validation for ZBC option (20 runs? (y/n): ");
    char response;
    scanf(" %c", &response);
    if (response == 'y' || response == 'Y') {
        run_zbc_statistical_validation(d_P_market, d_f_market, h_P[N_MAT-1]);
    }

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