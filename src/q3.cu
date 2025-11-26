#include "common.cuh"
#include "output.cuh"

/*
 * Hull-White Model: Sensitivity Analysis:
 * 1. Compute sensitivity of ZBC option w.r.t sigma
 * 2. Compare with Finite Difference approximation
 */

// Constant memory for the specific drift term in the sensitivity process
__constant__ float d_sigma_drift_table[N_STEPS]; 

/**
 * Precompute the drift increment for the sensitivity process sensitivity_r(t).
 * Based on PDF Equation (390): 
 * Increment = (2*sigma * e^(-at) * [cosh(at) - cosh(as)]) / a^2
 * where t = s + dt
 */
void compute_sigma_drift_table() {
    float h_sigma_drift[N_STEPS];
    
    for (int i = 0; i < N_STEPS; i++) {
        float s = i * H_DT;
        float t = (i + 1) * H_DT;
        
        // Term from eq 390 M_{s,t}
        // The process is: dr_sig(t) = dr_sig(s)*exp(-a*dt) + INCREMENT
        // INCREMENT = (2*sigma*e^(-at) * [cosh(at) - cosh(as)]) / a^2
        
        float term = (2.0f * H_SIGMA * expf(-H_A * t)) * (coshf(H_A * t) - coshf(H_A * s));
        h_sigma_drift[i] = term / (H_A * H_A);
    }
    
    cudaMemcpyToSymbol(d_sigma_drift_table, h_sigma_drift, N_STEPS * sizeof(float));
}

/**
 * Analytical derivative of Bond Price P(S1, S2) with respect to sigma: partial_sigma P(S1, S2)
 */
__device__ float compute_dP_dsigma(float S1, float S2, float P_S1_S2, float d_sigma_r_S1, float a, float sigma) {
    float B = (1.0f - expf(-a * (S2 - S1))) / a;
    float one_minus_exp = 1.0f - expf(-2.0f * a * S2);
    return - P_S1_S2 * B * (sigma / (2.0f * a) * one_minus_exp * B + d_sigma_r_S1);
}

/**
 * Kernel for Sensitivity (Monte Carlo).
 * * Simulates:
 * 1. r(t): Short rate
 * 2. partial_sigma r(t): Sensitivity process
 * * Computes Equation (388):
 * E [ dP_dsigma * discount * Ind(P>K)  -  (Integral(partial_sigma r(t))) * discount * (P-K)+ ]
 */
__global__ void simulate_sensitivity(
    float* sens_sum, 
    curandState* states, 
    float S1, float S2, float K,
    const float* __restrict__ d_P_market,
    const float* __restrict__ d_f_market
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    __shared__ float warp_sums[WARPS_PER_BLOCK];
    float thread_sum = 0.0f;

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r = d_r0;
        float d_sigma_r = 0.0f; // This is partial_sigma r(t), starts at 0
        
        float int_r = 0.0f; // Integral of r(s) ds
        float int_d_sigma_r = 0.0f; // Integral of partial_sigma r(s) ds

        int n_steps_S1 = (int)(S1 / d_dt);

        for (int i = 1; i <= n_steps_S1; i++) {
            float drift_r = d_drift_table[i - 1];
            float drift_d_sigma_r = d_sigma_drift_table[i - 1]; // The specific drift for sensitivity process
            
            float G = curand_normal(&local);

            float r_next = r * d_exp_adt + drift_r + d_sig_st * G; 
            float d_sigma_r_next = d_sigma_r * d_exp_adt + drift_d_sigma_r + (d_sig_st / d_sigma) * G;

            int_r += 0.5f * d_dt * (r + r_next);
            int_d_sigma_r += 0.5f * d_dt * (d_sigma_r + d_sigma_r_next);

            r = r_next;
            d_sigma_r = d_sigma_r_next;
        }

        // Calculate Prices at S1
        float P_val = compute_P_HW(S1, S2, r, d_a, d_sigma, d_P_market, d_f_market);
        
        // Calculate Discount Factor
        float discount = expf(-int_r);
        
        // Calculate Term 1: dP/dsigma * discount * Indicator(P > K)
        float term1 = (P_val > K) ?
                      compute_dP_dsigma(S1, S2, P_val, r, d_a, d_sigma) * discount
                      : 0.0f;

        // Calculate Term 2: (Integral(partial_sigma r(t))) * discount * (P - K)+
        float payoff = fmaxf(P_val - K, 0.0f);
        float term2 = int_d_sigma_r * discount * payoff;

        // Result = E[ Term1 - Term2 ]
        thread_sum = term1 - term2;
        
        states[pid] = local;
    }

    // Warp Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane == 0) warp_sums[warp_id] = thread_sum;
    __syncthreads();

    // Block Reduction
    if (warp_id == 0) {
        float warp_sum = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (lane == 0) atomicAdd(sens_sum, warp_sum);
    }
}

__global__ void simulate_ZBC_simple(
    float* ZBC_sum, curandState* states, 
    float S1, float S2, float K,
    const float* __restrict__ d_P_market, const float* __restrict__ d_f_market
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid < N_PATHS) {
        curandState local = states[pid]; // Use copies to not mess up main stream
        float r = d_r0;
        float integral = 0.0f;
        int n_steps = (int)(S1 / d_dt);

        for (int i = 0; i < n_steps; i++) {
            float drift = d_drift_table[i];
            float G = curand_normal(&local);
            float r_next = r * d_exp_adt + drift + d_sig_st * G;
            integral += 0.5f * d_dt * (r + r_next);
            r = r_next;
        }
        
        float P = compute_P_HW(S1, S2, r, d_a, d_sigma, d_P_market, d_f_market);
        float val = expf(-integral) * fmaxf(P - K, 0.0f);
        
        atomicAdd(ZBC_sum, val);
    }
}

float run_zbc_price(float S1, float S2, float K, 
                    const float* d_P_market, const float* d_f_market, 
                    curandState* d_states) {
    float* d_sum;
    float h_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    
    // We re-use d_states. Note: To be perfectly rigorous for FD, 
    // we should re-seed or reset states to ensure we use the SAME random numbers 
    // for both sigma and sigma+eps (Common Random Numbers), which reduces variance of difference.
    // However, since we modify the global constant d_sig_st inside the kernel logic implicitly
    // via compute_constants(), we must ensure the G values are identical.
    
    // We will re-initialize RNG with the SAME seed before every call to this function 
    // in the FD wrapper.

    simulate_ZBC_simple<<<NB, NTPB>>>(d_sum, d_states, S1, S2, K, d_P_market, d_f_market);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    
    return h_sum / N_PATHS;
}


void run_sensitivity_mc(const float* d_P_market, const float* d_f_market, 
                        curandState* d_states, float* h_result_sens) {
    printf("\n=== Running Pathwise Sensitivity (Monte Carlo) ===\n");
    
    float S1 = 5.0f;
    float S2 = 10.0f;
    float K = expf(-0.1f);
    
    float* d_sens_sum;
    cudaMalloc(&d_sens_sum, sizeof(float));
    cudaMemset(d_sens_sum, 0, sizeof(float));
    
    // Upload the specific drift table for the sensitivity process
    compute_sigma_drift_table();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_sensitivity<<<NB, NTPB>>>(d_sens_sum, d_states, S1, S2, K, d_P_market, d_f_market);
    cudaEventRecord(stop);
    check_cuda("simulate_sensitivity");
    cudaDeviceSynchronize();
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float sum;
    cudaMemcpy(&sum, d_sens_sum, sizeof(float), cudaMemcpyDeviceToHost);
    *h_result_sens = sum / N_PATHS;
    
    printf("Sensitivity (d_ZBC / d_sigma) = %.6f\n", *h_result_sens);
    printf("Computation time: %.2f ms\n", ms);
    
    cudaFree(d_sens_sum);
}

void run_finite_difference(const float* d_P_market, const float* d_f_market, 
                           curandState* d_states, float* h_result_fd) {
    printf("\n=== Running Finite Difference Check ===\n");
    
    float S1 = 5.0f;
    float S2 = 10.0f;
    float K = expf(-0.1f);
    float epsilon = 0.001f;
    float original_sigma = H_SIGMA;
    
    // 1. Price with original Sigma
    // Re-init RNG to ensure CRN (Common Random Numbers)
    init_rng<<<NB, NTPB>>>(d_states, 12345); 
    cudaDeviceSynchronize();
    float price_base = run_zbc_price(S1, S2, K, d_P_market, d_f_market, d_states);
    
    // 2. Price with Sigma + Epsilon
    // We must update the global constants on GPU!
    // This is hacky but necessary given the architecture of common.cuh
    
    // Temporarily cast away const-ness of the global H_SIGMA to change it 
    // (Note: H_SIGMA is const in common.cuh, but we need to change the value passed to compute_constants)
    // Actually, common.cuh uses H_SIGMA directly. We cannot change a const. 
    // We must manually upload the new values to the symbols.
    
    float new_sigma = original_sigma + epsilon;
    
    // Recompute derived constants on host
    float h_sig_st = new_sigma * sqrtf((1.0f - expf(-2.0f * H_A * H_DT)) / (2.0f * H_A));
    // Note: A, dt, r0 do not change. Only sigma and sig_st change.
    
    cudaMemcpyToSymbol(d_sigma, &new_sigma, sizeof(float));
    cudaMemcpyToSymbol(d_sig_st, &h_sig_st, sizeof(float));
    // Important: Drift table for r(t) does NOT depend on sigma in Hull-White?
    // Let's check compute_constants in common.cuh.
    // Yes, drift table depends on 'h_one_minus_exp_adt_over_a' and 'first_term'.
    // 'first_term' depends on 'h_one_minus_exp_adt_over_a_sq'.
    // These depend on A and DT. NOT on Sigma.
    // So d_drift_table remains valid!
    
    // Re-init RNG with SAME seed for CRN
    init_rng<<<NB, NTPB>>>(d_states, 12345);
    cudaDeviceSynchronize();
    float price_bump = run_zbc_price(S1, S2, K, d_P_market, d_f_market, d_states);
    
    *h_result_fd = (price_bump - price_base) / epsilon;
    
    printf("Price(sigma)     = %.6f\n", price_base);
    printf("Price(sigma+eps) = %.6f\n", price_bump);
    printf("FD Sensitivity   = %.6f\n", *h_result_fd);
    
    // Restore Constants
    float old_sig_st = original_sigma * sqrtf((1.0f - expf(-2.0f * H_A * H_DT)) / (2.0f * H_A));
    cudaMemcpyToSymbol(d_sigma, &original_sigma, sizeof(float));
    cudaMemcpyToSymbol(d_sig_st, &old_sig_st, sizeof(float));
}

int main() {
    select_gpu();
    printf("HULL-WHITE MODEL: QUESTION 3 (Sensitivity)\n");
    printf("==========================================\n");

    // Load Market Data
    float h_P[N_MAT], h_f[N_MAT];
    load_array(P_FILE, h_P, N_MAT);
    load_array(F_FILE, h_f, N_MAT);

    float *d_P_market, *d_f_market;
    cudaMalloc(&d_P_market, N_MAT * sizeof(float));
    cudaMalloc(&d_f_market, N_MAT * sizeof(float));
    cudaMemcpy(d_P_market, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_market, h_f, N_MAT * sizeof(float), cudaMemcpyHostToDevice);

    // Setup RNG
    curandState *d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    check_cuda("init_rng");

    // Initialize Standard Constants
    compute_constants();

    // Run MC Sensitivity
    float sens_mc;
    run_sensitivity_mc(d_P_market, d_f_market, d_states, &sens_mc);

    // Run FD Sensitivity
    float sens_fd;
    run_finite_difference(d_P_market, d_f_market, d_states, &sens_fd);

    // Comparison
    printf("\n=== SUMMARY ===\n");
    printf("Monte Carlo (Pathwise): %.6f\n", sens_mc);
    printf("Finite Difference:      %.6f\n", sens_fd);
    printf("Difference:             %.2e\n", fabsf(sens_mc - sens_fd));

    // Save Results
    FILE* json = json_open("data/q3_results.json", "Q3: Sensitivity Analysis");
    if (json) {
        fprintf(json, "  \"results\": {\n");
        fprintf(json, "    \"sensitivity_mc\": %.6f,\n", sens_mc);
        fprintf(json, "    \"sensitivity_fd\": %.6f,\n", sens_fd);
        fprintf(json, "    \"abs_diff\": %.2e\n", fabsf(sens_mc - sens_fd));
        fprintf(json, "  }\n");
        json_close(json);
        printf("Saved data/q3_results.json\n");
    }

    summary_append("data/summary.txt", "Q3: SENSITIVITY ANALYSIS");
    FILE* sum = fopen("data/summary.txt", "a");
    fprintf(sum, "  Sens (MC): %.6f\n", sens_mc);
    fprintf(sum, "  Sens (FD): %.6f\n", sens_fd);
    fclose(sum);

    cudaFree(d_P_market);
    cudaFree(d_f_market);
    cudaFree(d_states);

    return 0;
}