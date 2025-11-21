#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define N_PATHS (1024 * 1024)
#define N_STEPS 1000
#define NTPB 1024
#define NB ((N_PATHS + NTPB - 1) / NTPB)
#define N_MAT 101
#define T 10.0f

#if (N_STEPS % (N_MAT - 1)) != 0
    #error "N_STEPS must be evenly divisible by (N_MAT - 1)"
#endif

#define SAVE_STRIDE (N_STEPS / (N_MAT - 1)) 

__constant__ float a; // mean reversion speed
__constant__ float sigma;// volatility
__constant__ float r0; // initial short rate
__constant__ float dt; // time step size (0.01 years for 1000 steps over 10 years)
__constant__ float exp_adt; // precomputed e^{-a*dt}
__constant__ float sig_st; // precomputed sigma_{s,t}
__constant__ float one_minus_exp_adt_over_a; // precomputed (1-e^{-a*dt})/a
__constant__ float one_minus_exp_adt_over_a_sq; // precomputed (1-e^{-a*dt})/a^2

__constant__ float drift_table[N_STEPS];  // Precomputed drift integral

// Precompute constants and copy to device constant memory

void compute_constants() {
    const float DT = T / N_STEPS;
    const float A = 1.0f;
    const float SIGMA = 0.1f;
    const float R0 = 0.012f;

    float h_exp_adt = expf(-A * DT);
    float h_sig_st = SIGMA * sqrtf((1.0f - expf(-2.0f * A * DT)) / (2.0f * A));
    float h_one_minus_exp_adt_over_a = (1.0f - h_exp_adt) / A;
    float h_one_minus_exp_adt_over_a_sq = h_one_minus_exp_adt_over_a / A;

    cudaMemcpyToSymbol(a, &A, sizeof(float));
    cudaMemcpyToSymbol(sigma, &SIGMA, sizeof(float));
    cudaMemcpyToSymbol(r0, &R0, sizeof(float));
    cudaMemcpyToSymbol(dt, &DT, sizeof(float));
    cudaMemcpyToSymbol(exp_adt, &h_exp_adt, sizeof(float));
    cudaMemcpyToSymbol(sig_st, &h_sig_st, sizeof(float));
    cudaMemcpyToSymbol(one_minus_exp_adt_over_a, &h_one_minus_exp_adt_over_a, sizeof(float));
    cudaMemcpyToSymbol(one_minus_exp_adt_over_a_sq, &h_one_minus_exp_adt_over_a_sq, sizeof(float));

    // Precompute drift integral for each time step
    float h_drift[N_STEPS];
    for (int i = 0; i < N_STEPS; i++) {
        float t = i * DT;
        float first_term = ((t + DT) - h_exp_adt * t) / A - h_one_minus_exp_adt_over_a_sq;
        h_drift[i] = (t < 5.0f) ? 
            (0.0014f * first_term + 0.012f * h_one_minus_exp_adt_over_a) :
            (0.001f * first_term + 0.014f * h_one_minus_exp_adt_over_a);
    }
    cudaMemcpyToSymbol(drift_table, h_drift, N_STEPS * sizeof(float));
}

// Time-dependent theta function 
__device__ float theta(float t) {
    return (t < 5.0f) ? (0.012f + 0.0014f * t) : (0.014f + 0.001f * t);
}


// rng initialization kernel
// curandState is a struct that holds the state for each path
__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void simulate_q1(float* P_sum, curandState* states) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_P_sum[N_MAT];
    if (threadIdx.x < N_MAT) {
        s_P_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (pid < N_PATHS) {
        curandState local = states[pid];

        // Base case: both antithetic paths contribute P(0,0) = 1
        atomicAdd(&s_P_sum[0], 2.0f);

        // Two paths for antithetic variates
        float r1 = r0, r2 = r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        for (int i = 1; i <= N_STEPS; i++) {
            float drift = drift_table[i - 1];
            float G = curand_normal(&local);

            // Path 1: +G
            float r1_next = r1 * exp_adt + drift + sig_st * G;
            integral1 += 0.5f * (r1 + r1_next) * dt;
            r1 = r1_next;

            // Path 2: -G (antithetic)
            float r2_next = r2 * exp_adt + drift - sig_st * G;
            integral2 += 0.5f * (r2 + r2_next) * dt;
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

__global__ void compute_average_and_forward(
    float* d_P,       // Output: averaged bond prices
    float* d_f,       // Output: forward rates
    float* d_P_sum,   // Input: raw sums from simulation
    int n_mat,        // Number of maturities
    int n_paths,      // Number of paths
    float dT          // Maturity spacing
) {
    __shared__ float s_P[N_MAT];
    
    int m = threadIdx.x;
    
    // Step 1: Compute average and store in shared memory
    if (m < n_mat) {
        float avg = d_P_sum[m] / (float)n_paths;
        s_P[m] = avg;
        d_P[m] = avg;  // Also write to global for output
    }
    __syncthreads();
    
    // Step 2: Compute forward rates using shared memory
    if (m < n_mat) {
        if (m == 0) {
            // Forward difference (first point)
            d_f[m] = -(logf(s_P[1]) - logf(s_P[0])) / dT;
            
        } else if (m == n_mat - 1) {
            // Backward difference (last point)
            d_f[m] = -(logf(s_P[m]) - logf(s_P[m-1])) / dT;
            
        } else {
            // Central difference (interior points)
            d_f[m] = -(logf(s_P[m+1]) - logf(s_P[m-1])) / (2.0f * dT);
        }
    }
}

int main() {

     cudaSetDevice(3);
    

    printf("=== Hull-White Monte Carlo - Q1 ===\n");
    printf("N_PATHS = %d, N_STEPS = %d, N_MAT = %d\n", N_PATHS, N_STEPS, N_MAT);
    printf("SAVE_STRIDE = %d\n\n", SAVE_STRIDE);
    
    
    float *d_P_sum;           // Raw sums from simulation
    float *d_P, *d_f;         // Averaged P and forward rates (device)
    float *h_P, *h_f;         // Host copies
    curandState *d_states;
    
     // NOTE: Unified memory would be too slow for this application
    cudaMalloc(&d_P_sum, N_MAT * sizeof(float));
    cudaMalloc(&d_P, N_MAT * sizeof(float));
    cudaMalloc(&d_f, N_MAT * sizeof(float));
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    
    h_P = (float*)malloc(N_MAT * sizeof(float));
    h_f = (float*)malloc(N_MAT * sizeof(float));
    
    // Initialize d_P_sum to zero
    cudaMemset(d_P_sum, 0, N_MAT * sizeof(float));
    
    
    compute_constants();
    
    printf("Initializing RNG...\n");
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("RNG Init Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    printf("RNG initialized ✓\n");
    
    
    // Run Monte Carlo simulation
    printf("Running simulation...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_q1<<<NB, NTPB>>>(d_P_sum, d_states);
    cudaEventRecord(stop);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Simulation Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    printf("Simulation complete ✓\n");
    
    float sim_ms = 0;
    cudaEventElapsedTime(&sim_ms, start, stop);
    
    
    printf("Computing averages and forward rates...\n");
    float maturity_spacing = T / (N_MAT - 1);  // 0.1 years
    
    // Launch with single block (N_MAT = 101 < 128 threads)
    compute_average_and_forward<<<1, 128>>>(
        d_P, d_f, d_P_sum, N_MAT, 2*N_PATHS, maturity_spacing
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Forward rates Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    printf("Forward rates computed ✓\n");
    

    cudaMemcpy(h_P, d_P, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f, d_f, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Memcpy Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    
    printf("\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("                      RESULTS                         \n");
    printf("══════════════════════════════════════════════════════\n");
    printf("T (years)    P(0,T)         f(0,T)\n");
    printf("──────────────────────────────────────────────────────\n");
    for (int i = 0; i <= 100; i += 10) {
        printf("%5.1f        %.6f       %7.4f%%\n", 
               i * 0.1f, h_P[i], h_f[i] * 100.0f);
    }
    printf("══════════════════════════════════════════════════════\n");
    
   
    printf("\n=== Sanity Checks ===\n");
    printf("P(0,0) should be ~1.0:       %.6f %s\n", 
           h_P[0], (h_P[0] > 0.99f && h_P[0] < 1.01f) ? "✓" : "✗");
    printf("P(0,10) should be ~0.3-0.9:  %.6f %s\n", 
           h_P[100], (h_P[100] > 0.3f && h_P[100] < 0.9f) ? "✓" : "✗");
    printf("f(0,0) should be ~1.2%%:      %.4f%% %s\n", 
           h_f[0] * 100.0f, (h_f[0] > 0.01f && h_f[0] < 0.02f) ? "✓" : "✗");
    printf("f(0,10) should be ~1-3%%:     %.4f%% %s\n", 
           h_f[100] * 100.0f, (h_f[100] > 0.01f && h_f[100] < 0.03f) ? "✓" : "✗");
    
    // Check monotonicity of P(0,T)
    int monotonic = 1;
    for (int i = 1; i < N_MAT; i++) {
        if (h_P[i] > h_P[i-1]) {
            monotonic = 0;
            break;
        }
    }
    printf("P(0,T) monotonically decreasing: %s\n", monotonic ? "✓" : "✗");
    
    // Check forward rates are positive
    int positive_rates = 1;
    for (int i = 0; i < N_MAT; i++) {
        if (h_f[i] < 0) {
            positive_rates = 0;
            break;
        }
    }
    printf("f(0,T) all positive: %s\n", positive_rates ? "✓" : "✗");
    

    printf("\n=== Performance ===\n");
    printf("Simulation time: %.2f ms\n", sim_ms);
    printf("Paths simulated: %d (x2 antithetic = %d effective)\n", N_PATHS, N_PATHS * 2);
    printf("Effective paths/second: %.2f million\n", (N_PATHS * 2.0f / sim_ms) / 1000.0f);
    
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
