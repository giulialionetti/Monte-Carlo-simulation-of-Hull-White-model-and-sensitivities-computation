#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define N_PATHS (1024 * 1024)
#define N_STEPS 1000
#define NTPB 1024
#define NB ((N_PATHS + NTPB - 1) / NTPB)
#define N_MAT 101
#define SAVE_STRIDE (N_STEPS / (N_MAT - 1)) 
#define T 10.0f

__constant__ float a; // mean reversion speed
__constant__ float sigma;// volatility
__constant__ float r0; // initial short rate
__constant__ float dt; // time step size (0.01 years for 1000 steps over 10 years)
__constant__ float exp_adt; // precomputed e^{-a*dt}
__constant__ float sig_st; // precomputed sigma_{s,t}
__constant__ float one_minus_exp_adt_over_a; // precomputed (1-e^{-a*dt})/a
__constant__ float one_minus_exp_adt_over_a_sq; // precomputed (1-e^{-a*dt})/a^2

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
}


// Time-dependent theta function 
__device__ float theta(float t) {
    return (t < 5.0f) ? (0.012f + 0.0014f * t) : (0.014f + 0.001f * t);
}

// Integral of e^{-a*(t-u)}*theta(u) du from t to t+dt required for m_{s,t}
__device__ float m_st_drift_integral(float t, float dt) {
    float first_term = ((t+dt) - expf(-a * dt) * t) / a - one_minus_exp_adt_over_a_sq;
    return (t < 5.0f) ? 
        (0.0014f * first_term + 0.012f * one_minus_exp_adt_over_a) :
        (0.001f * first_term + 0.014f * one_minus_exp_adt_over_a);
}

// rng initialization kernel
// curandState is a struct that holds the state for each path
__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void simulate_q1(float* P_sum, curandState* states) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x; // path index 

    __shared__ float s_P_sum[N_MAT];
    if (threadIdx.x < N_MAT) {
        s_P_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (pid < N_PATHS) {
        curandState local = states[pid];  // local (on each thread's private memory) copy of RNG state

        // base case: T=0 means P(0,0) = 1
        atomicAdd(&s_P_sum[0], 1.0f);

        float r = r0;
        // to calculate the integral of r(t) from 0 to T
        float integral = 0.0f;

        // Simulate path from 0 to T = 10
        for (int i = 1; i <= N_STEPS; i++) {
            float t = (i - 1) * dt; // current time at start of step
            // float theta_t = theta(t); NOT USED

            /*      
            Expected value of r(t+dt) given r(t)

            - r*e^{-a*dt}: old rate decays exponentially
            - theta(t)*m_st_drift_integral(t, dt): pull towards mean reversion level theta(t)
            */
            float m_st = r * exp_adt + m_st_drift_integral(t, dt);

            // r(t+dt) = m_{s,t} + sigma_{s,t}*G, where G ~ N(0,1)
            float G = curand_normal(&local);
            float r_next = m_st + sig_st * G;

            // Trapezoidal rule (inherently sequential)
            integral += 0.5f * (r + r_next) * dt;
            
            r = r_next;

            // Check if we reached a maturity
            if (i % SAVE_STRIDE == 0) {
                int m = i / SAVE_STRIDE;
                if (m < N_MAT) {
                    // Discount factor
                    float discount = expf(-integral);
                    atomicAdd(&s_P_sum[m], discount);
                }
            }
            states[pid] = local;
        }
    }
    __syncthreads();
    if (threadIdx.x < N_MAT) {
        atomicAdd(&P_sum[threadIdx.x], s_P_sum[threadIdx.x]);
    }
}

int main() {
    printf("=== Hull-White Monte Carlo - Q1 ===\n");
    printf("N_PATHS = %d, N_STEPS = %d, N_MAT = %d\n\n", N_PATHS, N_STEPS, N_MAT);
    
    // Allocate
    float *d_P_sum, *h_P;
    curandState *d_states;
    
    // NOTE: Unified memory would be too slow for this application
    cudaMalloc(&d_P_sum, N_MAT * sizeof(float));
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    h_P = (float*)malloc(N_MAT * sizeof(float));
    
    // Initialize to zero
    cudaMemset(d_P_sum, 0, N_MAT * sizeof(float));
    
    // Initialize RNG
    printf("Initializing RNG...\n");
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("RNG Init Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    printf("RNG initialized\n");

    // Precompute simulation constants
    compute_constants();
    
    // Run simulation
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
    printf("Simulation complete\n");
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results
    cudaMemcpy(h_P, d_P_sum, N_MAT * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Memcpy Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Print results
    printf("\nResults (P(0,T)):\n");
    printf("T\tP(0,T)\t\tRaw Sum\n");
    for (int i = 0; i <= 100; i += 10) {
        float avg = h_P[i] / N_PATHS;
        printf("%.1f\t%.6f\t%.2f\n", i * 0.1f, avg, h_P[i]);
    }
    
    // Sanity checks
    printf("\n=== Sanity Checks ===\n");
    printf("P(0,0) should be ~1.0: %.6f\n", h_P[0] / N_PATHS);
    printf("P(0,10) should be ~0.3-0.9: %.6f\n", h_P[100] / N_PATHS);
    printf("Raw sum for P(0,0): %.2f (should be ~%d)\n", h_P[0], N_PATHS);
    
    printf("\nTime: %.2f ms\n", milliseconds);
    
    // Cleanup
    free(h_P);
    cudaFree(d_P_sum);
    cudaFree(d_states);
    
    return 0;
}