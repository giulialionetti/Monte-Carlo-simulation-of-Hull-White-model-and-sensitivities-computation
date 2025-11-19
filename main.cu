#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define N_PATHS 65536
#define N_STEPS 1000
#define N_MAT 101

__constant__ float a = 1.0f; // mean reversion speed
__constant__ float sigma = 0.1f; // volatility
__constant__ float r0 = 0.012f; // initial short rate

// Time-dependent theta function 
__device__ float theta(float t) {
    return (t < 5.0f) ? (0.012f + 0.0014f * t) : (0.019f + 0.001f * (t - 5.0f));
}

// rng initialization kernel
// curandState is a struct that holds the state for each path
__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void simulate_q1(float* P_sum, curandState* states, int n_paths) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x; // path index 
    if (pid >= n_paths) return; 
    
    curandState local = states[pid];  // local (on each thread's private memory) copy of RNG state
    float dt = 10.0f / N_STEPS; // time step size (0.01 years for 1000 steps over 10 years)
    
    // Simulate each maturity
    for (int m = 0; m < N_MAT; m++) {
        float T = m * 0.1f; // convert index to actual years
        int steps = (int)(T / dt);  // To reach T=5.0 years, I need to take 500 steps of size 0.01
        
        // base case: T=0 means P(0,0) = 1
        if (steps == 0) {
            atomicAdd(&P_sum[m], 1.0f);
            continue;
        }
        
        float r = r0;
        // to calculate the integral of r(t) from 0 to T
        float integral = 0.0f;
        float r_prev = r; // trapezoidal rule needs previous r
        
        // Simulate path from 0 to T
        for (int i = 0; i < steps; i++) {
            float t = i * dt;
            float theta_t = theta(t);
            
            // e^{-a·dt} = e^{-1.0 × 0.01} = e^{-0.01} ≈ 0.99
            float exp_adt = expf(-a * dt);
            /*m_{s,t} =r(t)·e^{-a·dt} + θ(t)·(1 - e^{-a·dt})/a
            
            Expected value of r(t+dt) given r(t)

            1. r·e^{-a·dt}: old rate decays exponentially
            2. θ(t)·(1-e^{-a·dt})/a: pull towards mean reversion level θ(t) */
            float m_st = r * exp_adt + theta_t * (1.0f - exp_adt) / a;

            // σ_{s,t} = σ·sqrt((1 - e^{-2a·dt})/(2a))
            // volatility of r(t+dt)
            float sig_st = sigma * sqrtf((1.0f - expf(-2.0f * a * dt)) / (2.0f * a));
            
            // Sample from N(m_{s,t}, σ_{s,t}^2)
            // r(t+dt) = m_{s,t} + σ_{s,t}·G, where G ~ N(0,1)
            // curand_normal generates standard normal variable
            // The bell curve around m_st has width sig_st
            // Random G determines where in the curve we land
            float G = curand_normal(&local); // NOTE: this is the only place in the code that has (minimal) divergence 
            r = m_st + sig_st * G;
            
            // Trapezoidal rule (inherently sequential)
            integral += 0.5f * (r_prev + r) * dt;
            r_prev = r;
        }
        
        // Discount factor
        float discount = expf(-integral);

        // must use atomic add since multiple threads write to same location
        atomicAdd(&P_sum[m], discount);
    }
    
    states[pid] = local;
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
    
    int blocks = (N_PATHS + 255) / 256;
    
    // Initialize RNG
    printf("Initializing RNG...\n");
    init_rng<<<blocks, 256>>>(d_states, time(NULL));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("RNG Init Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    printf("RNG initialized ✓\n");
    
    // Run simulation
    printf("Running simulation...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    simulate_q1<<<blocks, 256>>>(d_P_sum, d_states, N_PATHS);
    cudaEventRecord(stop);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Simulation Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    printf("Simulation complete ✓\n");
    
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