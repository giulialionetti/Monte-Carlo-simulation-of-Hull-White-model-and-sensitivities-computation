#ifndef COMMON_CUH
#define COMMON_CUH

/*
 * common.cuh
 * 
 * Shared definitions, constants, and utility functions for Hull-White 
 * interest rate model Monte Carlo simulation.
 * 
 * This header provides:
 * - Simulation parameters and grid configuration
 * - Hull-White model parameters (mean reversion, volatility, initial rate)
 * - GPU constant memory declarations for simulation coefficients
 * - Device functions for Hull-White analytical formulas
 * - Utility functions for GPU management and file I/O
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Monte Carlo simulation configuration
#define N_PATHS (1024 * 1024)  // Number of Monte Carlo paths
#define N_STEPS 1000 // Time discretization steps
#define NTPB 1024  // Threads per block
#define NB ((N_PATHS + NTPB - 1) / NTPB) // Number of blocks
#define N_MAT 101 // Number of maturities 
#define T_FINAL 10.0f  // Final maturity (years)

// Ensure N_STEPS is evenly divisible by (N_MAT - 1) for uniform sampling
#if (N_STEPS % (N_MAT - 1)) != 0
    #error "N_STEPS must be evenly divisible by (N_MAT - 1)"
#endif

#define SAVE_STRIDE (N_STEPS / (N_MAT - 1))

// Derived simulation parameters
// Time step and maturity spacing
const float H_DT = T_FINAL / N_STEPS;
const float H_MAT_SPACING = T_FINAL / (N_MAT - 1);

// Hull-White model parameters
const float H_A = 1.0f;
const float H_SIGMA = 0.1f;
const float H_R0 = 0.012f;

// File paths for data persistence
#define DATA_DIR "data/"
#define P_FILE DATA_DIR "P.bin" // Zero-coupon bond prices P(0,T)
#define F_FILE DATA_DIR "f.bin" // Forward rates f(0,T)


__constant__ float d_a; // Mean reversion speed
__constant__ float d_sigma;  // Volatility
__constant__ float d_r0; // Initial short rate
__constant__ float d_dt; // Time step
__constant__ float d_mat_spacing; // Maturity spacing
__constant__ float d_exp_adt; // e^{-aΔt}
__constant__ float d_sig_st; // σ·√[(1-e^{-2aΔt})/(2a)]
__constant__ float d_one_minus_exp_adt_over_a; // (1 - e^{-aΔt})/a
__constant__ float d_one_minus_exp_adt_over_a_sq; // ((1 - e^{-aΔt})/a)²
__constant__ float d_drift_table[N_STEPS]; // Precomputed drift integral table

// Initialize constant memory with precomputed values
void compute_constants() {
    float h_exp_adt = expf(-H_A * H_DT);
    float h_sig_st = H_SIGMA * sqrtf((1.0f - expf(-2.0f * H_A * H_DT)) / (2.0f * H_A));
    float h_one_minus_exp_adt_over_a = (1.0f - h_exp_adt) / H_A;
    float h_one_minus_exp_adt_over_a_sq = h_one_minus_exp_adt_over_a / H_A;

    cudaMemcpyToSymbol(d_a, &H_A, sizeof(float));
    cudaMemcpyToSymbol(d_sigma, &H_SIGMA, sizeof(float));
    cudaMemcpyToSymbol(d_r0, &H_R0, sizeof(float));
    cudaMemcpyToSymbol(d_dt, &H_DT, sizeof(float));
    cudaMemcpyToSymbol(d_mat_spacing, &H_MAT_SPACING, sizeof(float));
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

// Utility functions for GPU management and file I/O




/**
 * Check for CUDA errors and exit if an error occurred.
 * 
 * @param msg Context message to display with error
 */

inline void check_cuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}


/**
 * Automatically selects the GPU with the most free memory.
 */

inline void select_gpu() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    int best = 0;
    size_t max_free = 0;
    
    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (free_mem > max_free) {
            max_free = free_mem;
            best = i;
        }
    }
    
    cudaSetDevice(best);
    printf("Using GPU %d (%.2f GB free)\n\n", best, max_free / 1e9);
}

/**
 * Save array to binary file.
 * 
 * @param filename Output file path
 * @param data Host array to save
 * @param n Number of floats to write
 */

inline void save_array(const char* filename, float* data, int n) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Cannot open %s for writing\n", filename);
        exit(1);
    }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    printf("Saved %s (%d floats)\n", filename, n);
}

/**
 * Load array from binary file.
 * 
 * @param filename Input file path
 * @param data Host array to populate
 * @param n Number of floats to read
 */

inline void load_array(const char* filename, float* data, int n) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        printf("Did you run Q1 first?\n");
        exit(1);
    }
    size_t read = fread(data, sizeof(float), n, f);
    if ((int)read != n) {
        printf("Error: Expected %d floats, got %zu\n", n, read);
        exit(1);
    }
    fclose(f);
    printf("Loaded %s (%d floats)\n", filename, n);
}

/* DEVICE FUNCTIONS
 * 
 * These inline device functions are used in CUDA kernels for:
 * - Hull-White model theta function (piecewise linear)
 * - Analytical zero-coupon bond pricing formulas
 * - Linear interpolation for market data
 * */


 /**
 * Hull-White B(t,T) function: (1 - e^(-a(T-t))) / a
 * 
 * Used in the analytical zero-coupon bond pricing formula.
 * Represents the sensitivity of bond price to interest rate changes.
 * 
 * @param t Current time
 * @param T Maturity time
 * @param a Mean reversion speed
 * @return B(t,T) coefficient
 */

__device__ inline float B_func(float t, float T, float a) {
    return (1.0f - expf(-a * (T - t))) / a;
}

/**
 * Linear interpolation for discrete market data.
 * 
 * Used to obtain P(0,t) or f(0,t) for arbitrary t from discrete grid.
 * 
 * @param data Array of market values (P or f)
 * @param T Time to interpolate at
 * @param spacing Grid spacing between data points
 * @return Interpolated value
 */


__device__ inline float interpolate(const float* data, float T, float spacing) {
    int idx = (int)(T / spacing);
    if (idx >= N_MAT - 1) return data[N_MAT - 1];
    
    float t0 = idx * spacing;
    float alpha = (T - t0) / spacing;
    return data[idx] * (1.0f - alpha) + data[idx + 1] * alpha;
}

/**
 * Hull-White A(t,T) function for zero-coupon bond pricing.
 * 
 * Analytical formula: A(t,T) = [P(0,T)/P(0,t)] × exp[B(t,T)f(0,t) - σ²/(4a)(1-e^(-2at))B(t,T)²]
 * 
 * @param t Current time
 * @param T Bond maturity
 * @param a Mean reversion speed
 * @param sigma Volatility
 * @param P_market Array of market zero-coupon bond prices P(0,·)
 * @param f_market Array of market forward rates f(0,·)
 * @return A(t,T) coefficient
 */

__device__ inline float compute_A_HW(float t, float T, float a, float sigma,
                                      const float* P_market, const float* f_market) {
    float B_val = B_func(t, T, a);
    
    float P0T = interpolate(P_market, T, H_MAT_SPACING);
    float P0t = interpolate(P_market, t, H_MAT_SPACING);
    float f0t = interpolate(f_market, t, H_MAT_SPACING);
    
    float ratio = P0T / P0t;
    float term2 = B_val * f0t;
    float term3 = (sigma * sigma / (4.0f * a)) * (1.0f - expf(-2.0f * a * t)) * B_val * B_val;
    
    return ratio * expf(term2 - term3);
}

/**
 * Analytical Hull-White zero-coupon bond price: P(t,T) = A(t,T)e^(-B(t,T)r(t))
 * 
 * This closed-form solution is used in Q2b for option pricing, avoiding
 * the need to simulate bond prices explicitly.
 * 
 * @param t Current time
 * @param T Bond maturity
 * @param rt Short rate at time t
 * @param a Mean reversion speed
 * @param sigma Volatility
 * @param P_market Array of market zero-coupon bond prices P(0,·)
 * @param f_market Array of market forward rates f(0,·)
 * @return Zero-coupon bond price P(t,T)
 */

__device__ inline float compute_P_HW(float t, float T, float rt, float a, float sigma,
                                      const float* P_market, const float* f_market) {
    float A = compute_A_HW(t, T, a, sigma, P_market, f_market);
    float B = B_func(t, T, a);
    return A * expf(-B * rt);
}


/**
 * Piecewise linear theta function from project equation (7).
 * 
 * θ(t) = 0.012 + 0.0014t  for t < 5
 * θ(t) = 0.014 + 0.001t   for t ≥ 5
 * 
 * @param t Time in years
 * @return Instantaneous forward rate drift
 */

__host__ __device__ inline float theta_func(float t) {
    return (t < 5.0f) ? (0.012f + 0.0014f * t) : (0.014f + 0.001f * t);
}


/**
 * Initialize cuRAND states for Monte Carlo simulation.
 * 
 * Each thread initializes its own RNG state with a unique sequence number
 * to ensure independent random streams across parallel paths.
 * 
 * @param states Array of cuRAND states (one per path)
 * @param seed Random seed for reproducibility
 */


__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

#endif // COMMON_CUH