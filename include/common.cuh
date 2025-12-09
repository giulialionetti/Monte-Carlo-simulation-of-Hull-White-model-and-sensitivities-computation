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
#define WARPS_PER_BLOCK (NTPB >> 5) // Number of warps per block
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
__constant__ float d_exp_adt; // e^{-adt}
__constant__ float d_sig_st; // sigma*sqrt[(1-e^{-2adt})/(2a)]
__constant__ float d_one_minus_exp_adt_over_a; // (1 - e^{-adt})/a
__constant__ float d_one_minus_exp_adt_over_a_sq; // ((1 - e^{-adt})/a)^2
__constant__ float d_drift_table[N_STEPS]; // Precomputed drift integral table
__constant__ float d_sigma_drift_table[N_STEPS]; // Drift term in the sensitivity process

// Precompute drift integral tables and copy to constant memory
void compute_drift_tables(float sigma) {
    float h_exp_adt = expf(-H_A * H_DT);
    float h_one_minus_exp_adt_over_a = (1.0f - h_exp_adt) / H_A;
    float h_one_minus_exp_adt_over_a_sq = h_one_minus_exp_adt_over_a / H_A;

    float h_drift[N_STEPS], h_sigma_drift[N_STEPS];
    for (int i = 0; i < N_STEPS; i++) {
        float s = i * H_DT;
        float t = (i + 1) * H_DT;

        float first_term = ((s + H_DT) - h_exp_adt * s) / H_A - h_one_minus_exp_adt_over_a_sq;
        h_drift[i] = (s < 5.0f) ? 
            (0.0014f * first_term + 0.012f * h_one_minus_exp_adt_over_a) :
            (0.001f * first_term + 0.014f * h_one_minus_exp_adt_over_a);
        
        float sigma_term = (2.0f * sigma * expf(-H_A * t)) * (coshf(H_A * t) - coshf(H_A * s));
        h_sigma_drift[i] = sigma_term / (H_A * H_A);
    }
    cudaMemcpyToSymbol(d_drift_table, h_drift, N_STEPS * sizeof(float));
    cudaMemcpyToSymbol(d_sigma_drift_table, h_sigma_drift, N_STEPS * sizeof(float));
}

// Compute sig_st for given sigma
float compute_h_sig_st(float sigma) {
    return sigma * sqrtf((1.0f - expf(-2.0f * H_A * H_DT)) / (2.0f * H_A));
}

// Initialize constant memory with precomputed values
void compute_constants() {
    float h_exp_adt = expf(-H_A * H_DT);
    float h_sig_st = compute_h_sig_st(H_SIGMA);
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

    // Precompute drift integral tables
    compute_drift_tables(H_SIGMA);
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

/**
 * Load market data (P and f) from host arrays to device arrays.
 * 
 * @param h_P Host array for P(0,T)
 * @param h_f Host array for f(0,T)
 * @param d_P Pointer to device array for P(0,T)
 * @param d_f Pointer to device array for f(0,T)
 */
void load_market_data_to_device(float h_P[N_MAT], float h_f[N_MAT], float** d_P, float** d_f) { 
    cudaMalloc(d_P, N_MAT * sizeof(float));
    cudaMalloc(d_f, N_MAT * sizeof(float));
    cudaMemcpy(*d_P, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_f, h_f, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
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
 * Analytical formula: A(t,T) = [P(0,T)/P(0,t)] * exp[B(t,T)f(0,t) - sigma^2/(4a)(1-e^(-2at))B(t,T)^2]
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
 * Evolve Hull-White short rate and integral for one time step.
 * r(t+dt) = r(t)e^(-a*dt) + drift_integral + sigma * sqrt[(1-e^(-2a*dt))/(2a)] * G
 * 
 * @param r Pointer to current short rate
 * @param integral Pointer to accumulated integral of short rate
 * @param drift Drift term for the current time step
 * @param sig_G Stochastic term (sigma * G)
 * @param exp_adt Precomputed e^{-a*dt}
 * @param dt Time step size
 */
__device__ inline void evolve_hull_white_step(
    float* r, float* integral, float drift, 
    float sig_G, float exp_adt, float dt
) {
    float r_next = __fmaf_rn(*r, exp_adt, drift + sig_G);
    *integral += 0.5f * (*r + r_next) * dt; // Trapezoidal rule
    *r = r_next;
}

 /**
 * Compute numerical derivative df/dT using finite differences.
 * 
 * Uses central differences for interior points and one-sided differences
 * at boundaries for improved accuracy.
 * 
 * @param f Array of forward rates
 * @param i Index at which to compute derivative
 * @param n Array length
 * @param spacing Grid spacing dT
 * @return df/dT at index i
 */
__device__ float compute_derivative(const float* f, int i, int n, float spacing) {
    if (i == 0) {
        return (f[1] - f[0]) / spacing;
    } else if (i == n - 1) {
        return (f[i] - f[i-1]) / spacing;
    } else {
        return (f[i+1] - f[i-1]) / (2.0f * spacing);
    }
}

/**
 * Warp-level reduction using shuffle intrinsics.
 * Reduces thread_sum across all 32 threads in a warp.
 * After this, lane 0 holds the warp sum.
 * 
 * @param thread_sum Value to reduce (modified in-place)
 */
__device__ inline void warp_reduce(float& thread_sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
}

/**
 * Block-level reduction from warp sums.
 * First warp reduces across all warp_sums in shared memory.
 * 
 * @param warp_sums Shared memory array of warp sums [WARPS_PER_BLOCK]
 * @param lane Thread index within warp (threadIdx.x & 31)
 * @param warp_id Warp index within block (threadIdx.x >> 5)
 * @return Block sum (valid only in lane 0 of warp 0)
 */
__device__ inline float block_reduce(float* warp_sums, int lane, int warp_id) {
    float warp_sum = (warp_id == 0) ? 
        ((lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f) : 0.0f;
    
    if (warp_id == 0) {
        warp_reduce(warp_sum);
    }
    return warp_sum;
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


__global__ void simulate_ZBC_control_variate(
    float* ZBC_sum,
    float* control_sum,
    float* ZBC_sq_sum,      
    float* control_sq_sum,  
    float* cross_prod_sum,  
    curandState* states, 
    float S1, float S2, float K,
    const float* d_P_market,
    const float* d_f_market
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    __shared__ float warp_ZBC_sums[WARPS_PER_BLOCK];
    __shared__ float warp_control_sums[WARPS_PER_BLOCK];
    __shared__ float warp_ZBC_sq_sums[WARPS_PER_BLOCK];
    __shared__ float warp_control_sq_sums[WARPS_PER_BLOCK];
    __shared__ float warp_cross_sums[WARPS_PER_BLOCK];

    float thread_ZBC = 0.0f;
    float thread_control = 0.0f;
    float thread_ZBC_sq = 0.0f;
    float thread_control_sq = 0.0f;
    float thread_cross = 0.0f;

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        int n_steps_S1 = (int)(S1 / d_dt);

        for (int i = 0; i < n_steps_S1; i++) {
            float drift = d_drift_table[i];
            float G = curand_normal(&local);
            float sig_G = d_sig_st * G;

            evolve_hull_white_step(&r1, &integral1, drift, sig_G, d_exp_adt, d_dt);
            evolve_hull_white_step(&r2, &integral2, drift, -sig_G, d_exp_adt, d_dt);
        }
        
        float P1 = compute_P_HW(S1, S2, r1, d_a, d_sigma, d_P_market, d_f_market);
        float P2 = compute_P_HW(S1, S2, r2, d_a, d_sigma, d_P_market, d_f_market);
        
        float discount1 = expf(-integral1);
        float discount2 = expf(-integral2);
        
        
        float control1 = discount1 * P1;
        float control2 = discount2 * P2;
        
       
        float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);
        
       
        thread_ZBC = payoff1 + payoff2;
        thread_control = control1 + control2;
        
       
        thread_ZBC_sq = payoff1 * payoff1 + payoff2 * payoff2;
        thread_control_sq = control1 * control1 + control2 * control2;
        thread_cross = payoff1 * control1 + payoff2 * control2;
        
        states[pid] = local;
    }
    
    // Warp reduction for all accumulators
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_ZBC += __shfl_down_sync(0xffffffff, thread_ZBC, offset);
        thread_control += __shfl_down_sync(0xffffffff, thread_control, offset);
        thread_ZBC_sq += __shfl_down_sync(0xffffffff, thread_ZBC_sq, offset);
        thread_control_sq += __shfl_down_sync(0xffffffff, thread_control_sq, offset);
        thread_cross += __shfl_down_sync(0xffffffff, thread_cross, offset);
    }
    
    if (lane == 0) {
        warp_ZBC_sums[warp_id] = thread_ZBC;
        warp_control_sums[warp_id] = thread_control;
        warp_ZBC_sq_sums[warp_id] = thread_ZBC_sq;
        warp_control_sq_sums[warp_id] = thread_control_sq;
        warp_cross_sums[warp_id] = thread_cross;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_ZBC = (lane < WARPS_PER_BLOCK) ? warp_ZBC_sums[lane] : 0.0f;
        float warp_control = (lane < WARPS_PER_BLOCK) ? warp_control_sums[lane] : 0.0f;
        float warp_ZBC_sq = (lane < WARPS_PER_BLOCK) ? warp_ZBC_sq_sums[lane] : 0.0f;
        float warp_control_sq = (lane < WARPS_PER_BLOCK) ? warp_control_sq_sums[lane] : 0.0f;
        float warp_cross = (lane < WARPS_PER_BLOCK) ? warp_cross_sums[lane] : 0.0f;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_ZBC += __shfl_down_sync(0xffffffff, warp_ZBC, offset);
            warp_control += __shfl_down_sync(0xffffffff, warp_control, offset);
            warp_ZBC_sq += __shfl_down_sync(0xffffffff, warp_ZBC_sq, offset);
            warp_control_sq += __shfl_down_sync(0xffffffff, warp_control_sq, offset);
            warp_cross += __shfl_down_sync(0xffffffff, warp_cross, offset);
        }
        
        if (lane == 0) {
            atomicAdd(ZBC_sum, warp_ZBC);
            atomicAdd(control_sum, warp_control);
            atomicAdd(ZBC_sq_sum, warp_ZBC_sq);
            atomicAdd(control_sq_sum, warp_control_sq);
            atomicAdd(cross_prod_sum, warp_cross);
        }
    }
}

#endif // COMMON_CUH