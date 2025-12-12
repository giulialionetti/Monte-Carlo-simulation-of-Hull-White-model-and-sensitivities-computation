#ifndef COMMON_CUH
#define COMMON_CUH


 //Shared definitions, constants, and utility functions for Hull-White interest rate model Monte Carlo simulation.


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
       
        // first term with 1/a factored out later
        // this is the result of integrating e^{-a(t - u)} * u du from s to t 
        // it arises from theta(u)= alpha + beta*u which solved by parts, leading to the expression below
        float first_term = ((s + H_DT) - h_exp_adt * s) / H_A - h_one_minus_exp_adt_over_a_sq;
        h_drift[i] = (s < 5.0f) ? 
            (0.0014f * first_term + 0.012f * h_one_minus_exp_adt_over_a) :
            (0.001f * first_term + 0.019f * h_one_minus_exp_adt_over_a);
        
        
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

inline void check_cuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

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

void load_market_data_to_device(float h_P[N_MAT], float h_f[N_MAT], float** d_P, float** d_f) { 
    cudaMalloc(d_P, N_MAT * sizeof(float));
    cudaMalloc(d_f, N_MAT * sizeof(float));
    cudaMemcpy(*d_P, h_P, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_f, h_f, N_MAT * sizeof(float), cudaMemcpyHostToDevice);
}

// this function computes the B(t,T) component of the Hull-White bond pricing formula
// using the closed-form expression
__device__ inline float B_func(float t, float T, float a) {
    return (1.0f - expf(-a * (T - t))) / a;
}

// this function retrives values from market data (bond prices, forward rates) arrays
// at arbitrary time points using linear interpolation
// assuming f is sampled at uniform intervals defined by spacing
__device__ inline float interpolate(const float* data, float T, float spacing) {
    int idx = (int)(T / spacing); // determine which array segment T falls into
    if (idx >= N_MAT - 1) // if T exceeds max maturity
    return data[N_MAT - 1]; // return last element
    
    float t0 = idx * spacing; // find time corresponding to idx
    float alpha = (T - t0) / spacing; // calculate how far T is between t0 and t0 + spacing
     // linear interpolation between data[idx] and data[idx + 1]
    return data[idx] * (1.0f - alpha) + data[idx + 1] * alpha;
}

// this function computes the A(t,T) component of the Hull-White bond pricing formula
// we first compute B(t,T) using the closed-form expression
// then retrieve P(0,T), P(0,t), and f(0,t) from market data using interpolation
__device__ inline float compute_A_HW(float t, float T, float a, float sigma,
                                      const float* P_market, const float* f_market) {
    float B_val = B_func(t, T, a);
    
    float P0T = interpolate(P_market, T, H_MAT_SPACING);
    float P0t = interpolate(P_market, t, H_MAT_SPACING);
    float f0t = interpolate(f_market, t, H_MAT_SPACING);
    
    float ratio = P0T / P0t; // forward discount factor from t to T
    float term2 = B_val * f0t; // adjust for the expected drift of the short rate
    float term3 = (sigma * sigma / (4.0f * a)) * (1.0f - expf(-2.0f * a * t)) * B_val * B_val; // convexity adjustment
    
    return ratio * expf(term2 - term3); // combine all and return A(t,T)
}

// Hull-White bond pricing formula P(t,T) = A(t,T) * exp(-B(t,T)*r_t)
// where A(t,T) and B(t,T) are computed as per above functions
// r_t is the short rate at time t 
// P_market and f_market are arrays of market bond prices and forward rates
__device__ inline float compute_P_HW(float t, float T, float rt, float a, float sigma,
                                      const float* P_market, const float* f_market) {
    float A = compute_A_HW(t, T, a, sigma, P_market, f_market);
    float B = B_func(t, T, a);
    return A * expf(-B * rt); // start with A(t,T) and discount by exp(-B(t,T)*r_t)
}

// Hull-White model theta(t) function 
__host__ __device__ inline float theta_func(float t) {
    return (t < 5.0f) ? (0.012f + 0.0014f * t) : (0.019f + 0.001f * t);
}

// exp_adt is e^{-adt} and sig_G is sigma*sqrt[(1-e^{-2adt})/(2a)]*G
// drift is the precomputed integral drift (see compute_drift_tables function)
// computes exact exponential integration to the deterministic part of the SDE
// then adds the stochastic shock
// updates the short rate r and the integral of r over the time step dt
__device__ inline void evolve_hull_white_step(
    float* r, float* integral, float drift, 
    float sig_G, float exp_adt, float dt
) {
    float r_next = __fmaf_rn(*r, exp_adt, drift + sig_G);
    *integral += 0.5f * (*r + r_next) * dt; // Trapezoidal rule
    *r = r_next;
}

// used by recover_theta kernel to compute numerical derivative of forward rate curve f(T) 
// the function uses finite difference methods adapted to positions in the array
// the boundary cases (i=0 and i=n-1) use forward and backward difference respectively
// to avoid accessing out-of-bounds memory
__device__ float compute_derivative(const float* f, int i, int n, float spacing) {
    if (i == 0) { // first element
        return (f[1] - f[0]) / spacing; // forward difference
    } else if (i == n - 1) { // last element
        return (f[i] - f[i-1]) / spacing; // backward difference
    } else { // all other cases
        return (f[i+1] - f[i-1]) / (2.0f * spacing); // central difference
    }
}

__device__ inline void warp_reduce(float& thread_sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
}

__device__ inline float block_reduce(float* warp_sums, int lane, int warp_id) {
    float warp_sum = (warp_id == 0) ? 
        ((lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f) : 0.0f;
    
    if (warp_id == 0) {
        warp_reduce(warp_sum);
    }
    return warp_sum;
}

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}


// function for pricing European call option on zero-coupon bond
// using control variate technique with antithetic variates
// beta is estimated within the kernel
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

    // path accumulators initialization
    float thread_ZBC = 0.0f; // Payoff accumulator
    float thread_control = 0.0f; // Control variate accumulator
    float thread_ZBC_sq = 0.0f; // Payoff squared accumulator (for variance)
    float thread_control_sq = 0.0f; // Control variate squared accumulator (for variance)
    float thread_cross = 0.0f; // Cross product accumulator (for covariance)

    if (pid < N_PATHS) {
        curandState local = states[pid];
        
        // Simulate two antithetic paths for control variate technique
        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;
        
        // Number of discrete time steps to reach exercise time S1
        int n_steps_S1 = (int)(S1 / d_dt);

        // Evolve both paths up to S1
        for (int i = 0; i < n_steps_S1; i++) {
            float drift = d_drift_table[i]; // Precomputed drift term
            float G = curand_normal(&local); // Random shock N(0,1)
            float sig_G = d_sig_st * G; // Scaled shock for our model

            evolve_hull_white_step(&r1, &integral1, drift, sig_G, d_exp_adt, d_dt);
            evolve_hull_white_step(&r2, &integral2, drift, -sig_G, d_exp_adt, d_dt);
        }

        // after loop both paths are at time S1 with short rates r1 and r2
        
        // comput bond prices for both paths
        float P1 = compute_P_HW(S1, S2, r1, d_a, d_sigma, d_P_market, d_f_market);
        float P2 = compute_P_HW(S1, S2, r2, d_a, d_sigma, d_P_market, d_f_market);
        
        // compute discount factors for both paths from 0 to S1 
        // to bring future payoffs to time 0 (present value)
        float discount1 = expf(-integral1);
        float discount2 = expf(-integral2);
        
        // compute control variate values Y for both paths
        // E[Y] is expected value of the discounted bond price P(0,S2) 
        float control1 = discount1 * P1;
        float control2 = discount2 * P2;
        
       // compute payoffs for both paths to estimate option price E[Xi] given
       // that E[Yi] = P(0,S2) is known exactly from market data
        float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);
        
       // sum up both antithetic contributions
        thread_ZBC = payoff1 + payoff2;
        thread_control = control1 + control2;
        
       // second moments for variance and covariance estimation
        thread_ZBC_sq = payoff1 * payoff1 + payoff2 * payoff2;
        thread_control_sq = control1 * control1 + control2 * control2;
        thread_cross = payoff1 * control1 + payoff2 * control2;
        
        states[pid] = local;
    }
    
    // tree-like warp reduction for all accumulators
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_ZBC += __shfl_down_sync(0xffffffff, thread_ZBC, offset);
        thread_control += __shfl_down_sync(0xffffffff, thread_control, offset);
        thread_ZBC_sq += __shfl_down_sync(0xffffffff, thread_ZBC_sq, offset);
        thread_control_sq += __shfl_down_sync(0xffffffff, thread_control_sq, offset);
        thread_cross += __shfl_down_sync(0xffffffff, thread_cross, offset);
    }
    
    // atomic addition of warp results to global memory
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