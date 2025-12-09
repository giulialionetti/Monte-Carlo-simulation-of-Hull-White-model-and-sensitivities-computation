#ifndef MARKET_DATA_CUH
#define MARKET_DATA_CUH

// Market Data for Hull-White Model Calibration
#include "common.cuh"

/**
 * Simulate zero-coupon bond prices using Hull-White model with antithetic variates.
 * 
 * For each path, we simulate two trajectories using G and -G (antithetic pair)
 * to reduce variance. The short rate evolution follows:
 * 
 *   r(t+dt) = r(t)e^(-a*dt) + drift_integral + sigma * √[(1-e^(-2a*dt))/(2a)] * G
 * 
 * Bond prices are computed via: P(0,T) = E[exp(-\int_0^T r(s)ds)]
 * 
 * The integral (\int_0^T r(s)ds) is approximated using the trapezoidal rule.
 * Results are accumulated in shared memory before writing to global memory
 * to reduce atomic operation overhead.
 * 
 * @param P_sum Global memory array to accumulate bond price sums [N_MAT]
 * @param states cuRAND states for random number generation [N_PATHS]
 */

__global__ void simulate_zcb(float* P_sum, curandState* states) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    __shared__ float s_P_sum[N_MAT];
    if (threadIdx.x < N_MAT) {
        s_P_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;
        const float exp_adt = d_exp_adt;      
        const float sig_st = d_sig_st;        

        for (int i = 1; i <= N_STEPS; i++) {
            float drift = d_drift_table[i - 1];
            float G = curand_normal(&local);
            const float sig_G = sig_st * G;

            evolve_hull_white_step(
                &r1, &integral1, drift, 
                sig_G, exp_adt, d_dt
            );
            evolve_hull_white_step(
                &r2, &integral2, drift, 
                -sig_G, exp_adt, d_dt
            );

            if (i % SAVE_STRIDE == 0) {
                int m = i / SAVE_STRIDE;
                if (m < N_MAT) {
                    float p0_m = expf(-integral1) + expf(-integral2);
                    warp_reduce(p0_m);
                    if (lane == 0) {
                        atomicAdd(&s_P_sum[m], p0_m);
                    }
                }
            }
        }
        states[pid] = local;
    }
    
    __syncthreads();
    if (threadIdx.x < N_MAT) {
        atomicAdd(&P_sum[threadIdx.x], s_P_sum[threadIdx.x]);
    }
    __syncthreads();
    if (pid == 0) {
       P_sum[0] = 2.0f * N_PATHS;
    }
}

/**
 * Compute zero-coupon bond prices and forward rates from simulation results.
 * 
 * Bond prices: P(0,T) = P_sum[T] / (2 * N_PATHS)
 * Forward rates: f(0,T) = -\partial ln(P(0,T)) / \partial T
 * 
 * Forward rates are computed using finite differences:
 * - Central difference for interior points: [ln P(T+dT) - ln P(T-dT)] / (2dT)
 * - Forward difference at T=0: [ln P(dT) - ln P(0)] / dT
 * - Backward difference at T=T_max: [ln P(T_max) - ln P(T_max-dT)] / dT
 * 
 * @param d_P Output array of averaged bond prices [N_MAT]
 * @param d_f Output array of forward rates [N_MAT]
 * @param d_P_sum Input array of accumulated sums from simulation [N_MAT]
 * @param n_mat Number of maturity points
 * @param n_paths Total number of paths (including antithetic)
 * @param dT Maturity grid spacing
 */


__global__ void compute_average_and_forward(
    float* d_P,  // Output: averaged bond prices
    float* d_f,  // Output: forward rates
    float* d_P_sum, // Input: raw sums from simulation
    int n_mat, // Number of maturities
    int n_paths, // Number of paths (2*N_PATHS for antithetic)
    float inv_dT // (1 / Maturity spacing)
) {
    __shared__ float s_P[N_MAT];
    int m = threadIdx.x;
    
     // Compute averages
    if (m < n_mat) {
        float avg = d_P_sum[m] / (float)n_paths;
        s_P[m] = avg;
        d_P[m] = avg;
    }
    __syncthreads();
    
    //Compute forward rates via finite differences
    if (m < n_mat) {
        int first_idx = (m == 0) ? 0 : m - 1;
        int last_idx = (m == n_mat - 1) ? n_mat - 1 : m + 1;
        float scale = ((m == 0) || (m == n_mat - 1)) ? 1.0f : 0.5f;
        d_f[m] = -scale * inv_dT * (logf(s_P[last_idx]) - logf(s_P[first_idx]));
    }
}

/**
 * Simulates a small number of paths and records the full r(t) trajectory.
 * 
 * @param d_paths Output array to store simulated paths [n_paths * (N_STEPS + 1)]
 * @param states cuRAND states for random number generation [n_paths]
 * @param n_paths Number of paths to simulate
 */
__global__ void simulate_paths_show(
    float* d_paths,      
    curandState* states, 
    int n_paths
) {
    int pid = threadIdx.x;
    if (pid >= n_paths) return;

    curandState local = states[pid];
    float r = d_r0;
    float integral = 0.0f;
    
    d_paths[pid * (N_STEPS + 1) + 0] = r;

    for (int i = 1; i <= N_STEPS; i++) {
        float drift = d_drift_table[i - 1];
        float G = curand_normal(&local);
        float sig_G = d_sig_st * G;

        evolve_hull_white_step(&r, &integral, drift, sig_G, d_exp_adt, d_dt);
        
        d_paths[pid * (N_STEPS + 1) + i] = r;
    }
    // states[pid] = local;
}

#endif // MARKET_DATA_CUH