#ifndef PERF_BENCHMARK_CUH
#define PERF_BENCHMARK_CUH

/*
 * Performance Benchmarking: Reduction Methods
 * 
 * Compares:
 * 1. Naive atomicAdd per thread (baseline)
 * 2. Shared memory reduction per block
 * 3. Warp reduction + block reduction (optimized)
 */

#include "common.cuh"

/**
 * NAIVE: Every thread atomicAdds directly to global memory (baseline).
 * Expected: High contention, slow.
 */
__global__ void simulate_ZBC_naive(
    float* ZBC_sum, 
    curandState* states, 
    float S1, float S2, float K,
    const float* __restrict__ d_P_market,
    const float* __restrict__ d_f_market
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        const int n_steps = (int)(S1 / d_dt);
        const float exp_adt = d_exp_adt;
        const float sig_st = d_sig_st;

        for (int i = 1; i <= n_steps; i++) {
            const float drift = d_drift_table[i - 1];
            const float G = curand_normal(&local);
            const float sig_G = sig_st * G;

            evolve_hull_white_step(&r1, &integral1, drift, sig_G, exp_adt, d_dt);
            evolve_hull_white_step(&r2, &integral2, drift, -sig_G, exp_adt, d_dt);
        }

        const float P1 = compute_P_HW(S1, S2, r1, d_a, d_sigma, d_P_market, d_f_market);
        const float P2 = compute_P_HW(S1, S2, r2, d_a, d_sigma, d_P_market, d_f_market);

        const float discount1 = __expf(-integral1);
        const float discount2 = __expf(-integral2);

        const float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        const float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);

        // NAIVE: Direct atomicAdd (high contention)
        atomicAdd(ZBC_sum, payoff1 + payoff2);

        states[pid] = local;
    }
}

/**
 * SHARED MEMORY: Use block-level shared memory reduction (good but not optimal).
 * Expected: Better than naive, but not as good as warp+block.
 */
__global__ void simulate_ZBC_shared_memory(
    float* ZBC_sum, 
    curandState* states, 
    float S1, float S2, float K,
    const float* __restrict__ d_P_market,
    const float* __restrict__ d_f_market
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_sum[1024];  // Shared memory accumulator
    s_sum[tid] = 0.0f;
    __syncthreads();

    if (pid < N_PATHS) {
        curandState local = states[pid];

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        const int n_steps = (int)(S1 / d_dt);
        const float exp_adt = d_exp_adt;
        const float sig_st = d_sig_st;

        for (int i = 1; i <= n_steps; i++) {
            const float drift = d_drift_table[i - 1];
            const float G = curand_normal(&local);
            const float sig_G = sig_st * G;

            evolve_hull_white_step(&r1, &integral1, drift, sig_G, exp_adt, d_dt);
            evolve_hull_white_step(&r2, &integral2, drift, -sig_G, exp_adt, d_dt);
        }

        const float P1 = compute_P_HW(S1, S2, r1, d_a, d_sigma, d_P_market, d_f_market);
        const float P2 = compute_P_HW(S1, S2, r2, d_a, d_sigma, d_P_market, d_f_market);

        const float discount1 = __expf(-integral1);
        const float discount2 = __expf(-integral2);

        const float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        const float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);

        // Shared memory accumulation
        atomicAdd(&s_sum[tid], payoff1 + payoff2);

        states[pid] = local;
    }

    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    // One atomic per block to global
    if (tid == 0) {
        atomicAdd(ZBC_sum, s_sum[0]);
    }
}

/**
 * WARP+BLOCK OPTIMIZED: Shuffle-based warp reduction + warp-tree block reduction.
 * Expected: Best performance (minimal atomics, fast shuffle ops).
 */
__global__ void simulate_ZBC_warp_optimized(
    float* ZBC_sum, 
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

        float r1 = d_r0, r2 = d_r0;
        float integral1 = 0.0f, integral2 = 0.0f;

        const int n_steps = (int)(S1 / d_dt);
        const float exp_adt = d_exp_adt;
        const float sig_st = d_sig_st;

        for (int i = 1; i <= n_steps; i++) {
            const float drift = d_drift_table[i - 1];
            const float G = curand_normal(&local);
            const float sig_G = sig_st * G;

            evolve_hull_white_step(&r1, &integral1, drift, sig_G, exp_adt, d_dt);
            evolve_hull_white_step(&r2, &integral2, drift, -sig_G, exp_adt, d_dt);
        }

        const float P1 = compute_P_HW(S1, S2, r1, d_a, d_sigma, d_P_market, d_f_market);
        const float P2 = compute_P_HW(S1, S2, r2, d_a, d_sigma, d_P_market, d_f_market);

        const float discount1 = __expf(-integral1);
        const float discount2 = __expf(-integral2);

        const float payoff1 = discount1 * fmaxf(P1 - K, 0.0f);
        const float payoff2 = discount2 * fmaxf(P2 - K, 0.0f);

        thread_sum = payoff1 + payoff2;

        states[pid] = local;
    }

    // Warp reduction
    warp_reduce(thread_sum);

    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Block reduction
    if (warp_id == 0) {
        float block_sum = block_reduce(warp_sums, lane, warp_id);
        if (lane == 0) {
            atomicAdd(ZBC_sum, block_sum);
        }
    }
}

#endif // PERF_BENCHMARK_CUH
