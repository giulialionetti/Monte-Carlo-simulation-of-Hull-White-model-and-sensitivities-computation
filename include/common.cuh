#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define N_PATHS (1024 * 1024)
#define N_STEPS 1000
#define NTPB 1024
#define NB ((N_PATHS + NTPB - 1) / NTPB)
#define N_MAT 101
#define T_FINAL 10.0f

#if (N_STEPS % (N_MAT - 1)) != 0
    #error "N_STEPS must be evenly divisible by (N_MAT - 1)"
#endif

#define SAVE_STRIDE (N_STEPS / (N_MAT - 1))

// Time step and maturity spacing
const float H_DT = T_FINAL / N_STEPS;
const float H_MAT_SPACING = T_FINAL / (N_MAT - 1);

// Model parameters
const float H_A = 1.0f;
const float H_SIGMA = 0.1f;
const float H_R0 = 0.012f;


#define DATA_DIR "data/"
#define P_FILE DATA_DIR "P.bin"
#define F_FILE DATA_DIR "f.bin"

// ═══════════════════════════════════════════════════════════════
// UTILITY: Error checking
// ═══════════════════════════════════════════════════════════════
inline void check_cuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ═══════════════════════════════════════════════════════════════
// UTILITY: GPU Selection
// ═══════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════
// UTILITY: File I/O
// ═══════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════
// DEVICE FUNCTIONS
// ═══════════════════════════════════════════════════════════════
__device__ inline float theta_func(float t) {
    return (t < 5.0f) ? (0.012f + 0.0014f * t) : (0.014f + 0.001f * t);
}

__device__ inline float B_func(float t, float T, float a) {
    return (1.0f - expf(-a * (T - t))) / a;
}

// ═══════════════════════════════════════════════════════════════
// KERNEL: RNG Initialization
// ═══════════════════════════════════════════════════════════════
__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_PATHS) curand_init(seed, idx, 0, &states[idx]);
}

#endif // COMMON_CUH