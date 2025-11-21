#ifndef COMMON_CUH
#define COMMON_CUH


#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>


//utility function to check for CUDA errors
inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// utility function to select the GPU with the most free memory
inline void select_best_gpu() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    int best_device = 0;
    size_t max_free = 0;
    
    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (free_mem > max_free) {
            max_free = free_mem;
            best_device = i;
        }
    }
    
    cudaSetDevice(best_device);
    printf("Using GPU %d (%.2f GB free)\n\n", best_device, max_free / 1e9);
}




#endif // COMMON_CUH