#include "common.cuh"
#include "perf_benchmark.cuh"
#include "output.cuh"
#include <vector>

/**
 * Benchmark harness: Runs all three reduction methods and compares performance.
 */

struct BenchmarkResult {
    const char* method;
    float time_ms;
    float throughput_Mpaths_per_sec;
    float price;
};

BenchmarkResult benchmark_kernel(
    const char* method_name,
    void (*kernel)(float*, curandState*, float, float, float, const float*, const float*),
    const float* d_P_market, const float* d_f_market,
    curandState* d_states,
    float S1, float S2, float K,
    int num_runs = 5
) {
    printf("  Benchmarking %s...\n", method_name);

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 2; i++) {
        cudaMemset(d_sum, 0, sizeof(float));
        kernel<<<NB, NTPB>>>(d_sum, d_states, S1, S2, K, d_P_market, d_f_market);
        cudaDeviceSynchronize();
    }

    // Benchmark (average over runs)
    float total_time = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        cudaMemset(d_sum, 0, sizeof(float));
        
        cudaEventRecord(start);
        kernel<<<NB, NTPB>>>(d_sum, d_states, S1, S2, K, d_P_market, d_f_market);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    float avg_time = total_time / num_runs;
    float throughput = (N_PATHS * 2.0f / avg_time) / 1000.0f;

    // Get result
    float h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float price = h_sum / (2.0f * N_PATHS);

    cudaFree(d_sum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("    Time: %.3f ms | Throughput: %.2f M paths/sec | Price: %.8f\n",
           avg_time, throughput, price);

    return {method_name, avg_time, throughput, price};
}

int main() {
    select_gpu();
    printf("\n");
    printf("================================================================================\n");
    printf("REDUCTION METHOD PERFORMANCE BENCHMARK\n");
    printf("================================================================================\n\n");

    // Load market data
    float h_P[N_MAT], h_f[N_MAT];
    load_array(P_FILE, h_P, N_MAT);
    load_array(F_FILE, h_f, N_MAT);

    float *d_P_market, *d_f_market;
    load_market_data_to_device(h_P, h_f, &d_P_market, &d_f_market);
    check_cuda("Load market data");

    compute_constants();

    // Setup RNG
    curandState *d_states;
    cudaMalloc(&d_states, N_PATHS * sizeof(curandState));
    init_rng<<<NB, NTPB>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    check_cuda("Init RNG");

    // Parameters
    float S1 = 5.0f;
    float S2 = 10.0f;
    float K = expf(-0.1f);

    printf("Test Parameters:\n");
    printf("  Option: ZBC(S1=%.1f, S2=%.1f, K=%.6f)\n", S1, S2, K);
    printf("  Paths: %d (x2 antithetic = %d effective)\n", N_PATHS, N_PATHS * 2);
    printf("  Block config: %d threads/block, %d blocks\n", NTPB, NB);
    printf("  Number of benchmark runs: 5 (average taken)\n\n");

    // Run benchmarks
    printf("Running benchmarks...\n\n");

    std::vector<BenchmarkResult> results;

    results.push_back(benchmark_kernel(
        "Naive (direct atomicAdd)",
        simulate_ZBC_naive,
        d_P_market, d_f_market, d_states,
        S1, S2, K
    ));

    results.push_back(benchmark_kernel(
        "Shared Memory Reduction",
        simulate_ZBC_shared_memory,
        d_P_market, d_f_market, d_states,
        S1, S2, K
    ));

    results.push_back(benchmark_kernel(
        "Warp+Block Optimized",
        simulate_ZBC_warp_optimized,
        d_P_market, d_f_market, d_states,
        S1, S2, K
    ));

    // Print summary
    printf("\n");
    printf("================================================================================\n");
    printf("BENCHMARK SUMMARY\n");
    printf("================================================================================\n\n");

    printf("%-30s | %10s | %15s\n", "Method", "Time (ms)", "Throughput (M/s)");
    printf("%-30s-+-----------+-----------------\n", "------------------------------");

    float baseline_time = results[0].time_ms;
    for (const auto& r : results) {
        float speedup = baseline_time / r.time_ms;
        printf("%-30s | %10.3f | %15.2f  (%.2fx)\n",
               r.method, r.time_ms, r.throughput_Mpaths_per_sec, speedup);
    }

    // Validation
    printf("\n");
    printf("================================================================================\n");
    printf("VALIDATION\n");
    printf("================================================================================\n\n");

    float price_diff_01 = fabsf(results[0].price - results[1].price);
    float price_diff_02 = fabsf(results[0].price - results[2].price);

    printf("Price consistency:\n");
    printf("  Naive vs Shared Memory:   %.2e (relative: %.4f%%)\n",
           price_diff_01, 100.0f * price_diff_01 / results[0].price);
    printf("  Naive vs Warp+Block:      %.2e (relative: %.4f%%)\n",
           price_diff_02, 100.0f * price_diff_02 / results[0].price);

    if (price_diff_01 < 1e-6 && price_diff_02 < 1e-6) {
        printf("\nAll methods produce identical results (within numerical precision)\n");
    }

    // Save results to JSON
    FILE* json = fopen("data/benchmark_reductions.json", "w");
    if (json) {
        fprintf(json, "{\n");
        fprintf(json, "  \"benchmark\": \"Reduction Methods Performance\",\n");
        fprintf(json, "  \"parameters\": {\n");
        fprintf(json, "    \"N_PATHS\": %d,\n", N_PATHS);
        fprintf(json, "    \"NTPB\": %d,\n", NTPB);
        fprintf(json, "    \"NB\": %d,\n", NB);
        fprintf(json, "    \"S1\": %.1f,\n", S1);
        fprintf(json, "    \"S2\": %.1f,\n", S2);
        fprintf(json, "    \"K\": %.6f\n", K);
        fprintf(json, "  },\n");
        fprintf(json, "  \"results\": [\n");

        for (size_t i = 0; i < results.size(); i++) {
            fprintf(json, "    {\n");
            fprintf(json, "      \"method\": \"%s\",\n", results[i].method);
            fprintf(json, "      \"time_ms\": %.3f,\n", results[i].time_ms);
            fprintf(json, "      \"throughput_Mpaths_per_sec\": %.2f,\n", results[i].throughput_Mpaths_per_sec);
            fprintf(json, "      \"price\": %.8f\n", results[i].price);
            fprintf(json, "    }%s\n", i < results.size() - 1 ? "," : "");
        }

        fprintf(json, "  ]\n");
        fprintf(json, "}\n");
        fclose(json);
        printf("\nSaved data/benchmark_reductions.json\n");
    }

    // Cleanup
    cudaFree(d_P_market);
    cudaFree(d_f_market);
    cudaFree(d_states);

    printf("\n");
    printf("================================================================================\n");
    printf("BENCHMARK COMPLETE\n");
    printf("================================================================================\n\n");

    return 0;
}
