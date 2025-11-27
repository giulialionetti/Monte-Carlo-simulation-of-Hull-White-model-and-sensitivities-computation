#ifndef OUTPUT_CUH
#define OUTPUT_CUH

#include <stdio.h>
#include <time.h>


inline FILE* json_open(const char* filename, const char* task_name) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Cannot create %s\n", filename);
        return NULL;
    }
    
    time_t now = time(NULL);
    char* timestamp = ctime(&now);
    timestamp[strlen(timestamp) - 1] = '\0';  
    
    fprintf(f, "{\n");
    fprintf(f, "  \"task\": \"%s\",\n", task_name);
    fprintf(f, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(f, "  \"parameters\": {\n");
    fprintf(f, "    \"N_PATHS\": %d,\n", N_PATHS);
    fprintf(f, "    \"N_STEPS\": %d,\n", N_STEPS);
    fprintf(f, "    \"N_MAT\": %d,\n", N_MAT);
    fprintf(f, "    \"T_FINAL\": %.1f,\n", T_FINAL);
    fprintf(f, "    \"a\": %.2f,\n", H_A);
    fprintf(f, "    \"sigma\": %.2f,\n", H_SIGMA);
    fprintf(f, "    \"r0\": %.4f\n", H_R0);
    fprintf(f, "  },\n");
     
    return f;
}


inline void json_close(FILE* f) {
    fprintf(f, "}\n");
    fclose(f);
}

inline void json_write_array(FILE* f, const char* name, const float* data, int n, 
                             const char* indent = "  ") {
    fprintf(f, "%s\"%s\": [", indent, name);
    for (int i = 0; i < n; i++) {
        if (i % 10 == 0) fprintf(f, "\n%s  ", indent);
        fprintf(f, "%.8f", data[i]);
        if (i < n - 1) fprintf(f, ", ");
    }
    fprintf(f, "\n%s]", indent);
}


inline void json_write_performance(FILE* f, float time_ms, int n_paths, 
                                   const char* indent = "  ") {
    fprintf(f, "%s\"performance\": {\n", indent);
    fprintf(f, "%s  \"simulation_time_ms\": %.2f,\n", indent, time_ms);
    fprintf(f, "%s  \"throughput_Mpaths_per_sec\": %.2f\n", indent, 
            (n_paths / time_ms) / 1000.0f);
    fprintf(f, "%s}", indent);
}

inline void csv_write_timeseries(const char* filename, const char* header,
                                 const float* data, int n, float spacing) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Cannot create %s\n", filename);
        return;
    }
    
    fprintf(f, "T,%s\n", header);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%.4f,%.8f\n", i * spacing, data[i]);
    }
    
    fclose(f);
    printf("Saved %s\n", filename);
}

inline void csv_write_comparison(const char* filename,
                                const float* x, const float* y1, const float* y2,
                                const char* x_name, const char* y1_name, const char* y2_name,
                                int n) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Cannot create %s\n", filename);
        return;
    }
    
    fprintf(f, "%s,%s,%s\n", x_name, y1_name, y2_name);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%.4f,%.8f,%.8f\n", x[i], y1[i], y2[i]);
    }
    
    fclose(f);
    printf("Saved %s\n", filename);
}


void save_q2a_json(const char* filename, float max_error, bool success) {
    FILE* json = json_open(filename, "q2a_results");
    if (!json) return;

    fprintf(json, "  \"error_metrics\": {\n");
    fprintf(json, "    \"max_error\": %.2e,\n", max_error);
    fprintf(json, "    \"success\": %s\n", success ? "true" : "false");
    fprintf(json, "  }\n");

    json_close(json);
    printf("Saved %s\n", filename);
}

void save_q2b_json(const char* filename, float zbc_val, float control_dev, float time_ms, int effective_paths) {
    FILE* json = json_open(filename, "q2b_results");
    if (!json) return;
    
    json_write_performance(json, time_ms, effective_paths);
    
    fprintf(json, ",\n");
    fprintf(json, "  \"results\": {\n");
    fprintf(json, "    \"ZBC_control_variate\": %.8f,\n", zbc_val);
    fprintf(json, "    \"control_deviation\": %.2e\n", control_dev);
    fprintf(json, "  }\n");

    json_close(json);
    printf("Saved %s\n", filename);
}

inline void summary_append(const char* filename, const char* section_title) {
    FILE* f = fopen(filename, "a");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        return;
    }
    
    fprintf(f, "\n");
    fprintf(f, "================================================================================\n");
    fprintf(f, "%s\n", section_title);
    fprintf(f, "================================================================================\n");
    
    fclose(f);
}


inline void summary_init(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Cannot create %s\n", filename);
        return;
    }
    
    time_t now = time(NULL);
    
    fprintf(f, "================================================================================\n");
    fprintf(f, "HULL-WHITE MODEL SIMULATION RESULTS\n");
    fprintf(f, "================================================================================\n");
    fprintf(f, "Generated: %s", ctime(&now));
    fprintf(f, "\n");
    fprintf(f, "Parameters:\n");
    fprintf(f, "  N_PATHS = %d (x2 antithetic = %d effective)\n", N_PATHS, N_PATHS * 2);
    fprintf(f, "  N_STEPS = %d\n", N_STEPS);
    fprintf(f, "  N_MAT = %d\n", N_MAT);
    fprintf(f, "  T_FINAL = %.1f years\n", T_FINAL);
    fprintf(f, "  a = %.2f, sigma = %.2f, r0 = %.4f\n", H_A, H_SIGMA, H_R0);
    
    fclose(f);
    printf("Initialized %s\n", filename);
}

#endif // OUTPUT_CUH