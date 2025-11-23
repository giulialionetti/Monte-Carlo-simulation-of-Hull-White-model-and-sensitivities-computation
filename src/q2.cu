#include "common.cuh"

/*
 * Recover theta(t) from forward rates using the Hull-White calibration formula
 * 
 * theta(T) = df/dT + a*f(0,T) + (sigma^2 / 2a) * (1 - exp(-2aT))
 * 
 * We verify that applying this formula to our Monte Carlo f(0,T) 
 * recovers the original piecewise linear theta(t) from equation (7).
 */

/*
 * Compute df/dT using finite differences
 */
float compute_derivative(const float* f, int i, int n, float spacing) {
    if (i == 0) {
        return (f[1] - f[0]) / spacing;
    } else if (i == n - 1) {
        return (f[i] - f[i-1]) / spacing;
    } else {
        return (f[i+1] - f[i-1]) / (2.0f * spacing);
    }
}

/*
 * Recover theta from forward rates
 * theta(T) = df/dT + a*f(0,T) + (sigma^2 / 2a) * (1 - exp(-2aT))
 */
void recover_theta(const float* f, float* theta_recovered, int n_mat) {
    for (int i = 0; i < n_mat; i++) {
        float T = i * H_MAT_SPACING;
        
        // Compute df/dT
        float df_dT = compute_derivative(f, i, n_mat, H_MAT_SPACING);
        
        // Convexity adjustment term
        float convexity = (H_SIGMA * H_SIGMA / (2.0f * H_A)) * 
                         (1.0f - expf(-2.0f * H_A * T));
        
        // Recover theta
        theta_recovered[i] = df_dT + H_A * f[i] + convexity;
    }
}


void print_theta_comparison(const float* theta_original, 
                           const float* theta_recovered, 
                           int n_mat) {
    printf("  T      theta_original   theta_recovered   error\n");
    
    float max_error = 0.0f;
    float sum_error = 0.0f;
    int n_printed = 0;
    
    for (int i = 0; i <= 100; i += 10) {
        float T = i * H_MAT_SPACING;
        float error = fabsf(theta_recovered[i] - theta_original[i]);
        max_error = fmaxf(max_error, error);
        sum_error += error;
        n_printed++;
        
        printf("%5.1f    %.6f         %.6f          %.2e\n",
               T, theta_original[i], theta_recovered[i], error);
    }
    printf("\n");
    
    // Summary
    printf("Max error:  %.2e\n", max_error);
    printf("Mean error: %.2e\n", sum_error / n_printed);
    printf("\nRecovery: %s\n", max_error < 0.01f ? "SUCCESS ✓" : "FAILED ✗");
}

int main() {
    printf("RECOVER THETA FROM FORWARD RATES\n");
    
    // Load data from Q1
    float h_P[N_MAT], h_f[N_MAT];
    load_array(P_FILE, h_P, N_MAT);
    load_array(F_FILE, h_f, N_MAT);
    printf("\n");
    
    // Recover theta and compute original
    float h_theta_recovered[N_MAT];
    float h_theta_original[N_MAT];
    
    recover_theta(h_f, h_theta_recovered, N_MAT);
    
    for (int i = 0; i < N_MAT; i++) {
        float T = i * H_MAT_SPACING;
        h_theta_original[i] = theta_func(T);
    }
    
    // Print comparison
    print_theta_comparison(h_theta_original, h_theta_recovered, N_MAT);
    
    return 0;
}