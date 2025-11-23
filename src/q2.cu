#include "common.cuh"

/*
 * Q2a: Recover theta(t) from forward rates using the Hull-White calibration formula
 * 
 * theta(T) = df/dT + a*f(0,T) + (sigma^2 / 2a) * (1 - exp(-2aT))
 * 
 * We verify that applying this formula to our Monte Carlo f(0,T) 
 * recovers the original piecewise linear theta(t) from equation (7).
 */

/*
 * Original theta function (for comparison)
 */
float theta_original(float t) {
    return (t < 5.0f) ? (0.012f + 0.0014f * t) : (0.014f + 0.001f * t);
}

int main() {
    
    printf("   Q2a: RECOVER THETA FROM FORWARD RATES                  \n");


    // Load Q1 results
    float h_P[N_MAT], h_f[N_MAT];
    load_array(P_FILE, h_P, N_MAT);
    load_array(F_FILE, h_f, N_MAT);
    printf("\n");

    // Recover theta using: theta(T) = df/dT + a*f + (sigma^2/2a)*(1 - exp(-2aT))
    float h_theta_recovered[N_MAT];
    float h_theta_original[N_MAT];

    for (int i = 0; i < N_MAT; i++) {
        float T = i * H_MAT_SPACING;

        // Compute df/dT using finite differences
        float df_dT;
        if (i == 0) {
            df_dT = (h_f[1] - h_f[0]) / H_MAT_SPACING;
        } else if (i == N_MAT - 1) {
            df_dT = (h_f[i] - h_f[i-1]) / H_MAT_SPACING;
        } else {
            df_dT = (h_f[i+1] - h_f[i-1]) / (2.0f * H_MAT_SPACING);
        }

        // Convexity adjustment term
        float convexity = (H_SIGMA * H_SIGMA / (2.0f * H_A)) * (1.0f - expf(-2.0f * H_A * T));

        // Recover theta
        h_theta_recovered[i] = df_dT + H_A * h_f[i] + convexity;
        h_theta_original[i] = theta_original(T);
    }

    // Print comparison
    printf("══════════════════════════════════════════════════════════\n");
    printf("   THETA RECOVERY RESULTS                                 \n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  T      theta_original   theta_recovered   error\n");
    printf("──────────────────────────────────────────────────────────\n");

    float max_error = 0.0f;
    float sum_error = 0.0f;

    for (int i = 0; i <= 100; i += 10) {
        float T = i * H_MAT_SPACING;
        float error = fabsf(h_theta_recovered[i] - h_theta_original[i]);
        max_error = fmaxf(max_error, error);
        sum_error += error;

        printf("%5.1f    %.6f         %.6f          %.2e\n",
               T, h_theta_original[i], h_theta_recovered[i], error);
    }
    printf("══════════════════════════════════════════════════════════\n");

    // Summary
    printf("\nMax error:  %.2e\n", max_error);
    printf("Mean error: %.2e\n", sum_error / 11.0f);
    printf("\nRecovery: %s\n", max_error < 0.01f ? "SUCCESS" : "FAILED");

    return 0;
}