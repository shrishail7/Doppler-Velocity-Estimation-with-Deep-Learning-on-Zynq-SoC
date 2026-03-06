

#include "inputs.h"
#include "mlp_run.h"
#include <stdio.h>
#include <math.h>

static inline void init_platform(void) {}
static inline void cleanup_platform(void) {}

static void build_input(const Complex *data, float *x)
{
    for (int i = 0; i < N_SAMPLES; i++) {
        x[2*i]     = (float)data[i].real;
        x[2*i + 1] = (float)data[i].imag;
    }
}

static void print_result(int id, int snr, float p1, float p2, double g1, double g2)
{
    double e1 = (double)p1 - g1, e2 = (double)p2 - g2;
    printf("\r\n+ Sample %2d (SNR %4d dB) +\r\n", id, snr);
    printf("  Pred: [%+.20f, %+.20f]\r\n", (double)p1, (double)p2);
    printf("  GT:   [%+.20f, %+.20f]\r\n", g1, g2);
    printf("  Err:  [%+.20f, %+.20f]\r\n", e1, e2);
}

int main(void)
{
    init_platform();
    printf("MLP Doppler Inference — 20+ Digit Precision\r\n");
    
    if (mlp_init() != 0) { printf("Init failed\r\n"); return -1; }
    
    // Accumulators for RMSE per SNR group (index: 0 = -20, 1 = 0, 2 = 20)
    int count_snr[3] = {0};
    double sum_sq_err_v1[3] = {0.0};
    double sum_sq_err_v2[3] = {0.0};
    
    for (int s = 0; s < NUM_SAMPLES; s++) {
        float x[MLP_N_IN], y[MLP_N_OUT];
        build_input(dataset[s].data, x);
        mlp_forward(x, y);
        
        double pred1 = (double)y[0];
        double pred2 = (double)y[1];
        double gt1 = dataset[s].gt_vel1;
        double gt2 = dataset[s].gt_vel2;
        
        print_result(dataset[s].sample_id, dataset[s].snr, y[0], y[1], gt1, gt2);
        
        // Determine SNR group index
        int snr_idx;
        switch (dataset[s].snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx < 0) continue; // should not happen
        
        // Accumulate squared error of magnitudes (|gt| - |pred|)^2
        double err_mag1 = fabs(gt1) - fabs(pred1);
        double err_mag2 = fabs(gt2) - fabs(pred2);
        sum_sq_err_v1[snr_idx] += err_mag1 * err_mag1;
        sum_sq_err_v2[snr_idx] += err_mag2 * err_mag2;
        count_snr[snr_idx]++;
    }
    
    // Compute and print RMSE for each SNR group
    const char *snr_labels[] = {"-20 dB", "0 dB", "+20 dB"};
    printf("\r\nRMSE (magnitude difference) per SNR group:\r\n");
    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            double rmse_v1 = sqrt(sum_sq_err_v1[i] / count_snr[i]);
            double rmse_v2 = sqrt(sum_sq_err_v2[i] / count_snr[i]);
            printf("SNR %s: Vel1 RMSE = %.20f m/s, Vel2 RMSE = %.20f m/s\r\n",
                   snr_labels[i], rmse_v1, rmse_v2);
        }
    }
    
    cleanup_platform();
    return 0;
}
