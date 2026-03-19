
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <xtime_l.h>
#include <math.h>

#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "inputs.h"
#include "mlp_run.h"

const char *snr_labels[3] = {"-20dB", "0dB", "20dB"};

// generating 40 real inputs from the 20 complex inputs
static void build_input(const Complex *data, float *x)
{
    for (int i = 0; i < N_SAMPLES; i++) {
        x[2*i]     = (float)data[i].real;
        x[2*i + 1] = (float)data[i].imag;
    }
}

// Add this helper function before main()
static void sort2_double(double *a, double *b) {
    if (*a > *b) { double tmp = *a; *a = *b; *b = tmp; }
}

// print function
static void print_result(int id, int snr, float p1, float p2, double g1, double g2)
{
    double e1 = fabs(g1) - fabs((double)p1), e2 =   fabs(g2)- fabs((double)p2);
    printf("\r\n+ Sample %2d (SNR %4d dB) +\r\n", id, snr);
    printf("  Pred: [%+.20f, %+.20f]\r\n", (double)p1, (double)p2);
    printf("  GT:   [%+.20f, %+.20f]\r\n", g1, g2);
    printf("  Err:  [%+.20f, %+.20f]\r\n", e1, e2);
}

int main(void)
{
    init_platform();

    // timing variables
    XTime PS_Start , PS_End;
    XTime_SetTime(0);
    XTime_GetTime(&PS_Start);
    
    // checking initialisation
    if (mlp_init() != 0) { printf("Init failed\r\n"); return -1; }
    
    // Accumulators for RMSE per SNR group (index: 0 = -20, 1 = 0, 2 = 20)
    int count_snr[3] = {0};
    double sum_sq_err_v1[3] = {0.0};
    double sum_sq_err_v2[3] = {0.0};
    
    // iterating all the inputs
    for (int s = 0; s < NUM_SAMPLES; s++) {
        float x[MLP_N_IN], y[MLP_N_OUT];

        // converting 20 comples inputs into 40 real inputs
        build_input(dataset[s].data, x);

        // applying MLP function to input
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
//        double err_mag1 = fabs(gt1) - fabs(pred1);
//        double err_mag2 = fabs(gt2) - fabs(pred2);
        double sp1 = pred1, sp2 = pred2;
        double sg1 = gt1,   sg2 = gt2;
        sort2_double(&sp1, &sp2);   // sort predictions ascending
        sort2_double(&sg1, &sg2);   // sort GT ascending

        double err_mag1 = sg1 - sp1;  // direct diff on sorted values
        double err_mag2 = sg2 - sp2;
        sum_sq_err_v1[snr_idx] += err_mag1 * err_mag1;
        sum_sq_err_v2[snr_idx] += err_mag2 * err_mag2;
        count_snr[snr_idx]++;
    }
    
    XTime_GetTime(&PS_End);



    // evaluating RMSE
    printf("\r\nRMSE per SNR group:\r\n");
    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            double rmse = sqrt((sum_sq_err_v1[i] + sum_sq_err_v2[i]) / (2.0 * count_snr[i]));
            printf("SNR %s: RMSE = %.20f m/s\r\n", snr_labels[i], rmse);
        }
    }
    
    float execution_time = (float)1.0*((PS_End - PS_Start))/(COUNTS_PER_SECOND/1000);

    printf("Average Execution Time per input ( in milli-sec )  = %f \n", execution_time/15);

    cleanup_platform();
    return 0;
}
