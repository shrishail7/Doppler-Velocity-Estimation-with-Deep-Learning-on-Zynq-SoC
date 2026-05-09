#include <math.h>
#include "inputs.h"
#include "mlp_run.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <xtime_l.h>

#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xil_cache.h"
#include "xmlp.h"

const char *snr_labels[3] = {"-20dB", "0dB", "20dB"};
float PL_Input[MLP_N_IN] __attribute__((aligned(32)));
float PL_Output[MLP_N_OUT] __attribute__((aligned(32)));

static void build_input(const Complex *data, float *x)
{
    for (int i = 0; i < N_SAMPLES; i++) {
        x[2*i]     = (float)data[i].real;
        x[2*i + 1] = (float)data[i].imag;
    }
}

static void sort2_double(double *a, double *b) {
    if (*a > *b) { double tmp = *a; *a = *b; *b = tmp; }
}

int main(void)
{
    init_platform();
    mlp_init();

    // timing variables
    XTime PS_Start, PS_End, PL_start, PL_end;
    float total_ps_time = 0;
    float total_pl_time = 0;

    // Accumulators for RMSE per SNR group (index: 0=-20, 1=0, 2=+20)
    int    count_snr[3]        = {0};
    double sum_sq_err_v1[3]    = {0.0};
    double sum_sq_err_v2[3]    = {0.0};
    double sum_sq_err_pl_v1[3] = {0.0};
    double sum_sq_err_pl_v2[3] = {0.0};

    // ---------------------------------------------------------------
    // Memory-Mapped IP Initialization (Replaces DMA)
    // ---------------------------------------------------------------
    XMlp do_mlp;
    XMlp_Config *cfg = XMlp_LookupConfig(XPAR_XMLP_0_DEVICE_ID);
    if (!cfg) {
        printf("ERROR: Could not find MLP IP in hardware!\n\r");
        return 0;
    }
    int status = XMlp_CfgInitialize(&do_mlp, cfg);
    if (status != XST_SUCCESS) {
        printf("ERROR: MLP IP Configuration Failed.\n\r");
        return 0;
    }

    // ---------------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------------
    for (int s = 0; s < NUM_SAMPLES; s++) {

        float x[MLP_N_IN], y[MLP_N_OUT];
        float y_PL[MLP_N_OUT];
        float local_ps_time = 0, local_pl_time = 0;

        // ---------- PS inference (Software Benchmark) ----------
        build_input(dataset[s].data, x);

        XTime_SetTime(0);
        XTime_GetTime(&PS_Start);

        mlp_forward(x, y);

        XTime_GetTime(&PS_End);
        local_ps_time = (float)1.0 * (PS_End - PS_Start) / (COUNTS_PER_SECOND / 1000);
        total_ps_time += local_ps_time;

        // ---------- PL inference via Memory-Mapped IP ----------

        // 1. Copy data into the aligned DDR buffer
        for(int i = 0; i < MLP_N_IN; i++) {
            PL_Input[i] = x[i];
        }

        XTime_SetTime(0);
        XTime_GetTime(&PL_start);

        // 2. Pass DDR memory addresses to the IP via AXI-Lite
        XMlp_Set_Input_r(&do_mlp, (u32)(UINTPTR)PL_Input);
        XMlp_Set_Output_r(&do_mlp, (u32)(UINTPTR)PL_Output);

        // 3. FLUSH CACHE
        //Xil_DCacheFlushRange((UINTPTR)PL_Input, MLP_N_IN * sizeof(float));

        // 4. Start the IP and wait for it to finish
        XMlp_Start(&do_mlp);
        while (!XMlp_IsDone(&do_mlp));

        // 5. INVALIDATE CACHE
        //Xil_DCacheInvalidateRange((UINTPTR)PL_Output, MLP_N_OUT * sizeof(float));

        XTime_GetTime(&PL_end);

        local_pl_time = (float)1.0 * (PL_end - PL_start) / (COUNTS_PER_SECOND / 1000);
        total_pl_time += local_pl_time;

        // 6. Copy hardware result back into local array for comparison
        for(int i = 0; i < MLP_N_OUT; i++) {
            y_PL[i] = PL_Output[i];
        }

        // ---------- PS vs PL comparison ----------
        int error = 0;
        int i;
        printf("\r\n+ Sample %2d (SNR %4d dB) +\r\n", dataset[s].sample_id, dataset[s].snr);
        for (i = 0; i < MLP_N_OUT; i++) {
            printf("  for velocity %d \n", i + 1);
            printf("  ps = %.9g , pl = %.9g \n\n", y[i], y_PL[i]);
            if (fabs(y_PL[i] - y[i]) > 0.0001) {
                error = 1;
                printf("Mismatch found\n");
                return 0;
            }
        }

        if (error) {
            printf(" i = %d , ps = %.9g , pl = %.9g  \n", i, y[i], y_PL[i]);
        }

        double pred1 = (double)y[0];
        double pred2 = (double)y[1];
        double gt1   = dataset[s].gt_vel1;
        double gt2   = dataset[s].gt_vel2;

        // Determine SNR group index
        int snr_idx;
        switch (dataset[s].snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx < 0) continue;

        double sp1 = pred1, sp2 = pred2;
        double sg1 = gt1,   sg2 = gt2;
        sort2_double(&sp1, &sp2);
        sort2_double(&sg1, &sg2);

        double err_mag1 = fabs(sg1) - fabs(sp1);
        double err_mag2 = fabs(sg2) - fabs(sp2);
        sum_sq_err_v1[snr_idx] += err_mag1 * err_mag1;
        sum_sq_err_v2[snr_idx] += err_mag2 * err_mag2;
        count_snr[snr_idx]++;

        double pred1_pl = (double)y_PL[0];
        double pred2_pl = (double)y_PL[1];
        double sp1_pl   = pred1_pl, sp2_pl = pred2_pl;

        sort2_double(&sp1_pl, &sp2_pl);
        sort2_double(&sg1,    &sg2);

        double err_mag1_pl = fabs(sg1) - fabs(sp1_pl);
        double err_mag2_pl = fabs(sg2) - fabs(sp2_pl);
        sum_sq_err_pl_v1[snr_idx] += err_mag1_pl * err_mag1_pl;
        sum_sq_err_pl_v2[snr_idx] += err_mag2_pl * err_mag2_pl;
    }

    // ---------------------------------------------------------------
    // RMSE summary
    // ---------------------------------------------------------------
    printf("--------------------------------------------------\r\n\n");
    printf("RMSE per SNR group (5 samples each):\r\n");
    printf("%-8s | %-16s | %-16s\r\n", "SNR", "PS RMSE (m/s)", "PL RMSE (m/s)");
    printf("---------|------------------|------------------\r\n");
    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            double rmse_ps = sqrt((sum_sq_err_v1[i]    + sum_sq_err_v2[i])    / (2.0 * count_snr[i]));
            double rmse_pl = sqrt((sum_sq_err_pl_v1[i] + sum_sq_err_pl_v2[i]) / (2.0 * count_snr[i]));
            printf("%-8s | %-16.9g | %-16.9g\r\n", snr_labels[i], rmse_ps, rmse_pl);
        }
    }

    // ---------------------------------------------------------------
    // Execution Times
    // ---------------------------------------------------------------
    printf("\nTotal execution time PS (ms): %.9g \n", total_ps_time);
    printf("Total execution time PL (ms): %.9g \n\n", total_pl_time);

    printf("Average PS (ms): %.9g \n", total_ps_time / 15.0);
    printf("Average PL (ms): %.9g \n\n", total_pl_time / 15.0);

    printf("Speed-up ratio: %.9g \n", total_ps_time / total_pl_time);

    cleanup_platform();
    return 0;
}
