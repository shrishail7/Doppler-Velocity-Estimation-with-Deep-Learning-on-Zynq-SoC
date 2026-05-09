


#include <math.h>
//
#include "inputs.h"
#include "mlp_run.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <xtime_l.h>

#include "platform.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xparameters.h"

const char *snr_labels[3] = {"-20dB", "0dB", "20dB"};

// generating 40 real inputs from the 20 complex inputs
static void build_input(const Complex *data, float *x)
{
    for (int i = 0; i < N_SAMPLES; i++) {
        x[2*i]     = (float)data[i].real;
        x[2*i + 1] = (float)data[i].imag;
    }
}

// sorting algo
static void sort2_double(double *a, double *b) {
    if (*a > *b) { double tmp = *a; *a = *b; *b = tmp; }
}

// print function
static void print_result(int id, int snr, float p1, float p2, double g1, double g2)
{
    double e1 = fabs(g1) - fabs((double)p1), e2 = fabs(g2) - fabs((double)p2);
    printf("  Pred: [%+.20f, %+.20f]\r\n", (double)p1, (double)p2);
    printf("  GT:   [%+.20f, %+.20f]\r\n", g1, g2);
    printf("  Err:  [%+.20f, %+.20f]\r\n", e1, e2);
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
    // DMA Initialisation
    // ---------------------------------------------------------------

    int status;
    XAxiDma_Config *DMA_Config_ACP;
    XAxiDma DMA_instance_ACP;
    DMA_Config_ACP = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    status = XAxiDma_CfgInitialize(&DMA_instance_ACP, DMA_Config_ACP);
    if (status != XST_SUCCESS) {
        printf("DMA Configuration Failed.\t\n");
        return 0;
    }


    // ---------------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------------
    for (int s = 0; s < NUM_SAMPLES; s++) {

        float x[MLP_N_IN], y[MLP_N_OUT];
        float y_PL[MLP_N_OUT];
        float local_ps_time =0 , local_pl_time = 0;

        // ---------- PS inference ----------
        XTime_SetTime(0);
        XTime_GetTime(&PS_Start);
        build_input(dataset[s].data, x);


        mlp_forward(x, y);   // safe now  mlp_init() was called above

        XTime_GetTime(&PS_End);
        local_ps_time = (float)1.0 * (PS_End - PS_Start) / (COUNTS_PER_SECOND / 1000);
        total_ps_time += local_ps_time;

        // ---------- PL inference via DMA  (same pattern as helloworld.c) ----------

        XTime_SetTime(0);
        XTime_GetTime(&PL_start);

        Xil_DCacheFlushRange((UINTPTR)x, (sizeof(float) * MLP_N_IN));
        Xil_DCacheInvalidateRange((UINTPTR)y_PL, (sizeof(float) * MLP_N_OUT));
        status = XAxiDma_SimpleTransfer(&DMA_instance_ACP, (UINTPTR)y_PL,
                                        MLP_N_OUT * sizeof(float), XAXIDMA_DEVICE_TO_DMA);
        status = XAxiDma_SimpleTransfer(&DMA_instance_ACP, (UINTPTR)x,
                                        MLP_N_IN  * sizeof(float), XAXIDMA_DMA_TO_DEVICE);



        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;
        while (status != 0x00000002)
        {
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;
        }

        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;
        while (status != 0x00000002)
        {
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;
        }
        Xil_DCacheInvalidateRange((UINTPTR)y_PL, (sizeof(float) * MLP_N_OUT));
        XTime_GetTime(&PL_end);


        local_pl_time = (float)1.0 * (PL_end - PL_start) / (COUNTS_PER_SECOND / 1000);
        total_pl_time += local_pl_time;

        // ---------- PS vs PL comparison ----------
        int error = 0;
        int i;
        printf("************************************  PS PL comparison  *********************************************\n");
        printf("\r\n+ Sample %2d (SNR %4d dB) +\r\n", dataset[s].sample_id, dataset[s].snr);
        for (i = 0; i < MLP_N_OUT; i++) {
            printf("  for velocity %d \n", i + 1);
            printf("  ps = %f , pl = %f \n\n\n", y[i], y_PL[i]);
            if (fabs(y_PL[i] - y[i]) > 0.0001) {
                error = 1;
                printf("Mismatch found\n");
                return 0;
            }
        }

        if (error) {
            printf(" i = %d , ps = %f , pl = %f  \n", i, y[i], y_PL[i]);
        }

        double pred1 = (double)y[0];
        double pred2 = (double)y[1];
        double gt1   = dataset[s].gt_vel1;
        double gt2   = dataset[s].gt_vel2;

        print_result(dataset[s].sample_id, dataset[s].snr, y[0], y[1], gt1, gt2);

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
    printf("\r\nRMSE per SNR group (5 samples each):\r\n");
    printf("%-8s | %-25s | %-25s\r\n", "SNR", "PS RMSE (m/s)", "PL RMSE (m/s)");
    printf("---------|---------------------------|---------------------------\r\n");
    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            double rmse_ps = sqrt((sum_sq_err_v1[i]    + sum_sq_err_v2[i])    / (2.0 * count_snr[i]));
            double rmse_pl = sqrt((sum_sq_err_pl_v1[i] + sum_sq_err_pl_v2[i]) / (2.0 * count_snr[i]));
            printf("%-8s | %.20f          | %.20f\r\n", snr_labels[i], rmse_ps, rmse_pl);
        }
    }

    printf("**************************** Total Execution time *********************************\n");
    printf("Total execution time for PS in (milli-sec) %f \n", total_ps_time);
    printf("Total execution time for PL in  (milli-sec) %f \n", total_pl_time);

    printf("****************************  Average execution Time ****************************\n");
    printf("Average execution time for PS in  (milli-sec) : %f \n", total_ps_time / 15);
    printf("Average execution time for PL in  (milli-sec) : %f \n", total_pl_time / 15);

    printf("Speed Up ratio %f \n", total_ps_time / total_pl_time);

    cleanup_platform();
    return 0;
}
