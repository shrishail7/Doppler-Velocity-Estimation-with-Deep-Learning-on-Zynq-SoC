

#include <math.h>
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

/* Build 40 real inputs from 20 complex samples */
static void build_input(const Complex *data, float *x)
{
    for (int i = 0; i < N_SAMPLES; i++) {
        x[2*i]     = (float)data[i].real;
        x[2*i + 1] = (float)data[i].imag;
    }
}

static void sort2_float(float *a, float *b) {
    if (*a > *b) { float tmp = *a; *a = *b; *b = tmp; }
}

int main(void)
{
    init_platform();
    mlp_init();

    /* Timing */
    XTime PS_Start, PS_End, PL_start, PL_end;
    float total_ps_time = 0.0f;
    float total_pl_time = 0.0f;

    /* RMSE accumulators*/
    int   count_snr[3]        = {0};
    float sum_sq_err_ps_v1[3] = {0.0f};
    float sum_sq_err_ps_v2[3] = {0.0f};
    float sum_sq_err_pl_v1[3] = {0.0f};
    float sum_sq_err_pl_v2[3] = {0.0f};

    /* ------------------------------------------------------------------ */
    /* DMA Initialisation                                                  */
    /* ------------------------------------------------------------------ */
    int status;
    XAxiDma_Config *DMA_Config_ACP;
    XAxiDma        DMA_instance_ACP;

    DMA_Config_ACP = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    status = XAxiDma_CfgInitialize(&DMA_instance_ACP, DMA_Config_ACP);
    if (status != XST_SUCCESS) {
        printf("DMA Configuration Failed.\r\n");
        return 0;
    }

    /* ------------------------------------------------------------------ */
    /* Main inference loop                                                 */
    /* ------------------------------------------------------------------ */
    for (int s = 0; s < NUM_SAMPLES; s++) {

        float x[MLP_N_IN];
        float y[MLP_N_OUT];
        float y_PL[MLP_N_OUT];

        /* ---------- PS inference ---------- */
        XTime_SetTime(0);
        XTime_GetTime(&PS_Start);

        build_input(dataset[s].data, x);
        mlp_forward(x, y);

        XTime_GetTime(&PS_End);
        float local_ps_time = (float)(PS_End - PS_Start) / (COUNTS_PER_SECOND / 1000);
        total_ps_time += local_ps_time;

        /* ---------- PL inference via DMA ---------- */
        XTime_SetTime(0);
        XTime_GetTime(&PL_start);

        status = XAxiDma_SimpleTransfer(&DMA_instance_ACP, (UINTPTR)y_PL,
                                        MLP_N_OUT * sizeof(float), XAXIDMA_DEVICE_TO_DMA);
        status = XAxiDma_SimpleTransfer(&DMA_instance_ACP, (UINTPTR)x,
                                        MLP_N_IN  * sizeof(float), XAXIDMA_DMA_TO_DEVICE);

        /* Pooling */
        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;
        while (status != 0x00000002)
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;


        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;
        while (status != 0x00000002)
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;

        XTime_GetTime(&PL_end);
        float local_pl_time = (float)(PL_end - PL_start) / (COUNTS_PER_SECOND / 1000);
        total_pl_time += local_pl_time;

        /* ---------- Mismatch check ---------- */
        for (int i = 0; i < MLP_N_OUT; i++) {
            if (fabsf(y_PL[i] - y[i]) > 0.001f) {
                printf("Mismatch at output %d: ps=%.9g pl=%.9g\r\n", i, y[i], y_PL[i]);
                cleanup_platform();
                return 0;
            }
        }

        /* -------------------------------------------------------------- */
        /* Print: sample number, SNR, then velocity lines                  */
        printf("+ Sample %2d (SNR %4d dB) +\r\n", dataset[s].sample_id, dataset[s].snr);
        for (int i = 0; i < MLP_N_OUT; i++) {
            printf("  for velocity %d\r\n", i + 1);
            printf("  ps = %.9g , pl = %.9g\r\n", y[i], y_PL[i]);
        }
        printf("================================================\r\n\r\n");

        /* ---------- RMSE accumulation (pure float, no double) ---------- */
        int snr_idx;
        switch (dataset[s].snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx < 0) continue;

        /* Sort predictions and ground-truth by magnitude before error */
        float sp1 = y[0],    sp2 = y[1];
        float sp1_pl = y_PL[0], sp2_pl = y_PL[1];
        float sg1 = (float)dataset[s].gt_vel1;
        float sg2 = (float)dataset[s].gt_vel2;

        sort2_float(&sp1,    &sp2);
        sort2_float(&sp1_pl, &sp2_pl);
        sort2_float(&sg1,    &sg2);

        float e_ps1 = fabsf(sg1) - fabsf(sp1);
        float e_ps2 = fabsf(sg2) - fabsf(sp2);
        float e_pl1 = fabsf(sg1) - fabsf(sp1_pl);
        float e_pl2 = fabsf(sg2) - fabsf(sp2_pl);

        sum_sq_err_ps_v1[snr_idx] += e_ps1 * e_ps1;
        sum_sq_err_ps_v2[snr_idx] += e_ps2 * e_ps2;
        sum_sq_err_pl_v1[snr_idx] += e_pl1 * e_pl1;
        sum_sq_err_pl_v2[snr_idx] += e_pl2 * e_pl2;
        count_snr[snr_idx]++;
    }

    /* ------------------------------------------------------------------ */
    /* RMSE summary — float sqrtf, printed with full float precision       */
    /* ------------------------------------------------------------------ */
    printf("\r\nRMSE per SNR group (5 samples each):\r\n");
    printf("%-8s | %-20s | %-20s\r\n", "SNR", "PS RMSE (m/s)", "PL RMSE (m/s)");
    printf("---------|---------------------|---------------------\r\n");

    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            float rmse_ps = sqrtf((sum_sq_err_ps_v1[i] + sum_sq_err_ps_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            float rmse_pl = sqrtf((sum_sq_err_pl_v1[i] + sum_sq_err_pl_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            printf("%-8s | %.9g          | %.9g\r\n",
                   snr_labels[i], rmse_ps, rmse_pl);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Timing summary                                                      */
    /* ------------------------------------------------------------------ */
    printf("\r\nTotal execution time PS (ms): %.9g\r\n", total_ps_time);
    printf("Total execution time PL (ms): %.9g\r\n", total_pl_time);
    printf("Average PS (ms): %.9g\r\n", total_ps_time / (float)NUM_SAMPLES);
    printf("Average PL (ms): %.9g\r\n", total_pl_time / (float)NUM_SAMPLES);
    printf("Speed-up ratio: %.9g\r\n",  total_ps_time  / total_pl_time);

    cleanup_platform();
    return 0;
}
