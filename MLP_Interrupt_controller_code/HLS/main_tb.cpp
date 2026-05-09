

#include "main.h"
#include "mlp_run.h"
#include "inputs.h"
#include <math.h>
#include <stdio.h>

// sort function
static void sort2(float *a, float *b)
{
    if (*a > *b) { float t = *a; *a = *b; *b = t; }
}

// benchmark function
void MLP_benchmark(float A[Input_Size], float Y[Output_Size])
{
    mlp_init();
    mlp_forward(A, Y);
}

// main function
int main(void)
{
    int total_errors  = 0;
    int sample_errors = 0;

    /* Per-SNR squared-error accumulators                              */
    /* index: 0 = -20 dB,  1 = 0 dB,  2 = +20 dB                    */
    const int    SNR_LEVELS[3]  = { -20, 0, 20 };
    double       sq_err_bm[3]   = { 0.0, 0.0, 0.0 };   /* benchmark  */
    double       sq_err_ip[3]   = { 0.0, 0.0, 0.0 };   /* IP stream  */
    int          n_samp[3]      = { 0,   0,   0   };
    const double GOLDEN_RMSE[3] = { 4.905501, 2.965027, 1.734260 };

    printf("=============================================================\n");
    printf(" Running MLP on %d samples from inputs.h\n", NUM_SAMPLES);
    printf("=============================================================\n\n");

    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        const ESPRIT_Sample *samp = &dataset[s];

        /* Flatten Complex[20] -> float[40]  */
        float A[Input_Size];
        for (int t = 0; t < 20; t++) {
            A[t * 2 + 0] = (float)samp->data[t].real;
            A[t * 2 + 1] = (float)samp->data[t].imag;
        }

        /* ── Benchmark function calling */
        float Y_benchmark[Output_Size];
        MLP_benchmark(A, Y_benchmark);

        /* ── Stream path (AXI-Stream MLP IP) ────────────────────── */
        hls::stream<axis_data> stream_in, stream_out;
        axis_data pkt;

        for (int i = 0; i < Input_Size; i++) {
            pkt.data = A[i];
            pkt.keep = -1;
            pkt.last = (i == Input_Size - 1) ? 1 : 0;
            stream_in.write(pkt);
        }

        // stream function calling
        MLP(stream_in, stream_out);

        /* ── Compare & print (original working format preserved) ─── */
        printf("-------------------------------------------------------------\n");
        printf("Sample %d | SNR %d dB | GT: [%f , %f]\n",
               samp->sample_id, samp->snr,
               samp->gt_vel1, samp->gt_vel2);


        sample_errors = 0;

        // vaariable to store output generated from IP
        float Y_ip[Output_Size];
        for (int i = 0; i < Output_Size; i++) {
            axis_data rd = stream_out.read();
            Y_ip[i] = rd.data;
            printf("           IP out[%d]  :  %f   Benchmark: %f",
                   i, Y_ip[i], Y_benchmark[i]);
            if (fabsf(Y_ip[i] - Y_benchmark[i]) >= 1e-4f) {
                printf("  <-- MISMATCH");
                sample_errors++;
                total_errors++;
            }
            printf("\n");
        }
        if (sample_errors == 0)
            printf("           Status     : PASS\n");
        else
            printf("           Status     : FAIL (%d mismatches)\n", sample_errors);

        /* ── Sort GT and predictions per-sample ──── */

        float gt0 = (float)samp->gt_vel1;
        float gt1 = (float)samp->gt_vel2;
        sort2(&gt0, &gt1);          /* sort GT ascending */

        /* --- benchmark --- */
        float bm0 = Y_benchmark[0];
        float bm1 = Y_benchmark[1];
        sort2(&bm0, &bm1);          /* sort benchmark ascending */

        double e0_bm = (double)(bm0 - gt0);
        double e1_bm = (double)(bm1 - gt1);

        printf("  Sorted GT  : [%f , %f]\n",(double)gt0, (double)gt1);
        printf("  Sorted BM  : [%f , %f]\n",(double)bm0, (double)bm1);

        /* --- IP stream --- */
        float ip0 = Y_ip[0];
        float ip1 = Y_ip[1];
        sort2(&ip0, &ip1);          /* sort IP ascending */

        double e0_ip = (double)(ip0 - gt0);
        double e1_ip = (double)(ip1 - gt1);

        printf("  Sorted IP  : [%f , %f] \n",(double)ip0, (double)ip1);

        /* --- map SNR to index and accumulate squared errors --- */
        int si = -1;
        for (int k = 0; k < 3; k++)
            if (samp->snr == SNR_LEVELS[k]) { si = k; break; }

        // square the error to calculate RMSE
        if (si >= 0) {
            sq_err_bm[si] += e0_bm * e0_bm + e1_bm * e1_bm;
            sq_err_ip[si] += e0_ip * e0_ip + e1_ip * e1_ip;
            n_samp[si]++;
        }
    }

    /* ── RMSE Summary ──────────────────────────────────────────────── */
    printf("\n=============================================================\n");
    printf(" RMSE per SNR  (sort2-matched, both velocity components)\n");
    printf(" Formula : sqrt( sum(e0^2 + e1^2) / (N_samples * 2) )\n");
    printf("-------------------------------------------------------------\n");
    printf("  %6s  %14s  %14s  %14s  %10s\n",
           "SNR", "BM_RMSE", "IP_RMSE", "Golden_RMSE", "Delta_BM");

    for (int k = 0; k < 3; k++) {
        if (n_samp[k] == 0) continue;
        double rmse_bm = sqrt(sq_err_bm[k] / (double)(n_samp[k] * 2));
        double rmse_ip = sqrt(sq_err_ip[k] / (double)(n_samp[k] * 2));
        double delta   = rmse_bm - GOLDEN_RMSE[k];
        printf("  %4d dB  %14.6f  %14.6f  %14.6f  %+10.6f\n", SNR_LEVELS[k], rmse_bm, rmse_ip, GOLDEN_RMSE[k], delta);
    }
    printf("=============================================================\n");

    if (total_errors == 0)
        printf("\n All %d samples PASSED (benchmark == IP stream).\n", NUM_SAMPLES);
    else
        printf("\n FAILED: %d stream-vs-benchmark mismatches.\n", total_errors);

    return (total_errors > 0) ? 1 : 0;
}

