//// // ***************************** Working Code **************************
////#include <stdio.h>
////#include <stdlib.h>
////#include <math.h>
////#include "main.h"
////#include "mlp_run.h"
////#include "inputs.h"
////
/////* Sort ascending (simple 2-element swap) */
////static void sort2(float *a)
////{
////    if (a[0] > a[1]) { float tmp = a[0]; a[0] = a[1]; a[1] = tmp; }
////}
////
////static void run_benchmark(float A[Input_Size], float Y[Output_Size])
////{
////    mlp_forward(A, Y);
////}
////
////int main()
////{
////    mlp_init();   /* weights are static — one init is enough */
////
////    /* ── Per-SNR accumulators ─────────────────────────────────────────── */
////    /*  3 SNR groups: index 0=-20dB, 1=0dB, 2=+20dB                      */
////    int   snr_levels[3]   = { -20, 0, 20 };
////    float bench_sse[3]    = { 0.f, 0.f, 0.f };  /* sum of squared errors */
////    float ip_sse   [3]    = { 0.f, 0.f, 0.f };
////    int   snr_count[3]    = {  0,   0,   0  };   /* samples per group     */
////    int   total_errors    = 0;
////
////    printf("\n");
////    printf("=========================================================================\n");
////    printf("  MLP Doppler Estimation — All %d ESPRIT Samples  (T=2 targets)\n",
////           NUM_SAMPLES);
////    printf("  Velocities sorted before RMSE (estimator output is unordered)\n");
////    printf("=========================================================================\n");
////
////    for (int s = 0; s < NUM_SAMPLES; s++)
////    {
////        const ESPRIT_Sample *smp = &dataset[s];
////
////        /* Identify SNR group */
////        int snr_idx = -1;
////        for (int k = 0; k < 3; k++)
////            if (smp->snr == snr_levels[k]) { snr_idx = k; break; }
////
////        /* Build flat real input: [re0,im0, re1,im1, ..., re19,im19] */
////        float A[Input_Size];
////        for (int i = 0; i < N_SAMPLES; i++) {
////            A[2*i]     = (float)smp->data[i].real;
////            A[2*i + 1] = (float)smp->data[i].imag;
////        }
////
////        /* Sorted GT velocities */
////        float gt[2] = { (float)smp->gt_vel1, (float)smp->gt_vel2 };
////        sort2(gt);
////
////        printf("\n+-- Sample %2d | SNR = %+3d dB -------------------------------------------+\n",
////               smp->sample_id, smp->snr);
////        printf("|  GT (sorted)  vel1 = %10.4f   vel2 = %10.4f                      |\n",
////               gt[0], gt[1]);
////        printf("+-------------------------------------------------------------------------+\n");
////
////        /* ── Benchmark path ───────────────────────────────────────────── */
////        float Y_bench[Output_Size];
////        run_benchmark(A, Y_bench);
////        sort2(Y_bench);   /* sort predicted velocities before comparing */
////
////        /* ── Stream / IP path ────────────────────────────────────────── */
////        hls::stream<axis_data> stream_in, stream_out;
////        axis_data pkt;
////        for (int i = 0; i < Input_Size; i++) {
////            pkt.data = A[i];  pkt.keep = -1;
////            pkt.last = (i == Input_Size - 1) ? 1 : 0;
////            stream_in.write(pkt);
////        }
////        MLP(stream_in, stream_out);
////
////        float Y_ip[Output_Size];
////        for (int i = 0; i < Output_Size; i++)
////            Y_ip[i] = stream_out.read().data;
////        sort2(Y_ip);      /* sort predicted velocities before comparing */
////
////        /* ── Per-sample output & match check ─────────────────────────── */
////        const char *labels[2] = {"vel1", "vel2"};
////        int sample_ok = 1;
////        for (int i = 0; i < Output_Size; i++) {
////            float diff  = fabsf(Y_ip[i] - Y_bench[i]);
////            int   match = (diff < 1e-4f);
////            if (!match) { sample_ok = 0; total_errors++; }
////            printf("|  %-4s | Benchmark: %10.4f | IP: %10.4f | %s |\n",
//////            printf("|  %f | Benchmark: %f | IP: %f | %s |\n",
////                   labels[i], Y_bench[i], Y_ip[i],
////                   match ? "  OK  " : " FAIL ");
////        }
////        printf("+-------------------------------------------------------------------------+\n");
////        if (!sample_ok) printf("  *** STREAM MISMATCH above ***\n");
////
////        /* ── Accumulate squared errors vs sorted GT ───────────────────── */
////        if (snr_idx >= 0) {
////            for (int i = 0; i < Output_Size; i++) {
////                float eb = Y_bench[i] - gt[i];
////                float ei = Y_ip[i]    - gt[i];
////                bench_sse[snr_idx] += eb * eb;
////                ip_sse   [snr_idx] += ei * ei;
////            }
////            snr_count[snr_idx]++;
////        }
////    }
////
////    /* ── RMSE Summary ─────────────────────────────────────────────────── */
////    /*  RMSE = sqrt( total_SSE / (N_samples * T_targets) )                */
////    int T = Output_Size;   /* T = 2 targets */
////
////    printf("\n");
////    printf("=========================================================================\n");
////    printf("  RMSE per SNR  [ sqrt(SSE / (N*T)),  N=5 samples, T=%d targets ]\n", T);
////    printf("  Velocities sorted before error — correct for unordered estimators\n");
////    printf("=========================================================================\n");
//////    printf("  %-9s | %14s | %14s | %14s | %s\n","SNR (dB)", "Bench RMSE", "IP RMSE", "Difference", "Match?");
////    printf("  %f | %f | %f | %f | %s\n","SNR (dB)", "Bench RMSE", "IP RMSE", "Difference", "Match?");
////    printf("  ----------|----------------|----------------|----------------|--------\n");
////
////    int rmse_all_match = 1;
////    for (int k = 0; k < 3; k++) {
////        int   n          = snr_count[k];
////        float denom      = (float)(n * T);
////        float rmse_bench = sqrtf(bench_sse[k] / denom);
////        float rmse_ip    = sqrtf(ip_sse   [k] / denom);
////        float diff       = fabsf(rmse_bench - rmse_ip);
////        int   match      = (diff < 1e-4f);
////        if (!match) rmse_all_match = 0;
////
////        printf("  %+9d | %14.6f | %14.6f | %14.8f | %s\n",
////               snr_levels[k], rmse_bench, rmse_ip, diff,
////               match ? "  OK  " : " FAIL ");
////    }
////
////    printf("\n=========================================================================\n");
////    if (total_errors == 0 && rmse_all_match)
////        printf("  ALL %d SAMPLES PASSED & ALL 3 RMSE VALUES MATCH  :)\n", NUM_SAMPLES);
////    else if (total_errors > 0)
////        printf("  STREAM MISMATCHES: %d\n", total_errors);
////    else
////        printf("  RMSE MISMATCH DETECTED\n");
////    printf("=========================================================================\n\n");
////
////    return (total_errors > 0 || !rmse_all_match) ? 1 : 0;
////}
//
//
//
//
//
//
//
///*
// * main_tb_fixed.cpp — float-only, all bits shown
// *
// * Printing:  %.9g   — 9 significant decimal digits, the exact minimum
// *            needed to uniquely round-trip any IEEE-754 single-precision
// *            float (FLT_DECIMAL_DIG = 9).  No bits are hidden or rounded.
// *
// * Accumulation: float throughout — bench_sse / ip_sse are float,
// *               sqrtf / fabsf used everywhere.
// *
// * Velocities sorted before comparison and RMSE.
// */
//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include "main.h"
//#include "mlp_run.h"
//#include "inputs.h"
//
//static void sort2f(float *a)
//{
//    if (a[0] > a[1]) { float t = a[0]; a[0] = a[1]; a[1] = t; }
//}
//
//int main()
//{
//    mlp_init();
//
//    int   snr_levels[3] = { -20, 0, 20 };
//    float bench_sse[3]  = { 0.0f, 0.0f, 0.0f };
//    float ip_sse[3]     = { 0.0f, 0.0f, 0.0f };
//    int   snr_count[3]  = {    0,    0,    0  };
//    int   total_errors  = 0;
//
//    printf("\n");
//    printf("=============================================================================\n");
//    printf("  MLP Doppler Estimation -- All %d ESPRIT Samples  (T=2 targets)\n", NUM_SAMPLES);
//    printf("  Print format: %%.9g (9 sig-digits = all IEEE-754 float bits, no compression)\n");
//    printf("  Accumulation: float only (fabsf / sqrtf)\n");
//    printf("  Velocities sorted before comparison / RMSE\n");
//    printf("=============================================================================\n");
//
//    for (int s = 0; s < NUM_SAMPLES; s++)
//    {
//        const ESPRIT_Sample *smp = &dataset[s];
//
//        /* identify SNR group */
//        int snr_idx = -1;
//        for (int k = 0; k < 3; k++)
//            if (smp->snr == snr_levels[k]) { snr_idx = k; break; }
//
//        /* interleaved real input */
//        float A[Input_Size];
//        for (int i = 0; i < N_SAMPLES; i++) {
//            A[2*i]     = (float)smp->data[i].real;
//            A[2*i + 1] = (float)smp->data[i].imag;
//        }
//
//        /* GT as float, sorted */
//        float gt[2] = { (float)smp->gt_vel1, (float)smp->gt_vel2 };
//        sort2f(gt);
//
//        printf("\n");
//        printf("+-- Sample %d | SNR = %+d dB\n", smp->sample_id, smp->snr);
//        printf("|  GT vel1 (sorted) = %.9g\n", gt[0]);
//        printf("|  GT vel2 (sorted) = %.9g\n", gt[1]);
//        printf("+-----------------------------------------------------------------------------\n");
//
//        /* ── Benchmark ───────────────────────────────────────────────── */
//        float Y_bench[Output_Size];
//        mlp_forward(A, Y_bench);
//        sort2f(Y_bench);
//
//        /* ── Stream / IP ─────────────────────────────────────────────── */
//        hls::stream<axis_data> stream_in, stream_out;
//        axis_data pkt;
//        for (int i = 0; i < Input_Size; i++) {
//            pkt.data = A[i]; pkt.keep = -1;
//            pkt.last = (i == Input_Size - 1) ? 1 : 0;
//            stream_in.write(pkt);
//        }
//        MLP(stream_in, stream_out);
//
//        float Y_ip[Output_Size];
//        for (int i = 0; i < Output_Size; i++)
//            Y_ip[i] = stream_out.read().data;
//        sort2f(Y_ip);
//
//        /* ── Per-output print & verification ─────────────────────────── */
//        const char *vlabel[2] = { "vel1", "vel2" };
//        int sample_ok = 1;
//
//        for (int i = 0; i < Output_Size; i++)
//        {
//            float diff  = fabsf(Y_ip[i] - Y_bench[i]);
//            int   match = (diff < 1e-6f);
//            if (!match) { sample_ok = 0; total_errors++; }
//
//            printf("|  %s\n",          vlabel[i]);
//            printf("|    Benchmark : %.9g\n", Y_bench[i]);
//            printf("|    IP output : %.9g\n", Y_ip[i]);
//            printf("|    Diff      : %.9g  --> %s\n", diff, match ? "OK" : "FAIL");
//
//            /* float-only SSE accumulation */
//            if (snr_idx >= 0) {
//                float eb = Y_bench[i] - gt[i];
//                float ei = Y_ip[i]    - gt[i];
//                bench_sse[snr_idx] += eb * eb;
//                ip_sse   [snr_idx] += ei * ei;
//            }
//        }
//
//        if (snr_idx >= 0) snr_count[snr_idx]++;
//        printf("+-----------------------------------------------------------------------------\n");
//        if (!sample_ok) printf("  *** STREAM MISMATCH in sample above ***\n");
//    }
//
//    /* ── RMSE per SNR ─────────────────────────────────────────────────── */
//    int T = Output_Size;   /* T = 2 */
//
//    printf("\n");
//    printf("=============================================================================\n");
//    printf("  RMSE per SNR  [ sqrtf(SSE / (N*T)),  N=5, T=%d ]\n", T);
//    printf("  Float-only accumulation and sqrtf, printed with %%.9g (all bits)\n");
//    printf("=============================================================================\n");
//
//    int rmse_all_match = 1;
//    for (int k = 0; k < 3; k++)
//    {
//        int   n          = snr_count[k];
//        float denom      = (float)(n * T);
//        float rmse_bench = sqrtf(bench_sse[k] / denom);
//        float rmse_ip    = sqrtf(ip_sse[k]    / denom);
//        float diff       = fabsf(rmse_bench - rmse_ip);
//        int   match      = (diff < 1e-6f);
//        if (!match) rmse_all_match = 0;
//
//        printf("\n  SNR = %+d dB  (N=%d samples, T=%d targets)\n",
//               snr_levels[k], n, T);
//        printf("    Bench RMSE : %.9g\n", rmse_bench);
//        printf("    IP    RMSE : %.9g\n", rmse_ip);
//        printf("    Diff       : %.9g  --> %s\n", diff, match ? "OK" : "FAIL");
//    }
//
//    printf("\n=============================================================================\n");
//    if (total_errors == 0 && rmse_all_match)
//        printf("  ALL %d SAMPLES PASSED & ALL 3 RMSE VALUES MATCH :)\n", NUM_SAMPLES);
//    else if (total_errors > 0)
//        printf("  STREAM MISMATCHES: %d\n", total_errors);
//    else
//        printf("  RMSE MISMATCH DETECTED\n");
//    printf("=============================================================================\n\n");
//
//    return (total_errors > 0 || !rmse_all_match) ? 1 : 0;
//}



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "main.h"
#include "mlp_run.h"
#include "inputs.h"

static void sort2f(float *a)
{
    if (a[0] > a[1]) { float t = a[0]; a[0] = a[1]; a[1] = t; }
}

int main()
{
    mlp_init();

    int   snr_levels[3] = { -20, 0, 20 };
    float bench_sse[3]  = { 0.0f, 0.0f, 0.0f };
    float ip_sse[3]     = { 0.0f, 0.0f, 0.0f };
    int   snr_count[3]  = {    0,    0,    0  };
    int   total_errors  = 0;

    printf("\n");
    printf("=============================================================================\n");
    printf("  MLP Doppler Estimation -- All %d ESPRIT Samples  (T=2 targets)\n", NUM_SAMPLES);
    printf("  Print format: %%.9g (9 sig-digits = all IEEE-754 float bits, no compression)\n");
    printf("  Accumulation: float only (fabsf / sqrtf)\n");
    printf("  Velocities sorted before comparison / RMSE\n");
    printf("=============================================================================\n");

    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        const ESPRIT_Sample *smp = &dataset[s];

        /* identify SNR group */
        int snr_idx = -1;
        for (int k = 0; k < 3; k++)
            if (smp->snr == snr_levels[k]) { snr_idx = k; break; }

        /* interleaved real input */
        float A[Input_Size];
        for (int i = 0; i < N_SAMPLES; i++) {
            A[2*i]     = (float)smp->data[i].real;
            A[2*i + 1] = (float)smp->data[i].imag;
        }

        /* GT as float, sorted */
        float gt[2] = { (float)smp->gt_vel1, (float)smp->gt_vel2 };
        sort2f(gt);

        printf("\n");
        printf("+-- Sample %d | SNR = %+d dB\n", smp->sample_id, smp->snr);
        printf("|  GT vel1 (sorted) = %.9g\n", gt[0]);
        printf("|  GT vel2 (sorted) = %.9g\n", gt[1]);
        printf("+-----------------------------------------------------------------------------\n");

        /* Benchmark */
        float Y_bench[Output_Size];
        mlp_forward(A, Y_bench);
        sort2f(Y_bench);

        /* Memory-Mapped / IP */
        float Y_ip[Output_Size];

        // Pass the flat arrays directly to the top-level IP
        MLP(A, Y_ip);

        sort2f(Y_ip);

        /* Per-output print & verification */
        const char *vlabel[2] = { "vel1", "vel2" };
        int sample_ok = 1;

        for (int i = 0; i < Output_Size; i++)
        {
            float diff  = fabsf(Y_ip[i] - Y_bench[i]);
            int   match = (diff < 1e-6f);
            if (!match) { sample_ok = 0; total_errors++; }

            printf("|  %s\n",          vlabel[i]);
            printf("|    Benchmark : %.9g\n", Y_bench[i]);
            printf("|    IP output : %.9g\n", Y_ip[i]);
            printf("|    Diff      : %.9g  --> %s\n", diff, match ? "OK" : "FAIL");

            /* float-only SSE accumulation */
            if (snr_idx >= 0) {
                float eb = Y_bench[i] - gt[i];
                float ei = Y_ip[i]    - gt[i];
                bench_sse[snr_idx] += eb * eb;
                ip_sse   [snr_idx] += ei * ei;
            }
        }

        if (snr_idx >= 0) snr_count[snr_idx]++;
        printf("+-----------------------------------------------------------------------------\n");
        if (!sample_ok) printf("  *** STREAM MISMATCH in sample above ***\n");
    }

    /* RMSE per SNR */
    int T = Output_Size;   /* T = 2 */

    printf("\n");
    printf("=============================================================================\n");
    printf("  RMSE per SNR  [ sqrtf(SSE / (N*T)),  N=5, T=%d ]\n", T);
    printf("  Float-only accumulation and sqrtf, printed with %%.9g (all bits)\n");
    printf("=============================================================================\n");

    int rmse_all_match = 1;
    for (int k = 0; k < 3; k++)
    {
        int   n          = snr_count[k];
        float denom      = (float)(n * T);
        float rmse_bench = sqrtf(bench_sse[k] / denom);
        float rmse_ip    = sqrtf(ip_sse[k]    / denom);
        float diff       = fabsf(rmse_bench - rmse_ip);
        int   match      = (diff < 1e-6f);
        if (!match) rmse_all_match = 0;

        printf("\n  SNR = %+d dB  (N=%d samples, T=%d targets)\n",
               snr_levels[k], n, T);
        printf("    Bench RMSE : %.9g\n", rmse_bench);
        printf("    IP    RMSE : %.9g\n", rmse_ip);
        printf("    Diff       : %.9g  --> %s\n", diff, match ? "OK" : "FAIL");
    }

    printf("\n=============================================================================\n");
    if (total_errors == 0 && rmse_all_match)
        printf("  ALL %d SAMPLES PASSED & ALL 3 RMSE VALUES MATCH :)\n", NUM_SAMPLES);
    else if (total_errors > 0)
        printf("  STREAM MISMATCHES: %d\n", total_errors);
    else
        printf("  RMSE MISMATCH DETECTED\n");
    printf("=============================================================================\n\n");

    return (total_errors > 0 || !rmse_all_match) ? 1 : 0;
}
