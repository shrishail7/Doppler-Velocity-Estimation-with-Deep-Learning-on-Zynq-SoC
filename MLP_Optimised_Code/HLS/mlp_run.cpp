
#include "mlp_run.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mlp_weights_data.h"
#include "ap_int.h"

#define LN_EPS     1e-5f
#define FADD_LAT   5

static int g_initialized = 0;

static inline float relu_f(float v) { return v > 0.0f ? v : 0.0f; }

static inline void linear_layer(
        const float * __restrict__ in,  int n_in,
        float       * __restrict__ out, int n_out,
        const float * __restrict__ W,
        const float * __restrict__ b)
{
#pragma HLS INLINE

    OUT_LOOP: for (int o = 0; o < n_out; ++o) {
#pragma HLS PIPELINE II=1
        float acc = b[o];
        const float *row = W + o * n_in;

        MAC_LOOP: for (int i = 0; i < n_in; ++i) {
#pragma HLS UNROLL factor=40
            acc += row[i] * in[i];
        }

        out[o] = acc;
    }
}


static inline void layer_norm_relu(
        const float * __restrict__ in,
        float       * __restrict__ out,
        int n,
        const float * __restrict__ gamma,
        const float * __restrict__ beta)
{
#pragma HLS INLINE

    // ── (1) mean ────────────────────────────────────────────────
    float sum_p[FADD_LAT];
#pragma HLS ARRAY_PARTITION variable=sum_p complete dim=1

    for (int k = 0; k < FADD_LAT; ++k) {
#pragma HLS UNROLL
        sum_p[k] = 0.0f;
    }

    MEAN_LOOP: for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        sum_p[i % FADD_LAT] += in[i];
    }
    float mean = sum_p[0] + sum_p[1] + sum_p[2] + sum_p[3] + sum_p[4];
    mean /= (float)n;

    // ── (2) variance ────────────────────────────────────────────
    float var_p[FADD_LAT];
#pragma HLS ARRAY_PARTITION variable=var_p complete dim=1

    for (int k = 0; k < FADD_LAT; ++k) {
#pragma HLS UNROLL
        var_p[k] = 0.0f;
    }

    VAR_LOOP: for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        float d = in[i] - mean;
        var_p[i % FADD_LAT] += d * d;
    }
    float var = var_p[0] + var_p[1] + var_p[2] + var_p[3] + var_p[4];
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + LN_EPS);


    NORM_LOOP: for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        float norm   = (in[i] - mean) * inv_std;
        float scaled = gamma[i] * norm + beta[i];
        out[i]       = relu_f(scaled);
    }
}

int mlp_init(void)
{
    if (g_initialized) return 0;
    printf("[MLP] Setting up pointers to weights...\r\n");
    g_initialized = 1;
    printf("[MLP] Ready.\r\n");
    return 0;
}

// ─────────────────────────────────────────────────────────────────────
//  mlp_forward
// ─────────────────────────────────────────────────────────────────────
void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT])
{
#pragma HLS BIND_STORAGE variable=fc1_W type=ROM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=fc2_W type=ROM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=fc3_W type=ROM_2P impl=BRAM

#pragma HLS ARRAY_PARTITION variable=x     type=complete           dim=1
#pragma HLS ARRAY_PARTITION variable=fc1_W type=cyclic factor=40   dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_W type=cyclic factor=40   dim=1
#pragma HLS ARRAY_PARTITION variable=fc3_W type=cyclic factor=40   dim=1
#pragma HLS ARRAY_PARTITION variable=fc1_b type=complete           dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_b type=complete           dim=1
#pragma HLS ARRAY_PARTITION variable=fc3_b type=complete           dim=1
#pragma HLS ARRAY_PARTITION variable=ln1_g type=cyclic factor=40   dim=1
#pragma HLS ARRAY_PARTITION variable=ln1_b type=cyclic factor=40   dim=1
#pragma HLS ARRAY_PARTITION variable=ln2_g type=cyclic factor=40   dim=1
#pragma HLS ARRAY_PARTITION variable=ln2_b type=cyclic factor=40   dim=1

    float h1[MLP_N_H1];
#pragma HLS ARRAY_PARTITION variable=h1 type=cyclic factor=40 dim=1

    float h2[MLP_N_H2];
#pragma HLS ARRAY_PARTITION variable=h2 type=cyclic factor=40 dim=1

    // ── Layer 1 ────────────────────────────────────────────────
    linear_layer(x,  MLP_N_IN, h1, MLP_N_H1, fc1_W, fc1_b);
    layer_norm_relu(h1, h1, MLP_N_H1, ln1_g, ln1_b);

    // ── Layer 2 ────────────────────────────────────────────────
    linear_layer(h1, MLP_N_H1, h2, MLP_N_H2, fc2_W, fc2_b);
    layer_norm_relu(h2, h2, MLP_N_H2, ln2_g, ln2_b);

    // ── Layer 3 ────────────────────────────────────────────────
    linear_layer(h2, MLP_N_H2, y, MLP_N_OUT, fc3_W, fc3_b);
}


