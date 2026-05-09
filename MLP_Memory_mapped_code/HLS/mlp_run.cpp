#include "mlp_run.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mlp_weights_data.h"
#include "ap_int.h"

#define LN_EPS    1e-5f
#define FADD_LAT  5

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

        float a0 = b[o], a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f;

        MAC_LOOP: for (int i = 0; i < n_in; i += FADD_LAT) {
#pragma HLS PIPELINE II=1
            a0 += W[o * n_in + i    ] * in[i    ];
            a1 += W[o * n_in + i + 1] * in[i + 1];
            a2 += W[o * n_in + i + 2] * in[i + 2];
            a3 += W[o * n_in + i + 3] * in[i + 3];
            a4 += W[o * n_in + i + 4] * in[i + 4];
        }

        out[o] = a0 + a1 + a2 + a3 + a4;
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

    float s0=0.f, s1=0.f, s2=0.f, s3=0.f, s4=0.f;

    MEAN_LOOP: for (int i = 0; i < n; i += FADD_LAT) {
#pragma HLS PIPELINE II=1
        s0 += in[i    ];
        s1 += in[i + 1];
        s2 += in[i + 2];
        s3 += in[i + 3];
        s4 += in[i + 4];
    }
    float mean = (s0 + s1 + s2 + s3 + s4) / (float)n;

    float v0=0.f, v1=0.f, v2=0.f, v3=0.f, v4=0.f;

    VAR_LOOP: for (int i = 0; i < n; i += FADD_LAT) {
#pragma HLS PIPELINE II=1
        float d0 = in[i    ] - mean;
        float d1 = in[i + 1] - mean;
        float d2 = in[i + 2] - mean;
        float d3 = in[i + 3] - mean;
        float d4 = in[i + 4] - mean;
        v0 += d0 * d0;
        v1 += d1 * d1;
        v2 += d2 * d2;
        v3 += d3 * d3;
        v4 += d4 * d4;
    }

    float var     = (v0 + v1 + v2 + v3 + v4) / (float)n;
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

void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT])
{
#pragma HLS BIND_STORAGE variable=fc1_W type=ROM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=fc2_W type=ROM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=fc3_W type=ROM_2P impl=BRAM

#pragma HLS ARRAY_PARTITION variable=x     type=complete         dim=1

#pragma HLS ARRAY_PARTITION variable=fc1_W type=cyclic factor=5  dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_W type=cyclic factor=5  dim=1
#pragma HLS ARRAY_PARTITION variable=fc3_W type=cyclic factor=5  dim=1

#pragma HLS ARRAY_PARTITION variable=fc1_b type=complete         dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_b type=complete         dim=1
#pragma HLS ARRAY_PARTITION variable=fc3_b type=complete         dim=1

#pragma HLS ARRAY_PARTITION variable=ln1_g type=cyclic factor=40  dim=1
#pragma HLS ARRAY_PARTITION variable=ln1_b type=cyclic factor=40  dim=1
#pragma HLS ARRAY_PARTITION variable=ln2_g type=cyclic factor=40  dim=1
#pragma HLS ARRAY_PARTITION variable=ln2_b type=cyclic factor=40  dim=1

    float h1[MLP_N_H1];
#pragma HLS ARRAY_PARTITION variable=h1 type=cyclic factor=40 dim=1

    float h2[MLP_N_H2];
#pragma HLS ARRAY_PARTITION variable=h2 type=cyclic factor=40 dim=1

    linear_layer(x,  MLP_N_IN,  h1, MLP_N_H1, fc1_W, fc1_b);
    layer_norm_relu(h1, h1, MLP_N_H1, ln1_g, ln1_b);

    linear_layer(h1, MLP_N_H1, h2, MLP_N_H2, fc2_W, fc2_b);
    layer_norm_relu(h2, h2, MLP_N_H2, ln2_g, ln2_b);

    linear_layer(h2, MLP_N_H2, y,  MLP_N_OUT, fc3_W, fc3_b);
}
