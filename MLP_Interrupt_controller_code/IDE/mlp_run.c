

#include "mlp_run.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mlp_weights_data.h"

#define LN_EPS 1e-5f

static MLP_Instance g_mlp;

static inline float relu_f(float v) { return v > 0.0f ? v : 0.0f; }

__attribute__((optimize("O3")))
static inline void linear_layer(const float * restrict in, int n_in,
                                float * restrict out, int n_out,
                                const float * restrict W, const float * restrict b)
{
    for (int o = 0; o < n_out; ++o) {
        float acc = b[o];
        const float *row = W + o * n_in;
		#pragma GCC unroll 8
        for (int i = 0; i < n_in; ++i)
            acc += row[i] * in[i];
        out[o] = acc;
    }
}

// Allow in-place operation: out may alias with in
__attribute__((optimize("O3")))
static inline void layer_norm_relu(const float * restrict in, float * out,
                                   int n, const float * restrict gamma,
                                   const float * restrict beta)
{
    float mean = 0.0f;
#pragma GCC unroll 8
    for (int i = 0; i < n; ++i) mean += in[i];
    mean /= (float)n;

    float var = 0.0f;
#pragma GCC unroll 8
    for (int i = 0; i < n; ++i) {
        float d = in[i] - mean;
        var += d * d;
    }

    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + LN_EPS);

#pragma GCC unroll 8
    for (int i = 0; i < n; ++i) {
        float norm = (in[i] - mean) * inv_std;
        float scaled = gamma[i] * norm + beta[i];
        out[i] = relu_f(scaled);
    }
}

int mlp_init(void)
{
    if (g_mlp.initialized) return 0;

    printf("[MLP] Setting up pointers to weights...\r\n");

    // Directly point to the external arrays (defined in mlp_weights_data.h)
    g_mlp.fc1_W = fc1_W;
    g_mlp.fc1_b = fc1_b;
    g_mlp.ln1_g = ln1_g;
    g_mlp.ln1_b = ln1_b;

    g_mlp.fc2_W = fc2_W;
    g_mlp.fc2_b = fc2_b;
    g_mlp.ln2_g = ln2_g;
    g_mlp.ln2_b = ln2_b;

    g_mlp.fc3_W = fc3_W;
    g_mlp.fc3_b = fc3_b;

    g_mlp.initialized = 1;
    printf("[MLP] Ready.\r\n");
    return 0;
}

__attribute__((optimize("O3")))
void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT])
{
    float h1[MLP_N_H1];
    float h2[MLP_N_H2];

    linear_layer(x, MLP_N_IN, h1, MLP_N_H1,
                 g_mlp.fc1_W, g_mlp.fc1_b);
    layer_norm_relu(h1, h1, MLP_N_H1, g_mlp.ln1_g, g_mlp.ln1_b);

    linear_layer(h1, MLP_N_H1, h2, MLP_N_H2,
                 g_mlp.fc2_W, g_mlp.fc2_b);
    layer_norm_relu(h2, h2, MLP_N_H2, g_mlp.ln2_g, g_mlp.ln2_b);

    linear_layer(h2, MLP_N_H2, y, MLP_N_OUT,
                 g_mlp.fc3_W, g_mlp.fc3_b);
}
