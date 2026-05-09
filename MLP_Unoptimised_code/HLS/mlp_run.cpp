
#include "mlp_run.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mlp_weights_data.h"
#include "ap_int.h"

#define LN_EPS 1e-5f

//static MLP_Instance g_mlp;
static int g_initialized = 0;

// activation function
static inline float relu_f(float v) { return v > 0.0f ? v : 0.0f; }

// linear layer processing
static inline void linear_layer(const float * __restrict__ in, int n_in, float * __restrict__ out, int n_out, const float * __restrict__ W, const float * __restrict__ b)
{
    for (int o = 0; o < n_out; ++o) {
        float acc = b[o];
        const float *row =  W + o * n_in;
        for (int i = 0; i < n_in; ++i)
            acc += row[i] * in[i];
        out[o] = acc;
    }
}

// normalisation function
static inline void layer_norm_relu(const float * __restrict__ in, float * out,
                                   int n, const float * __restrict__ gamma,
                                   const float * __restrict__ beta)
{
    float mean = 0.0f;

    // mean calculation
    for (int i = 0; i < n; ++i) mean += in[i];
    mean /= (float)n;

    // varience calculation
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + LN_EPS);

    // weight and bias processing using gamma and beta in normalisation
    for (int i = 0; i < n; ++i) {
        float norm = (in[i] - mean) * inv_std;
        float scaled = (float) gamma[i] * norm + (float) beta[i];
        out[i] = relu_f(scaled);
    }
}

int mlp_init(void)
{
	// initialisation
	if(g_initialized) return 0;

    printf("[MLP] Setting up pointers to weights...\r\n");

    g_initialized =1;
    printf("[MLP] Ready.\r\n");
    return 0;
}

void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT])
{
	// used BIND STORAGE to map data to BRAM
#pragma HLS BIND_STORAGE variable=fc1_W type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=fc2_W type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=fc3_W type=ram_2p impl=bram

	// internal variables to store hidden layer 1 and layer 2 output
    float h1[MLP_N_H1];
    float h2[MLP_N_H2];

    // ************************* Processing Hidden layer 1 *************************

    linear_layer(x,MLP_N_IN,h1,MLP_N_H1 , fc1_W , fc1_b);
    layer_norm_relu(h1, h1, MLP_N_H1,ln1_g,ln1_b);

    // ************************* Processing Hidden layer 2 *************************

    linear_layer(h1, MLP_N_H1, h2, MLP_N_H2,fc2_W, fc2_b);
    layer_norm_relu(h2, h2, MLP_N_H2, ln2_g, ln2_b);

    // ************************* Processing Hidden layer 3 *************************

    linear_layer(h2, MLP_N_H2, y, MLP_N_OUT,fc3_W, fc3_b);
}
