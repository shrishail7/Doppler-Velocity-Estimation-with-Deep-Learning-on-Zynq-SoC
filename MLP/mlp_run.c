#include "mlp_run.h"
#include "mlp_weights_data.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

#define LN_EPS 1e-5f
#define MLP_PRINT printf

static MLP_Instance g_mlp;

// activation function (ignore negative value and make it 0)
static inline float relu_f(float v) { return v > 0.0f ? v : 0.0f; }

// normalisation of layer
static void layer_norm(const float * restrict in, float * restrict out, int n,
                       const float *gamma, const float *beta)
{
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mean += (double)in[i];
    mean /= (double)n;
    for (int i = 0; i < n; i++) { double d = (double)in[i] - mean; var += d * d; }
    var /= (double)n;
    float inv_std = 1.0f / sqrtf((float)var + LN_EPS);
    for (int i = 0; i < n; i++)
        out[i] = gamma[i] * ((in[i] - (float)mean) * inv_std) + beta[i];
}

// linear function to add scale and bias ( y = W*in+b )
static void linear_layer(const float * restrict in, int n_in,
                         float * restrict out, int n_out,
                         const float *W, const float *b)
{
    for (int o = 0; o < n_out; o++) {
        double acc = (double)b[o];
        for (int i = 0; i < n_in; i++)
            acc += (double)W[o * n_in + i] * (double)in[i];
        out[o] = (float)acc;
    }
}


// reading inputs 
int mlp_init(void)
{
    if (g_mlp.initialized) return 0;
    
    MLP_PRINT("[MLP] Loading weights...\r\n");
    
    /* fc1 : for layer 1 */
    for (int i = 0; i < MLP_N_H1; i++)
        for (int j = 0; j < MLP_N_IN; j++)
            g_mlp.fc1_W[i][j] = fc1_W[i * MLP_N_IN + j];
    memcpy(g_mlp.fc1_b, fc1_b, sizeof(float) * MLP_N_H1);
    memcpy(g_mlp.ln1_g, ln1_g, sizeof(float) * MLP_N_H1);
    memcpy(g_mlp.ln1_b, ln1_b, sizeof(float) * MLP_N_H1);
    
    /* fc2 : for layer 2  */
    for (int i = 0; i < MLP_N_H2; i++)
        for (int j = 0; j < MLP_N_H1; j++)
            g_mlp.fc2_W[i][j] = fc2_W[i * MLP_N_H1 + j];
    memcpy(g_mlp.fc2_b, fc2_b, sizeof(float) * MLP_N_H2);
    memcpy(g_mlp.ln2_g, ln2_g, sizeof(float) * MLP_N_H2);
    memcpy(g_mlp.ln2_b, ln2_b, sizeof(float) * MLP_N_H2);
    
    /* fc3 : for layer 3  */
    for (int i = 0; i < MLP_N_OUT; i++)
        for (int j = 0; j < MLP_N_H2; j++)
            g_mlp.fc3_W[i][j] = fc3_W[i * MLP_N_H2 + j];
    memcpy(g_mlp.fc3_b, fc3_b, sizeof(float) * MLP_N_OUT);
    
    g_mlp.initialized = 1;
    MLP_PRINT("[MLP] Ready.\r\n");
    return 0;
}

// running mlp algorithm by passing inputs , weights bias to various layers

void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT])
{
    float h1_pre[MLP_N_H1], h1_ln[MLP_N_H1], h1[MLP_N_H1];
    float h2_pre[MLP_N_H2], h2_ln[MLP_N_H2], h2[MLP_N_H2];
    
    linear_layer(x, MLP_N_IN, h1_pre, MLP_N_H1, &g_mlp.fc1_W[0][0], g_mlp.fc1_b);
    layer_norm(h1_pre, h1_ln, MLP_N_H1, g_mlp.ln1_g, g_mlp.ln1_b);
    for (int i = 0; i < MLP_N_H1; i++) h1[i] = relu_f(h1_ln[i]);
    
    linear_layer(h1, MLP_N_H1, h2_pre, MLP_N_H2, &g_mlp.fc2_W[0][0], g_mlp.fc2_b);
    layer_norm(h2_pre, h2_ln, MLP_N_H2, g_mlp.ln2_g, g_mlp.ln2_b);
    for (int i = 0; i < MLP_N_H2; i++) h2[i] = relu_f(h2_ln[i]);
    
    linear_layer(h2, MLP_N_H2, y, MLP_N_OUT, &g_mlp.fc3_W[0][0], g_mlp.fc3_b);
}
