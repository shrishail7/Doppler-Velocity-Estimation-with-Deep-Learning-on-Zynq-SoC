#ifndef MLP_RUN_H
#define MLP_RUN_H

#define MLP_N_IN   40
#define MLP_N_H1  400
#define MLP_N_H2  400
#define MLP_N_OUT   2

typedef struct {
    float fc1_W[MLP_N_H1][MLP_N_IN];
    float fc1_b[MLP_N_H1];
    float ln1_g[MLP_N_H1], ln1_b[MLP_N_H1];
    float fc2_W[MLP_N_H2][MLP_N_H1];
    float fc2_b[MLP_N_H2];
    float ln2_g[MLP_N_H2], ln2_b[MLP_N_H2];
    float fc3_W[MLP_N_OUT][MLP_N_H2];
    float fc3_b[MLP_N_OUT];
    int initialized;
} MLP_Instance;

int mlp_init(void);
void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT]);

#endif
