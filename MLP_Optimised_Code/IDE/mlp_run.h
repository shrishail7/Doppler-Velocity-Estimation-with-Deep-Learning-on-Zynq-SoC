
#ifndef MLP_RUN_H
#define MLP_RUN_H

#define MLP_N_IN   40
#define MLP_N_H1  400
#define MLP_N_H2  400
#define MLP_N_OUT   2

typedef struct {
    const float *fc1_W;    // point to external arrays
    const float *fc1_b;
    const float *ln1_g;
    const float *ln1_b;
    const float *fc2_W;
    const float *fc2_b;
    const float *ln2_g;
    const float *ln2_b;
    const float *fc3_W;
    const float *fc3_b;
    int initialized;
} MLP_Instance;

int mlp_init(void);
void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT]);

#endif
