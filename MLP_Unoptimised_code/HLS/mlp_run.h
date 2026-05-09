
#ifndef MLP_RUN_H
#define MLP_RUN_H
#include <ap_fixed.h>
#define MLP_N_IN   40
#define MLP_N_H1  400
#define MLP_N_H2  400
#define MLP_N_OUT   2


int mlp_init(void);
void mlp_forward(const float x[MLP_N_IN], float y[MLP_N_OUT]);

#endif
