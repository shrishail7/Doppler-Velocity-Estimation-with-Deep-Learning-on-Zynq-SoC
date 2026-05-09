#ifndef ESPRIT_H
#define ESPRIT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ap_int.h>

#define N_SAMPLES 20

typedef struct {
    float real;
    float imag;
} Complex;

//  Memory-Mapped
void esprit_hls(Complex *in_mem, float *out_mem);

#endif
