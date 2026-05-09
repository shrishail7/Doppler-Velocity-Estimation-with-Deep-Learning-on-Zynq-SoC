#ifndef ESPRIT_H
#define ESPRIT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"
#include <ap_int.h>

#define N_SAMPLES 20

typedef float ESPRIT_Dtype;

typedef struct {
    float real;
    float imag;
} Complex;

typedef hls::axis<ESPRIT_Dtype, 0, 0, 0> axis_data;

#define ESPRIT_IP
#ifdef ESPRIT_IP
void esprit_hls(hls::stream<axis_data> &in_stream, hls::stream<axis_data> &out_stream);
#endif

#endif
