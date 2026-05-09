#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"
#include <ap_int.h>

typedef float Data_type;

typedef hls::axis<Data_type, 0, 0, 0> axis_data;

#define Input_Size  40
#define Output_Size  2

void MLP(hls::stream<axis_data> &Input, hls::stream<axis_data> &Output);

#endif
