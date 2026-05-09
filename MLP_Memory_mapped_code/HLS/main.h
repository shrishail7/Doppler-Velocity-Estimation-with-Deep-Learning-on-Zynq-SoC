#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <ap_int.h>
#include <string.h> // Required for memcpy

typedef float Data_type;

#define Input_Size  40
#define Output_Size  2

// Top-level signature uses memory pointers for AXI4 Master
void MLP(Data_type *Input, Data_type *Output);

#endif
