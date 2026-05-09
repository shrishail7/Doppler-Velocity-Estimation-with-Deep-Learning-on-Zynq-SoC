#include "main.h"
#include "mlp_run.h"

void MLP(Data_type *Input, Data_type *Output)
{
#pragma HLS INTERFACE m_axi port=Input depth=40 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=Output depth=2 offset=slave bundle=gmem0

#pragma HLS INTERFACE s_axilite port=Input bundle=control
#pragma HLS INTERFACE s_axilite port=Output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    mlp_init();

    Data_type InputA[Input_Size];
#pragma HLS ARRAY_PARTITION variable=InputA type=complete dim=1
    Data_type OutputY[Output_Size];
#pragma HLS ARRAY_PARTITION variable=OutputY type=complete dim=1

    /* ---- Step 1: Burst-read input data from DDR via AXI Master ----- */
    memcpy(InputA, (const Data_type*)Input, Input_Size * sizeof(Data_type));

    /* ---- Step 2: Inference ----------------------------------------- */
    mlp_forward(InputA, OutputY);

    /* ---- Step 3: Burst-write result data back to DDR --------------- */
    memcpy((Data_type*)Output, OutputY, Output_Size * sizeof(Data_type));
}
