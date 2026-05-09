
#include "main.h"
#include "mlp_run.h"

void MLP(hls::stream<axis_data> &Input, hls::stream<axis_data> &Output)
{
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=Input
#pragma HLS INTERFACE axis register both port=Output

    /* Host-only banner; completely absent from RTL thanks to
     * the __SYNTHESIS__ guard inside mlp_init.                           */
    mlp_init();

    Data_type InputA[Input_Size];
#pragma HLS ARRAY_PARTITION variable=InputA type=complete dim=1

    Data_type OutputY[Output_Size];
#pragma HLS ARRAY_PARTITION variable=OutputY type=complete dim=1

    axis_data pkt_in, pkt_out;

    /* ---- Stream -> scalar array (1 beat / cycle, II=1) ------------- */
    READ_IN:
    for (int i = 0; i < Input_Size; ++i) {
#pragma HLS PIPELINE II=1
        pkt_in     = Input.read();
        InputA[i]  = pkt_in.data;
    }

    /* ---- Inference --------------------------------------------------- */
    mlp_forward(InputA, OutputY);

    /* ---- Scalar array -> stream (1 beat / cycle, II=1) ------------- */
    WRITE_OUT:
    for (int i = 0; i < Output_Size; ++i) {
#pragma HLS PIPELINE II=1
        pkt_out.data = OutputY[i];
        pkt_out.keep = -1;
        pkt_out.strb = -1;
        pkt_out.last = (i == Output_Size - 1) ? 1 : 0;
        Output.write(pkt_out);
    }
}
